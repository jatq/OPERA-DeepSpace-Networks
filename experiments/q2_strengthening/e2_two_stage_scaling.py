"""E2 end-to-end t0/t1 scalability experiment for OPTIONCAST-CAP.

The historical scalable experiments freeze the t0 portfolio to all options.
This module closes that gap: every compared method must construct a legal t0
portfolio and a complete, causal, Z-indexed t1 policy under one *total* wall
clock budget.  The t1 search reuses the V10.10 LNS-2 neighbourhood and the
frozen V10.3 plan/realisation value interfaces, but accepts an explicit
portfolio instead of silently substituting the full-option portfolio.

The outer methods are intentionally conventional and plainly named:

``task_aware_greedy``
    Add the contract with the largest fully evaluated marginal plan value.
``multistart_1swap``
    A strong baseline: greedy plus deterministic singleton starts and
    best-improvement add/delete/change/exchange moves.
``candidate_outer_search``
    Best-first candidate search over the same legal moves plus bounded
    two-exchanges.  It is a candidate algorithm, not assumed to dominate.

All accepted candidates have every observation branch evaluated.  A timeout
while evaluating a candidate discards that partial candidate and returns the
last fully evaluated legal incumbent.  The parent runner launches methods in
fresh processes, writes one JSONL row per scenario/method, and produces a JSON
summary.  Formal experiments are opt-in; the default CLI mode is smoke.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import resource
except ImportError:  # Windows
    resource = None

from experimental.run_equal_budget_benchmark import _paired_summary
from experimental.run_v10_8_scaling import _validate_scaling_extension
from experimental.run_v10_10_lns2_audit import (
    MAX_CANDIDATES_PER_ROUND,
    MAX_ROUNDS,
    SHORTLIST_SIZE,
)
from experimental.run_v10_9_milp_scaling import (
    _bootstrap_ci,
    _pack_posterior_tasks,
    build_congested_scenario,
    score_policy,
)
from experimental.v10_3_optioncast_eval import (
    ContactActivation,
    T1Decision,
    _plan_policy_value,
    decision_plan_value,
    enumerate_portfolios,
    enumerate_t1_decisions,
)
from experimental.v10_3_optioncast_model import (
    CONTROL_MESSAGE_CAPACITY_UNITS,
    CONTROL_MESSAGE_ENERGY_MILLIUNITS,
    CONTROL_MESSAGE_PRICE_MILLIUNITS,
    ContractKind,
    OptioncastScenario,
    PROBABILITY_SCALE,
    T0ContractChoice,
    contact_can_serve_task,
    exercise_fee,
    nominal_contact_capacity_units,
    observation_ids,
    radio_conflict,
    validate_scenario,
    validate_t0_contract_choices,
)
from experimental.v10_4_mismatch_eval import select_best
from q2_strengthening.e1_pricing import (
    BudgetModel,
    EconomicConfig,
    round_half_up,
    staged_budgets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "q2_strengthening/e2_two_stage_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "q2_strengthening/e2_two_stage_summary.json"
METHODS = ("task_aware_greedy", "multistart_1swap", "candidate_outer_search")
SCALES = ((6, 8), (10, 20), (20, 50), (30, 100))
# Frozen E2 evaluation seeds are disjoint from the historical 100--129 suite
# and E1's 20_000--20_029 suite.
FORMAL_SEEDS = tuple(range(30_000, 30_030))
SMOKE_SCALES = ((6, 8),)
SMOKE_SEEDS = (0,)
TOLERANCE = 1e-9
DEFAULT_A = 0.15
DEFAULT_B = 1.0
DEFAULT_S = 0.5
DEFAULT_TOTAL_BUDGET = 255
RUN_VERSION = "e2-two-stage-scaling-v1"
HASHED_SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "q2_strengthening/e1_pricing.py",
    ROOT / "experimental/v10_3_optioncast_model.py",
    ROOT / "experimental/v10_3_optioncast_eval.py",
    ROOT / "experimental/run_v10_9_milp_scaling.py",
    ROOT / "experimental/run_v10_10_lns2_audit.py",
)


class CandidateTimeout(RuntimeError):
    """A portfolio was not completely evaluated before the global deadline."""


class InvalidPortfolio(RuntimeError):
    """A t0 portfolio has no legal mandatory t1 state."""


@dataclass(frozen=True)
class PortfolioEvaluation:
    portfolio: tuple[T0ContractChoice, ...]
    policy: Mapping[str, T1Decision]
    planned_net_milliunits: float
    branch_runtime_seconds: tuple[float, ...]
    candidate_evaluations: int


@dataclass(frozen=True)
class SearchResult:
    method: str
    evaluation: PortfolioEvaluation
    runtime_seconds: float
    timed_out: bool
    timeout_candidate_discards: int
    portfolios_completed: int
    cache_hits: int
    outer_moves_accepted: int
    fallback_used: bool


def adapt_e2_economics(
    scenario: OptioncastScenario,
    *,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    s: float = DEFAULT_S,
    total_budget_milliunits: int = DEFAULT_TOTAL_BUDGET,
) -> OptioncastScenario:
    """Apply the E1 staged price/budget mapping to any E2 scale.

    E1's public ``adapt_scenario`` intentionally invokes the frozen Stage-0
    validator, whose 6x8 size ceiling rejects E2's larger scenarios.  This is
    the same price mapping with the scaling validator substituted; calendars,
    tasks, worlds, forecasts, energy, authorization, and radio constraints are
    unchanged.
    """
    config = EconomicConfig(a, b, s, total_budget_milliunits)
    b0, b1 = staged_budgets(config)
    contacts = []
    for contact in scenario.contacts:
        try:
            reference_fee = contact.exercise_fee_milliunits_by_mode[config.reference_mode_id]
        except KeyError as exc:
            raise ValueError(
                f"contact {contact.contact_id} has no reference mode {config.reference_mode_id}"
            ) from exc
        option_fee = max(1, round_half_up(config.a * contact.commit_fee_milliunits))
        exercise_fees = {
            mode_id: max(
                1,
                round_half_up(
                    config.b
                    * contact.commit_fee_milliunits
                    * (original_fee / reference_fee)
                ),
            )
            for mode_id, original_fee in contact.exercise_fee_milliunits_by_mode.items()
        }
        contacts.append(replace(
            contact,
            option_fee_milliunits=option_fee,
            exercise_fee_milliunits_by_mode=exercise_fees,
        ))
    adapted = replace(
        scenario,
        scenario_id=(
            f"{scenario.scenario_id}-e2-a{a:g}-b{b:g}-s{s:g}-bt{total_budget_milliunits}"
        ),
        contacts=tuple(contacts),
        contract_budget_milliunits=b0,
        exercise_budget_milliunits=b1,
    )
    if 4 <= len(adapted.contacts) <= 6 and 6 <= len(adapted.tasks) <= 8:
        validate_scenario(adapted)
    else:
        _validate_scaling_extension(adapted)
    return adapted


def _peak_rss_bytes() -> int:
    if resource is not None:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    import psutil
    return int(psutil.Process().memory_info().rss)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind each result summary to the exact solver and model sources used."""
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in HASHED_SOURCE_PATHS
    }


def _atomic_write_json(path: Path, value: object, *, overwrite: bool) -> None:
    """Atomically publish a JSON summary without silently replacing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _formal_output_is_in_source(path: Path) -> bool:
    source_directory = (ROOT / "q2_strengthening").resolve()
    return path.resolve().is_relative_to(source_directory)


def _portfolio_key(portfolio: Sequence[T0ContractChoice]) -> tuple[T0ContractChoice, ...]:
    return tuple(sorted(portfolio, key=lambda item: item.contact_id))


def _contract_fee(scenario: OptioncastScenario, choice: T0ContractChoice) -> int:
    contact = next(item for item in scenario.contacts if item.contact_id == choice.contact_id)
    return (
        contact.option_fee_milliunits
        if choice.kind is ContractKind.OPTION
        else contact.commit_fee_milliunits
    )


def _contract_universe(scenario: OptioncastScenario) -> tuple[T0ContractChoice, ...]:
    choices: list[T0ContractChoice] = []
    for contact in sorted(scenario.contacts, key=lambda item: item.contact_id):
        choices.append(T0ContractChoice(contact.contact_id, ContractKind.OPTION))
        choices.extend(
            T0ContractChoice(contact.contact_id, ContractKind.COMMIT, mode.mode_id)
            for mode in sorted(scenario.vcm_modes, key=lambda item: item.mode_id)
        )
    return tuple(choices)


def _mandatory_activations(
    portfolio: Sequence[T0ContractChoice],
) -> tuple[ContactActivation, ...]:
    return tuple(sorted(
        ContactActivation(choice.contact_id, str(choice.commit_mode_id))
        for choice in portfolio if choice.kind is ContractKind.COMMIT
    ))


def _available_activations(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> tuple[ContactActivation, ...]:
    output: list[ContactActivation] = []
    for choice in portfolio:
        if choice.kind is ContractKind.COMMIT:
            output.append(ContactActivation(choice.contact_id, str(choice.commit_mode_id)))
        else:
            output.extend(
                ContactActivation(choice.contact_id, mode.mode_id)
                for mode in scenario.vcm_modes
            )
    return tuple(sorted(output))


def portfolio_decision_violations(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decision: T1Decision,
) -> tuple[str, ...]:
    """Return complete nominal t0/t1 legality violations.

    This validator is portfolio-aware.  In particular, committed contacts are
    mandatory and do not consume the option exercise budget.
    """
    try:
        validate_t0_contract_choices(portfolio, scenario)
    except ValueError as exc:
        return (f"t0:{exc}",)
    contacts = {item.contact_id: item for item in scenario.contacts}
    modes = {item.mode_id: item for item in scenario.vcm_modes}
    tasks = {item.task_id: item for item in scenario.tasks}
    capsules = {item.task_id: item for item in scenario.policy_capsules}
    choices = {item.contact_id: item for item in portfolio}
    issues: list[str] = []
    active_by_contact: dict[str, str] = {}
    for activation in decision.activations:
        if activation.contact_id in active_by_contact:
            issues.append("multiple modes for one contact")
        active_by_contact[activation.contact_id] = activation.mode_id
        choice = choices.get(activation.contact_id)
        if choice is None:
            issues.append("activation without purchased contract")
        elif choice.kind is ContractKind.COMMIT and choice.commit_mode_id != activation.mode_id:
            issues.append("commit mode changed at t1")
    for choice in portfolio:
        if choice.kind is ContractKind.COMMIT and active_by_contact.get(choice.contact_id) != choice.commit_mode_id:
            issues.append("committed contact not activated")
    selected_contacts = [contacts[item.contact_id] for item in decision.activations if item.contact_id in contacts]
    if any(
        radio_conflict(left, right)
        for index, left in enumerate(selected_contacts)
        for right in selected_contacts[index + 1:]
    ):
        issues.append("radio conflict")
    option_ids = {item.contact_id for item in portfolio if item.kind is ContractKind.OPTION}
    exercise = sum(
        exercise_fee(contacts[item.contact_id], item.mode_id)
        for item in decision.activations if item.contact_id in option_ids
    )
    execution_spend = exercise + len(decision.assignments) * CONTROL_MESSAGE_PRICE_MILLIUNITS
    if execution_spend > scenario.exercise_budget_milliunits:
        issues.append("execution budget exceeded")
    activation_energy = sum(
        contacts[item.contact_id].duration_epochs * modes[item.mode_id].energy_milliunits_per_epoch
        for item in decision.activations
        if item.contact_id in contacts and item.mode_id in modes
    )
    total_energy = activation_energy + len(decision.assignments) * CONTROL_MESSAGE_ENERGY_MILLIUNITS
    if total_energy > scenario.energy_budget_milliunits:
        issues.append("energy budget exceeded")
    if len({item.task_id for item in decision.assignments}) != len(decision.assignments):
        issues.append("task assigned more than once")
    if len(decision.assignments) > scenario.final_grant_budget:
        issues.append("final grant budget exceeded")
    used: dict[str, int] = {}
    for assignment in decision.assignments:
        task = tasks.get(assignment.task_id)
        contact = contacts.get(assignment.contact_id)
        mode_id = active_by_contact.get(assignment.contact_id)
        if task is None or contact is None or mode_id is None:
            issues.append("assignment references inactive or unknown object")
            continue
        if not contact_can_serve_task(contact, task):
            issues.append("task lifetime incompatible with contact")
        capsule = capsules.get(task.task_id)
        if capsule is None or not capsule.permits(
            contact_id=contact.contact_id,
            mode_id=mode_id,
            epoch=scenario.t1_epoch,
            contract_version=scenario.contract_version,
        ):
            issues.append("policy capsule rejects assignment")
        used[contact.contact_id] = used.get(contact.contact_id, 0) + task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
    for contact_id, units in used.items():
        if units > nominal_contact_capacity_units(contacts[contact_id], modes[active_by_contact[contact_id]]):
            issues.append("nominal capacity exceeded")
    return tuple(sorted(set(issues)))


def _active_set_legal(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    active: Sequence[ContactActivation],
) -> bool:
    return not portfolio_decision_violations(scenario, portfolio, T1Decision(tuple(sorted(active)), ()))


def _pack_portfolio_tasks(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    observation_id: str,
    active: Sequence[ContactActivation],
) -> T1Decision:
    """Run the shared posterior packer, then enforce E1's staged B1 rule."""
    packed = _pack_posterior_tasks(scenario, observation_id, active)
    contacts = {item.contact_id: item for item in scenario.contacts}
    tasks = {item.task_id: item for item in scenario.tasks}
    option_ids = {item.contact_id for item in portfolio if item.kind is ContractKind.OPTION}
    exercise_spend = sum(
        exercise_fee(contacts[item.contact_id], item.mode_id)
        for item in packed.activations if item.contact_id in option_ids
    )
    control_slots = max(
        0,
        (scenario.exercise_budget_milliunits - exercise_spend)
        // CONTROL_MESSAGE_PRICE_MILLIUNITS,
    )
    if len(packed.assignments) <= control_slots:
        return packed
    # The shared packer has already established capacity/energy legality.  If
    # B1 binds, retain its highest-value selected tasks deterministically.
    kept = sorted(
        packed.assignments,
        key=lambda item: (-tasks[item.task_id].value, item.task_id, item.contact_id),
    )[:control_slots]
    return T1Decision(packed.activations, tuple(sorted(kept)))


def _initial_t1_decision(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    observation_id: str,
    *,
    deadline: float,
) -> tuple[T1Decision, int]:
    """Build a task-aware legal incumbent for one observation."""
    active = _mandatory_activations(portfolio)
    if not _active_set_legal(scenario, portfolio, active):
        raise InvalidPortfolio("mandatory commitments have no joint legal activation")
    incumbent = _pack_portfolio_tasks(scenario, portfolio, observation_id, active)
    incumbent_value = decision_plan_value(scenario, portfolio, incumbent, observation_id)
    evaluations = 0
    remaining = [item for item in _available_activations(scenario, portfolio) if item not in active]
    while remaining and time.perf_counter() < deadline:
        best = incumbent
        best_value = incumbent_value
        best_activation: ContactActivation | None = None
        for activation in remaining:
            if time.perf_counter() >= deadline:
                break
            candidate_active = tuple(sorted(active + (activation,)))
            if not _active_set_legal(scenario, portfolio, candidate_active):
                continue
            candidate = _pack_portfolio_tasks(
                scenario, portfolio, observation_id, candidate_active,
            )
            value = decision_plan_value(scenario, portfolio, candidate, observation_id)
            evaluations += 1
            if value > best_value + TOLERANCE or (
                abs(value - best_value) <= TOLERANCE and repr(candidate) < repr(best)
            ):
                best, best_value, best_activation = candidate, value, activation
        if best_activation is None or best_value <= incumbent_value + TOLERANCE:
            break
        incumbent, incumbent_value = best, best_value
        active = incumbent.activations
        remaining = [item for item in remaining if item.contact_id != best_activation.contact_id]
    return incumbent, evaluations


def lns2_for_portfolio(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    observation_id: str,
    *,
    deadline: float,
    shortlist_size: int = SHORTLIST_SIZE,
    max_candidates_per_round: int = MAX_CANDIDATES_PER_ROUND,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[T1Decision, int]:
    """Portfolio-aware V10.10 remove/add-two LNS with a hard shared deadline."""
    incumbent, evaluations = _initial_t1_decision(
        scenario, portfolio, observation_id, deadline=deadline,
    )
    incumbent_value = decision_plan_value(scenario, portfolio, incumbent, observation_id)
    mandatory_ids = {
        item.contact_id for item in portfolio if item.kind is ContractKind.COMMIT
    }
    available = _available_activations(scenario, portfolio)
    for round_index in range(max_rounds):
        if time.perf_counter() >= deadline:
            break
        round_best = incumbent
        round_best_value = incumbent_value
        removable = tuple(item for item in incumbent.activations if item.contact_id not in mandatory_ids)
        removal_sets: list[tuple[ContactActivation, ...]] = [()]
        removal_sets.extend((item,) for item in removable)
        removal_sets.extend(combinations(removable, 2))
        seen: set[tuple[ContactActivation, ...]] = {incumbent.activations}
        round_evaluations = 0
        exhausted = False
        for removed in removal_sets:
            reduced = tuple(item for item in incumbent.activations if item not in removed)
            ranked: list[tuple[float, str, ContactActivation]] = []
            for activation in available:
                if time.perf_counter() >= deadline:
                    exhausted = True
                    break
                if any(item.contact_id == activation.contact_id for item in reduced):
                    continue
                candidate_active = tuple(sorted(reduced + (activation,)))
                if not _active_set_legal(scenario, portfolio, candidate_active):
                    continue
                candidate = _pack_portfolio_tasks(
                    scenario, portfolio, observation_id, candidate_active,
                )
                value = decision_plan_value(scenario, portfolio, candidate, observation_id)
                ranked.append((value, repr(activation), activation))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            shortlist = tuple(item[2] for item in ranked[:shortlist_size])
            additions: list[tuple[ContactActivation, ...]] = [()]
            additions.extend((item,) for item in shortlist)
            additions.extend(combinations(shortlist, 2))
            for added in additions:
                if time.perf_counter() >= deadline:
                    exhausted = True
                    break
                candidate_active = tuple(sorted(reduced + added))
                if candidate_active in seen or not _active_set_legal(scenario, portfolio, candidate_active):
                    continue
                seen.add(candidate_active)
                candidate = _pack_portfolio_tasks(
                    scenario, portfolio, observation_id, candidate_active,
                )
                value = decision_plan_value(scenario, portfolio, candidate, observation_id)
                evaluations += 1
                round_evaluations += 1
                if value > round_best_value + TOLERANCE or (
                    abs(value - round_best_value) <= TOLERANCE and repr(candidate) < repr(round_best)
                ):
                    round_best, round_best_value = candidate, value
                if round_evaluations >= max_candidates_per_round:
                    exhausted = True
                    break
            if exhausted:
                break
        if round_best_value <= incumbent_value + TOLERANCE:
            break
        incumbent, incumbent_value = round_best, round_best_value
    violations = portfolio_decision_violations(scenario, portfolio, incumbent)
    if violations:
        raise AssertionError(f"portfolio-aware LNS emitted illegal decision: {violations}")
    return incumbent, evaluations


class PortfolioEvaluator:
    """Memoized atomic evaluator for complete t0 portfolios."""

    def __init__(self, scenario: OptioncastScenario, deadline: float):
        self.scenario = scenario
        self.deadline = deadline
        self.cache: dict[tuple[T0ContractChoice, ...], PortfolioEvaluation] = {}
        self.cache_hits = 0
        self.timeout_discards = 0

    def evaluate(self, portfolio: Sequence[T0ContractChoice]) -> PortfolioEvaluation:
        key = _portfolio_key(portfolio)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        validate_t0_contract_choices(key, self.scenario)
        if time.perf_counter() >= self.deadline and key:
            self.timeout_discards += 1
            raise CandidateTimeout("global t0/t1 budget expired before candidate evaluation")
        policy: dict[str, T1Decision] = {}
        runtimes: list[float] = []
        candidate_evaluations = 0
        for observation_id in sorted(observation_ids(self.scenario)):
            if time.perf_counter() >= self.deadline and key:
                self.timeout_discards += 1
                raise CandidateTimeout("partial portfolio discarded at observation boundary")
            started = time.perf_counter()
            decision, count = lns2_for_portfolio(
                self.scenario, key, observation_id, deadline=self.deadline,
            )
            runtimes.append(time.perf_counter() - started)
            candidate_evaluations += count
            policy[observation_id] = decision
        if set(policy) != set(observation_ids(self.scenario)):
            raise AssertionError("only complete Z-indexed policies may enter the cache")
        violations = tuple(
            issue
            for observation_id, decision in policy.items()
            for issue in portfolio_decision_violations(self.scenario, key, decision)
        )
        if violations:
            raise AssertionError(f"accepted portfolio policy is illegal: {violations}")
        result = PortfolioEvaluation(
            key,
            policy,
            _plan_policy_value(self.scenario, key, policy),
            tuple(runtimes),
            candidate_evaluations,
        )
        self.cache[key] = result
        return result


def _better(left: PortfolioEvaluation, right: PortfolioEvaluation) -> bool:
    return (
        left.planned_net_milliunits > right.planned_net_milliunits + TOLERANCE
        or (
            abs(left.planned_net_milliunits - right.planned_net_milliunits) <= TOLERANCE
            and repr(left.portfolio) < repr(right.portfolio)
        )
    )


def _contract_priority(scenario: OptioncastScenario, choice: T0ContractChoice) -> float:
    """Score-blind task-aware ordering; final choices use full policy values."""
    contact = next(item for item in scenario.contacts if item.contact_id == choice.contact_id)
    modes = (
        tuple(mode for mode in scenario.vcm_modes if mode.mode_id == choice.commit_mode_id)
        if choice.kind is ContractKind.COMMIT
        else scenario.vcm_modes
    )
    capsule_by_task = {item.task_id: item for item in scenario.policy_capsules}
    task_scores: list[float] = []
    for task in scenario.tasks:
        if not contact_can_serve_task(contact, task):
            continue
        best_success = 0.0
        for mode in modes:
            capsule = capsule_by_task[task.task_id]
            if not capsule.permits(
                contact_id=contact.contact_id,
                mode_id=mode.mode_id,
                epoch=scenario.t1_epoch,
                contract_version=scenario.contract_version,
            ):
                continue
            success = sum(
                world.probability_units / PROBABILITY_SCALE
                for world in scenario.worlds
                if world.beacon_arrival_epoch_by_contact[contact.contact_id] is not None
                and nominal_contact_capacity_units(contact, mode)
                >= task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
            )
            best_success = max(best_success, success)
        if best_success:
            task_scores.append(task.value * scenario.value_milliunits_per_point * best_success)
    return sum(sorted(task_scores, reverse=True)[: scenario.final_grant_budget]) - _contract_fee(scenario, choice)


def _valid_portfolio(
    scenario: OptioncastScenario, portfolio: Iterable[T0ContractChoice],
) -> tuple[T0ContractChoice, ...] | None:
    key = _portfolio_key(tuple(portfolio))
    try:
        validate_t0_contract_choices(key, scenario)
    except ValueError:
        return None
    if not _active_set_legal(scenario, key, _mandatory_activations(key)):
        return None
    return key


def _one_change_neighbours(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    *,
    limit: int,
) -> tuple[tuple[T0ContractChoice, ...], ...]:
    current = {item.contact_id: item for item in portfolio}
    universe = _contract_universe(scenario)
    by_contact: dict[str, list[T0ContractChoice]] = {}
    for choice in universe:
        by_contact.setdefault(choice.contact_id, []).append(choice)
    raw: set[tuple[T0ContractChoice, ...]] = set()
    # Delete or change contract kind/mode at one selected contact.
    for contact_id in sorted(current):
        reduced = {key: value for key, value in current.items() if key != contact_id}
        valid = _valid_portfolio(scenario, reduced.values())
        if valid is not None:
            raw.add(valid)
        for replacement in by_contact[contact_id]:
            if replacement == current[contact_id]:
                continue
            changed = dict(current)
            changed[contact_id] = replacement
            valid = _valid_portfolio(scenario, changed.values())
            if valid is not None:
                raw.add(valid)
    # Add an absent contact.
    absent = sorted(set(by_contact) - set(current))
    for contact_id in absent:
        for addition in by_contact[contact_id]:
            changed = dict(current)
            changed[contact_id] = addition
            valid = _valid_portfolio(scenario, changed.values())
            if valid is not None:
                raw.add(valid)
    # One-for-one exchange between selected and absent contacts.
    for removed in sorted(current):
        for added in absent:
            for addition in by_contact[added]:
                changed = {key: value for key, value in current.items() if key != removed}
                changed[added] = addition
                valid = _valid_portfolio(scenario, changed.values())
                if valid is not None:
                    raw.add(valid)
    ranked = sorted(
        raw,
        key=lambda item: (
            -sum(_contract_priority(scenario, choice) for choice in item),
            repr(item),
        ),
    )
    return tuple(ranked[:limit])


def _two_exchange_neighbours(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    *,
    shortlist: int = 6,
    limit: int = 32,
) -> tuple[tuple[T0ContractChoice, ...], ...]:
    current = {item.contact_id: item for item in portfolio}
    absent_choices = [
        choice for choice in _contract_universe(scenario)
        if choice.contact_id not in current
    ]
    absent_choices.sort(key=lambda item: (-_contract_priority(scenario, item), repr(item)))
    raw: set[tuple[T0ContractChoice, ...]] = set()
    selected_ids = sorted(current)[:shortlist]
    additions = absent_choices[:shortlist]
    for removed_ids in combinations(selected_ids, 2):
        for added in combinations(additions, 2):
            if added[0].contact_id == added[1].contact_id:
                continue
            changed = {key: value for key, value in current.items() if key not in removed_ids}
            changed[added[0].contact_id] = added[0]
            changed[added[1].contact_id] = added[1]
            valid = _valid_portfolio(scenario, changed.values())
            if valid is not None:
                raw.add(valid)
    ranked = sorted(
        raw,
        key=lambda item: (
            -sum(_contract_priority(scenario, choice) for choice in item), repr(item),
        ),
    )
    return tuple(ranked[:limit])


def _greedy_search(
    evaluator: PortfolioEvaluator, *, candidate_limit_per_round: int = 24,
) -> tuple[PortfolioEvaluation, int]:
    scenario = evaluator.scenario
    incumbent = evaluator.evaluate(())
    accepted = 0
    universe = sorted(
        _contract_universe(scenario),
        key=lambda item: (-_contract_priority(scenario, item), repr(item)),
    )
    while time.perf_counter() < evaluator.deadline:
        present = {item.contact_id for item in incumbent.portfolio}
        best = incumbent
        round_candidates = [choice for choice in universe if choice.contact_id not in present]
        for choice in round_candidates[:candidate_limit_per_round]:
            if choice.contact_id in present or time.perf_counter() >= evaluator.deadline:
                continue
            candidate = _valid_portfolio(scenario, incumbent.portfolio + (choice,))
            if candidate is None:
                continue
            try:
                result = evaluator.evaluate(candidate)
            except CandidateTimeout:
                break
            if _better(result, best):
                best = result
        if not _better(best, incumbent):
            break
        incumbent = best
        accepted += 1
    return incumbent, accepted


def _local_improve(
    evaluator: PortfolioEvaluator,
    start: PortfolioEvaluation,
    *,
    neighbour_limit: int,
) -> tuple[PortfolioEvaluation, int]:
    incumbent = start
    accepted = 0
    while time.perf_counter() < evaluator.deadline:
        best = incumbent
        for portfolio in _one_change_neighbours(
            evaluator.scenario, incumbent.portfolio, limit=neighbour_limit,
        ):
            if time.perf_counter() >= evaluator.deadline:
                break
            try:
                candidate = evaluator.evaluate(portfolio)
            except CandidateTimeout:
                break
            if _better(candidate, best):
                best = candidate
        if not _better(best, incumbent):
            break
        incumbent = best
        accepted += 1
    return incumbent, accepted


def _run_search(
    scenario: OptioncastScenario,
    method: str,
    *,
    budget_seconds: float,
) -> SearchResult:
    if method not in METHODS:
        raise ValueError(f"unknown E2 method: {method}")
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    started = time.perf_counter()
    evaluator = PortfolioEvaluator(scenario, started + budget_seconds)
    # The empty strategy is evaluated first and is the always-legal fallback.
    empty = evaluator.evaluate(())
    incumbent = empty
    moves = 0
    if method == "task_aware_greedy":
        incumbent, moves = _greedy_search(evaluator)
    elif method == "multistart_1swap":
        greedy, greedy_moves = _greedy_search(evaluator, candidate_limit_per_round=16)
        incumbent = greedy
        moves += greedy_moves
        singleton_starts: list[PortfolioEvaluation] = []
        for choice in sorted(
            _contract_universe(scenario),
            key=lambda item: (-_contract_priority(scenario, item), repr(item)),
        )[:8]:
            if time.perf_counter() >= evaluator.deadline:
                break
            valid = _valid_portfolio(scenario, (choice,))
            if valid is None:
                continue
            try:
                singleton_starts.append(evaluator.evaluate(valid))
            except CandidateTimeout:
                break
        for start in sorted(singleton_starts + [greedy], key=lambda item: (-item.planned_net_milliunits, repr(item.portfolio))):
            if time.perf_counter() >= evaluator.deadline:
                break
            improved, count = _local_improve(evaluator, start, neighbour_limit=64)
            moves += count
            if _better(improved, incumbent):
                incumbent = improved
    else:
        greedy, greedy_moves = _greedy_search(evaluator, candidate_limit_per_round=16)
        incumbent = greedy
        moves += greedy_moves
        # Best-first frontier preserves complete candidates only.  The wider
        # neighbourhood is the sole algorithmic difference from the strong
        # multistart baseline.
        frontier: list[PortfolioEvaluation] = [greedy]
        expanded: set[tuple[T0ContractChoice, ...]] = set()
        while frontier and time.perf_counter() < evaluator.deadline:
            frontier.sort(key=lambda item: (-item.planned_net_milliunits, repr(item.portfolio)))
            current = frontier.pop(0)
            if current.portfolio in expanded:
                continue
            expanded.add(current.portfolio)
            neighbours = list(_one_change_neighbours(scenario, current.portfolio, limit=96))
            neighbours.extend(_two_exchange_neighbours(scenario, current.portfolio))
            for portfolio in neighbours:
                if time.perf_counter() >= evaluator.deadline:
                    break
                try:
                    candidate = evaluator.evaluate(portfolio)
                except CandidateTimeout:
                    break
                if _better(candidate, incumbent):
                    incumbent = candidate
                    moves += 1
                if candidate.portfolio not in expanded:
                    frontier.append(candidate)
            frontier = sorted(
                {item.portfolio: item for item in frontier}.values(),
                key=lambda item: (-item.planned_net_milliunits, repr(item.portfolio)),
            )[:6]
    elapsed = time.perf_counter() - started
    timed_out = elapsed >= budget_seconds or evaluator.timeout_discards > 0
    return SearchResult(
        method,
        incumbent,
        elapsed,
        timed_out,
        evaluator.timeout_discards,
        len(evaluator.cache),
        evaluator.cache_hits,
        moves,
        # On timeout the incumbent is always the last completely evaluated
        # legal strategy, whether that incumbent is empty or non-empty.
        timed_out,
    )


def _lookup_validation_times(
    scenario: OptioncastScenario, evaluation: PortfolioEvaluation,
) -> tuple[float, ...]:
    timings = []
    for observation_id in sorted(observation_ids(scenario)):
        started = time.perf_counter()
        decision = evaluation.policy[observation_id]
        violations = portfolio_decision_violations(scenario, evaluation.portfolio, decision)
        timings.append(time.perf_counter() - started)
        if violations:
            raise AssertionError(f"stored online action failed validation: {violations}")
    return tuple(timings)


def _run_worker(
    *,
    method: str,
    seed: int,
    contact_count: int,
    task_count: int,
    budget_seconds: float,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    s: float = DEFAULT_S,
    total_budget_milliunits: int = DEFAULT_TOTAL_BUDGET,
) -> dict[str, object]:
    setup_started = time.perf_counter()
    base_scenario = build_congested_scenario(seed, contact_count, task_count)
    scenario = adapt_e2_economics(
        base_scenario,
        a=a,
        b=b,
        s=s,
        total_budget_milliunits=total_budget_milliunits,
    )
    setup_seconds = time.perf_counter() - setup_started
    result = _run_search(scenario, method, budget_seconds=budget_seconds)
    evaluation = result.evaluation
    violations = {
        observation_id: list(portfolio_decision_violations(scenario, evaluation.portfolio, decision))
        for observation_id, decision in evaluation.policy.items()
        if portfolio_decision_violations(scenario, evaluation.portfolio, decision)
    }
    online = _lookup_validation_times(scenario, evaluation)
    net = score_policy(scenario, evaluation.portfolio, evaluation.policy)
    denominator = sum(task.value for task in scenario.tasks) * scenario.value_milliunits_per_point
    affordable_singletons = [
        choice for choice in _contract_universe(scenario)
        if _valid_portfolio(scenario, (choice,)) is not None
    ]
    portfolio_json = [
        {
            "contact_id": item.contact_id,
            "kind": item.kind.value,
            "commit_mode_id": item.commit_mode_id,
        }
        for item in evaluation.portfolio
    ]
    return {
        "status": "ok",
        "method": method,
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "contact_count": contact_count,
        "task_count": task_count,
        "budget_seconds_total_strategy": budget_seconds,
        "economic_config": {
            "a": a,
            "b": b,
            "s": s,
            "total_budget_milliunits": total_budget_milliunits,
            "b0_contract_milliunits": scenario.contract_budget_milliunits,
            "b1_execution_milliunits": scenario.exercise_budget_milliunits,
        },
        "common_setup_seconds_excluded": setup_seconds,
        "strategy_runtime_seconds": result.runtime_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "planned_net_milliunits": evaluation.planned_net_milliunits,
        "scored_net_milliunits": net,
        "scored_net_vbs": net / denominator * 100.0 if denominator else 0.0,
        "t0_portfolio": portfolio_json,
        "t0_contract_count": len(evaluation.portfolio),
        "t0_contract_fee_milliunits": sum(_contract_fee(scenario, item) for item in evaluation.portfolio),
        "affordable_singleton_option_count": sum(
            item.kind is ContractKind.OPTION for item in affordable_singletons
        ),
        "affordable_singleton_commit_count": sum(
            item.kind is ContractKind.COMMIT for item in affordable_singletons
        ),
        "t1_branch_runtime_median_seconds": median(evaluation.branch_runtime_seconds),
        "t1_branch_runtime_p95_seconds": float(np.quantile(evaluation.branch_runtime_seconds, 0.95)),
        "online_lookup_validation_median_seconds": median(online),
        "online_lookup_validation_p95_seconds": float(np.quantile(online, 0.95)),
        "inner_candidate_evaluations": evaluation.candidate_evaluations,
        "outer_portfolios_completed": result.portfolios_completed,
        "outer_cache_hits": result.cache_hits,
        "outer_moves_accepted": result.outer_moves_accepted,
        "timed_out": result.timed_out,
        "timeout_candidate_discards": result.timeout_candidate_discards,
        "fallback_used": result.fallback_used,
        "legal": not violations,
        "violations_by_observation": violations,
        "observation_count": len(evaluation.policy),
    }


def _invoke_worker(
    method: str,
    *,
    seed: int,
    contact_count: int,
    task_count: int,
    budget_seconds: float,
    a: float,
    b: float,
    s: float,
    total_budget_milliunits: int,
) -> dict[str, object]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    command = [
        sys.executable,
        "-m",
        "q2_strengthening.e2_two_stage_scaling",
        "--worker",
        method,
        "--seed",
        str(seed),
        "--contacts",
        str(contact_count),
        "--tasks",
        str(task_count),
        "--budget-seconds",
        str(budget_seconds),
        "--a",
        str(a),
        "--b",
        str(b),
        "--s",
        str(s),
        "--total-budget",
        str(total_budget_milliunits),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=budget_seconds + 60.0,
    )
    return json.loads(completed.stdout)


def exact_small_crosscheck(
    *,
    seed: int = 0,
    budget_seconds: float = 2.0,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    s: float = DEFAULT_S,
    total_budget_milliunits: int = DEFAULT_TOTAL_BUDGET,
) -> dict[str, object]:
    """Compare the candidate search with a tiny exact t0/t1 problem.

    The contract budget is reduced to the cheapest single option.  This keeps
    the exact universe small while preserving the same model and evaluator.
    """
    from dataclasses import replace

    base = adapt_e2_economics(
        build_congested_scenario(seed, 4, 6),
        a=a,
        b=b,
        s=s,
        total_budget_milliunits=total_budget_milliunits,
    )
    cheapest = min(contact.option_fee_milliunits for contact in base.contacts)
    scenario = replace(
        base,
        scenario_id=f"{base.scenario_id}-exact-check",
        contract_budget_milliunits=cheapest,
        option_count_budget=1,
    )
    exact_candidates = []
    exact_started = time.perf_counter()
    for portfolio in enumerate_portfolios(scenario):
        decisions = tuple(
            decision for decision in enumerate_t1_decisions(scenario, portfolio)
            if not portfolio_decision_violations(scenario, portfolio, decision)
        )
        if not decisions:
            continue
        policy = {
            observation_id: select_best(
                decisions,
                lambda decision, observation_id=observation_id: decision_plan_value(
                    scenario, portfolio, decision, observation_id,
                ),
            )
            for observation_id in sorted(observation_ids(scenario))
        }
        exact_candidates.append((portfolio, policy))
    exact_portfolio, exact_policy = select_best(
        exact_candidates,
        lambda item: _plan_policy_value(scenario, item[0], item[1]),
    )
    exact_value = _plan_policy_value(scenario, exact_portfolio, exact_policy)
    exact_runtime = time.perf_counter() - exact_started
    heuristic = _run_search(
        scenario, "candidate_outer_search", budget_seconds=budget_seconds,
    )
    heuristic_value = heuristic.evaluation.planned_net_milliunits
    if heuristic_value > exact_value + TOLERANCE:
        raise AssertionError("heuristic exceeded the same-information exact optimum")
    violations = {
        observation_id: portfolio_decision_violations(
            scenario, heuristic.evaluation.portfolio, decision,
        )
        for observation_id, decision in heuristic.evaluation.policy.items()
    }
    if any(violations.values()):
        raise AssertionError(f"heuristic cross-check policy is illegal: {violations}")
    return {
        "scenario_id": scenario.scenario_id,
        "exact_portfolio_count": len(exact_candidates),
        "exact_planned_net_milliunits": exact_value,
        "heuristic_planned_net_milliunits": heuristic_value,
        "absolute_gap_milliunits": exact_value - heuristic_value,
        "heuristic_not_above_exact": True,
        "exact_runtime_seconds": exact_runtime,
        "heuristic_runtime_seconds": heuristic.runtime_seconds,
    }


def _summarise(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for contact_count, task_count in sorted({
        (int(row["contact_count"]), int(row["task_count"])) for row in rows
    }):
        selected = [
            row for row in rows
            if int(row["contact_count"]) == contact_count
            and int(row["task_count"]) == task_count
        ]
        by_method = {
            method: [row for row in selected if row["method"] == method]
            for method in METHODS
        }
        method_summary = {}
        for method, method_rows in by_method.items():
            successful = [row for row in method_rows if row.get("status", "ok") == "ok"]
            values = [float(row["scored_net_vbs"]) for row in successful]
            if not successful:
                method_summary[method] = {
                    "seed_count": 0,
                    "failure_count": len(method_rows),
                    "failure_rate": 1.0 if method_rows else None,
                    "scored_net_vbs_mean": None,
                    "scored_net_vbs_median": None,
                    "scored_net_vbs_mean_95pct_bootstrap_ci": None,
                    "runtime_median_seconds": None,
                    "runtime_p95_seconds": None,
                    "peak_rss_median_bytes": None,
                    "timeout_rate": None,
                    "fallback_rate": None,
                    "legal_rate": 0.0 if method_rows else None,
                }
                continue
            method_summary[method] = {
                "seed_count": len(successful),
                "failure_count": len(method_rows) - len(successful),
                "failure_rate": (len(method_rows) - len(successful)) / len(method_rows),
                "scored_net_vbs_mean": float(np.mean(values)),
                "scored_net_vbs_median": median(values),
                "scored_net_vbs_mean_95pct_bootstrap_ci": list(
                    _bootstrap_ci(values, seed=20261201 + contact_count + task_count + METHODS.index(method)),
                ),
                "runtime_median_seconds": median(float(row["strategy_runtime_seconds"]) for row in successful),
                "runtime_p95_seconds": float(np.quantile([row["strategy_runtime_seconds"] for row in successful], 0.95)),
                "peak_rss_median_bytes": median(int(row["peak_rss_bytes"]) for row in successful),
                "timeout_rate": sum(bool(row["timed_out"]) for row in successful) / len(successful),
                "fallback_rate": sum(bool(row["fallback_used"]) for row in successful) / len(successful),
                "legal_rate": sum(bool(row["legal"]) for row in method_rows) / len(method_rows),
            }
        paired_rows = []
        indexed = {
            (int(row["seed"]), str(row["method"])): row for row in selected
        }
        for seed in sorted({int(row["seed"]) for row in selected}):
            if all(
                (seed, method) in indexed
                and indexed[(seed, method)].get("status", "ok") == "ok"
                for method in METHODS
            ):
                paired_rows.append({
                    method: float(indexed[(seed, method)]["scored_net_vbs"])
                    for method in METHODS
                })
        comparisons = (
            {
                "candidate_minus_multistart_1swap": _paired_summary(
                    paired_rows,
                    "candidate_outer_search",
                    "multistart_1swap",
                    seed=20261301 + contact_count + task_count,
                ),
                "multistart_1swap_minus_greedy": _paired_summary(
                    paired_rows,
                    "multistart_1swap",
                    "task_aware_greedy",
                    seed=20261401 + contact_count + task_count,
                ),
            }
            if paired_rows
            else {
                "candidate_minus_multistart_1swap": {"paired_seed_count": 0},
                "multistart_1swap_minus_greedy": {"paired_seed_count": 0},
            }
        )
        summaries.append({
            "contact_count": contact_count,
            "task_count": task_count,
            "methods": method_summary,
            "paired_comparisons_vbs_points": comparisons,
        })
    return summaries


def run(
    *,
    scales: Sequence[tuple[int, int]],
    seeds: Sequence[int],
    budget_seconds: float,
    rows_path: Path,
    exact_check: bool = False,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    s: float = DEFAULT_S,
    total_budget_milliunits: int = DEFAULT_TOTAL_BUDGET,
    overwrite: bool = False,
) -> dict[str, object]:
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    # Validate economics before opening an artifact.
    EconomicConfig(a, b, s, total_budget_milliunits).validate(BudgetModel.STAGED)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace existing artifact: {rows_path}")
    rows: list[dict[str, object]] = []
    with rows_path.open("w", encoding="utf-8") as handle:
        for contact_count, task_count in scales:
            for seed in seeds:
                # Rotate order to reduce systematic server-load ordering bias.
                offset = seed % len(METHODS)
                method_order = METHODS[offset:] + METHODS[:offset]
                for method in method_order:
                    try:
                        row = _invoke_worker(
                            method,
                            seed=seed,
                            contact_count=contact_count,
                            task_count=task_count,
                            budget_seconds=budget_seconds,
                            a=a,
                            b=b,
                            s=s,
                            total_budget_milliunits=total_budget_milliunits,
                        )
                    except (subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
                        row = {
                            "status": "error",
                            "method": method,
                            "seed": seed,
                            "scenario_id": f"worker-failed-s{seed}-c{contact_count}-t{task_count}",
                            "contact_count": contact_count,
                            "task_count": task_count,
                            "budget_seconds_total_strategy": budget_seconds,
                            "economic_config": {
                                "a": a,
                                "b": b,
                                "s": s,
                                "total_budget_milliunits": total_budget_milliunits,
                            },
                            "legal": False,
                            "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                            "fallback_used": False,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:2000],
                        }
                    row["worker_order"] = list(method_order)
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
    return {
        "run_version": RUN_VERSION,
        "study": "E2 complete t0/t1 two-stage equal-total-wall-clock scaling",
        "budget": {
            "mode": "total_wall_clock_per_scenario_method",
            "seconds": budget_seconds,
            "includes": [
                "initial_solution",
                "t0_outer_search",
                "all_observation_branch_t1_search",
                "portfolio_cache_construction",
            ],
            "excludes": ["common deterministic scenario construction", "fresh worker startup"],
            "thread_limits": 1,
        },
        "causal_boundary": "t0 sees scenario and P_plan; t1 policy is indexed only by public Z; W is scoring-only",
        "statistical_unit": "one seeded scenario; Z branches are not independent samples",
        "methods": list(METHODS),
        "economic_config": {
            "model": "staged",
            "a": a,
            "b": b,
            "s": s,
            "total_budget_milliunits": total_budget_milliunits,
            "rounding": "decimal_half_up_to_integer_milliunit",
        },
        "scales": [list(item) for item in scales],
        "seeds": list(seeds),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "processor": platform.processor(),
        },
        "rows_path": str(rows_path),
        "rows_sha256": _sha256(rows_path),
        "source_sha256": _source_hashes(),
        "rows": rows,
        "summary_by_scale": _summarise(rows),
        "small_exact_crosscheck": exact_small_crosscheck(
            a=a,
            b=b,
            s=s,
            total_budget_milliunits=total_budget_milliunits,
        ) if exact_check else None,
        "claim_guardrail": (
            "LNS-2 and the outer searches are bounded heuristics. Only the reduced 4x6 cross-check is exact; "
            "large-scale runs must not be described as globally optimal."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="one 6x8 development scenario (default)")
    mode.add_argument("--formal", action="store_true", help="all preregistered scales and 30 frozen seeds")
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--exact-check", action="store_true")
    parser.add_argument("--a", type=float, default=DEFAULT_A)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--s", type=float, default=DEFAULT_S)
    parser.add_argument("--total-budget", type=int, default=DEFAULT_TOTAL_BUDGET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--contacts", type=int)
    parser.add_argument("--tasks", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.seed is None or args.contacts is None or args.tasks is None:
            parser.error("worker mode requires --seed, --contacts, and --tasks")
        budget = 30.0 if args.budget_seconds is None else args.budget_seconds
        row = _run_worker(
            method=args.worker,
            seed=args.seed,
            contact_count=args.contacts,
            task_count=args.tasks,
            budget_seconds=budget,
            a=args.a,
            b=args.b,
            s=args.s,
            total_budget_milliunits=args.total_budget,
        )
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))
        return 0
    formal = bool(args.formal)
    if formal and (
        _formal_output_is_in_source(args.rows_output)
        or _formal_output_is_in_source(args.summary_output)
    ):
        parser.error(
            "formal artifacts must be written outside the q2_strengthening source directory"
        )
    if args.summary_output.exists() and not args.overwrite:
        parser.error(f"refusing to replace existing artifact: {args.summary_output}")
    budget = (30.0 if formal else 0.25) if args.budget_seconds is None else args.budget_seconds
    result = run(
        scales=SCALES if formal else SMOKE_SCALES,
        seeds=FORMAL_SEEDS if formal else SMOKE_SEEDS,
        budget_seconds=budget,
        rows_path=args.rows_output,
        exact_check=args.exact_check,
        a=args.a,
        b=args.b,
        s=args.s,
        total_budget_milliunits=args.total_budget,
        overwrite=args.overwrite,
    )
    _atomic_write_json(
        args.summary_output,
        result,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["summary_by_scale"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
