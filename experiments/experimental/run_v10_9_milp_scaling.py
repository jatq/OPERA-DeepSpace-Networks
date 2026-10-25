"""V10.9 congestion-aware scalability experiment for OPTIONCAST-CAP.

The experiment is additive and leaves the frozen V10.3--V10.5 artifacts
untouched.  It addresses the main limitation of V10.8 by introducing
overlapping contacts, binding budgets and partial policy-capsule whitelists.

For every public observation Z, ``joint_milp_ec`` solves a sparse binary
surrogate with posterior expected capacities.  The resulting *fixed causal
policy* is then scored by the frozen realization state machine under P_score.
Consequently, a reported MIP gap certifies only the surrogate, never the
original stochastic protocol objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from experimental.run_v10_8_scaling import (
    _activation_legal,
    _validate_scaling_extension,
    build_scaling_scenario,
    score_policy,
)
from experimental.v10_3_optioncast_eval import (
    ContactActivation,
    T1Decision,
    TaskAssignment,
    decision_plan_value,
)
from experimental.v10_3_optioncast_model import (
    CONTROL_MESSAGE_CAPACITY_UNITS,
    CONTROL_MESSAGE_ENERGY_MILLIUNITS,
    CONTROL_MESSAGE_PRICE_MILLIUNITS,
    OptioncastScenario,
    T0ContractChoice,
    beacon_arrival_deadline_epoch,
    contact_can_serve_task,
    contact_capacity_units,
    exercise_fee,
    observation_ids,
    posterior_world_weights,
    radio_conflict,
    validate_t0_contract_choices,
)
from experimental.v10_4_mismatch_eval import full_option_portfolio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experimental/v10_9_milp_scaling_summary.json"
SCALES = ((10, 20), (20, 50), (30, 100))
# Seeds 0--9 were used during implementation; 100--129 are the frozen
# score-blind evaluation set.
SEEDS = tuple(range(100, 130))


@dataclass(frozen=True)
class MilpPolicyResult:
    policy: Mapping[str, T1Decision]
    raw_policy: Mapping[str, T1Decision]
    statuses: tuple[str, ...]
    runtimes: tuple[float, ...]
    mip_gaps: tuple[float, ...]
    node_counts: tuple[int, ...]
    variable_counts: tuple[int, ...]
    constraint_counts: tuple[int, ...]
    fallback_count: int


def build_congested_scenario(seed: int, contact_count: int, task_count: int) -> OptioncastScenario:
    """Create a score-blind stress family with real radio/authorization pressure."""
    base = build_scaling_scenario(seed, contact_count, task_count)
    # Roughly one quarter of the horizon is simultaneously visible.  This
    # creates a stable, non-zero conflict density as the instance grows.
    duration = 2 * max(2, math.ceil(contact_count / 4)) + 1
    contacts = tuple(
        replace(contact, start_epoch=8 + 2 * index, end_epoch=8 + 2 * index + duration)
        for index, contact in enumerate(base.contacts)
    )
    contact_ids = tuple(contact.contact_id for contact in contacts)
    last_end = max(contact.end_epoch for contact in contacts)

    tasks = []
    capsules = []
    capsule_template = base.policy_capsules[0]
    for index, task in enumerate(base.tasks):
        # Three deterministic lifetime classes create heterogeneous eligibility
        # without inspecting P_score or any method's result.
        release = 0 if index % 3 == 0 else 8 + 2 * (index % max(1, contact_count // 2))
        deadline = last_end - 2 * (index % max(2, contact_count // 5))
        deadline = max(deadline, release + duration + 1)
        new_task = replace(task, release_epoch=release, deadline_epoch=deadline)
        tasks.append(new_task)
        eligible = [
            contact.contact_id for contact in contacts
            if contact_can_serve_task(contact, new_task)
        ]
        # A deterministic 50--70% subset models agency/policy restrictions.
        stride = 2 + (index + seed) % 2
        whitelist = tuple(
            cid for rank, cid in enumerate(eligible)
            if (rank + index + seed) % stride != 0
        )
        if not whitelist and eligible:
            whitelist = (eligible[(index + seed) % len(eligible)],)
        capsules.append(replace(
            capsule_template,
            capsule_id=f"v109-cap:{new_task.task_id}",
            task_id=new_task.task_id,
            contact_whitelist=whitelist,
            expires_epoch=deadline,
        ))

    worlds = []
    for world in base.worlds:
        arrivals = {
            contact.contact_id: (
                None if world.beacon_arrival_epoch_by_contact[contact.contact_id] is None
                else contact.start_epoch - base.validation_latency_epochs
            )
            for contact in contacts
        }
        worlds.append(replace(world, beacon_arrival_epoch_by_contact=arrivals))

    max_nonoverlap = max(1, math.ceil(contact_count / max(1, math.ceil(duration / 2))))
    typical_fee = median(
        exercise_fee(contact, mode.mode_id)
        for contact in contacts for mode in base.vcm_modes
    )
    typical_energy = median(
        contact.duration_epochs * mode.energy_milliunits_per_epoch
        for contact in contacts for mode in base.vcm_modes
    )
    scenario = replace(
        base,
        scenario_id=f"v109-congested-s{seed}-c{contact_count}-t{task_count}",
        contacts=contacts,
        tasks=tuple(tasks),
        worlds=tuple(worlds),
        policy_capsules=tuple(capsules),
        exercise_budget_milliunits=max(1, int(0.72 * max_nonoverlap * typical_fee)),
        energy_budget_milliunits=max(
            CONTROL_MESSAGE_ENERGY_MILLIUNITS,
            int(0.72 * max_nonoverlap * typical_energy)
            + CONTROL_MESSAGE_ENERGY_MILLIUNITS * max(2, math.ceil(task_count * 0.18)),
        ),
        final_grant_budget=max(2, math.ceil(task_count * 0.18)),
    )
    _validate_scaling_extension(scenario)
    validate_t0_contract_choices(full_option_portfolio(scenario), scenario)
    if conflict_density(scenario) <= 0:
        raise ValueError("congested scenario unexpectedly contains no radio conflicts")
    if any(not capsule.contact_whitelist for capsule in scenario.policy_capsules):
        raise ValueError("every task must retain at least one whitelisted contact")
    return scenario


def conflict_density(scenario: OptioncastScenario) -> float:
    pairs = len(scenario.contacts) * (len(scenario.contacts) - 1) // 2
    if pairs == 0:
        return 0.0
    conflicts = sum(
        radio_conflict(left, right)
        for index, left in enumerate(scenario.contacts)
        for right in scenario.contacts[index + 1:]
    )
    return conflicts / pairs


def _posterior_terms(
    scenario: OptioncastScenario, observation_id: str,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    weights = posterior_world_weights(scenario, observation_id, planning=True)
    expected_capacity: dict[tuple[str, str], float] = {}
    success_probability: dict[tuple[str, str], float] = {}
    for contact in scenario.contacts:
        for mode in scenario.vcm_modes:
            key = (contact.contact_id, mode.mode_id)
            expected_capacity[key] = sum(
                weights[world.world_id] * contact_capacity_units(contact, mode, world)
                for world in scenario.worlds
            )
            probability = 0.0
            for world in scenario.worlds:
                arrival = world.beacon_arrival_epoch_by_contact[contact.contact_id]
                valid_arrival = (
                    arrival is not None
                    and arrival >= scenario.t1_epoch + scenario.uplink_latency_epochs
                    and arrival <= beacon_arrival_deadline_epoch(scenario, contact)
                )
                if valid_arrival and contact_capacity_units(contact, mode, world) > 0:
                    probability += weights[world.world_id]
            success_probability[key] = probability
    return expected_capacity, success_probability


def _eligible_triplets(scenario: OptioncastScenario) -> list[tuple[str, str, str]]:
    capsules = {capsule.task_id: capsule for capsule in scenario.policy_capsules}
    output = []
    for task in scenario.tasks:
        capsule = capsules[task.task_id]
        for contact in scenario.contacts:
            if not contact_can_serve_task(contact, task):
                continue
            for mode in scenario.vcm_modes:
                if capsule.permits(
                    contact_id=contact.contact_id,
                    mode_id=mode.mode_id,
                    epoch=contact.start_epoch,
                    contract_version=scenario.contract_version,
                ):
                    output.append((task.task_id, contact.contact_id, mode.mode_id))
    return output


def solve_observation_milp(
    scenario: OptioncastScenario,
    observation_id: str,
    *,
    time_limit_seconds: float,
    mip_rel_gap: float,
) -> tuple[T1Decision, dict[str, float | int | str]]:
    """Solve the expected-capacity binary surrogate for one public Z."""
    contacts = {contact.contact_id: contact for contact in scenario.contacts}
    tasks = {task.task_id: task for task in scenario.tasks}
    modes = {mode.mode_id: mode for mode in scenario.vcm_modes}
    activation_keys = [
        (contact.contact_id, mode.mode_id)
        for contact in scenario.contacts for mode in scenario.vcm_modes
    ]
    triplets = _eligible_triplets(scenario)
    activation_index = {key: index for index, key in enumerate(activation_keys)}
    assignment_index = {
        key: len(activation_keys) + index for index, key in enumerate(triplets)
    }
    variable_count = len(activation_keys) + len(triplets)
    expected_capacity, success_probability = _posterior_terms(scenario, observation_id)

    objective = np.zeros(variable_count, dtype=float)
    for key, index in activation_index.items():
        objective[index] = exercise_fee(contacts[key[0]], key[1])
    for key, index in assignment_index.items():
        task_id, contact_id, mode_id = key
        gross = tasks[task_id].value * scenario.value_milliunits_per_point
        objective[index] = -(gross * success_probability[(contact_id, mode_id)] - CONTROL_MESSAGE_PRICE_MILLIUNITS)

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        row = len(lower)
        for column, coefficient in coefficients.items():
            if coefficient:
                row_indices.append(row)
                col_indices.append(column)
                data.append(float(coefficient))
        lower.append(lb)
        upper.append(ub)

    # One VCM per contact.
    for contact in scenario.contacts:
        add_constraint({activation_index[(contact.contact_id, mode.mode_id)]: 1 for mode in scenario.vcm_modes}, -np.inf, 1)
    # Shared spacecraft radio.
    for left_index, left in enumerate(scenario.contacts):
        for right in scenario.contacts[left_index + 1:]:
            if radio_conflict(left, right):
                coefficients = {
                    activation_index[(left.contact_id, mode.mode_id)]: 1 for mode in scenario.vcm_modes
                }
                coefficients.update({
                    activation_index[(right.contact_id, mode.mode_id)]: 1 for mode in scenario.vcm_modes
                })
                add_constraint(coefficients, -np.inf, 1)
    add_constraint({
        activation_index[key]: exercise_fee(contacts[key[0]], key[1]) for key in activation_keys
    }, -np.inf, scenario.exercise_budget_milliunits)
    energy_coefficients = {
        activation_index[key]: contacts[key[0]].duration_epochs * modes[key[1]].energy_milliunits_per_epoch
        for key in activation_keys
    }
    energy_coefficients.update({index: CONTROL_MESSAGE_ENERGY_MILLIUNITS for index in assignment_index.values()})
    add_constraint(energy_coefficients, -np.inf, scenario.energy_budget_milliunits)
    # Assignment implies its exact contact-mode activation.
    for key, index in assignment_index.items():
        add_constraint({index: 1, activation_index[(key[1], key[2])]: -1}, -np.inf, 0)
    for task in scenario.tasks:
        indices = [index for key, index in assignment_index.items() if key[0] == task.task_id]
        if indices:
            add_constraint({index: 1 for index in indices}, -np.inf, 1)
    # Posterior expected capacity, including one control unit per assignment.
    for key in activation_keys:
        indices = [
            index for triplet, index in assignment_index.items()
            if triplet[1:] == key
        ]
        coefficients = {
            index: tasks[triplets[index - len(activation_keys)][0]].payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
            for index in indices
        }
        coefficients[activation_index[key]] = -expected_capacity[key]
        add_constraint(coefficients, -np.inf, 0)
    add_constraint({index: 1 for index in assignment_index.values()}, -np.inf, scenario.final_grant_budget)

    matrix = coo_matrix((data, (row_indices, col_indices)), shape=(len(lower), variable_count)).tocsr()
    started = time.perf_counter()
    result = milp(
        objective,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": mip_rel_gap, "presolve": True},
    )
    elapsed = time.perf_counter() - started
    if result.x is None:
        decision = T1Decision((), ())
        status = "no_incumbent"
    else:
        activations = tuple(sorted(
            ContactActivation(contact_id, mode_id)
            for (contact_id, mode_id), index in activation_index.items()
            if result.x[index] >= 0.5
        ))
        assignments = tuple(sorted(
            TaskAssignment(task_id, contact_id)
            for (task_id, contact_id, _mode_id), index in assignment_index.items()
            if result.x[index] >= 0.5
        ))
        decision = T1Decision(activations, assignments)
        # With a positive mip_rel_gap, HiGHS status 0 means the requested
        # termination target was met; it need not mean a zero-gap proof.
        status = "target_gap_met" if result.status == 0 else "time_limit_feasible"
    raw_mip_gap = getattr(result, "mip_gap", math.nan)
    metadata: dict[str, float | int | str] = {
        "status": status,
        "solver_status": int(result.status),
        "runtime_seconds": elapsed,
        "mip_gap": math.nan if raw_mip_gap is None else float(raw_mip_gap),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "variable_count": variable_count,
        "constraint_count": len(lower),
    }
    return decision, metadata


def joint_milp_policy(
    scenario: OptioncastScenario,
    *,
    time_limit_seconds: float = 1.0,
    mip_rel_gap: float = 0.01,
    baseline_policy: Mapping[str, T1Decision] | None = None,
) -> MilpPolicyResult:
    policy: dict[str, T1Decision] = {}
    raw_policy: dict[str, T1Decision] = {}
    metadata = []
    portfolio = full_option_portfolio(scenario)
    fallback_count = 0
    for observation_id in sorted(observation_ids(scenario)):
        raw_decision, row = solve_observation_milp(
            scenario, observation_id,
            time_limit_seconds=time_limit_seconds,
            mip_rel_gap=mip_rel_gap,
        )
        # Fair main comparison: MILP and sequential baseline share the same
        # posterior-aware integral task packer.  The MILP assignment variables
        # therefore guide activation selection but do not receive a different
        # downstream allocator.
        projected = _pack_posterior_tasks(scenario, observation_id, raw_decision.activations)
        raw_policy[observation_id] = projected
        baseline = (
            baseline_policy[observation_id]
            if baseline_policy is not None
            else _strong_baseline_decision(scenario, observation_id)
        )
        projected_value = decision_plan_value(scenario, portfolio, projected, observation_id)
        baseline_value = decision_plan_value(scenario, portfolio, baseline, observation_id)
        if baseline_value > projected_value + 1e-9:
            policy[observation_id] = baseline
            fallback_count += 1
        else:
            policy[observation_id] = projected
        metadata.append(row)
    return MilpPolicyResult(
        policy=policy,
        raw_policy=raw_policy,
        statuses=tuple(str(row["status"]) for row in metadata),
        runtimes=tuple(float(row["runtime_seconds"]) for row in metadata),
        mip_gaps=tuple(float(row["mip_gap"]) for row in metadata),
        node_counts=tuple(int(row["mip_node_count"]) for row in metadata),
        variable_counts=tuple(int(row["variable_count"]) for row in metadata),
        constraint_counts=tuple(int(row["constraint_count"]) for row in metadata),
        fallback_count=fallback_count,
    )


def decision_violations(scenario: OptioncastScenario, decision: T1Decision) -> tuple[str, ...]:
    """Return complete structural/resource violations for one projected policy action."""
    contacts = {item.contact_id: item for item in scenario.contacts}
    tasks = {item.task_id: item for item in scenario.tasks}
    modes = {item.mode_id: item for item in scenario.vcm_modes}
    capsules = {item.task_id: item for item in scenario.policy_capsules}
    violations: list[str] = []
    if len({item.contact_id for item in decision.activations}) != len(decision.activations):
        violations.append("multiple modes selected for one contact")
    if len({item.task_id for item in decision.assignments}) != len(decision.assignments):
        violations.append("task assigned more than once")
    active_by_contact = {item.contact_id: item.mode_id for item in decision.activations}
    selected_contacts = [contacts[item.contact_id] for item in decision.activations]
    if any(
        radio_conflict(left, right)
        for index, left in enumerate(selected_contacts) for right in selected_contacts[index + 1:]
    ):
        violations.append("radio conflict")
    exercise = sum(exercise_fee(contacts[item.contact_id], item.mode_id) for item in decision.activations)
    if exercise > scenario.exercise_budget_milliunits:
        violations.append("exercise budget exceeded")
    energy = sum(
        contacts[item.contact_id].duration_epochs * modes[item.mode_id].energy_milliunits_per_epoch
        for item in decision.activations
    ) + CONTROL_MESSAGE_ENERGY_MILLIUNITS * len(decision.assignments)
    if energy > scenario.energy_budget_milliunits:
        violations.append("energy budget exceeded")
    if len(decision.assignments) > scenario.final_grant_budget:
        violations.append("final grant budget exceeded")
    used_capacity: dict[str, int] = {}
    for assignment in decision.assignments:
        if assignment.task_id not in tasks or assignment.contact_id not in contacts:
            violations.append("unknown task or contact")
            continue
        mode_id = active_by_contact.get(assignment.contact_id)
        if mode_id is None:
            violations.append("assignment has no matching activation")
            continue
        task, contact = tasks[assignment.task_id], contacts[assignment.contact_id]
        if not contact_can_serve_task(contact, task):
            violations.append("task lifetime incompatible with contact")
        if not capsules[task.task_id].permits(
            contact_id=contact.contact_id,
            mode_id=mode_id,
            epoch=contact.start_epoch,
            contract_version=scenario.contract_version,
        ):
            violations.append("policy capsule rejects assignment")
        used_capacity[contact.contact_id] = used_capacity.get(contact.contact_id, 0) + task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
    for contact_id, used in used_capacity.items():
        mode_id = active_by_contact[contact_id]
        nominal = min(
            contacts[contact_id].capacity_units,
            contacts[contact_id].duration_epochs * modes[mode_id].capacity_units_per_epoch,
        )
        if used > nominal:
            violations.append("nominal contact capacity exceeded")
    return tuple(sorted(set(violations)))


def _pack_posterior_tasks(
    scenario: OptioncastScenario,
    observation_id: str,
    active: Sequence[ContactActivation],
) -> T1Decision:
    """Shared posterior-aware integral packer for both compared methods."""
    contacts = {contact.contact_id: contact for contact in scenario.contacts}
    tasks = {task.task_id: task for task in scenario.tasks}
    modes = {mode.mode_id: mode for mode in scenario.vcm_modes}
    triplets = _eligible_triplets(scenario)
    expected_capacity, success_probability = _posterior_terms(scenario, observation_id)
    active_tuple = tuple(sorted(active))
    remaining = {
        (item.contact_id, item.mode_id): expected_capacity[(item.contact_id, item.mode_id)]
        for item in active_tuple
    }
    activation_energy = sum(
        contacts[item.contact_id].duration_epochs * modes[item.mode_id].energy_milliunits_per_epoch
        for item in active_tuple
    )
    control_limit = max(0, (scenario.energy_budget_milliunits - activation_energy) // CONTROL_MESSAGE_ENERGY_MILLIUNITS)
    assignment_limit = min(scenario.final_grant_budget, control_limit)
    assignments = []
    assignment_candidates = []
    for task_id, contact_id, mode_id in triplets:
        key = (contact_id, mode_id)
        if key not in remaining:
            continue
        task = tasks[task_id]
        expected_value = task.value * scenario.value_milliunits_per_point * success_probability[key]
        size = task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
        assignment_candidates.append((expected_value / size, expected_value, task_id, key, size))
    assignment_candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used_tasks = set()
    for _density, _value, task_id, key, size in assignment_candidates:
        if len(assignments) >= assignment_limit:
            break
        if task_id in used_tasks or remaining[key] + 1e-9 < size:
            continue
        assignments.append(TaskAssignment(task_id, key[0]))
        used_tasks.add(task_id)
        remaining[key] -= size
    return T1Decision(active_tuple, tuple(sorted(assignments)))


def _greedy_decision(scenario: OptioncastScenario, observation_id: str) -> T1Decision:
    tasks = {task.task_id: task for task in scenario.tasks}
    triplets = _eligible_triplets(scenario)
    _expected_capacity, success_probability = _posterior_terms(scenario, observation_id)
    candidates = []
    for contact in scenario.contacts:
        for mode in scenario.vcm_modes:
            key = (contact.contact_id, mode.mode_id)
            eligible = [item for item in triplets if item[1:] == key]
            potential = sum(sorted((
                tasks[item[0]].value * scenario.value_milliunits_per_point
                * success_probability[key]
                / (tasks[item[0]].payload_units + CONTROL_MESSAGE_CAPACITY_UNITS)
                for item in eligible
            ), reverse=True)[:scenario.final_grant_budget])
            candidates.append((potential - exercise_fee(contact, mode.mode_id), key))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    active: tuple[ContactActivation, ...] = ()
    for utility, key in candidates:
        candidate = ContactActivation(*key)
        if utility > 0 and _activation_legal(scenario, active, candidate):
            active = tuple(sorted(active + (candidate,)))
    return _pack_posterior_tasks(scenario, observation_id, active)


def _strong_baseline_decision(scenario: OptioncastScenario, observation_id: str) -> T1Decision:
    """Greedy sequential policy followed by one deterministic 1-swap pass."""
    portfolio = full_option_portfolio(scenario)
    incumbent = _greedy_decision(scenario, observation_id)
    incumbent_value = decision_plan_value(scenario, portfolio, incumbent, observation_id)
    active = incumbent.activations
    candidates: set[tuple[ContactActivation, ...]] = {active}
    # Remove one selected activation.
    for selected in active:
        candidates.add(tuple(item for item in active if item != selected))
    all_activations = tuple(
        ContactActivation(contact.contact_id, mode.mode_id)
        for contact in scenario.contacts for mode in scenario.vcm_modes
    )
    # Add one activation, or replace one selected activation with one candidate.
    for candidate in all_activations:
        if _activation_legal(scenario, active, candidate):
            candidates.add(tuple(sorted(active + (candidate,))))
        for selected in active:
            reduced = tuple(item for item in active if item != selected)
            if _activation_legal(scenario, reduced, candidate):
                candidates.add(tuple(sorted(reduced + (candidate,))))
    for candidate_active in sorted(candidates, key=repr):
        decision = _pack_posterior_tasks(scenario, observation_id, candidate_active)
        value = decision_plan_value(scenario, portfolio, decision, observation_id)
        if value > incumbent_value + 1e-9 or (
            abs(value - incumbent_value) <= 1e-9 and repr(decision) < repr(incumbent)
        ):
            incumbent = decision
            incumbent_value = value
    return incumbent


def posterior_greedy_policy(scenario: OptioncastScenario) -> Mapping[str, T1Decision]:
    """Strong causal sequential baseline with one 1-swap local-search pass."""
    return {
        observation_id: _strong_baseline_decision(scenario, observation_id)
        for observation_id in sorted(observation_ids(scenario))
    }


def _bootstrap_ci(values: Sequence[float], seed: int = 20261001, draws: int = 10_000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(data, size=(draws, len(data)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run(
    *,
    scales: Sequence[tuple[int, int]] = SCALES,
    seeds: Sequence[int] = SEEDS,
    time_limit_seconds: float = 1.0,
    mip_rel_gap: float = 0.01,
) -> dict[str, object]:
    rows = []
    for contact_count, task_count in scales:
        for seed in seeds:
            scenario = build_congested_scenario(seed, contact_count, task_count)
            portfolio = full_option_portfolio(scenario)
            started = time.perf_counter()
            greedy_policy = posterior_greedy_policy(scenario)
            greedy_runtime = time.perf_counter() - started
            started = time.perf_counter()
            milp_result = joint_milp_policy(
                scenario,
                time_limit_seconds=time_limit_seconds,
                mip_rel_gap=mip_rel_gap,
                baseline_policy=greedy_policy,
            )
            milp_solver_runtime = time.perf_counter() - started
            # The joint method is anytime-safe only because it first constructs
            # the strong baseline; report that deployable end-to-end cost.
            milp_runtime = greedy_runtime + milp_solver_runtime
            milp_value = score_policy(scenario, portfolio, milp_result.policy)
            greedy_value = score_policy(scenario, portfolio, greedy_policy)
            contacts_by_id = {contact.contact_id: contact for contact in scenario.contacts}
            modes_by_id = {mode.mode_id: mode for mode in scenario.vcm_modes}
            activation_counts = []
            grant_utilizations = []
            exercise_utilizations = []
            energy_utilizations = []
            for decision in milp_result.policy.values():
                activation_counts.append(len(decision.activations))
                grant_utilizations.append(len(decision.assignments) / scenario.final_grant_budget)
                exercise_used = sum(
                    exercise_fee(contacts_by_id[item.contact_id], item.mode_id)
                    for item in decision.activations
                )
                exercise_utilizations.append(exercise_used / scenario.exercise_budget_milliunits)
                energy_used = sum(
                    contacts_by_id[item.contact_id].duration_epochs
                    * modes_by_id[item.mode_id].energy_milliunits_per_epoch
                    for item in decision.activations
                ) + CONTROL_MESSAGE_ENERGY_MILLIUNITS * len(decision.assignments)
                energy_utilizations.append(energy_used / scenario.energy_budget_milliunits)
            finite_gaps = [value for value in milp_result.mip_gaps if math.isfinite(value)]
            rows.append({
                "seed": seed,
                "contact_count": contact_count,
                "task_count": task_count,
                "conflict_density": conflict_density(scenario),
                "exercise_budget_milliunits": scenario.exercise_budget_milliunits,
                "energy_budget_milliunits": scenario.energy_budget_milliunits,
                "final_grant_budget": scenario.final_grant_budget,
                "milp_expected_net_milliunits": milp_value,
                "greedy_expected_net_milliunits": greedy_value,
                "paired_gain_milliunits": milp_value - greedy_value,
                "milp_runtime_seconds": milp_runtime,
                "greedy_runtime_seconds": greedy_runtime,
                "milp_status_counts": {
                    status: milp_result.statuses.count(status) for status in sorted(set(milp_result.statuses))
                },
                "milp_max_finite_surrogate_gap": max(finite_gaps, default=None),
                "milp_nonfinite_surrogate_gap_count": sum(
                    not math.isfinite(value) for value in milp_result.mip_gaps
                ),
                "milp_median_surrogate_gap": median(finite_gaps),
                "milp_max_variables": max(milp_result.variable_counts),
                "milp_max_constraints": max(milp_result.constraint_counts),
                "milp_total_nodes": sum(milp_result.node_counts),
                "milp_planning_fallback_count": milp_result.fallback_count,
                "milp_activation_count_median": median(activation_counts),
                "milp_grant_utilization_median": median(grant_utilizations),
                "milp_exercise_budget_utilization_median": median(exercise_utilizations),
                "milp_energy_budget_utilization_median": median(energy_utilizations),
            })
    summaries = []
    for contact_count, task_count in scales:
        selected = [row for row in rows if row["contact_count"] == contact_count and row["task_count"] == task_count]
        gains = [float(row["paired_gain_milliunits"]) for row in selected]
        relative_gains = [
            float(row["paired_gain_milliunits"]) / max(1.0, abs(float(row["greedy_expected_net_milliunits"])))
            for row in selected
        ]
        ci_low, ci_high = _bootstrap_ci(gains, seed=20261001 + contact_count + task_count)
        relative_ci_low, relative_ci_high = _bootstrap_ci(
            relative_gains, seed=20262001 + contact_count + task_count,
        )
        summaries.append({
            "contact_count": contact_count,
            "task_count": task_count,
            "seed_count": len(selected),
            "conflict_density_median": median(float(row["conflict_density"]) for row in selected),
            "milp_runtime_seconds_median": median(float(row["milp_runtime_seconds"]) for row in selected),
            "milp_runtime_seconds_p95": float(np.quantile([row["milp_runtime_seconds"] for row in selected], 0.95)),
            "greedy_runtime_seconds_median": median(float(row["greedy_runtime_seconds"]) for row in selected),
            "paired_gain_milliunits_mean": float(np.mean(gains)),
            "paired_gain_milliunits_median": median(gains),
            "paired_gain_95pct_bootstrap_ci": [ci_low, ci_high],
            "relative_gain_median": median(relative_gains),
            "relative_gain_mean": float(np.mean(relative_gains)),
            "relative_gain_mean_95pct_bootstrap_ci": [relative_ci_low, relative_ci_high],
            "win_rate": sum(value > 0 for value in gains) / len(gains),
            "all_observations_feasible": all(
                "no_incumbent" not in row["milp_status_counts"] for row in selected
            ),
            "max_finite_surrogate_mip_gap": max(
                float(row["milp_max_finite_surrogate_gap"])
                for row in selected if row["milp_max_finite_surrogate_gap"] is not None
            ),
            "nonfinite_surrogate_gap_count": sum(
                int(row["milp_nonfinite_surrogate_gap_count"]) for row in selected
            ),
            "median_of_observation_median_surrogate_gaps": median(
                float(row["milp_median_surrogate_gap"]) for row in selected
            ),
            "solver_target_status_observation_rate": sum(
                int(row["milp_status_counts"].get("target_gap_met", 0)) for row in selected
            ) / (len(selected) * len(observation_ids(build_congested_scenario(int(selected[0]["seed"]), contact_count, task_count)))),
            "planning_fallback_rate": sum(int(row["milp_planning_fallback_count"]) for row in selected)
            / (len(selected) * len(observation_ids(build_congested_scenario(int(selected[0]["seed"]), contact_count, task_count)))),
            "grant_utilization_median": median(float(row["milp_grant_utilization_median"]) for row in selected),
            "exercise_budget_utilization_median": median(
                float(row["milp_exercise_budget_utilization_median"]) for row in selected
            ),
            "energy_budget_utilization_median": median(
                float(row["milp_energy_budget_utilization_median"]) for row in selected
            ),
        })
    return {
        "study": "V10.9 congestion-aware observation-decomposed Joint-MILP-EC",
        "causal_boundary": "both methods use only P_plan and public Z; frozen policies are scored under P_score",
        "surrogate_disclosure": (
            "posterior expected capacity replaces per-world capacity; MIP gap certifies only the surrogate, "
            "not the original stochastic CAP state-machine objective"
        ),
        "scope": (
            "fixed full-option t0 portfolio; both methods use one shared posterior-aware task packer; "
            "baseline is posterior greedy plus deterministic one-swap local search"
        ),
        "time_limit_seconds_per_observation": time_limit_seconds,
        "mip_relative_gap_target": mip_rel_gap,
        "seeds": list(seeds),
        "rows": rows,
        "summary": summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        scales=((10, 20),) if args.quick else SCALES,
        seeds=(100,) if args.quick else SEEDS,
        time_limit_seconds=args.time_limit,
        mip_rel_gap=args.mip_gap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
