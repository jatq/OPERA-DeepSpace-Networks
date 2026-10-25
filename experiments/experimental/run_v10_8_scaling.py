"""Bounded scalability audit for the V10.3/V10.4 OPTIONCAST-CAP model.

This file is deliberately additive: it imports the frozen public model and
evaluation interfaces but does not change them.  Exact enumeration is used as
an optimality calibration on a small instance.  Larger instances compare two
bounded policies on the same portfolio, observations, feasibility checks and
downstream task packer:

* ``joint_beam`` jointly ranks activation sets by their resulting task value;
* ``task_aware_sequential`` first ranks activations by a fractional-knapsack
  proxy, then invokes the identical integral task packer.

Large-instance rows are explicitly labelled ``bounded_heuristic`` or
``time_limited`` and never presented as global optima.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
import tracemalloc
from statistics import median
from typing import Callable, Mapping, Sequence

from experimental.v10_3_optioncast_eval import (
    ContactActivation,
    PolicyEvaluation,
    T1Decision,
    TaskAssignment,
    decision_plan_value,
    enumerate_t1_decisions,
)
from experimental.v10_3_optioncast_model import (
    CONTROL_MESSAGE_CAPACITY_UNITS,
    CONTROL_MESSAGE_ENERGY_MILLIUNITS,
    PROBABILITY_SCALE,
    ContractKind,
    OptioncastScenario,
    T0ContractChoice,
    contact_can_serve_task,
    contact_capacity_units,
    exercise_fee,
    likelihood,
    nominal_contact_capacity_units,
    observation_ids,
    posterior_world_weights,
    radio_conflict,
    validate_scenario,
    validate_t0_contract_choices,
)
from experimental.v10_3_optioncast_suite import make_scenario
from experimental.v10_4_mismatch_eval import (
    evaluate_fixed_portfolio_joint,
    full_option_portfolio,
    realization_trace,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experimental/v10_8_scaling_summary.json"
CONTACT_COUNTS = (10, 20, 30)
TASK_COUNTS = (20, 50, 100)
SCALE_SEEDS = (0, 1, 2)
BEAM_WIDTH = 12
# Frozen V10.8 stop condition used by the published result artifact.
METHOD_TIME_LIMIT_SECONDS = 3.0


@dataclass(frozen=True)
class SearchResult:
    policy: Mapping[str, T1Decision]
    candidate_states: int
    status: str
    observations_completed: int


def build_scaling_scenario(seed: int, contact_count: int, task_count: int) -> OptioncastScenario:
    """Mechanically expand a frozen score-blind scenario without score tuning."""
    if contact_count < 1 or task_count < 1:
        raise ValueError("contact_count and task_count must be positive")
    base = make_scenario(seed, mismatch="matching")
    stations = tuple(sorted(base.station_slot_limits))
    contacts = []
    for index in range(contact_count):
        template = base.contacts[index % len(base.contacts)]
        start = 8 + 3 * index
        duration = 2 + index % 2
        contacts.append(replace(
            template,
            contact_id=f"scale-c{seed}-{index:02d}",
            station_id=stations[index % len(stations)],
            start_epoch=start,
            end_epoch=start + duration,
            capacity_units=8 + 2 * (index % 4),
            option_fee_milliunits=12 + index % 5,
        ))
    contacts_tuple = tuple(contacts)
    contact_ids = tuple(item.contact_id for item in contacts_tuple)

    tasks = []
    for index in range(task_count):
        template = base.tasks[index % len(base.tasks)]
        release = 0 if index % 4 else 3 * (index % max(1, contact_count // 2))
        deadline = contacts_tuple[-1].end_epoch + 3 - (index % 3)
        tasks.append(replace(
            template,
            task_id=f"scale-t{seed}-{index:03d}",
            release_epoch=release,
            deadline_epoch=max(release + 4, deadline),
            payload_units=2 + index % 5,
            value=5 + (7 * index + seed) % 17,
        ))
    tasks_tuple = tuple(tasks)

    capsule_template = base.policy_capsules[0]
    capsules = tuple(replace(
        capsule_template,
        capsule_id=f"scale-cap:{task.task_id}",
        task_id=task.task_id,
        contact_whitelist=contact_ids,
        expires_epoch=contacts_tuple[-1].end_epoch + 10,
    ) for task in tasks_tuple)

    source_by_station = {contact.station_id: contact for contact in base.contacts}
    worlds = []
    for world in base.worlds:
        arrivals: dict[str, int | None] = {}
        for contact in contacts_tuple:
            source = source_by_station.get(contact.station_id, base.contacts[0])
            original = world.beacon_arrival_epoch_by_contact[source.contact_id]
            arrivals[contact.contact_id] = None if original is None else contact.start_epoch - 2
        worlds.append(replace(world, beacon_arrival_epoch_by_contact=arrivals))

    option_cost = sum(contact.option_fee_milliunits for contact in contacts_tuple)
    grant_budget = min(12, max(4, math.ceil(task_count / 10)))
    scenario = replace(
        base,
        scenario_id=f"v108-scale-s{seed}-c{contact_count}-t{task_count}",
        contacts=contacts_tuple,
        tasks=tasks_tuple,
        worlds=tuple(worlds),
        policy_capsules=capsules,
        station_slot_limits={station: contact_count for station in stations},
        contract_budget_milliunits=option_cost,
        option_count_budget=contact_count,
        exercise_budget_milliunits=600,
        energy_budget_milliunits=2_500,
        final_grant_budget=grant_budget,
    )
    if 4 <= contact_count <= 6 and 6 <= task_count <= 8:
        validate_scenario(scenario)
    else:
        _validate_scaling_extension(scenario)
    validate_t0_contract_choices(full_option_portfolio(scenario), scenario)
    return scenario


def _validate_scaling_extension(scenario: OptioncastScenario) -> None:
    """Validate the large adapter after the frozen Stage-0 size guard rejects it.

    ``validate_scenario`` intentionally caps the canonical model at 4--6
    contacts and 6--8 tasks.  This audit changes no frozen file and discloses
    that boundary; all operational primitives are still the public V10.3 API.
    """
    contact_ids = [item.contact_id for item in scenario.contacts]
    task_ids = [item.task_id for item in scenario.tasks]
    capsule_tasks = [item.task_id for item in scenario.policy_capsules]
    if len(set(contact_ids)) != len(contact_ids):
        raise ValueError("duplicate scaling contact id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate scaling task id")
    if set(capsule_tasks) != set(task_ids) or len(capsule_tasks) != len(task_ids):
        raise ValueError("scaling capsules must map one-to-one to tasks")
    if any(item.start_epoch >= item.end_epoch for item in scenario.contacts):
        raise ValueError("non-positive scaling contact duration")
    if any(item.release_epoch >= item.deadline_epoch for item in scenario.tasks):
        raise ValueError("invalid scaling task lifetime")
    if set(observation_ids(scenario)) != {
        key for row in scenario.plan_likelihood_by_world.values() for key in row
    }:
        raise ValueError("planning likelihood support differs from observations")
    if set(observation_ids(scenario)) != {
        key for row in scenario.score_likelihood_by_world.values() for key in row
    }:
        raise ValueError("scoring likelihood support differs from observations")


def _activation_legal(
    scenario: OptioncastScenario,
    active: tuple[ContactActivation, ...],
    candidate: ContactActivation,
) -> bool:
    if any(item.contact_id == candidate.contact_id for item in active):
        return False
    contacts = {item.contact_id: item for item in scenario.contacts}
    modes = {item.mode_id: item for item in scenario.vcm_modes}
    selected = active + (candidate,)
    if any(
        radio_conflict(contacts[left.contact_id], contacts[right.contact_id])
        for index, left in enumerate(selected) for right in selected[index + 1:]
    ):
        return False
    if sum(exercise_fee(contacts[item.contact_id], item.mode_id) for item in selected) > scenario.exercise_budget_milliunits:
        return False
    energy = sum(
        contacts[item.contact_id].duration_epochs
        * modes[item.mode_id].energy_milliunits_per_epoch
        for item in selected
    )
    return energy <= scenario.energy_budget_milliunits


def _pack_tasks(
    scenario: OptioncastScenario,
    activations: tuple[ContactActivation, ...],
) -> T1Decision:
    """Shared deterministic integral CAP packer used by both methods."""
    contacts = {item.contact_id: item for item in scenario.contacts}
    modes = {item.mode_id: item for item in scenario.vcm_modes}
    remaining = {
        item.contact_id: nominal_contact_capacity_units(
            contacts[item.contact_id], modes[item.mode_id],
        )
        for item in activations
    }
    activation_energy = sum(
        contacts[item.contact_id].duration_epochs
        * modes[item.mode_id].energy_milliunits_per_epoch
        for item in activations
    )
    control_limit = max(
        0,
        (scenario.energy_budget_milliunits - activation_energy)
        // CONTROL_MESSAGE_ENERGY_MILLIUNITS,
    )
    count_limit = min(scenario.final_grant_budget, control_limit)
    assignments: list[TaskAssignment] = []
    tasks = sorted(
        scenario.tasks,
        key=lambda item: (
            -(item.value / (item.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS)),
            item.deadline_epoch,
            item.task_id,
        ),
    )
    for task in tasks:
        if len(assignments) >= count_limit:
            break
        needed = task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
        feasible = [
            item for item in activations
            if contact_can_serve_task(contacts[item.contact_id], task)
            and remaining[item.contact_id] >= needed
        ]
        if not feasible:
            continue
        chosen = min(
            feasible,
            key=lambda item: (
                remaining[item.contact_id] - needed,
                contacts[item.contact_id].end_epoch,
                item.contact_id,
                item.mode_id,
            ),
        )
        assignments.append(TaskAssignment(task.task_id, chosen.contact_id))
        remaining[chosen.contact_id] -= needed
    return T1Decision(tuple(sorted(activations)), tuple(sorted(assignments)))


def _fractional_proxy(
    scenario: OptioncastScenario,
    observation_id: str,
    activation: ContactActivation,
) -> float:
    contacts = {item.contact_id: item for item in scenario.contacts}
    modes = {item.mode_id: item for item in scenario.vcm_modes}
    contact = contacts[activation.contact_id]
    weights = posterior_world_weights(scenario, observation_id, planning=True)
    capacity = sum(
        weights[world.world_id] * contact_capacity_units(contact, modes[activation.mode_id], world)
        for world in scenario.worlds
    )
    eligible = sorted(
        (task for task in scenario.tasks if contact_can_serve_task(contact, task)),
        key=lambda task: (-(task.value / (task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS)), task.task_id),
    )
    value = 0.0
    remaining = capacity
    for task in eligible:
        size = task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
        fraction = min(1.0, remaining / size)
        value += fraction * task.value * scenario.value_milliunits_per_point
        remaining -= fraction * size
        if remaining <= 1e-12:
            break
    return value - exercise_fee(contact, activation.mode_id)


def joint_beam_policy(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    *, beam_width: int = BEAM_WIDTH,
    time_limit_seconds: float = METHOD_TIME_LIMIT_SECONDS,
) -> SearchResult:
    """Bounded task-value-aware activation beam; no global-optimality claim."""
    if beam_width < 1 or time_limit_seconds <= 0:
        raise ValueError("beam_width and time limit must be positive")
    deadline = time.perf_counter() + time_limit_seconds
    policy: dict[str, T1Decision] = {}
    candidate_states = 0
    status = "bounded_heuristic"
    observations_completed = 0
    contacts = sorted(scenario.contacts, key=lambda item: (item.start_epoch, item.contact_id))
    for observation_id in sorted(observation_ids(scenario)):
        beam: tuple[tuple[ContactActivation, ...], ...] = ((),)
        best_decision = T1Decision((), ())
        best_score = decision_plan_value(scenario, portfolio, best_decision, observation_id)
        for contact in contacts:
            expanded: set[tuple[ContactActivation, ...]] = set(beam)
            for active in beam:
                for mode in scenario.vcm_modes:
                    candidate = ContactActivation(contact.contact_id, mode.mode_id)
                    if _activation_legal(scenario, active, candidate):
                        expanded.add(tuple(sorted(active + (candidate,))))
            scored = []
            for active in expanded:
                decision = _pack_tasks(scenario, active)
                score = decision_plan_value(scenario, portfolio, decision, observation_id)
                candidate_states += 1
                scored.append((score, repr(active), active, decision))
                if score > best_score or (abs(score - best_score) <= 1e-9 and repr(decision) < repr(best_decision)):
                    best_score, best_decision = score, decision
                if time.perf_counter() >= deadline:
                    status = "time_limited_with_sequential_fallback"
                    break
            scored.sort(key=lambda item: (-item[0], item[1]))
            beam = tuple(item[2] for item in scored[:beam_width]) or beam
            if status.startswith("time_limited"):
                break
        policy[observation_id] = best_decision
        observations_completed += 1
        if status.startswith("time_limited"):
            break
    # Preserve anytime usefulness: unfinished observations inherit the fair
    # sequential baseline, while the row remains explicitly time-limited.
    if status.startswith("time_limited"):
        fallback = task_aware_sequential_policy(
            scenario, portfolio, time_limit_seconds=max(1.0, time_limit_seconds),
        )
        candidate_states += fallback.candidate_states
        for observation_id in observation_ids(scenario):
            policy.setdefault(observation_id, fallback.policy[observation_id])
    else:
        for observation_id in observation_ids(scenario):
            policy.setdefault(observation_id, T1Decision((), ()))
    return SearchResult(policy, candidate_states, status, observations_completed)


def task_aware_sequential_policy(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    *, time_limit_seconds: float = METHOD_TIME_LIMIT_SECONDS,
) -> SearchResult:
    """Scalable analogue of relaxed-then-CAP with the same final packer."""
    deadline = time.perf_counter() + time_limit_seconds
    policy: dict[str, T1Decision] = {}
    candidate_states = 0
    status = "bounded_heuristic"
    observations_completed = 0
    for observation_id in sorted(observation_ids(scenario)):
        candidates = []
        for contact in scenario.contacts:
            for mode in scenario.vcm_modes:
                activation = ContactActivation(contact.contact_id, mode.mode_id)
                candidates.append((_fractional_proxy(scenario, observation_id, activation), repr(activation), activation))
                candidate_states += 1
        candidates.sort(key=lambda item: (-item[0], item[1]))
        active: tuple[ContactActivation, ...] = ()
        for score, _key, candidate in candidates:
            if score <= 0:
                continue
            if _activation_legal(scenario, active, candidate):
                active = tuple(sorted(active + (candidate,)))
            if time.perf_counter() >= deadline:
                status = "time_limited"
                break
        policy[observation_id] = _pack_tasks(scenario, active)
        observations_completed += 1
        if status == "time_limited":
            break
    for observation_id in observation_ids(scenario):
        policy.setdefault(observation_id, T1Decision((), ()))
    return SearchResult(policy, candidate_states, status, observations_completed)


def score_policy(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    policy: Mapping[str, T1Decision],
) -> float:
    """Score a fixed causal policy under P_score using the public trace API."""
    expected_net = 0.0
    for world in scenario.worlds:
        for observation_id in observation_ids(scenario):
            mass = (
                world.probability_units / PROBABILITY_SCALE
                * likelihood(scenario, world, observation_id, planning=False)
            )
            expected_net += mass * realization_trace(
                scenario, portfolio, policy[observation_id], world, observation_id,
            ).net_value_milliunits
    return expected_net


def _profile(function: Callable[[], SearchResult]) -> tuple[SearchResult, float, int]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = function()
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak


def _method_row(
    scenario: OptioncastScenario,
    name: str,
    runner: Callable[[], SearchResult],
) -> dict[str, object]:
    portfolio = full_option_portfolio(scenario)
    result, elapsed, peak = _profile(runner)
    return {
        "method": name,
        "runtime_seconds": elapsed,
        "peak_tracemalloc_bytes": peak,
        "candidate_states": result.candidate_states,
        "status": result.status,
        "globally_optimal": False,
        "observations_completed": result.observations_completed,
        "observation_count": len(observation_ids(scenario)),
        "expected_net_milliunits": score_policy(scenario, portfolio, result.policy),
    }


def exact_calibration(seed: int = 0) -> dict[str, object]:
    base = make_scenario(seed, mismatch="matching")
    if len(base.contacts) != 4 or len(base.tasks) != 6:
        raise ValueError("the frozen exact-calibration seed must remain a 4-contact/6-task case")
    scenario = replace(
        base,
        scenario_id=f"{base.scenario_id}-v108-exact-calibration",
        contract_budget_milliunits=sum(item.option_fee_milliunits for item in base.contacts),
        option_count_budget=len(base.contacts),
        station_slot_limits={station: len(base.contacts) for station in base.station_slot_limits},
    )
    validate_scenario(scenario)
    portfolio = full_option_portfolio(scenario)
    validate_t0_contract_choices(portfolio, scenario)
    decisions = enumerate_t1_decisions(scenario, portfolio)
    tracemalloc.start()
    started = time.perf_counter()
    try:
        exact: PolicyEvaluation = evaluate_fixed_portfolio_joint(scenario, portfolio)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    beam = joint_beam_policy(scenario, portfolio, time_limit_seconds=5.0)
    beam_value = score_policy(scenario, portfolio, beam.policy)
    gap = max(0.0, exact.expected_net_milliunits - beam_value)
    denominator = max(1.0, abs(exact.expected_net_milliunits))
    return {
        "contact_count": 4,
        "task_count": 6,
        "exact_status": "optimal_exhaustive",
        "exact_globally_optimal": True,
        "exact_candidate_states": len(decisions),
        "exact_runtime_seconds": elapsed,
        "exact_peak_tracemalloc_bytes": peak,
        "exact_expected_net_milliunits": exact.expected_net_milliunits,
        "beam_expected_net_milliunits": beam_value,
        "beam_relative_gap": gap / denominator,
    }


def run(
    *,
    contact_counts: Sequence[int] = CONTACT_COUNTS,
    task_counts: Sequence[int] = TASK_COUNTS,
    seeds: Sequence[int] = SCALE_SEEDS,
    time_limit_seconds: float = METHOD_TIME_LIMIT_SECONDS,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for contact_count in contact_counts:
            for task_count in task_counts:
                scenario = build_scaling_scenario(seed, contact_count, task_count)
                portfolio = full_option_portfolio(scenario)
                methods = (
                    _method_row(
                        scenario,
                        "joint_beam",
                        lambda scenario=scenario, portfolio=portfolio: joint_beam_policy(
                            scenario, portfolio, time_limit_seconds=time_limit_seconds,
                        ),
                    ),
                    _method_row(
                        scenario,
                        "task_aware_sequential",
                        lambda scenario=scenario, portfolio=portfolio: task_aware_sequential_policy(
                            scenario, portfolio, time_limit_seconds=time_limit_seconds,
                        ),
                    ),
                )
                rows.append({
                    "seed": seed,
                    "contact_count": contact_count,
                    "task_count": task_count,
                    "world_count": len(scenario.worlds),
                    "observation_count": len(scenario.forecast_observations),
                    "final_grant_budget": scenario.final_grant_budget,
                    "methods": methods,
                })
    summary: list[dict[str, object]] = []
    for contact_count in contact_counts:
        for task_count in task_counts:
            selected = [
                row for row in rows
                if row["contact_count"] == contact_count and row["task_count"] == task_count
            ]
            methods_summary: list[dict[str, object]] = []
            for method_name in ("joint_beam", "task_aware_sequential"):
                method_rows = [
                    next(item for item in row["methods"] if item["method"] == method_name)  # type: ignore[union-attr]
                    for row in selected
                ]
                methods_summary.append({
                    "method": method_name,
                    "runtime_seconds_median": median(float(item["runtime_seconds"]) for item in method_rows),
                    "peak_tracemalloc_bytes_median": median(float(item["peak_tracemalloc_bytes"]) for item in method_rows),
                    "candidate_states_median": median(float(item["candidate_states"]) for item in method_rows),
                    "expected_net_milliunits_median": median(float(item["expected_net_milliunits"]) for item in method_rows),
                    "status_counts": {
                        status: sum(item["status"] == status for item in method_rows)
                        for status in sorted({str(item["status"]) for item in method_rows})
                    },
                })
            summary.append({
                "contact_count": contact_count,
                "task_count": task_count,
                "seed_count": len(selected),
                "methods": methods_summary,
            })
    return {
        "study": "V10.8 bounded scalability and exact-boundary audit",
        "fairness": (
            "same score-blind full-option portfolio, P_plan/P_score tables, "
            "feasibility constraints, integral task packer, time limit and scorer"
        ),
        "large_instance_optimality_claim": "none; bounded heuristic or time-limited only",
        "time_limit_seconds_per_method": time_limit_seconds,
        "beam_width": BEAM_WIDTH,
        "memory_scope": "peak Python-managed allocations measured by tracemalloc; excludes interpreter baseline and native-library RSS",
        "exact_calibration": exact_calibration(),
        "rows": rows,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time-limit", type=float, default=METHOD_TIME_LIMIT_SECONDS)
    parser.add_argument("--quick", action="store_true", help="run one 10-contact/20-task smoke cell")
    args = parser.parse_args(argv)
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    result = run(
        contact_counts=(10,) if args.quick else CONTACT_COUNTS,
        task_counts=(20,) if args.quick else TASK_COUNTS,
        seeds=(0,) if args.quick else SCALE_SEEDS,
        time_limit_seconds=args.time_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact_calibration": result["exact_calibration"], "row_count": len(result["rows"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
