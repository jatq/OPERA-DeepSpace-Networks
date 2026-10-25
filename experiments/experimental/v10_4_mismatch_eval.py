"""V10.4 evaluators for the capacity--utility mismatch experiment.

This module intentionally reuses the frozen V10.3 protocol primitives.  The
new strong baseline uses a fractional task-loading relaxation *only* to pick
an activation set.  It then freezes that set and runs the same integral CAP
downstream decision and the same P_score evaluator as the proposed method.

All selectors in this module use the same rule: scores differing by at most
``TIE_TOLERANCE`` are tied, and the lexicographically smallest representation
wins.  No truth world is available at either the t0 or t1 decision node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np
from scipy.optimize import linprog

from experimental.v10_3_optioncast_eval import (
    FORECAST_OPTION_THEN_CAP,
    OPTIONCAST_CAP_EXACT,
    ContactActivation,
    PolicyEvaluation,
    T1Decision,
    _active_sets,
    _fees,
    _maps,
    _plan_policy_value,
    _realise,
    _score_policy,
    decision_plan_value,
    enumerate_portfolios,
    enumerate_t1_decisions,
)
from experimental.v10_3_optioncast_model import (
    CONTROL_MESSAGE_CAPACITY_UNITS,
    CONTROL_MESSAGE_ENERGY_MILLIUNITS,
    CONTROL_MESSAGE_PRICE_MILLIUNITS,
    PROBABILITY_SCALE,
    ContractKind,
    OptioncastScenario,
    T0ContractChoice,
    World,
    beacon_reaches_source,
    contact_can_serve_task,
    contact_capacity_units,
    exercise_fee,
    likelihood,
    posterior_world_weights,
    validate_scenario,
)


DEMAND_AWARE_RELAXED_THEN_CAP = "DemandAwareRelaxedThenCAP"
FIXED_PORTFOLIO_JOINT = "FixedPortfolioJoint"
FIXED_PORTFOLIO_CAPACITY_THEN_CAP = "FixedPortfolioCapacityThenCAP"
FIXED_PORTFOLIO_RELAXED_THEN_CAP = "FixedPortfolioRelaxedThenCAP"
DOWNSTREAM_VALUE_THEN_CAP = "DownstreamValueThenCAP"
TIE_TOLERANCE = 1e-9

_T = TypeVar("_T")


def _lex_key(item: object) -> str:
    return repr(item)


def select_best(
    items: Sequence[_T], scorer: Callable[[_T], float], *, tolerance: float = TIE_TOLERANCE,
) -> _T:
    """Maximize ``scorer`` with a tolerance tie and ascending lexical tie-break."""
    if tolerance < 0:
        raise ValueError("tie tolerance must be non-negative")
    ordered = sorted(items, key=_lex_key)
    if not ordered:
        raise ValueError("candidate universe is empty")
    scored = tuple((item, float(scorer(item))) for item in ordered)
    maximum = max(score for _item, score in scored)
    return next(item for item, score in scored if score >= maximum - tolerance)


@dataclass(frozen=True)
class RelaxedActivationScore:
    """Solution certificate for one activation-set loading relaxation."""

    activations: tuple[ContactActivation, ...]
    objective_milliunits: float
    expected_capacity_units: Mapping[str, float]
    task_fractions: Mapping[tuple[str, str], float]
    solver_status: int
    solver_message: str


@dataclass(frozen=True)
class TaskOutcomeTrace:
    task_id: str
    contact_id: str
    mode_id: str
    payload_units: int
    value: int
    completed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class RealizationTrace:
    world_id: str
    observation_id: str
    task_outcomes: tuple[TaskOutcomeTrace, ...]
    completed_task_ids: tuple[str, ...]
    attempted_payload_units: int
    completed_payload_units: int
    gross_value: int
    t0_fees_milliunits: int
    t1_fees_milliunits: int
    net_value_milliunits: int


@dataclass(frozen=True)
class ExpectedTrace:
    expected_completed_tasks: float
    expected_attempted_payload_units: float
    expected_completed_payload_units: float
    expected_gross_value: float
    expected_t0_fees_milliunits: float
    expected_t1_fees_milliunits: float
    expected_net_value_milliunits: float
    expected_completed_value_by_task: Mapping[str, float]
    realizations: tuple[RealizationTrace, ...]


@dataclass(frozen=True)
class TracedPolicyEvaluation:
    evaluation: PolicyEvaluation
    trace: ExpectedTrace


def _option_exercise_fees(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    activations: Sequence[ContactActivation],
) -> int:
    contacts, _, _ = _maps(scenario)
    options = {choice.contact_id for choice in portfolio if choice.kind is ContractKind.OPTION}
    return sum(
        exercise_fee(contacts[item.contact_id], item.mode_id)
        for item in activations if item.contact_id in options
    )


def relaxed_activation_score(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    observation_id: str,
    activations: Sequence[ContactActivation],
) -> RelaxedActivationScore:
    """Score an activation set with fractional, task-aware loading.

    Variables ``x[j,c]`` are fractional task completions.  The relaxation
    respects task deadlines, posterior expected contact capacity, receipt
    budget B, residual control-message energy, and the same exercise/control
    prices as V10.3.  Contract fees paid at t0 are sunk here and are included
    exactly once by the outer portfolio objective.
    """
    validate_scenario(scenario)
    if observation_id not in {item.observation_id for item in scenario.forecast_observations}:
        raise ValueError("unknown observation")
    contacts, tasks, modes = _maps(scenario)
    active = tuple(sorted(activations))
    active_by_contact = {item.contact_id: item for item in active}
    legal_sets = set(_active_sets(portfolio, scenario))
    if active not in legal_sets:
        raise ValueError("activation set is not legal for the fixed portfolio")

    weights = posterior_world_weights(scenario, observation_id, planning=True)
    expected_capacity = {
        activation.contact_id: sum(
            weights[world.world_id]
            * contact_capacity_units(
                contacts[activation.contact_id], modes[activation.mode_id], world,
            )
            for world in scenario.worlds
        )
        for activation in active
    }
    variables = tuple(
        (task.task_id, activation.contact_id)
        for task in sorted(scenario.tasks, key=lambda item: item.task_id)
        for activation in active
        if contact_can_serve_task(contacts[activation.contact_id], task)
    )
    fixed_fee = _option_exercise_fees(scenario, portfolio, active)
    if not variables:
        return RelaxedActivationScore(
            active, float(-fixed_fee), expected_capacity, {}, 0, "empty feasible loading",
        )

    # scipy minimizes.  Every completed fraction earns task value and pays a
    # proportional control message; activation exercise fees are fixed.
    c = np.array([
        -(tasks[task_id].value * scenario.value_milliunits_per_point
          - CONTROL_MESSAGE_PRICE_MILLIUNITS)
        for task_id, _contact_id in variables
    ], dtype=float)
    rows: list[list[float]] = []
    bounds: list[float] = []

    # A task can be completed at most once across active contacts.
    for task in sorted(scenario.tasks, key=lambda item: item.task_id):
        rows.append([1.0 if task_id == task.task_id else 0.0 for task_id, _ in variables])
        bounds.append(1.0)
    # Each active contact receives only its posterior expected usable capacity.
    for activation in active:
        rows.append([
            float(tasks[task_id].payload_units + CONTROL_MESSAGE_CAPACITY_UNITS)
            if contact_id == activation.contact_id else 0.0
            for task_id, contact_id in variables
        ])
        bounds.append(float(expected_capacity[activation.contact_id]))
    # Receipt grants and control energy use the same protocol budgets.
    rows.append([1.0] * len(variables))
    bounds.append(float(scenario.final_grant_budget))
    activation_energy = sum(
        contacts[item.contact_id].duration_epochs * modes[item.mode_id].energy_milliunits_per_epoch
        for item in active
    )
    residual_control_energy = scenario.energy_budget_milliunits - activation_energy
    rows.append([float(CONTROL_MESSAGE_ENERGY_MILLIUNITS)] * len(variables))
    bounds.append(float(max(0, residual_control_energy)))

    result = linprog(
        c, A_ub=np.asarray(rows), b_ub=np.asarray(bounds),
        bounds=[(0.0, 1.0)] * len(variables), method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"fractional task-loading relaxation failed: {result.message}")
    if len(result.x) != len(variables):
        raise RuntimeError("fractional task-loading relaxation returned the wrong vector length")
    fractions = {
        variable: float(value)
        for variable, value in zip(variables, result.x)
        if value > TIE_TOLERANCE
    }
    return RelaxedActivationScore(
        active, float(-result.fun - fixed_fee), expected_capacity, fractions,
        int(result.status), str(result.message),
    )


def _integral_downstream_decision(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    observation_id: str,
    activations: tuple[ContactActivation, ...],
) -> T1Decision:
    decisions = tuple(
        decision for decision in enumerate_t1_decisions(scenario, portfolio)
        if decision.activations == activations
    )
    return select_best(
        decisions,
        lambda decision: decision_plan_value(
            scenario, portfolio, decision, observation_id,
        ),
    )


def demand_aware_policy_for_portfolio(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> Mapping[str, T1Decision]:
    """Choose activation by relaxation, freeze it, then run integral CAP."""
    activation_sets = _active_sets(portfolio, scenario)
    policy: dict[str, T1Decision] = {}
    for observation_id in sorted(
        item.observation_id for item in scenario.forecast_observations
    ):
        certificates = tuple(
            relaxed_activation_score(
                scenario, portfolio, observation_id, activations,
            )
            for activations in activation_sets
        )
        chosen = select_best(
            certificates, lambda certificate: certificate.objective_milliunits,
        )
        policy[observation_id] = _integral_downstream_decision(
            scenario, portfolio, observation_id, chosen.activations,
        )
    return policy


def evaluate_demand_aware_relaxed_then_cap(
    scenario: OptioncastScenario,
) -> PolicyEvaluation:
    """Evaluate the strong task-aware sequential baseline end to end."""
    candidates: list[
        tuple[tuple[T0ContractChoice, ...], Mapping[str, T1Decision]]
    ] = []
    for portfolio in enumerate_portfolios(scenario):
        candidates.append((portfolio, demand_aware_policy_for_portfolio(scenario, portfolio)))
    portfolio, policy = select_best(
        candidates, lambda item: _plan_policy_value(scenario, item[0], item[1]),
    )
    return _score_policy(
        DEMAND_AWARE_RELAXED_THEN_CAP, scenario, portfolio, policy,
    )


def joint_policy_for_portfolio(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> Mapping[str, T1Decision]:
    """Joint exact t1 policy under the V10.4 common tie rule."""
    decisions = enumerate_t1_decisions(scenario, portfolio)
    return {
        observation_id: select_best(
            decisions,
            lambda decision, observation_id=observation_id: decision_plan_value(
                scenario, portfolio, decision, observation_id,
            ),
        )
        for observation_id in sorted(
            item.observation_id for item in scenario.forecast_observations
        )
    }


def evaluate_joint_exact(scenario: OptioncastScenario) -> PolicyEvaluation:
    """End-to-end proposed method with the V10.4 common tie rule."""
    candidates = tuple(
        (portfolio, joint_policy_for_portfolio(scenario, portfolio))
        for portfolio in enumerate_portfolios(scenario)
    )
    portfolio, policy = select_best(
        candidates, lambda item: _plan_policy_value(scenario, item[0], item[1]),
    )
    return _score_policy(OPTIONCAST_CAP_EXACT, scenario, portfolio, policy)


def capacity_then_cap_policy_for_portfolio(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> Mapping[str, T1Decision]:
    """Capacity-first policy under the same tolerance and lexical tie rule."""
    contacts, _tasks, modes = _maps(scenario)
    option_contacts = {
        item.contact_id for item in portfolio if item.kind is ContractKind.OPTION
    }
    activation_sets = _active_sets(portfolio, scenario)
    policy: dict[str, T1Decision] = {}
    for observation_id in sorted(
        item.observation_id for item in scenario.forecast_observations
    ):
        weights = posterior_world_weights(scenario, observation_id, planning=True)

        def capacity_score(activations: tuple[ContactActivation, ...]) -> float:
            expected_capacity = sum(
                weights[world.world_id]
                * sum(
                    contact_capacity_units(
                        contacts[item.contact_id], modes[item.mode_id], world,
                    )
                    for item in activations
                )
                for world in scenario.worlds
            )
            fees = sum(
                exercise_fee(contacts[item.contact_id], item.mode_id)
                for item in activations if item.contact_id in option_contacts
            )
            return expected_capacity * scenario.value_milliunits_per_point - fees

        chosen = select_best(activation_sets, capacity_score)
        policy[observation_id] = _integral_downstream_decision(
            scenario, portfolio, observation_id, chosen,
        )
    return policy


def evaluate_capacity_then_cap(scenario: OptioncastScenario) -> PolicyEvaluation:
    """End-to-end original capacity-first baseline under the common tie rule."""
    candidates = tuple(
        (portfolio, capacity_then_cap_policy_for_portfolio(scenario, portfolio))
        for portfolio in enumerate_portfolios(scenario)
    )
    portfolio, policy = select_best(
        candidates, lambda item: _plan_policy_value(scenario, item[0], item[1]),
    )
    return _score_policy(FORECAST_OPTION_THEN_CAP, scenario, portfolio, policy)


def evaluate_fixed_portfolio_joint(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> PolicyEvaluation:
    """Exact joint activation/assignment choice for a caller-fixed portfolio."""
    policy = joint_policy_for_portfolio(scenario, portfolio)
    return _score_policy(FIXED_PORTFOLIO_JOINT, scenario, portfolio, policy)


def full_option_portfolio(
    scenario: OptioncastScenario,
) -> tuple[T0ContractChoice, ...]:
    """Return the common, score-blind portfolio containing every option.

    The mismatch experiment freezes this portfolio for all methods when it
    wants to isolate the t1 activation/assignment property.  Scenario
    validation and the shared scorer remain responsible for rejecting an
    infeasible contract budget or reservation-slot construction.
    """
    return tuple(
        T0ContractChoice(contact.contact_id, ContractKind.OPTION)
        for contact in sorted(scenario.contacts, key=lambda item: item.contact_id)
    )


def evaluate_fixed_portfolio_capacity_then_cap(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> PolicyEvaluation:
    """Capacity-first activation followed by integral CAP on a fixed portfolio."""
    policy = capacity_then_cap_policy_for_portfolio(scenario, portfolio)
    return _score_policy(
        FIXED_PORTFOLIO_CAPACITY_THEN_CAP, scenario, portfolio, policy,
    )


def evaluate_fixed_portfolio_relaxed_then_cap(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> PolicyEvaluation:
    """Task-aware fractional activation followed by CAP on a fixed portfolio."""
    policy = demand_aware_policy_for_portfolio(scenario, portfolio)
    return _score_policy(
        FIXED_PORTFOLIO_RELAXED_THEN_CAP, scenario, portfolio, policy,
    )


def evaluate_downstream_value_then_cap(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> PolicyEvaluation:
    """Factor the exact fixed-portfolio maximum by activation then assignment.

    This is an equivalence/control implementation, not another claimed
    performance curve.  With the shared tolerance/tie rule its policy should
    equal :func:`evaluate_fixed_portfolio_joint` exactly.
    """
    all_decisions = enumerate_t1_decisions(scenario, portfolio)
    activation_sets = _active_sets(portfolio, scenario)
    policy: dict[str, T1Decision] = {}
    for observation_id in sorted(
        item.observation_id for item in scenario.forecast_observations
    ):
        best_by_activation = tuple(
            select_best(
                tuple(d for d in all_decisions if d.activations == activations),
                lambda decision: decision_plan_value(
                    scenario, portfolio, decision, observation_id,
                ),
            )
            for activations in activation_sets
        )
        policy[observation_id] = select_best(
            best_by_activation,
            lambda decision: decision_plan_value(
                scenario, portfolio, decision, observation_id,
            ),
        )
    return _score_policy(DOWNSTREAM_VALUE_THEN_CAP, scenario, portfolio, policy)


def assert_downstream_equivalence(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> None:
    joint = evaluate_fixed_portfolio_joint(scenario, portfolio)
    factored = evaluate_downstream_value_then_cap(scenario, portfolio)
    if joint.decisions_by_observation != factored.decisions_by_observation:
        raise AssertionError("factored downstream policy differs from joint policy")
    if abs(joint.expected_net_milliunits - factored.expected_net_milliunits) > TIE_TOLERANCE:
        raise AssertionError("factored downstream value differs from joint value")


def realization_trace(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decision: T1Decision,
    world: World,
    observation_id: str,
) -> RealizationTrace:
    """Return per-task/byte/fee outcomes for one already-fixed decision."""
    contacts, tasks, modes = _maps(scenario)
    active = {item.contact_id: item for item in decision.activations}
    t0_fee, t1_fee = _fees(portfolio, decision, scenario)
    outcomes: list[TaskOutcomeTrace] = []
    used_capacity: dict[str, int] = {}
    completed_count = 0
    for assignment in sorted(
        decision.assignments,
        key=lambda item: (-tasks[item.task_id].value, item.task_id, item.contact_id),
    ):
        task = tasks[assignment.task_id]
        activation = active[assignment.contact_id]
        contact = contacts[assignment.contact_id]
        completed = False
        reason: str | None = None
        # A lightweight beacon-shaped object is sufficient because the shared
        # reachability primitive reads only contact_id and issuance time.
        beacon = type(
            "TraceBeacon", (),
            {"contact_id": assignment.contact_id, "issued_epoch": scenario.t1_epoch},
        )()
        if not beacon_reaches_source(beacon, world, scenario):
            reason = "beacon_unavailable_or_late"
        else:
            needed = task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
            capacity = contact_capacity_units(contact, modes[activation.mode_id], world)
            if used_capacity.get(contact.contact_id, 0) + needed > capacity:
                reason = "realized_capacity"
            elif completed_count >= scenario.final_grant_budget:
                reason = "receipt_budget"
            else:
                used_capacity[contact.contact_id] = used_capacity.get(contact.contact_id, 0) + needed
                completed_count += 1
                completed = True
        outcomes.append(TaskOutcomeTrace(
            task.task_id, contact.contact_id, activation.mode_id,
            task.payload_units, task.value, completed, reason,
        ))
    gross = sum(item.value for item in outcomes if item.completed)
    shared_gross, shared_net, shared_completed, _ = _realise(
        scenario, portfolio, decision, world,
    )
    if (gross, completed_count) != (shared_gross, shared_completed):
        raise AssertionError("trace diverged from shared V10.3 protocol realization")
    return RealizationTrace(
        world.world_id, observation_id, tuple(outcomes),
        tuple(sorted(item.task_id for item in outcomes if item.completed)),
        sum(item.payload_units for item in outcomes),
        sum(item.payload_units for item in outcomes if item.completed),
        gross, t0_fee, t1_fee, shared_net,
    )


def trace_policy(
    scenario: OptioncastScenario, evaluation: PolicyEvaluation,
) -> TracedPolicyEvaluation:
    """Attach probability-weighted task/byte/fee traces to an evaluation."""
    realizations: list[RealizationTrace] = []
    completed_tasks = attempted_bytes = completed_bytes = gross = t1_fees = net = 0.0
    by_task = {task.task_id: 0.0 for task in scenario.tasks}
    for world in scenario.worlds:
        for observation_id in sorted(evaluation.decisions_by_observation):
            trace = realization_trace(
                scenario, evaluation.portfolio,
                evaluation.decisions_by_observation[observation_id], world,
                observation_id,
            )
            mass = (
                world.probability_units / PROBABILITY_SCALE
                * likelihood(scenario, world, observation_id, planning=False)
            )
            realizations.append(trace)
            completed_tasks += mass * len(trace.completed_task_ids)
            attempted_bytes += mass * trace.attempted_payload_units
            completed_bytes += mass * trace.completed_payload_units
            gross += mass * trace.gross_value
            t1_fees += mass * trace.t1_fees_milliunits
            net += mass * trace.net_value_milliunits
            for item in trace.task_outcomes:
                if item.completed:
                    by_task[item.task_id] += mass * item.value
    expected = ExpectedTrace(
        completed_tasks, attempted_bytes, completed_bytes, gross,
        evaluation.expected_t0_fees, t1_fees, net, by_task,
        tuple(realizations),
    )
    if abs(expected.expected_net_value_milliunits - evaluation.expected_net_milliunits) > TIE_TOLERANCE:
        raise AssertionError("weighted trace net value differs from policy evaluation")
    if abs(expected.expected_gross_value - evaluation.expected_gross_value) > TIE_TOLERANCE:
        raise AssertionError("weighted trace gross value differs from policy evaluation")
    return TracedPolicyEvaluation(evaluation, expected)


__all__ = [
    "DEMAND_AWARE_RELAXED_THEN_CAP", "FIXED_PORTFOLIO_JOINT",
    "FIXED_PORTFOLIO_CAPACITY_THEN_CAP", "FIXED_PORTFOLIO_RELAXED_THEN_CAP",
    "DOWNSTREAM_VALUE_THEN_CAP", "TIE_TOLERANCE",
    "RelaxedActivationScore", "TaskOutcomeTrace", "RealizationTrace",
    "ExpectedTrace", "TracedPolicyEvaluation", "select_best",
    "relaxed_activation_score", "demand_aware_policy_for_portfolio",
    "joint_policy_for_portfolio", "evaluate_joint_exact",
    "capacity_then_cap_policy_for_portfolio", "evaluate_capacity_then_cap",
    "evaluate_demand_aware_relaxed_then_cap", "evaluate_fixed_portfolio_joint",
    "full_option_portfolio", "evaluate_fixed_portfolio_capacity_then_cap",
    "evaluate_fixed_portfolio_relaxed_then_cap",
    "evaluate_downstream_value_then_cap", "assert_downstream_equivalence",
    "realization_trace", "trace_policy",
]
