"""Exact E1 evaluator with staged and path-wise unified budget semantics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import experimental.v10_3_optioncast_eval as v103_eval
from experimental.v10_3_optioncast_eval import (
    CAP_THEN_FORECAST_OPTION,
    COMMIT_ONLY_DYNAMIC_EXACT,
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    FORECAST_OPTION_THEN_CAP,
    OPTION_STATIC_EXACT,
    OPTIONCAST_CAP_EXACT,
    PolicyEvaluation,
    T1Decision,
    decision_plan_value,
    enumerate_portfolios,
    enumerate_t1_decisions,
)
from experimental.v10_3_optioncast_model import (
    ContractKind,
    OptioncastScenario,
    T0ContractChoice,
    contact_capacity_units,
    exercise_fee,
    observation_ids,
    posterior_world_weights,
    validate_scenario,
)
from q2_strengthening.e1_pricing import (
    BudgetModel,
    path_spend_milliunits,
    t0_fee_milliunits,
    t1_fee_milliunits,
)


PRIMARY_METHODS = (OPTIONCAST_CAP_EXACT, COMMIT_PORTFOLIO_DYNAMIC_EXACT)
MAIN_METHODS = (
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    OPTION_STATIC_EXACT,
    FORECAST_OPTION_THEN_CAP,
    CAP_THEN_FORECAST_OPTION,
    OPTIONCAST_CAP_EXACT,
)
UNIFIED_SUPPORTED_METHODS = (
    OPTIONCAST_CAP_EXACT,
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    COMMIT_ONLY_DYNAMIC_EXACT,
    OPTION_STATIC_EXACT,
    FORECAST_OPTION_THEN_CAP,
    CAP_THEN_FORECAST_OPTION,
)


@dataclass(frozen=True)
class EvaluationAudit:
    method: str
    budget_model: str
    elapsed_seconds: float
    enumerated_portfolios: int
    portfolios_with_legal_path: int
    max_selected_path_spend_milliunits: int
    hard_budget_milliunits: int
    hard_budget_violations: int


def _deterministic_best(items: Sequence[object], score: object) -> object:
    if not items:
        raise ValueError("exact action universe is empty")
    scorer = score
    if not callable(scorer):
        raise TypeError("score must be callable")
    return max(items, key=lambda item: (scorer(item), repr(item)))


def _method_semantics(method: str) -> tuple[bool, bool, bool, bool, str]:
    """Return action-space flags and the audited phase order."""
    if method == OPTIONCAST_CAP_EXACT:
        return True, True, True, False, "joint"
    if method == COMMIT_PORTFOLIO_DYNAMIC_EXACT:
        return True, False, True, True, "joint"
    if method == COMMIT_ONLY_DYNAMIC_EXACT:
        return True, False, True, False, "joint"
    if method == OPTION_STATIC_EXACT:
        return True, True, False, False, "joint"
    if method == FORECAST_OPTION_THEN_CAP:
        return True, True, True, False, "capacity_then_cap"
    if method == CAP_THEN_FORECAST_OPTION:
        return True, True, True, False, "cap_then_forecast"
    raise ValueError(
        f"unified-hard E1 does not implement {method}; supported methods are "
        f"{', '.join(UNIFIED_SUPPORTED_METHODS)}"
    )


def _policy_for_legal_decisions(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decisions: Sequence[T1Decision],
    *,
    dynamic: bool,
    commit_activation_optional: bool,
    phase_order: str,
) -> Mapping[str, T1Decision]:
    if dynamic:
        if phase_order == "capacity_then_cap":
            contacts = {contact.contact_id: contact for contact in scenario.contacts}
            modes = {mode.mode_id: mode for mode in scenario.vcm_modes}
            option_ids = {
                choice.contact_id
                for choice in portfolio
                if choice.kind is ContractKind.OPTION
            }
            output: dict[str, T1Decision] = {}
            for observation_id in observation_ids(scenario):
                weights = posterior_world_weights(
                    scenario, observation_id, planning=True
                )
                activation_sets = tuple(
                    sorted({decision.activations for decision in decisions}, key=repr)
                )

                def capacity_score(activations: object) -> float:
                    active = activations
                    assert isinstance(active, tuple)
                    option_fees = sum(
                        exercise_fee(
                            contacts[activation.contact_id], activation.mode_id
                        )
                        for activation in active
                        if activation.contact_id in option_ids
                    )
                    expected_capacity = sum(
                        weights[world.world_id]
                        * sum(
                            contact_capacity_units(
                                contacts[activation.contact_id],
                                modes[activation.mode_id],
                                world,
                            )
                            for activation in active
                        )
                        for world in scenario.worlds
                    )
                    return (
                        expected_capacity * scenario.value_milliunits_per_point
                        - option_fees
                    )

                selected = _deterministic_best(activation_sets, capacity_score)
                candidates = tuple(
                    decision
                    for decision in decisions
                    if decision.activations == selected
                )
                output[observation_id] = _deterministic_best(
                    candidates,
                    lambda decision, observation_id=observation_id: decision_plan_value(
                        scenario,
                        portfolio,
                        decision,
                        observation_id,
                        commit_activation_optional=commit_activation_optional,
                    ),
                )
            return output
        if phase_order not in ("joint", "cap_then_forecast"):
            raise ValueError(f"unknown E1 phase order: {phase_order}")
        return {
            observation_id: _deterministic_best(
                decisions,
                lambda decision, observation_id=observation_id: decision_plan_value(
                    scenario,
                    portfolio,
                    decision,
                    observation_id,
                    commit_activation_optional=commit_activation_optional,
                ),
            )
            for observation_id in observation_ids(scenario)
        }
    decision = _deterministic_best(
        decisions,
        lambda candidate: v103_eval._plan_policy_value(  # noqa: SLF001 - pinned audited dependency
            scenario,
            portfolio,
            {observation_id: candidate for observation_id in observation_ids(scenario)},
            commit_activation_optional=commit_activation_optional,
        ),
    )
    return {observation_id: decision for observation_id in observation_ids(scenario)}


def _evaluate_filtered_exact(
    scenario: OptioncastScenario,
    method: str,
    *,
    budget_model: BudgetModel,
    total_budget_milliunits: int,
) -> tuple[PolicyEvaluation, EvaluationAudit]:
    """Optimize subject to the declared budget rule on every Z path.

    A portfolio remains a candidate only if it has at least one legal t1
    decision.  For staged budgets the filter includes control-message prices
    in B1; the legacy V10.3 activation filter counts exercise fees but not
    those control prices.  For unified budgets it enforces
    ``t0 + path-specific t1 <= B_total`` directly.
    """
    validate_scenario(scenario)
    allow_commit, allow_option, dynamic, optional_commit, phase_order = _method_semantics(method)
    portfolios = enumerate_portfolios(
        scenario, allow_commit=allow_commit, allow_option=allow_option
    )
    candidates: list[
        tuple[tuple[T0ContractChoice, ...], Mapping[str, T1Decision]]
    ] = []
    for portfolio in portfolios:
        allowed_task_ids = None
        if phase_order == "cap_then_forecast":
            ordered_tasks = sorted(
                scenario.tasks,
                key=lambda task: (-task.value, task.deadline_epoch, task.task_id),
            )
            allowed_task_ids = frozenset(
                task.task_id
                for task in ordered_tasks[: scenario.final_grant_budget]
            )
        decisions = enumerate_t1_decisions(
            scenario,
            portfolio,
            allowed_task_ids=allowed_task_ids,
            commit_activation_optional=optional_commit,
        )
        if budget_model is BudgetModel.STAGED:
            legal = tuple(
                decision
                for decision in decisions
                if t0_fee_milliunits(scenario, portfolio)
                <= scenario.contract_budget_milliunits
                and t1_fee_milliunits(scenario, portfolio, decision)
                <= scenario.exercise_budget_milliunits
            )
        else:
            legal = tuple(
                decision
                for decision in decisions
                if path_spend_milliunits(scenario, portfolio, decision)
                <= total_budget_milliunits
            )
        if not legal:
            continue
        policy = _policy_for_legal_decisions(
            scenario,
            portfolio,
            legal,
            dynamic=dynamic,
            commit_activation_optional=optional_commit,
            phase_order=phase_order,
        )
        candidates.append((tuple(portfolio), policy))
    if not candidates:
        raise RuntimeError("budget-filtered search produced no legal policy, including empty")

    portfolio, policy = _deterministic_best(
        candidates,
        lambda item: v103_eval._plan_policy_value(  # noqa: SLF001 - pinned audited dependency
            scenario,
            item[0],
            item[1],
            commit_activation_optional=optional_commit,
        ),
    )
    evaluation = v103_eval._score_policy(  # noqa: SLF001 - pinned audited dependency
        method,
        scenario,
        portfolio,
        policy,
        commit_activation_optional=optional_commit,
    )
    spends = [
        path_spend_milliunits(scenario, portfolio, decision)
        for decision in policy.values()
    ]
    if budget_model is BudgetModel.STAGED:
        violations = sum(
            t0_fee_milliunits(scenario, portfolio)
            > scenario.contract_budget_milliunits
            or t1_fee_milliunits(scenario, portfolio, decision)
            > scenario.exercise_budget_milliunits
            for decision in policy.values()
        )
    else:
        violations = sum(spend > total_budget_milliunits for spend in spends)
    if violations:
        raise AssertionError("E1 evaluator selected an over-budget path")
    return evaluation, EvaluationAudit(
        method=method,
        budget_model=budget_model.value,
        elapsed_seconds=0.0,
        enumerated_portfolios=len(portfolios),
        portfolios_with_legal_path=len(candidates),
        max_selected_path_spend_milliunits=max(spends, default=0),
        hard_budget_milliunits=total_budget_milliunits,
        hard_budget_violations=violations,
    )


def evaluate_e1_method(
    scenario: OptioncastScenario,
    method: str,
    *,
    budget_model: BudgetModel,
    total_budget_milliunits: int,
) -> tuple[PolicyEvaluation, EvaluationAudit]:
    """Evaluate one method and return a separate budget/solver audit record."""
    start = perf_counter()
    evaluation, audit = _evaluate_filtered_exact(
        scenario,
        method,
        budget_model=budget_model,
        total_budget_milliunits=total_budget_milliunits,
    )
    elapsed = perf_counter() - start
    audit = EvaluationAudit(**{**audit.__dict__, "elapsed_seconds": elapsed})
    if evaluation.violations:
        raise RuntimeError(
            f"{method} returned evaluator violations: {evaluation.violations}"
        )
    if audit.hard_budget_violations:
        raise RuntimeError(f"{method} returned an over-budget execution path")
    return evaluation, audit


def evaluation_record(
    evaluation: PolicyEvaluation, audit: EvaluationAudit
) -> dict[str, object]:
    """Serialize result metrics without dumping large policy objects."""
    return {
        "method": evaluation.method,
        "status": "ok",
        "net_value_pp": evaluation.net_value_pp,
        "net_value_ppm": evaluation.net_value_ppm,
        "expected_net_milliunits": evaluation.expected_net_milliunits,
        "expected_gross_value": evaluation.expected_gross_value,
        "expected_t0_fees": evaluation.expected_t0_fees,
        "expected_execution_fees": evaluation.expected_exercise_fees,
        "expected_receipt_grants_consumed": evaluation.expected_receipt_grants_consumed,
        "candidate_noncompletion_rate": evaluation.candidate_noncompletion_rate,
        "authorization_waste_rate": evaluation.authorization_waste_rate,
        "selected_portfolio": [
            {
                "contact_id": choice.contact_id,
                "kind": choice.kind.value,
                "commit_mode_id": choice.commit_mode_id,
            }
            for choice in evaluation.portfolio
        ],
        "audit": {
            "budget_model": audit.budget_model,
            "elapsed_seconds": audit.elapsed_seconds,
            "enumerated_portfolios": audit.enumerated_portfolios,
            "portfolios_with_legal_path": audit.portfolios_with_legal_path,
            "max_selected_path_spend_milliunits": audit.max_selected_path_spend_milliunits,
            "hard_budget_milliunits": audit.hard_budget_milliunits,
            "hard_budget_violations": audit.hard_budget_violations,
        },
        "violations": list(evaluation.violations),
    }


def record_with_path_spends(
    scenario: OptioncastScenario,
    evaluation: PolicyEvaluation,
    audit: EvaluationAudit,
) -> dict[str, object]:
    record = evaluation_record(evaluation, audit)
    record["selected_path_spend_milliunits_by_observation"] = {
        observation_id: path_spend_milliunits(
            scenario, evaluation.portfolio, decision
        )
        for observation_id, decision in evaluation.decisions_by_observation.items()
    }
    return record


__all__ = [
    "EvaluationAudit",
    "MAIN_METHODS",
    "PRIMARY_METHODS",
    "UNIFIED_SUPPORTED_METHODS",
    "evaluate_e1_method",
    "evaluation_record",
    "record_with_path_spends",
]
