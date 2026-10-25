"""Economic-parameter adapter for the E1 price/budget boundary experiment.

The adapter changes only prices and the two monetary budgets of a frozen
V10.3 scenario.  Calendars, tasks, truth worlds, observations, authorization
rules, radio/energy limits, and forecast channels are preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Mapping, Sequence

from experimental.v10_3_optioncast_eval import T1Decision
from experimental.v10_3_optioncast_model import (
    CONTROL_MESSAGE_PRICE_MILLIUNITS,
    CandidateContact,
    ContractKind,
    OptioncastScenario,
    T0ContractChoice,
    exercise_fee,
    validate_scenario,
)


DEFAULT_TOTAL_BUDGET_MILLIUNITS = 255
DEFAULT_REFERENCE_MODE_ID = "robust-qpsk"


class BudgetModel(str, Enum):
    """The two economic environments reported by E1."""

    STAGED = "staged"
    UNIFIED_HARD = "unified_hard"


@dataclass(frozen=True)
class EconomicConfig:
    """One predeclared E1 grid point.

    ``a`` is the option fee divided by the contact commitment fee. ``b`` is
    the reference-mode exercise fee divided by the same commitment fee.
    Other mode prices preserve their original multiplicative spread relative
    to ``reference_mode_id``.  Integer prices use decimal half-up rounding.
    """

    a: float
    b: float
    s: float | None
    total_budget_milliunits: int = DEFAULT_TOTAL_BUDGET_MILLIUNITS
    reference_mode_id: str = DEFAULT_REFERENCE_MODE_ID

    def validate(self, budget_model: BudgetModel) -> None:
        if not 0.0 < self.a:
            raise ValueError("a must be positive")
        if not 0.0 < self.b:
            raise ValueError("b must be positive")
        if self.total_budget_milliunits <= 1:
            raise ValueError("total budget must be greater than one")
        if budget_model is BudgetModel.STAGED:
            if self.s is None or not 0.0 < self.s < 1.0:
                raise ValueError("staged budgets require 0 < s < 1")
        elif self.s is not None:
            raise ValueError("the unified-hard environment has no s parameter")


def round_half_up(value: float) -> int:
    """Round a nonnegative price to the integer minimum accounting unit."""
    if value < 0.0:
        raise ValueError("prices cannot be negative")
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def staged_budgets(config: EconomicConfig) -> tuple[int, int]:
    """Return ``(B0, B1)`` while keeping ``B0 + B1`` exactly fixed."""
    config.validate(BudgetModel.STAGED)
    assert config.s is not None
    b0 = round_half_up(config.total_budget_milliunits * config.s)
    b0 = min(max(1, b0), config.total_budget_milliunits - 1)
    return b0, config.total_budget_milliunits - b0


def _adapt_contact(contact: CandidateContact, config: EconomicConfig) -> CandidateContact:
    try:
        reference_fee = contact.exercise_fee_milliunits_by_mode[config.reference_mode_id]
    except KeyError as exc:
        raise ValueError(
            f"contact {contact.contact_id} has no reference mode {config.reference_mode_id}"
        ) from exc
    if reference_fee <= 0:
        raise ValueError("reference exercise fee must be positive")

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
    return replace(
        contact,
        option_fee_milliunits=option_fee,
        exercise_fee_milliunits_by_mode=exercise_fees,
    )


def adapt_scenario(
    scenario: OptioncastScenario,
    config: EconomicConfig,
    budget_model: BudgetModel,
) -> OptioncastScenario:
    """Return a price/budget-only variant of a frozen V10.3 scenario.

    In the unified environment, both legacy component caps are set to the
    total budget so they do not remove a path prematurely.  The evaluator
    additionally enforces ``t0 fee + path-specific t1 fee <= B_total`` for
    every observation path; setting the component caps alone is not treated
    as sufficient.
    """
    config.validate(budget_model)
    contacts = tuple(_adapt_contact(contact, config) for contact in scenario.contacts)
    if budget_model is BudgetModel.STAGED:
        b0, b1 = staged_budgets(config)
        suffix = f"e1-a{config.a:g}-b{config.b:g}-s{config.s:g}-staged"
    else:
        b0 = b1 = config.total_budget_milliunits
        suffix = f"e1-a{config.a:g}-b{config.b:g}-unified-hard"
    adapted = replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}-{suffix}",
        contacts=contacts,
        contract_budget_milliunits=b0,
        exercise_budget_milliunits=b1,
    )
    validate_scenario(adapted)
    return adapted


def t0_fee_milliunits(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice]
) -> int:
    contacts = {contact.contact_id: contact for contact in scenario.contacts}
    return sum(
        contacts[choice.contact_id].commit_fee_milliunits
        if choice.kind is ContractKind.COMMIT
        else contacts[choice.contact_id].option_fee_milliunits
        for choice in portfolio
    )


def t1_fee_milliunits(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decision: T1Decision,
) -> int:
    contacts = {contact.contact_id: contact for contact in scenario.contacts}
    choices = {choice.contact_id: choice for choice in portfolio}
    exercise_spend = sum(
        exercise_fee(contacts[activation.contact_id], activation.mode_id)
        for activation in decision.activations
        if choices[activation.contact_id].kind is ContractKind.OPTION
    )
    control_spend = len(decision.assignments) * CONTROL_MESSAGE_PRICE_MILLIUNITS
    return exercise_spend + control_spend


def path_spend_milliunits(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decision: T1Decision,
) -> int:
    return t0_fee_milliunits(scenario, portfolio) + t1_fee_milliunits(
        scenario, portfolio, decision
    )


def price_mapping_record(
    original: OptioncastScenario, adapted: OptioncastScenario, config: EconomicConfig
) -> list[dict[str, object]]:
    """Return the auditable old/new fee mapping stored with every run."""
    old_by_id = {contact.contact_id: contact for contact in original.contacts}
    records: list[dict[str, object]] = []
    for new in adapted.contacts:
        old = old_by_id[new.contact_id]
        records.append(
            {
                "contact_id": new.contact_id,
                "commit_fee_milliunits": new.commit_fee_milliunits,
                "old_option_fee_milliunits": old.option_fee_milliunits,
                "new_option_fee_milliunits": new.option_fee_milliunits,
                "old_exercise_fees_milliunits_by_mode": dict(
                    old.exercise_fee_milliunits_by_mode
                ),
                "new_exercise_fees_milliunits_by_mode": dict(
                    new.exercise_fee_milliunits_by_mode
                ),
                "realised_a": new.option_fee_milliunits / new.commit_fee_milliunits,
                "realised_b_by_mode": {
                    mode: fee / new.commit_fee_milliunits
                    for mode, fee in new.exercise_fee_milliunits_by_mode.items()
                },
                "reference_mode_id": config.reference_mode_id,
            }
        )
    return records


def budget_record(
    scenario: OptioncastScenario,
    config: EconomicConfig,
    budget_model: BudgetModel,
) -> Mapping[str, object]:
    if budget_model is BudgetModel.STAGED:
        b0, b1 = staged_budgets(config)
        return {
            "model": budget_model.value,
            "s": config.s,
            "b0_contract_milliunits": b0,
            "b1_execution_milliunits": b1,
            "b_total_milliunits": config.total_budget_milliunits,
            "hard_rule": "t0<=B0 and each t1 path<=B1",
        }
    return {
        "model": budget_model.value,
        "s": None,
        "legacy_component_caps_milliunits": {
            "contract": scenario.contract_budget_milliunits,
            "execution": scenario.exercise_budget_milliunits,
        },
        "b_total_milliunits": config.total_budget_milliunits,
        "hard_rule": "for every observation path: t0+t1<=B_total",
    }


__all__ = [
    "BudgetModel",
    "EconomicConfig",
    "DEFAULT_REFERENCE_MODE_ID",
    "DEFAULT_TOTAL_BUDGET_MILLIUNITS",
    "adapt_scenario",
    "budget_record",
    "path_spend_milliunits",
    "price_mapping_record",
    "round_half_up",
    "staged_budgets",
    "t0_fee_milliunits",
    "t1_fee_milliunits",
]
