"""Unit tests for the E1 economic adapter."""

from __future__ import annotations

from experimental.v10_3_optioncast_eval import ContactActivation, T1Decision
from experimental.v10_3_optioncast_model import ContractKind, T0ContractChoice
from experimental.v10_3_optioncast_suite import DEVELOPMENT_SEEDS, make_scenario
from q2_strengthening.e1_pricing import (
    BudgetModel,
    EconomicConfig,
    adapt_scenario,
    path_spend_milliunits,
    price_mapping_record,
    round_half_up,
    staged_budgets,
)


def test_rounding_and_staged_budget_preserve_total() -> None:
    assert round_half_up(12.5) == 13
    config = EconomicConfig(0.15, 1.0, 0.5, total_budget_milliunits=255)
    assert staged_budgets(config) == (128, 127)
    assert sum(staged_budgets(config)) == 255


def test_adapter_changes_only_economics_and_preserves_mode_spread() -> None:
    base = make_scenario(DEVELOPMENT_SEEDS[0], mismatch="matching")
    config = EconomicConfig(0.15, 1.0, 0.5)
    adapted = adapt_scenario(base, config, BudgetModel.STAGED)

    assert adapted.tasks == base.tasks
    assert adapted.worlds == base.worlds
    assert adapted.forecast_observations == base.forecast_observations
    assert adapted.score_likelihood_by_world == base.score_likelihood_by_world
    assert adapted.plan_likelihood_by_world == base.plan_likelihood_by_world
    assert adapted.contract_budget_milliunits == 128
    assert adapted.exercise_budget_milliunits == 127
    assert len(base.contacts) == len(adapted.contacts)
    for old, new in zip(base.contacts, adapted.contacts):
        assert new.option_fee_milliunits == round_half_up(
            0.15 * old.commit_fee_milliunits
        )
        assert new.exercise_fee_milliunits_by_mode["robust-qpsk"] == old.commit_fee_milliunits
        old_ratio = (
            old.exercise_fee_milliunits_by_mode["fast-8psk"]
            / old.exercise_fee_milliunits_by_mode["robust-qpsk"]
        )
        new_ratio = (
            new.exercise_fee_milliunits_by_mode["fast-8psk"]
            / new.exercise_fee_milliunits_by_mode["robust-qpsk"]
        )
        assert abs(old_ratio - new_ratio) <= 1.0 / new.commit_fee_milliunits

    mapping = price_mapping_record(base, adapted, config)
    assert len(mapping) == len(base.contacts)
    assert all(row["reference_mode_id"] == "robust-qpsk" for row in mapping)


def test_unified_adapter_does_not_mistake_component_caps_for_combined_cap() -> None:
    base = make_scenario(DEVELOPMENT_SEEDS[0], mismatch="matching")
    config = EconomicConfig(0.15, 0.60, None, total_budget_milliunits=255)
    adapted = adapt_scenario(base, config, BudgetModel.UNIFIED_HARD)
    assert adapted.contract_budget_milliunits == 255
    assert adapted.exercise_budget_milliunits == 255

    contact = adapted.contacts[0]
    portfolio = (T0ContractChoice(contact.contact_id, ContractKind.OPTION),)
    decision = T1Decision(
        (ContactActivation(contact.contact_id, "robust-qpsk"),), ()
    )
    assert path_spend_milliunits(adapted, portfolio, decision) == (
        contact.option_fee_milliunits
        + contact.exercise_fee_milliunits_by_mode["robust-qpsk"]
    )


def test_invalid_budget_semantics_fail_closed() -> None:
    base = make_scenario(DEVELOPMENT_SEEDS[0], mismatch="matching")
    try:
        adapt_scenario(base, EconomicConfig(0.15, 1.0, None), BudgetModel.STAGED)
    except ValueError as exc:
        assert "staged budgets" in str(exc)
    else:
        raise AssertionError("staged model accepted missing s")

    try:
        adapt_scenario(base, EconomicConfig(0.15, 1.0, 0.5), BudgetModel.UNIFIED_HARD)
    except ValueError as exc:
        assert "no s" in str(exc)
    else:
        raise AssertionError("unified model accepted an s parameter")
