"""Tests for E1 hard-budget evaluation, statistics, and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experimental.v10_3_optioncast_eval import (
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    OPTIONCAST_CAP_EXACT,
)
from experimental.v10_3_optioncast_suite import DEVELOPMENT_SEEDS, make_scenario
from q2_strengthening.e1_evaluator import evaluate_e1_method
from q2_strengthening.e1_pricing import BudgetModel, EconomicConfig, adapt_scenario
from q2_strengthening.e1_run import (
    E1_FORMAL_SEEDS,
    build_manifest,
    experiment_configs,
    paired_bootstrap_ci,
    run_e1,
)


def test_formal_seeds_are_new_and_disjoint_from_development() -> None:
    assert len(E1_FORMAL_SEEDS) == 30
    assert not set(E1_FORMAL_SEEDS) & set(range(31))
    assert not set(E1_FORMAL_SEEDS) & set(DEVELOPMENT_SEEDS)


def test_unified_exact_policy_satisfies_every_path_budget() -> None:
    config = EconomicConfig(0.15, 1.0, None, total_budget_milliunits=255)
    scenario = adapt_scenario(
        make_scenario(DEVELOPMENT_SEEDS[0], mismatch="matching"),
        config,
        BudgetModel.UNIFIED_HARD,
    )
    evaluation, audit = evaluate_e1_method(
        scenario,
        OPTIONCAST_CAP_EXACT,
        budget_model=BudgetModel.UNIFIED_HARD,
        total_budget_milliunits=255,
    )
    assert evaluation.method == OPTIONCAST_CAP_EXACT
    assert audit.enumerated_portfolios > 0
    assert audit.portfolios_with_legal_path > 0
    assert audit.hard_budget_violations == 0
    assert audit.max_selected_path_spend_milliunits <= 255


def test_manifest_is_deterministic_and_contains_no_results() -> None:
    configs = experiment_configs(
        a_values=(0.15,),
        b_values=(1.0,),
        s_values=(0.5,),
        total_budget_milliunits=255,
    )
    kwargs = {
        "seeds": (DEVELOPMENT_SEEDS[0],),
        "configs": configs,
        "methods": (OPTIONCAST_CAP_EXACT, COMMIT_PORTFOLIO_DYNAMIC_EXACT),
        "bootstrap_replicates": 20,
        "bootstrap_seed": 7,
        "smoke": True,
    }
    first = build_manifest(**kwargs)
    second = build_manifest(**kwargs)
    assert first == second
    assert len(first["scenario_records"]) == 2
    assert "net_value_pp" not in json.dumps(first)
    without_hash = {key: value for key, value in first.items() if key != "manifest_sha256"}
    from q2_strengthening.e1_run import sha256_object

    assert first["manifest_sha256"] == sha256_object(without_hash)


def test_cluster_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_ci((1.0, 2.0, 3.0), replicates=100, seed=19)
    second = paired_bootstrap_ci((1.0, 2.0, 3.0), replicates=100, seed=19)
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_smoke_run_writes_hashed_jsonl_and_summary(tmp_path: Path) -> None:
    summary = run_e1(
        output_dir=tmp_path,
        seeds=(DEVELOPMENT_SEEDS[0],),
        a_values=(0.15,),
        b_values=(1.0,),
        s_values=(0.5,),
        methods=(OPTIONCAST_CAP_EXACT, COMMIT_PORTFOLIO_DYNAMIC_EXACT),
        bootstrap_replicates=20,
        bootstrap_seed=7,
        smoke=True,
    )
    manifest_path = tmp_path / "e1_manifest.json"
    rows_path = tmp_path / "e1_rows.jsonl"
    summary_path = tmp_path / "e1_summary.json"
    assert manifest_path.is_file() and rows_path.is_file() and summary_path.is_file()
    assert summary["row_count"] == 2
    assert summary["hard_budget_violations"] == 0
    assert summary["primary_endpoint"] is not None
    assert summary["primary_unified_robustness"] is not None
    assert summary["rows_sha256"] == hashlib.sha256(rows_path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert {row["budget_model"] for row in rows} == {"staged", "unified_hard"}
    assert all(set(row["methods"]) == set((OPTIONCAST_CAP_EXACT, COMMIT_PORTFOLIO_DYNAMIC_EXACT)) for row in rows)

    try:
        run_e1(
            output_dir=tmp_path,
            seeds=(DEVELOPMENT_SEEDS[0],),
            a_values=(0.15,),
            b_values=(1.0,),
            s_values=(0.5,),
            methods=(OPTIONCAST_CAP_EXACT, COMMIT_PORTFOLIO_DYNAMIC_EXACT),
            bootstrap_replicates=20,
            smoke=True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("formal artifacts were overwritten without opt-in")
