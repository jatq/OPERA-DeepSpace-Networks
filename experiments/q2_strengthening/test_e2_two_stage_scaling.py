from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest

from experimental.run_v10_9_milp_scaling import build_congested_scenario
from experimental.v10_3_optioncast_model import ContractKind, T0ContractChoice, observation_ids
from q2_strengthening.e1_pricing import BudgetModel, EconomicConfig, adapt_scenario
from q2_strengthening.e2_two_stage_scaling import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_S,
    DEFAULT_TOTAL_BUDGET,
    FORMAL_SEEDS,
    METHODS,
    PortfolioEvaluator,
    _atomic_write_json,
    _formal_output_is_in_source,
    _run_worker,
    _run_search,
    _summarise,
    adapt_e2_economics,
    exact_small_crosscheck,
    lns2_for_portfolio,
    main,
    portfolio_decision_violations,
    run,
)


class E2TwoStageScalingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = build_congested_scenario(0, 6, 8)

    def test_formal_seeds_are_disjoint_from_prior_suites(self) -> None:
        self.assertEqual(len(FORMAL_SEEDS), 30)
        self.assertFalse(set(FORMAL_SEEDS) & set(range(100, 130)))
        self.assertFalse(set(FORMAL_SEEDS) & set(range(20_000, 20_030)))

    def test_e2_economics_matches_e1_and_scales_past_stage0_guard(self) -> None:
        adapted = adapt_e2_economics(self.scenario)
        e1 = adapt_scenario(
            self.scenario,
            EconomicConfig(DEFAULT_A, DEFAULT_B, DEFAULT_S, DEFAULT_TOTAL_BUDGET),
            BudgetModel.STAGED,
        )
        self.assertEqual(adapted.contacts, e1.contacts)
        self.assertEqual(adapted.contract_budget_milliunits, 128)
        self.assertEqual(adapted.exercise_budget_milliunits, 127)
        large = adapt_e2_economics(build_congested_scenario(0, 30, 100))
        self.assertEqual(len(large.contacts), 30)
        self.assertEqual(len(large.tasks), 100)
        self.assertTrue(any(
            contact.commit_fee_milliunits <= large.contract_budget_milliunits
            for contact in large.contacts
        ))

    def test_default_worker_records_economics_and_affordable_commit(self) -> None:
        row = _run_worker(
            method="task_aware_greedy",
            seed=0,
            contact_count=6,
            task_count=8,
            budget_seconds=0.02,
        )
        self.assertEqual(row["economic_config"]["b0_contract_milliunits"], 128)
        self.assertEqual(row["economic_config"]["b1_execution_milliunits"], 127)
        self.assertGreater(row["affordable_singleton_commit_count"], 0)
        self.assertTrue(row["legal"])

    def test_output_guards_and_atomic_summary(self) -> None:
        self.assertTrue(_formal_output_is_in_source(
            Path(__file__).resolve().parent / "formal.json",
        ))
        with self.assertRaises(SystemExit):
            main(["--formal"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            _atomic_write_json(summary_path, {"value": 1}, overwrite=False)
            self.assertEqual(json.loads(summary_path.read_text())["value"], 1)
            with self.assertRaises(FileExistsError):
                _atomic_write_json(summary_path, {"value": 2}, overwrite=False)
            rows_path = root / "rows.jsonl"
            rows_path.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(
                    scales=((6, 8),),
                    seeds=(0,),
                    budget_seconds=0.01,
                    rows_path=rows_path,
                )

    def test_explicit_portfolio_lns_is_complete_and_legal(self) -> None:
        contact = self.scenario.contacts[0]
        portfolio = (T0ContractChoice(contact.contact_id, ContractKind.OPTION),)
        observation_id = sorted(observation_ids(self.scenario))[0]
        decision, _evaluations = lns2_for_portfolio(
            self.scenario,
            portfolio,
            observation_id,
            deadline=time.perf_counter() + 1.0,
        )
        self.assertEqual(portfolio_decision_violations(self.scenario, portfolio, decision), ())
        self.assertTrue(all(item.contact_id == contact.contact_id for item in decision.activations))

    def test_commit_is_mandatory_and_keeps_its_t0_mode(self) -> None:
        contact = self.scenario.contacts[0]
        mode = self.scenario.vcm_modes[0]
        scenario = replace(
            self.scenario,
            contract_budget_milliunits=max(
                self.scenario.contract_budget_milliunits,
                contact.commit_fee_milliunits,
            ),
        )
        portfolio = (
            T0ContractChoice(contact.contact_id, ContractKind.COMMIT, mode.mode_id),
        )
        observation_id = sorted(observation_ids(scenario))[0]
        decision, _evaluations = lns2_for_portfolio(
            scenario,
            portfolio,
            observation_id,
            deadline=time.perf_counter() + 1.0,
        )
        self.assertEqual(portfolio_decision_violations(scenario, portfolio, decision), ())
        self.assertIn(
            (contact.contact_id, mode.mode_id),
            {(item.contact_id, item.mode_id) for item in decision.activations},
        )

    def test_timeout_keeps_last_complete_legal_portfolio(self) -> None:
        for method in METHODS:
            result = _run_search(self.scenario, method, budget_seconds=0.02)
            self.assertEqual(set(result.evaluation.policy), set(observation_ids(self.scenario)))
            for decision in result.evaluation.policy.values():
                self.assertEqual(
                    portfolio_decision_violations(
                        self.scenario, result.evaluation.portfolio, decision,
                    ),
                    (),
                )

    def test_atomic_cache_never_keeps_partial_candidate(self) -> None:
        evaluator = PortfolioEvaluator(self.scenario, time.perf_counter() + 1.0)
        result = evaluator.evaluate(())
        self.assertEqual(set(result.policy), set(observation_ids(self.scenario)))
        self.assertIn((), evaluator.cache)

    def test_small_exact_crosscheck_is_an_upper_bound(self) -> None:
        result = exact_small_crosscheck(seed=0, budget_seconds=0.25)
        self.assertTrue(result["heuristic_not_above_exact"])
        self.assertGreaterEqual(result["absolute_gap_milliunits"], -1e-9)
        self.assertGreater(result["exact_portfolio_count"], 0)

    def test_summary_pairs_by_seed_not_observation(self) -> None:
        rows = []
        for seed in (1, 2):
            for index, method in enumerate(METHODS):
                rows.append({
                    "seed": seed,
                    "contact_count": 6,
                    "task_count": 8,
                    "method": method,
                    "scored_net_vbs": float(seed + index),
                    "strategy_runtime_seconds": 0.1,
                    "peak_rss_bytes": 100,
                    "timed_out": False,
                    "fallback_used": False,
                    "legal": True,
                })
        summary = _summarise(rows)[0]
        comparison = summary["paired_comparisons_vbs_points"]["candidate_minus_multistart_1swap"]
        self.assertEqual(comparison["win_tie_loss"], [2, 0, 0])


if __name__ == "__main__":
    unittest.main()
