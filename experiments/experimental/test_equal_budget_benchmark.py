import unittest

from experimental.run_equal_budget_benchmark import (
    _paired_summary,
    _run_hybrid_worker,
    _run_lns_worker,
)


class EqualBudgetBenchmarkTest(unittest.TestCase):
    def test_workers_cover_all_observations_and_return_finite_scores(self) -> None:
        lns = _run_lns_worker(
            seed=0,
            contact_count=10,
            task_count=20,
            budget_seconds=0.05,
        )
        hybrid = _run_hybrid_worker(
            seed=0,
            contact_count=10,
            task_count=20,
            budget_seconds=0.05,
        )
        self.assertEqual(lns["observation_count"], hybrid["observation_count"])
        self.assertGreater(lns["observation_count"], 0)
        self.assertGreaterEqual(
            lns["lns2_expected_net_milliunits"] + 1e-9,
            lns["sequential_1swap_expected_net_milliunits"],
        )
        self.assertGreater(lns["peak_rss_bytes"], 0)
        self.assertGreater(hybrid["peak_rss_bytes"], 0)

    def test_paired_summary_uses_scenario_level_differences(self) -> None:
        rows = [
            {"left": 12.0, "right": 10.0},
            {"left": 9.0, "right": 10.0},
            {"left": 10.0, "right": 10.0},
        ]
        summary = _paired_summary(rows, "left", "right", seed=7)
        self.assertEqual(summary["win_tie_loss"], [1, 1, 1])
        self.assertAlmostEqual(summary["paired_gain_mean_milliunits"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
