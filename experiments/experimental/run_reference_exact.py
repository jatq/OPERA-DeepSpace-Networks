"""Regenerate the 31-seed exact reference comparison.

The Stage-0 evaluator intentionally does not build a suite.  This runner
supplies the preregistered seeds 0--30, evaluates every deployable policy on
the same matching scenario, and writes row-level results plus paired summary
statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from typing import Sequence

import numpy as np

from experimental.run_v10_9_milp_scaling import _bootstrap_ci
from experimental.v10_3_optioncast_eval import (
    CAP_THEN_FORECAST_OPTION,
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    DEPLOYABLE_METHODS,
    FORECAST_OPTION_THEN_CAP,
    OPTION_STATIC_EXACT,
    OPTIONCAST_CAP_EXACT,
    PREBOUND_GRANT_OPTION,
    evaluate_method,
)
from experimental.v10_3_optioncast_suite import make_scenario


REFERENCE_SEEDS = tuple(range(31))
PROPOSED = OPTIONCAST_CAP_EXACT
BASELINES = (
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    CAP_THEN_FORECAST_OPTION,
    FORECAST_OPTION_THEN_CAP,
    PREBOUND_GRANT_OPTION,
    OPTION_STATIC_EXACT,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: Sequence[float], *, seed: int) -> dict[str, object]:
    tolerance = 1e-12
    low, high = _bootstrap_ci(values, seed=seed)
    return {
        "seed_count": len(values),
        "mean_gain_pp": float(np.mean(values)),
        "median_gain_pp": median(values),
        "mean_95pct_bootstrap_ci_pp": [low, high],
        "wins_ties_losses": [
            sum(value > tolerance for value in values),
            sum(abs(value) <= tolerance for value in values),
            sum(value < -tolerance for value in values),
        ],
    }


def run(*, output_dir: Path, seeds: Sequence[int] = REFERENCE_SEEDS) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        scenario = make_scenario(seed, mismatch="matching")
        evaluations = {method: evaluate_method(scenario, method) for method in DEPLOYABLE_METHODS}
        proposed = evaluations[PROPOSED]
        row: dict[str, object] = {
            "seed": seed,
            "scenario_id": scenario.scenario_id,
            "proposed_method": PROPOSED,
            "methods": {
                method: {
                    "net_value_pp": evaluation.net_value_pp,
                    "expected_net_milliunits": evaluation.expected_net_milliunits,
                    "authorization_waste_rate": evaluation.authorization_waste_rate,
                    "violations": list(evaluation.violations),
                }
                for method, evaluation in evaluations.items()
            },
            "paired_gain_pp": {
                baseline: proposed.net_value_pp - evaluations[baseline].net_value_pp
                for baseline in BASELINES
            },
        }
        if any(evaluation.violations for evaluation in evaluations.values()):
            raise RuntimeError(f"exact reference produced a violation for seed {seed}")
        rows.append(row)

    summary_by_baseline = {
        baseline: _summary(
            [float(row["paired_gain_pp"][baseline]) for row in rows],
            seed=20261231 + index,
        )
        for index, baseline in enumerate(BASELINES)
    }
    rows_path = output_dir / "reference_exact_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "study": "31-seed exact reference comparison",
        "seeds": list(seeds),
        "statistical_unit": "seed/scenario; worlds and observations remain clustered",
        "proposed_method": PROPOSED,
        "summary_by_baseline": summary_by_baseline,
        "row_count": len(rows),
        "rows_sha256": _sha256(rows_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (output_dir / "reference_exact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(output_dir=args.output_dir)
    print(json.dumps(summary["summary_by_baseline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
