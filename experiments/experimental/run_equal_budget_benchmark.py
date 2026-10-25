"""Equal-wall-clock comparison of scalable post-observation planners.

Each public observation receives the same wall-clock cap.  Both competing
deployable methods first construct the same Sequential-1Swap incumbent and
spend only the remaining time on either LNS-2 or the expected-capacity MILP.
The outer runner executes each method in a fresh subprocess so peak RSS is not
contaminated by methods evaluated earlier in the same Python process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np

try:
    import resource
except ImportError:  # Windows
    resource = None

from experimental.run_v10_10_lns2_audit import lns2_decision
from experimental.run_v10_9_milp_scaling import (
    SCALES,
    _bootstrap_ci,
    _pack_posterior_tasks,
    _strong_baseline_decision,
    build_congested_scenario,
    score_policy,
    solve_observation_milp,
)
from experimental.v10_3_optioncast_eval import T1Decision, decision_plan_value
from experimental.v10_3_optioncast_model import observation_ids
from experimental.v10_4_mismatch_eval import full_option_portfolio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experimental/equal_budget_benchmark_summary.json"
FORMAL_SEEDS = tuple(range(100, 130))
SMOKE_SEEDS = (0, 1, 2)
MIN_SOLVER_SECONDS = 1e-3


def _peak_rss_bytes() -> int:
    if resource is not None:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    import psutil
    return int(psutil.Process().memory_info().rss)


def _run_lns_worker(
    *, seed: int, contact_count: int, task_count: int, budget_seconds: float,
) -> dict[str, object]:
    scenario = build_congested_scenario(seed, contact_count, task_count)
    portfolio = full_option_portfolio(scenario)
    sequential_policy: dict[str, T1Decision] = {}
    lns_policy: dict[str, T1Decision] = {}
    sequential_runtime = 0.0
    candidate_evaluations = 0
    budget_exhausted = 0
    started = time.perf_counter()
    for observation_id in sorted(observation_ids(scenario)):
        observation_started = time.perf_counter()
        sequential_started = time.perf_counter()
        incumbent = _strong_baseline_decision(scenario, observation_id)
        sequential_runtime += time.perf_counter() - sequential_started
        sequential_policy[observation_id] = incumbent
        deadline = observation_started + budget_seconds
        improved, evaluations = lns2_decision(
            scenario,
            observation_id,
            incumbent=incumbent,
            deadline_monotonic=deadline,
        )
        candidate_evaluations += evaluations
        if time.perf_counter() >= deadline:
            budget_exhausted += 1
        lns_policy[observation_id] = improved
    runtime = time.perf_counter() - started
    return {
        "worker": "lns2",
        "sequential_1swap_expected_net_milliunits": score_policy(
            scenario, portfolio, sequential_policy,
        ),
        "lns2_expected_net_milliunits": score_policy(scenario, portfolio, lns_policy),
        "sequential_runtime_seconds": sequential_runtime,
        "lns2_runtime_seconds": runtime,
        "lns2_candidate_evaluations": candidate_evaluations,
        "budget_exhausted_observations": budget_exhausted,
        "observation_count": len(lns_policy),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _run_hybrid_worker(
    *, seed: int, contact_count: int, task_count: int, budget_seconds: float,
) -> dict[str, object]:
    scenario = build_congested_scenario(seed, contact_count, task_count)
    portfolio = full_option_portfolio(scenario)
    raw_policy: dict[str, T1Decision] = {}
    hybrid_policy: dict[str, T1Decision] = {}
    sequential_runtime = 0.0
    fallback_count = 0
    statuses: list[str] = []
    budget_exhausted = 0
    started = time.perf_counter()
    for observation_id in sorted(observation_ids(scenario)):
        observation_started = time.perf_counter()
        sequential_started = time.perf_counter()
        incumbent = _strong_baseline_decision(scenario, observation_id)
        sequential_runtime += time.perf_counter() - sequential_started
        remaining = budget_seconds - (time.perf_counter() - observation_started)
        if remaining <= MIN_SOLVER_SECONDS:
            projected = T1Decision((), ())
            status = "not_run_budget_exhausted"
            budget_exhausted += 1
        else:
            raw, metadata = solve_observation_milp(
                scenario,
                observation_id,
                time_limit_seconds=remaining,
                mip_rel_gap=0.01,
            )
            projected = _pack_posterior_tasks(
                scenario, observation_id, raw.activations,
            )
            status = str(metadata["status"])
            if time.perf_counter() >= observation_started + budget_seconds:
                budget_exhausted += 1
        raw_policy[observation_id] = projected
        if decision_plan_value(scenario, portfolio, incumbent, observation_id) > decision_plan_value(
            scenario, portfolio, projected, observation_id,
        ) + 1e-9:
            hybrid_policy[observation_id] = incumbent
            fallback_count += 1
        else:
            hybrid_policy[observation_id] = projected
        statuses.append(status)
    runtime = time.perf_counter() - started
    return {
        "worker": "od_jmilp_ec_hybrid",
        "projection_expected_net_milliunits": score_policy(scenario, portfolio, raw_policy),
        "hybrid_expected_net_milliunits": score_policy(scenario, portfolio, hybrid_policy),
        "sequential_runtime_seconds": sequential_runtime,
        "hybrid_runtime_seconds": runtime,
        "fallback_count": fallback_count,
        "budget_exhausted_observations": budget_exhausted,
        "observation_count": len(hybrid_policy),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _invoke_worker(
    worker: str,
    *,
    seed: int,
    contact_count: int,
    task_count: int,
    budget_seconds: float,
) -> dict[str, object]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    command = [
        sys.executable,
        "-m",
        "experimental.run_equal_budget_benchmark",
        "--worker",
        worker,
        "--seed",
        str(seed),
        "--contacts",
        str(contact_count),
        "--tasks",
        str(task_count),
        "--budget-seconds",
        str(budget_seconds),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _paired_summary(rows: Sequence[Mapping[str, object]], left: str, right: str, seed: int) -> dict[str, object]:
    differences = [float(row[left]) - float(row[right]) for row in rows]
    relative = [
        difference / max(1.0, abs(float(row[right])))
        for row, difference in zip(rows, differences)
    ]
    tolerance = 1e-9
    return {
        "left_metric": left,
        "right_metric": right,
        "paired_gain_mean_milliunits": float(np.mean(differences)),
        "paired_gain_median_milliunits": median(differences),
        "paired_gain_mean_95pct_bootstrap_ci": list(_bootstrap_ci(differences, seed=seed)),
        "relative_gain_mean": float(np.mean(relative)),
        "relative_gain_median": median(relative),
        "relative_gain_mean_95pct_bootstrap_ci": list(_bootstrap_ci(relative, seed=seed + 1)),
        "win_tie_loss": [
            sum(value > tolerance for value in differences),
            sum(abs(value) <= tolerance for value in differences),
            sum(value < -tolerance for value in differences),
        ],
    }


def run(
    *,
    scales: Sequence[tuple[int, int]] = SCALES,
    seeds: Sequence[int] = FORMAL_SEEDS,
    budget_seconds: float = 1.0,
) -> dict[str, object]:
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    rows: list[dict[str, object]] = []
    for contact_count, task_count in scales:
        for seed in seeds:
            worker_order = ("lns2", "hybrid") if seed % 2 == 0 else ("hybrid", "lns2")
            worker_results = {
                worker: _invoke_worker(
                    worker,
                    seed=seed,
                    contact_count=contact_count,
                    task_count=task_count,
                    budget_seconds=budget_seconds,
                )
                for worker in worker_order
            }
            lns = worker_results["lns2"]
            hybrid = worker_results["hybrid"]
            rows.append({
                "instance_id": f"c{contact_count}-t{task_count}-s{seed}",
                "seed": seed,
                "contact_count": contact_count,
                "task_count": task_count,
                "worker_order": list(worker_order),
                "observation_count": int(lns["observation_count"]),
                "sequential_1swap_expected_net_milliunits": float(
                    lns["sequential_1swap_expected_net_milliunits"],
                ),
                "lns2_expected_net_milliunits": float(lns["lns2_expected_net_milliunits"]),
                "projection_expected_net_milliunits": float(
                    hybrid["projection_expected_net_milliunits"],
                ),
                "hybrid_expected_net_milliunits": float(hybrid["hybrid_expected_net_milliunits"]),
                "sequential_runtime_seconds": float(lns["sequential_runtime_seconds"]),
                "lns2_runtime_seconds": float(lns["lns2_runtime_seconds"]),
                "hybrid_runtime_seconds": float(hybrid["hybrid_runtime_seconds"]),
                "lns2_candidate_evaluations": int(lns["lns2_candidate_evaluations"]),
                "lns2_budget_exhausted_observations": int(lns["budget_exhausted_observations"]),
                "hybrid_budget_exhausted_observations": int(hybrid["budget_exhausted_observations"]),
                "hybrid_fallback_count": int(hybrid["fallback_count"]),
                "hybrid_status_counts": hybrid["status_counts"],
                "lns2_peak_rss_bytes": int(lns["peak_rss_bytes"]),
                "hybrid_peak_rss_bytes": int(hybrid["peak_rss_bytes"]),
            })
    summaries = []
    for contact_count, task_count in scales:
        selected = [
            row for row in rows
            if row["contact_count"] == contact_count and row["task_count"] == task_count
        ]
        summaries.append({
            "contact_count": contact_count,
            "task_count": task_count,
            "seed_count": len(selected),
            "hybrid_minus_lns2": _paired_summary(
                selected,
                "hybrid_expected_net_milliunits",
                "lns2_expected_net_milliunits",
                seed=20261101 + contact_count + task_count,
            ),
            "projection_minus_lns2": _paired_summary(
                selected,
                "projection_expected_net_milliunits",
                "lns2_expected_net_milliunits",
                seed=20261201 + contact_count + task_count,
            ),
            "lns2_minus_sequential": _paired_summary(
                selected,
                "lns2_expected_net_milliunits",
                "sequential_1swap_expected_net_milliunits",
                seed=20261301 + contact_count + task_count,
            ),
            "lns2_runtime_median_seconds": median(float(row["lns2_runtime_seconds"]) for row in selected),
            "lns2_runtime_p95_seconds": float(np.quantile([row["lns2_runtime_seconds"] for row in selected], 0.95)),
            "hybrid_runtime_median_seconds": median(float(row["hybrid_runtime_seconds"]) for row in selected),
            "hybrid_runtime_p95_seconds": float(np.quantile([row["hybrid_runtime_seconds"] for row in selected], 0.95)),
            "lns2_peak_rss_median_bytes": median(int(row["lns2_peak_rss_bytes"]) for row in selected),
            "hybrid_peak_rss_median_bytes": median(int(row["hybrid_peak_rss_bytes"]) for row in selected),
            "lns2_budget_exhausted_rate": sum(
                int(row["lns2_budget_exhausted_observations"]) for row in selected
            ) / sum(int(row["observation_count"]) for row in selected),
            "hybrid_budget_exhausted_rate": sum(
                int(row["hybrid_budget_exhausted_observations"]) for row in selected
            ) / sum(int(row["observation_count"]) for row in selected),
        })
    return {
        "study": "Equal-wall-clock scalable post-observation planner comparison",
        "budget": {
            "mode": "wall_clock_per_observation",
            "seconds": budget_seconds,
            "sequential_incumbent_time_included": True,
            "thread_limits": 1,
        },
        "statistical_unit": "one seeded scenario; observations are not treated as independent samples",
        "seeds": list(seeds),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "processor": platform.processor(),
        },
        "rows": rows,
        "summary_by_scale": summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=1.0)
    parser.add_argument("--worker", choices=("lns2", "hybrid"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--contacts", type=int)
    parser.add_argument("--tasks", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.seed is None or args.contacts is None or args.tasks is None:
            parser.error("worker mode requires --seed, --contacts, and --tasks")
        function = _run_lns_worker if args.worker == "lns2" else _run_hybrid_worker
        result = function(
            seed=args.seed,
            contact_count=args.contacts,
            task_count=args.tasks,
            budget_seconds=args.budget_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    result = run(
        seeds=FORMAL_SEEDS if args.formal else SMOKE_SEEDS,
        budget_seconds=args.budget_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary_by_scale"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
