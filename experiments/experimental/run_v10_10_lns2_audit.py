"""V10.10 stop-loss audit against a budgeted deterministic 2-exchange LNS.

This module imports the frozen V10.9 congestion family and MILP.  It only
strengthens the sequential baseline: one-swap is followed by at most two
best-improvement rounds over a bounded remove/add-two neighbourhood.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Mapping, Sequence

import numpy as np

from experimental.run_v10_9_milp_scaling import (
    SCALES,
    _activation_legal,
    _pack_posterior_tasks,
    _strong_baseline_decision,
    _bootstrap_ci,
    build_congested_scenario,
    joint_milp_policy,
    score_policy,
)
from experimental.v10_3_optioncast_eval import ContactActivation, T1Decision, decision_plan_value
from experimental.v10_3_optioncast_model import OptioncastScenario, observation_ids
from experimental.v10_4_mismatch_eval import full_option_portfolio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experimental/v10_10_lns2_audit_summary.json"
DEVELOPMENT_SEEDS = (100, 101, 102)
FORMAL_SEEDS = tuple(range(100, 130))
SHORTLIST_SIZE = 8
MAX_CANDIDATES_PER_ROUND = 512
MAX_ROUNDS = 2


@dataclass(frozen=True)
class LnsPolicyResult:
    policy: Mapping[str, T1Decision]
    candidate_evaluations: int
    improving_observations: int


def _legal_set(scenario: OptioncastScenario, active: Sequence[ContactActivation]) -> bool:
    selected: tuple[ContactActivation, ...] = ()
    for item in sorted(active):
        if not _activation_legal(scenario, selected, item):
            return False
        selected = tuple(sorted(selected + (item,)))
    return True


def lns2_decision(
    scenario: OptioncastScenario,
    observation_id: str,
    *,
    shortlist_size: int = SHORTLIST_SIZE,
    max_candidates_per_round: int = MAX_CANDIDATES_PER_ROUND,
    max_rounds: int = MAX_ROUNDS,
    incumbent: T1Decision | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[T1Decision, int]:
    """Improve the one-swap incumbent with a bounded remove/add-two LNS."""
    if shortlist_size < 2 or max_candidates_per_round < 1 or max_rounds < 1:
        raise ValueError("LNS budgets must be positive and shortlist_size at least two")
    portfolio = full_option_portfolio(scenario)
    if incumbent is None:
        incumbent = _strong_baseline_decision(scenario, observation_id)
    incumbent_value = decision_plan_value(scenario, portfolio, incumbent, observation_id)
    evaluations = 0
    all_activations = tuple(sorted(
        ContactActivation(contact.contact_id, mode.mode_id)
        for contact in scenario.contacts for mode in scenario.vcm_modes
    ))

    for _round in range(max_rounds):
        if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
            break
        round_best = incumbent
        round_best_value = incumbent_value
        seen: set[tuple[ContactActivation, ...]] = {incumbent.activations}
        active = incumbent.activations
        removal_sets = [()] + [(item,) for item in active] + list(combinations(active, 2))
        exhausted = False
        for removed in removal_sets:
            reduced = tuple(item for item in active if item not in removed)
            ranked_additions = []
            for candidate in all_activations:
                if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
                    exhausted = True
                    break
                if candidate in reduced or not _activation_legal(scenario, reduced, candidate):
                    continue
                candidate_active = tuple(sorted(reduced + (candidate,)))
                decision = _pack_posterior_tasks(scenario, observation_id, candidate_active)
                value = decision_plan_value(scenario, portfolio, decision, observation_id)
                ranked_additions.append((value, repr(candidate), candidate))
            ranked_additions.sort(key=lambda item: (-item[0], item[1]))
            shortlist = tuple(item[2] for item in ranked_additions[:shortlist_size])
            additions = [()] + [(item,) for item in shortlist] + list(combinations(shortlist, 2))
            for added in additions:
                if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
                    exhausted = True
                    break
                candidate_active = tuple(sorted(reduced + added))
                if candidate_active in seen or not _legal_set(scenario, candidate_active):
                    continue
                seen.add(candidate_active)
                decision = _pack_posterior_tasks(scenario, observation_id, candidate_active)
                value = decision_plan_value(scenario, portfolio, decision, observation_id)
                evaluations += 1
                if value > round_best_value + 1e-9 or (
                    abs(value - round_best_value) <= 1e-9 and repr(decision) < repr(round_best)
                ):
                    round_best, round_best_value = decision, value
                if evaluations >= max_candidates_per_round * (_round + 1):
                    exhausted = True
                    break
            if exhausted:
                break
        if round_best_value <= incumbent_value + 1e-9:
            break
        incumbent, incumbent_value = round_best, round_best_value
    return incumbent, evaluations


def lns2_policy(scenario: OptioncastScenario) -> LnsPolicyResult:
    policy = {}
    evaluations = 0
    improved = 0
    portfolio = full_option_portfolio(scenario)
    for observation_id in sorted(observation_ids(scenario)):
        one_swap = _strong_baseline_decision(scenario, observation_id)
        decision, count = lns2_decision(
            scenario,
            observation_id,
            incumbent=one_swap,
        )
        evaluations += count
        if decision_plan_value(scenario, portfolio, decision, observation_id) > decision_plan_value(
            scenario, portfolio, one_swap, observation_id,
        ) + 1e-9:
            improved += 1
        policy[observation_id] = decision
    return LnsPolicyResult(policy, evaluations, improved)


def run(
    *,
    scales: Sequence[tuple[int, int]] = SCALES,
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
    time_limit_seconds: float = 1.0,
) -> dict[str, object]:
    rows = []
    for contact_count, task_count in scales:
        for seed in seeds:
            scenario = build_congested_scenario(seed, contact_count, task_count)
            portfolio = full_option_portfolio(scenario)
            started = time.perf_counter()
            baseline = lns2_policy(scenario)
            baseline_runtime = time.perf_counter() - started
            started = time.perf_counter()
            hybrid = joint_milp_policy(
                scenario,
                time_limit_seconds=time_limit_seconds,
                baseline_policy=baseline.policy,
            )
            solver_runtime = time.perf_counter() - started
            baseline_value = score_policy(scenario, portfolio, baseline.policy)
            raw_value = score_policy(scenario, portfolio, hybrid.raw_policy)
            hybrid_value = score_policy(scenario, portfolio, hybrid.policy)
            rows.append({
                "seed": seed,
                "contact_count": contact_count,
                "task_count": task_count,
                "lns2_expected_net_milliunits": baseline_value,
                "raw_milp_expected_net_milliunits": raw_value,
                "hybrid_expected_net_milliunits": hybrid_value,
                "raw_paired_gain_milliunits": raw_value - baseline_value,
                "raw_relative_gain": (raw_value - baseline_value) / max(1.0, abs(baseline_value)),
                "hybrid_paired_gain_milliunits": hybrid_value - baseline_value,
                "hybrid_relative_gain": (hybrid_value - baseline_value) / max(1.0, abs(baseline_value)),
                "lns2_runtime_seconds": baseline_runtime,
                "hybrid_runtime_seconds": baseline_runtime + solver_runtime,
                "lns2_candidate_evaluations": baseline.candidate_evaluations,
                "lns2_improving_observations": baseline.improving_observations,
                "hybrid_fallback_count": hybrid.fallback_count,
                "hybrid_status_counts": {
                    status: hybrid.statuses.count(status) for status in sorted(set(hybrid.statuses))
                },
            })
    summary = []
    for contact_count, task_count in scales:
        selected = [row for row in rows if row["contact_count"] == contact_count and row["task_count"] == task_count]
        raw_gains = [float(row["raw_paired_gain_milliunits"]) for row in selected]
        raw_relative = [float(row["raw_relative_gain"]) for row in selected]
        hybrid_gains = [float(row["hybrid_paired_gain_milliunits"]) for row in selected]
        hybrid_relative = [float(row["hybrid_relative_gain"]) for row in selected]
        raw_gain_ci = _bootstrap_ci(raw_gains, seed=20261010 + contact_count + task_count)
        raw_relative_ci = _bootstrap_ci(raw_relative, seed=20262010 + contact_count + task_count)
        hybrid_gain_ci = _bootstrap_ci(hybrid_gains, seed=20263010 + contact_count + task_count)
        hybrid_relative_ci = _bootstrap_ci(hybrid_relative, seed=20264010 + contact_count + task_count)
        summary.append({
            "contact_count": contact_count,
            "task_count": task_count,
            "seed_count": len(selected),
            "raw_relative_gain_median": median(raw_relative),
            "raw_relative_gain_mean": float(np.mean(raw_relative)),
            "raw_relative_gain_mean_95pct_bootstrap_ci": list(raw_relative_ci),
            "raw_paired_gain_mean_95pct_bootstrap_ci": list(raw_gain_ci),
            "raw_win_rate": sum(value > 1e-12 for value in raw_gains) / len(raw_gains),
            "hybrid_relative_gain_median": median(hybrid_relative),
            "hybrid_relative_gain_mean": float(np.mean(hybrid_relative)),
            "hybrid_relative_gain_mean_95pct_bootstrap_ci": list(hybrid_relative_ci),
            "hybrid_paired_gain_mean_95pct_bootstrap_ci": list(hybrid_gain_ci),
            "hybrid_win_rate": sum(value > 1e-12 for value in hybrid_gains) / len(hybrid_gains),
            "lns2_runtime_seconds_median": median(float(row["lns2_runtime_seconds"]) for row in selected),
            "hybrid_runtime_seconds_median": median(float(row["hybrid_runtime_seconds"]) for row in selected),
            "hybrid_runtime_seconds_p95": float(np.quantile([row["hybrid_runtime_seconds"] for row in selected], 0.95)),
            "hybrid_fallback_rate": sum(int(row["hybrid_fallback_count"]) for row in selected)
            / (len(selected) * len(observation_ids(build_congested_scenario(int(selected[0]["seed"]), contact_count, task_count)))),
        })
    return {
        "study": "V10.10 stop-loss audit against budgeted deterministic 2-exchange LNS",
        "lns_budget": {
            "shortlist_size": SHORTLIST_SIZE,
            "max_candidates_per_round": MAX_CANDIDATES_PER_ROUND,
            "max_rounds": MAX_ROUNDS,
        },
        "seeds": list(seeds),
        "time_limit_seconds_per_milp_observation": time_limit_seconds,
        "claim_guardrail": "LNS is bounded and deterministic; it is not an exhaustive two-swap optimum",
        "rows": rows,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--time-limit", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = run(
        seeds=FORMAL_SEEDS if args.formal else DEVELOPMENT_SEEDS,
        time_limit_seconds=args.time_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
