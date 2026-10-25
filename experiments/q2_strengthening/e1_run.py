"""Run the E1 option-price and budget-boundary experiment.

Examples
--------
Smoke test (one development seed, one grid point, both budget models)::

    python -m q2_strengthening.e1_run --smoke --output-dir /tmp/e1-smoke

Formal default grid (30 independent seeds)::

    python -m q2_strengthening.e1_run --output-dir results/e1-formal

The runner freezes a result-free manifest before evaluating any method.  It
emits JSONL scenario/cell details and a JSON paired summary.  Existing formal
artifacts are never replaced unless ``--overwrite`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from experimental.v10_3_optioncast_eval import (
    COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    OPTIONCAST_CAP_EXACT,
)
from experimental.v10_3_optioncast_suite import (
    DEVELOPMENT_SEEDS,
    SUITE_VERSION,
    make_scenario,
)
from q2_strengthening.e1_evaluator import (
    MAIN_METHODS,
    UNIFIED_SUPPORTED_METHODS,
    evaluate_e1_method,
    record_with_path_spends,
)
from q2_strengthening.e1_pricing import (
    BudgetModel,
    EconomicConfig,
    adapt_scenario,
    budget_record,
    price_mapping_record,
)


E1_RUN_VERSION = "e1-price-budget-boundary-1"
DEFAULT_A_VALUES = (0.05, 0.15, 0.30, 0.50)
DEFAULT_B_VALUES = (0.60, 1.00, 1.40)
DEFAULT_S_VALUES = (0.25, 0.50, 0.75)
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 10_311
# Disjoint from both the V10.3 score seeds (0--30) and its development seeds
# (10_000--10_004).  The values are frozen here before any E1 scoring.
E1_FORMAL_SEEDS = tuple(range(20_000, 20_030))
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experimental/v10_3_optioncast_model.py",
    "experimental/v10_3_optioncast_suite.py",
    "experimental/v10_3_optioncast_eval.py",
    "q2_strengthening/e1_pricing.py",
    "q2_strengthening/e1_evaluator.py",
    "q2_strengthening/e1_run.py",
)


def _normalise(value: object) -> object:
    if is_dataclass(value):
        return _normalise(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalise(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_normalise(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot canonicalise {type(value)!r}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_object(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def source_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"declared E1 source is missing: {path}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object, *, overwrite: bool) -> None:
    data = json.dumps(
        _normalise(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ).encode("utf-8") + b"\n"
    _atomic_write(path, data, overwrite=overwrite)


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(
    values: Sequence[float], *, replicates: int, seed: int
) -> tuple[float, float]:
    """Deterministic percentile CI resampling whole paired seed clusters."""
    if not values:
        raise ValueError("bootstrap needs at least one seed cluster")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    )
    return _linear_quantile(means, 0.025), _linear_quantile(means, 0.975)


def _cell_id(
    budget_model: BudgetModel, a: float, b: float, s: float | None
) -> str:
    if budget_model is BudgetModel.STAGED:
        return f"staged:a={a:g}:b={b:g}:s={s:g}"
    return f"unified_hard:a={a:g}:b={b:g}"


def experiment_configs(
    *,
    a_values: Sequence[float],
    b_values: Sequence[float],
    s_values: Sequence[float],
    total_budget_milliunits: int,
) -> tuple[tuple[BudgetModel, EconomicConfig], ...]:
    configs: list[tuple[BudgetModel, EconomicConfig]] = []
    for a in a_values:
        for b in b_values:
            for s in s_values:
                config = EconomicConfig(a, b, s, total_budget_milliunits)
                config.validate(BudgetModel.STAGED)
                configs.append((BudgetModel.STAGED, config))
            unified = EconomicConfig(a, b, None, total_budget_milliunits)
            unified.validate(BudgetModel.UNIFIED_HARD)
            configs.append((BudgetModel.UNIFIED_HARD, unified))
    return tuple(configs)


def build_manifest(
    *,
    seeds: Sequence[int],
    configs: Sequence[tuple[BudgetModel, EconomicConfig]],
    methods: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    smoke: bool,
) -> dict[str, object]:
    """Build the result-free freeze, including every adapted input hash."""
    scenario_records: list[dict[str, object]] = []
    for seed in seeds:
        base = make_scenario(seed, mismatch="matching")
        for budget_model, config in configs:
            adapted = adapt_scenario(base, config, budget_model)
            scenario_records.append(
                {
                    "seed": seed,
                    "cell_id": _cell_id(
                        budget_model, config.a, config.b, config.s
                    ),
                    "budget": budget_record(adapted, config, budget_model),
                    "adapted_scenario_sha256": sha256_object(adapted),
                    "price_mapping": price_mapping_record(base, adapted, config),
                }
            )
    manifest: dict[str, object] = {
        "run_version": E1_RUN_VERSION,
        "formal_run_contract": (
            "freeze_before_scoring; paired_seed_clusters; abort_on_solver_error; "
            "no_posthoc_grid_selection"
        ),
        "smoke": smoke,
        "base_suite_version": SUITE_VERSION,
        "base_condition": "matching",
        "seeds": list(seeds),
        "independent_seed_cluster_count": len(seeds),
        "methods": list(methods),
        "proposed_method": OPTIONCAST_CAP_EXACT,
        "primary_baseline": COMMIT_PORTFOLIO_DYNAMIC_EXACT,
        "pricing_rule": {
            "a": "option_fee / contact_commit_fee",
            "b": "reference_mode_exercise_fee / contact_commit_fee",
            "other_modes": (
                "preserve the original exercise-price ratio to robust-qpsk"
            ),
            "integer_rounding": "decimal ROUND_HALF_UP; minimum positive fee is 1",
            "zero_option_fee_in_primary_grid": False,
        },
        "scenario_records": scenario_records,
        "scenario_records_sha256": sha256_object(scenario_records),
        "source_sha256": source_hashes(),
        "bootstrap": {
            "unit": "whole seed/scenario cluster; all price-budget variants move together",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "ci": "two-sided 95% percentile",
        },
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
    }
    manifest["manifest_sha256"] = sha256_object(manifest)
    return manifest


def _summarise_values(
    values: Sequence[float], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, object]:
    low, high = paired_bootstrap_ci(
        values, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    return {
        "independent_seed_count": len(values),
        "mean_gain_pp": statistics.fmean(values),
        "median_gain_pp": statistics.median(values),
        "bootstrap_percentile_95_ci_pp": [low, high],
        "wins_ties_losses": {
            "wins": sum(value > 1e-12 for value in values),
            "ties": sum(abs(value) <= 1e-12 for value in values),
            "losses": sum(value < -1e-12 for value in values),
        },
    }


def summarise_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    by_cell: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell_id"]), []).append(row)

    cell_summaries: dict[str, dict[str, object]] = {}
    for index, (cell_id, cell_rows) in enumerate(sorted(by_cell.items())):
        ordered = sorted(cell_rows, key=lambda row: int(row["seed"]))
        gains = [float(row["paired_gain_pp"]) for row in ordered]
        first = ordered[0]
        cell_summaries[cell_id] = {
            "budget_model": first["budget_model"],
            "a": first["a"],
            "b": first["b"],
            "s": first["s"],
            **_summarise_values(
                gains,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + index,
            ),
        }

    aggregate: dict[str, dict[str, object]] = {}
    for model in (BudgetModel.STAGED.value, BudgetModel.UNIFIED_HARD.value):
        selected = [row for row in rows if row["budget_model"] == model]
        by_seed: dict[int, list[float]] = {}
        for row in selected:
            by_seed.setdefault(int(row["seed"]), []).append(
                float(row["paired_gain_pp"])
            )
        seed_grid_means = [statistics.fmean(by_seed[seed]) for seed in sorted(by_seed)]
        aggregate[model] = {
            "interpretation": "equal-weight mean over the declared design grid, not a mission prior",
            "cell_count": len({str(row["cell_id"]) for row in selected}),
            **_summarise_values(
                seed_grid_means,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + (100 if model == BudgetModel.STAGED.value else 200),
            ),
        }

    primary_cell = _cell_id(BudgetModel.STAGED, 0.15, 1.0, 0.5)
    primary_unified = _cell_id(BudgetModel.UNIFIED_HARD, 0.15, 1.0, None)
    non_extreme_cells = [
        cell_id
        for cell_id, summary in cell_summaries.items()
        if summary["budget_model"] == BudgetModel.STAGED.value
        and float(summary["a"]) in (0.15, 0.30)
        and float(summary["b"]) == 1.0
        and float(summary["s"]) in (0.50, 0.75)
    ]
    non_extreme_by_seed: dict[int, list[float]] = {}
    for row in rows:
        if row["cell_id"] in non_extreme_cells:
            non_extreme_by_seed.setdefault(int(row["seed"]), []).append(
                float(row["paired_gain_pp"])
            )
    non_extreme = None
    if non_extreme_by_seed:
        values = [
            statistics.fmean(non_extreme_by_seed[seed])
            for seed in sorted(non_extreme_by_seed)
        ]
        non_extreme = {
            "cell_ids": non_extreme_cells,
            **_summarise_values(
                values,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 300,
            ),
        }
    return {
        "cell_summaries": cell_summaries,
        "equal_weight_grid_summaries": aggregate,
        "primary_endpoint": cell_summaries.get(primary_cell),
        "primary_unified_robustness": cell_summaries.get(primary_unified),
        "predeclared_non_extreme_region": non_extreme,
    }


def run_e1(
    *,
    output_dir: Path,
    seeds: Sequence[int] = E1_FORMAL_SEEDS,
    a_values: Sequence[float] = DEFAULT_A_VALUES,
    b_values: Sequence[float] = DEFAULT_B_VALUES,
    s_values: Sequence[float] = DEFAULT_S_VALUES,
    methods: Sequence[str] = MAIN_METHODS,
    total_budget_milliunits: int = 255,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    smoke: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if OPTIONCAST_CAP_EXACT not in methods or COMMIT_PORTFOLIO_DYNAMIC_EXACT not in methods:
        raise ValueError("methods must contain the proposed method and primary baseline")
    unsupported = set(methods) - set(UNIFIED_SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(
            "unified-hard run has no audited implementation for: "
            + ", ".join(sorted(unsupported))
        )
    configs = experiment_configs(
        a_values=a_values,
        b_values=b_values,
        s_values=s_values,
        total_budget_milliunits=total_budget_milliunits,
    )
    manifest = build_manifest(
        seeds=seeds,
        configs=configs,
        methods=methods,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        smoke=smoke,
    )
    manifest_path = output_dir / "e1_manifest.json"
    rows_path = output_dir / "e1_rows.jsonl"
    summary_path = output_dir / "e1_summary.json"
    for path in (manifest_path, rows_path, summary_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace existing artifact: {path}")
    _write_json(manifest_path, manifest, overwrite=overwrite)

    expected_hashes = {
        (int(record["seed"]), str(record["cell_id"])): str(
            record["adapted_scenario_sha256"]
        )
        for record in manifest["scenario_records"]  # type: ignore[union-attr]
    }
    rows: list[dict[str, object]] = []
    for seed in seeds:
        base = make_scenario(seed, mismatch="matching")
        for budget_model, config in configs:
            scenario = adapt_scenario(base, config, budget_model)
            cell_id = _cell_id(budget_model, config.a, config.b, config.s)
            observed_hash = sha256_object(scenario)
            if observed_hash != expected_hashes[(seed, cell_id)]:
                raise RuntimeError("adapted scenario differs from frozen manifest")
            method_records: dict[str, object] = {}
            for method in methods:
                evaluation, audit = evaluate_e1_method(
                    scenario,
                    method,
                    budget_model=budget_model,
                    total_budget_milliunits=config.total_budget_milliunits,
                )
                method_records[method] = record_with_path_spends(
                    scenario, evaluation, audit
                )
            proposed = method_records[OPTIONCAST_CAP_EXACT]  # type: ignore[assignment]
            baseline = method_records[COMMIT_PORTFOLIO_DYNAMIC_EXACT]  # type: ignore[assignment]
            paired_gains = {
                method: float(proposed["net_value_pp"])  # type: ignore[index]
                - float(record["net_value_pp"])  # type: ignore[index]
                for method, record in method_records.items()
                if method != OPTIONCAST_CAP_EXACT
            }
            rows.append(
                {
                    "run_version": E1_RUN_VERSION,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "seed": seed,
                    "scenario_id": scenario.scenario_id,
                    "scenario_sha256": observed_hash,
                    "cell_id": cell_id,
                    "budget_model": budget_model.value,
                    "a": config.a,
                    "b": config.b,
                    "s": config.s,
                    "budget": budget_record(scenario, config, budget_model),
                    "methods": method_records,
                    "paired_gains_pp_by_baseline": paired_gains,
                    "paired_gain_pp": float(proposed["net_value_pp"])  # type: ignore[index]
                    - float(baseline["net_value_pp"]),  # type: ignore[index]
                }
            )

    rows_data = _jsonl_bytes(rows)
    _atomic_write(rows_path, rows_data, overwrite=overwrite)
    primary_summary = summarise_rows(
        rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    paired_comparisons: dict[str, object] = {}
    for baseline_index, baseline_method in enumerate(
        method for method in methods if method != OPTIONCAST_CAP_EXACT
    ):
        comparison_rows = [
            {
                **row,
                "paired_gain_pp": row["paired_gains_pp_by_baseline"][baseline_method],  # type: ignore[index]
            }
            for row in rows
        ]
        paired_comparisons[baseline_method] = summarise_rows(
            comparison_rows,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + (baseline_index + 1) * 1_000,
        )
    summary: dict[str, object] = {
        "run_version": E1_RUN_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "rows_sha256": hashlib.sha256(rows_data).hexdigest(),
        "row_count": len(rows),
        "seed_count": len(seeds),
        "solver_failures": 0,
        "hard_budget_violations": sum(
            int(method_record["audit"]["hard_budget_violations"])  # type: ignore[index]
            for row in rows
            for method_record in row["methods"].values()  # type: ignore[union-attr]
        ),
        "proposed_method": OPTIONCAST_CAP_EXACT,
        "primary_baseline": COMMIT_PORTFOLIO_DYNAMIC_EXACT,
        **primary_summary,
        "paired_comparisons": paired_comparisons,
    }
    _write_json(summary_path, summary, overwrite=overwrite)
    return summary


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        output = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not output:
        raise argparse.ArgumentTypeError("at least one number is required")
    return output


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        output = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not output:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return output


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    output = tuple(item.strip() for item in value.split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("at least one method is required")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_csv_ints)
    parser.add_argument("--a-values", type=_parse_csv_floats)
    parser.add_argument("--b-values", type=_parse_csv_floats)
    parser.add_argument("--s-values", type=_parse_csv_floats)
    parser.add_argument("--methods", type=_parse_csv_strings)
    parser.add_argument("--total-budget", type=int, default=255)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    seeds = args.seeds
    a_values = args.a_values
    b_values = args.b_values
    s_values = args.s_values
    bootstrap_replicates = args.bootstrap_replicates
    if args.smoke:
        seeds = seeds or (DEVELOPMENT_SEEDS[0],)
        a_values = a_values or (0.15,)
        b_values = b_values or (1.0,)
        s_values = s_values or (0.5,)
        if args.bootstrap_replicates == DEFAULT_BOOTSTRAP_REPLICATES:
            bootstrap_replicates = 200
    summary = run_e1(
        output_dir=args.output_dir,
        seeds=seeds or E1_FORMAL_SEEDS,
        a_values=a_values or DEFAULT_A_VALUES,
        b_values=b_values or DEFAULT_B_VALUES,
        s_values=s_values or DEFAULT_S_VALUES,
        methods=args.methods or MAIN_METHODS,
        total_budget_milliunits=args.total_budget,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        smoke=args.smoke,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": summary["manifest_sha256"],
                "row_count": summary["row_count"],
                "hard_budget_violations": summary["hard_budget_violations"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_A_VALUES",
    "DEFAULT_B_VALUES",
    "DEFAULT_S_VALUES",
    "E1_RUN_VERSION",
    "E1_FORMAL_SEEDS",
    "build_manifest",
    "canonical_bytes",
    "experiment_configs",
    "main",
    "paired_bootstrap_ci",
    "run_e1",
    "sha256_object",
    "summarise_rows",
]
