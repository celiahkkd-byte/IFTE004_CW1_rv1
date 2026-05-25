#!/usr/bin/env python3
"""Run h=1 MCS robustness checks and summarize inclusion rates.

This runner deliberately writes to a separate robustness output directory and
only calls the existing checkpointed MCS implementation. It does not modify or
reuse the main MCS output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


METHODS = ("max", "R")
BLOCK_SIZES = (5, 10, 20)
BOOTSTRAPS = ("stationary", "circular", "moving block")


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_setting_name(method: str, bootstrap: str, block_size: int) -> str:
    safe_bootstrap = bootstrap.replace(" ", "_")
    return f"method-{method}_bootstrap-{safe_bootstrap}_block-{block_size:02d}"


def parse_args() -> argparse.Namespace:
    repo = resolve_repo_root()
    ap = argparse.ArgumentParser(description="Run h=1 MCS robustness checks.")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=repo
        / "outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523"
        / "predictions"
        / "model_predictions.csv",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=repo / "outputs_mcs_robustness_h1_20260523",
    )
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--min-valid-rows", type=int, default=200)
    ap.add_argument(
        "--allow-existing-output-root",
        action="store_true",
        help="Resume/rebuild summaries in an existing robustness output directory.",
    )
    ap.add_argument(
        "--skip-existing-settings",
        action="store_true",
        help="Skip settings that already have mcs_inclusion_rates.csv.",
    )
    return ap.parse_args()


def run_setting(
    repo: Path,
    predictions: Path,
    output_dir: Path,
    method: str,
    bootstrap: str,
    block_size: int,
    confidence: float,
    reps: int,
    min_valid_rows: int,
) -> None:
    cmd = [
        sys.executable,
        str(repo / "scripts" / "06e_compute_mcs.py"),
        "--predictions",
        str(predictions),
        "--output-dir",
        str(output_dir),
        "--confidence",
        f"{confidence:.2f}",
        "--reps",
        str(reps),
        "--block-size",
        str(block_size),
        "--method",
        method,
        "--bootstrap",
        bootstrap,
        "--datasets",
        "MHAR",
        "PARTIAL_MALL",
        "--horizons",
        "1",
        "--min-valid-rows",
        str(min_valid_rows),
    ]
    if output_dir.exists():
        cmd.append("--allow-existing-output-dir")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(repo / ".matplotlib-cache"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=repo, env=env)


def collect_setting(output_dir: Path, method: str, bootstrap: str, block_size: int) -> pd.DataFrame:
    path = output_dir / "tables" / "mcs_inclusion_rates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MCS inclusion table: {path}")
    data = pd.read_csv(path)
    data.insert(0, "setting", safe_setting_name(method, bootstrap, block_size))
    data.insert(1, "method", method)
    data.insert(2, "bootstrap", bootstrap)
    data.insert(3, "block_size", int(block_size))
    return data


def summarize(inclusion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = inclusion.groupby(["setting", "method", "bootstrap", "block_size", "dataset", "horizon"], sort=True)
    for keys, group in grouped:
        setting, method, bootstrap, block_size, dataset, horizon = keys
        rates = group["inclusion_rate"].astype(float)
        rows.append(
            {
                "setting": setting,
                "method": method,
                "bootstrap": bootstrap,
                "block_size": int(block_size),
                "dataset": dataset,
                "horizon": int(horizon),
                "n_models": int(group["model"].nunique()),
                "n_tickers_total_min": int(group["n_tickers_total"].min()),
                "mean_inclusion_rate": rates.mean(),
                "median_inclusion_rate": rates.median(),
                "min_inclusion_rate": rates.min(),
                "max_inclusion_rate": rates.max(),
                "share_models_ge_0_80": float((rates >= 0.80).mean()),
                "share_models_ge_0_90": float((rates >= 0.90).mean()),
                "share_models_equal_1": float((rates == 1.00).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "method", "bootstrap", "block_size"]).reset_index(drop=True)


def summarize_by_method(setting_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = setting_summary.groupby(["dataset", "method"], sort=True)
    for (dataset, method), group in grouped:
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_settings": int(len(group)),
                "mean_of_setting_mean_inclusion": group["mean_inclusion_rate"].mean(),
                "min_setting_mean_inclusion": group["mean_inclusion_rate"].min(),
                "max_setting_mean_inclusion": group["mean_inclusion_rate"].max(),
                "mean_of_setting_median_inclusion": group["median_inclusion_rate"].mean(),
                "min_share_models_ge_0_80": group["share_models_ge_0_80"].min(),
                "max_share_models_ge_0_80": group["share_models_ge_0_80"].max(),
                "min_share_models_ge_0_90": group["share_models_ge_0_90"].min(),
                "max_share_models_ge_0_90": group["share_models_ge_0_90"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def dataframe_to_markdown(data: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if data.empty:
        return "_No rows._"
    formatted = data.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: format(float(value), floatfmt))
        else:
            formatted[col] = formatted[col].astype(str)

    headers = list(formatted.columns)
    rows = formatted.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def write_markdown_report(
    output_root: Path,
    summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    inclusion: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    report_path = output_root / "MCS_ROBUSTNESS_H1_SUMMARY.md"
    lines = [
        "# H=1 MCS Robustness Summary",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Scope:",
        "- horizon: 1 only",
        "- datasets: MHAR, PARTIAL_MALL",
        "- confidence: 0.90",
        f"- reps: {args.reps}",
        "- methods: max, R",
        "- block sizes: 5, 10, 20",
        "- bootstraps: stationary, circular, moving block",
        "",
        "Interpretation rule used here: inclusion rates are treated as overall high when the",
        "mean inclusion rate is high and most models have inclusion rate at or above 0.80 or 0.90.",
        "",
        "## Method-Level Summary",
        "",
        dataframe_to_markdown(method_summary, floatfmt=".3f"),
        "",
        "## Setting-Level Summary",
        "",
        dataframe_to_markdown(summary, floatfmt=".3f"),
        "",
        "## Lowest Inclusion Rates By Dataset",
        "",
    ]
    for dataset, group in inclusion.groupby("dataset", sort=True):
        low = (
            group.sort_values(["inclusion_rate", "setting", "model"])
            .loc[:, ["setting", "method", "bootstrap", "block_size", "dataset", "model", "inclusion_rate"]]
            .head(15)
        )
        lines.extend([f"### {dataset}", "", dataframe_to_markdown(low, floatfmt=".3f"), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = resolve_repo_root()
    predictions = args.predictions.resolve()
    output_root = args.output_root.resolve()

    if not predictions.exists():
        raise SystemExit(f"Predictions file not found: {predictions}")
    if output_root.exists() and not args.allow_existing_output_root:
        raise SystemExit(
            f"Output root already exists: {output_root}. "
            "Use --allow-existing-output-root to resume or choose a new --output-root."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)

    setting_frames = []
    manifest = []
    for method in METHODS:
        for block_size in BLOCK_SIZES:
            for bootstrap in BOOTSTRAPS:
                setting = safe_setting_name(method, bootstrap, block_size)
                setting_dir = output_root / "settings" / setting
                inclusion_path = setting_dir / "tables" / "mcs_inclusion_rates.csv"
                skipped = False
                if args.skip_existing_settings and inclusion_path.exists():
                    skipped = True
                else:
                    run_setting(
                        repo=repo,
                        predictions=predictions,
                        output_dir=setting_dir,
                        method=method,
                        bootstrap=bootstrap,
                        block_size=block_size,
                        confidence=args.confidence,
                        reps=args.reps,
                        min_valid_rows=args.min_valid_rows,
                    )
                setting_frames.append(collect_setting(setting_dir, method, bootstrap, block_size))
                manifest.append(
                    {
                        "setting": setting,
                        "method": method,
                        "bootstrap": bootstrap,
                        "block_size": block_size,
                        "status": "skipped_existing" if skipped else "completed",
                        "output_dir": str(setting_dir),
                    }
                )

    inclusion = pd.concat(setting_frames, ignore_index=True)
    summary = summarize(inclusion)
    method_summary = summarize_by_method(summary)

    inclusion.to_csv(output_root / "tables" / "mcs_robustness_inclusion_rates_h1_all_settings.csv", index=False)
    summary.to_csv(output_root / "tables" / "mcs_robustness_setting_summary_h1.csv", index=False)
    method_summary.to_csv(output_root / "tables" / "mcs_robustness_method_summary_h1.csv", index=False)
    pd.DataFrame(manifest).to_csv(output_root / "tables" / "mcs_robustness_manifest_h1.csv", index=False)
    (output_root / "run_provenance.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "predictions": str(predictions),
                "output_root": str(output_root),
                "confidence": args.confidence,
                "reps": args.reps,
                "horizons": [1],
                "datasets": ["MHAR", "PARTIAL_MALL"],
                "methods": list(METHODS),
                "block_sizes": list(BLOCK_SIZES),
                "bootstraps": list(BOOTSTRAPS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(output_root, summary, method_summary, inclusion, args)


if __name__ == "__main__":
    main()
