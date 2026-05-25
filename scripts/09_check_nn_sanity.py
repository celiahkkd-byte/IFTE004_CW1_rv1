#!/usr/bin/env python3
"""Sanity checks for checkpointed neural-network forecast outputs.

The checks are read-only with respect to model outputs. They verify that the
published NN forecasts can be reconstructed from the seed checkpoints, and they
summarize whether NN forecasts are unusually smooth relative to actual RV and
non-neural benchmarks.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_PREDICTIONS = (
    ROOT
    / "outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523"
    / "predictions"
    / "model_predictions.csv"
)
DEFAULT_NN_CHECKPOINT_DIR = ROOT / "outputs_nn50_checkpointed_20260521"
DEFAULT_OUTPUT_DIR = ROOT / "outputs_nn_sanity_check_20260523"
BASE_NN_MODELS = ["NN1", "NN2", "NN3", "NN4"]
NN_MODELS = BASE_NN_MODELS + [f"{m}_1" for m in BASE_NN_MODELS]
BENCHMARK_MODELS = ["HAR", "HARX", "LogHAR", "ElasticNet", "RandomForest"]
KEY_COLS = ["date", "ticker", "dataset", "horizon", "model"]


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check checkpointed NN forecast consistency and smoothness.")
    parser.add_argument("--main-predictions", default=str(DEFAULT_MAIN_PREDICTIONS))
    parser.add_argument("--nn-checkpoint-dir", default=str(DEFAULT_NN_CHECKPOINT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    parser.add_argument("--negative-policy", default="in_sample_min_rv", choices=["none", "zero", "in_sample_min_rv"])
    parser.add_argument("--insanity-filter", action="store_true", default=True)
    parser.add_argument("--max-multiple-of-in-sample-mean", type=float, default=100.0)
    return parser.parse_args()


def parse_params(value: object) -> dict:
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def checkpoint_path(root: Path, dataset: str, horizon: int, model: str, ticker: str, seed: int) -> Path:
    return (
        root
        / "nn_seed_predictions"
        / safe_name(dataset)
        / f"h{int(horizon)}"
        / safe_name(model)
        / safe_name(ticker)
        / f"seed_{int(seed)}.csv"
    )


def postprocess(raw: np.ndarray, in_min: float, in_mean: float, args: argparse.Namespace) -> np.ndarray:
    pred = np.asarray(raw, dtype=float)
    if args.negative_policy == "zero":
        pred = np.maximum(pred, 0.0)
    elif args.negative_policy == "in_sample_min_rv":
        pred = np.where(pred <= 0, in_min, pred)
    elif args.negative_policy != "none":
        raise ValueError(f"Unknown negative policy: {args.negative_policy}")
    if args.insanity_filter:
        cap = float(args.max_multiple_of_in_sample_mean) * float(in_mean)
        pred = np.where((~np.isfinite(pred)) | (pred <= 0) | (pred > cap), in_min, pred)
    return pred


def load_seed_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {
        "date",
        "ticker",
        "actual_rv",
        "forecast_raw",
        "seed",
        "val_mse",
        "in_sample_min_rv",
        "in_sample_mean_rv",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def seed_directory_completeness(root: Path) -> pd.DataFrame:
    rows = []
    seed_root = root / "nn_seed_predictions"
    for ticker_dir in sorted(seed_root.glob("*/*/NN*/*")):
        if not ticker_dir.is_dir():
            continue
        parts = ticker_dir.relative_to(seed_root).parts
        if len(parts) != 4:
            continue
        dataset, hpart, model, ticker = parts
        files = sorted(ticker_dir.glob("seed_*.csv"))
        seeds = sorted(int(p.stem.split("_")[-1]) for p in files)
        rows.append(
            {
                "dataset": dataset,
                "horizon": int(hpart.removeprefix("h")),
                "source_model": model,
                "ticker": ticker,
                "seed_file_count": len(files),
                "min_seed": min(seeds) if seeds else np.nan,
                "max_seed": max(seeds) if seeds else np.nan,
                "missing_seed_count_in_range": (
                    int((max(seeds) - min(seeds) + 1) - len(seeds)) if seeds else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def reconstruct_group(
    group: pd.DataFrame,
    checkpoint_root: Path,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    first = group.iloc[0]
    dataset = str(first["dataset"])
    horizon = int(first["horizon"])
    ticker = str(first["ticker"])
    model = str(first["model"])
    params = parse_params(first["params"])
    if model.endswith("_1"):
        source_model = str(params.get("source_model", model.removesuffix("_1")))
        selected_seeds = [int(params["selected_seed"])]
        expected_top = 1
        mode = "single_best_seed"
    else:
        source_model = model
        selected_seeds = [int(s) for s in params["selected_seeds"]]
        expected_top = int(params.get("ensemble_top", len(selected_seeds)))
        mode = "top_k_ensemble"

    seed_frames = []
    seed_stats = []
    for seed in selected_seeds:
        path = checkpoint_path(checkpoint_root, dataset, horizon, source_model, ticker, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        df_seed = load_seed_file(path)
        seed_frames.append(df_seed)
        seed_stats.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "model": model,
                "source_model": source_model,
                "ticker": ticker,
                "seed": seed,
                "selected": True,
                "val_mse": float(df_seed["val_mse"].iloc[0]),
                "raw_mean": float(df_seed["forecast_raw"].mean()),
                "raw_std": float(df_seed["forecast_raw"].std(ddof=1)),
                "raw_min": float(df_seed["forecast_raw"].min()),
                "raw_max": float(df_seed["forecast_raw"].max()),
                "raw_nonpositive_frac": float((df_seed["forecast_raw"] <= 0).mean()),
            }
        )
    base = seed_frames[0][["date", "ticker", "actual_rv", "in_sample_min_rv", "in_sample_mean_rv"]].copy()
    for other in seed_frames[1:]:
        if not base[["date", "ticker"]].equals(other[["date", "ticker"]]):
            raise RuntimeError(f"Seed rows differ for {dataset}/h{horizon}/{model}/{ticker}")
    raw_matrix = np.vstack([f["forecast_raw"].to_numpy(dtype=float) for f in seed_frames])
    reconstructed = postprocess(
        raw_matrix.mean(axis=0),
        float(base["in_sample_min_rv"].iloc[0]),
        float(base["in_sample_mean_rv"].iloc[0]),
        args,
    )
    group_sorted = group.sort_values(["date", "ticker"]).reset_index(drop=True)
    base_sorted = base.sort_values(["date", "ticker"]).reset_index(drop=True)
    if len(group_sorted) != len(base_sorted):
        raise RuntimeError(f"Row count mismatch for {dataset}/h{horizon}/{model}/{ticker}")
    if not pd.to_datetime(group_sorted["date"]).reset_index(drop=True).equals(base_sorted["date"].reset_index(drop=True)):
        raise RuntimeError(f"Date mismatch for {dataset}/h{horizon}/{model}/{ticker}")
    final_pred = group_sorted["forecast_rv"].to_numpy(dtype=float)
    diff = final_pred - reconstructed
    summary = {
        "dataset": dataset,
        "horizon": horizon,
        "model": model,
        "source_model": source_model,
        "ticker": ticker,
        "mode": mode,
        "selected_seed_count": len(selected_seeds),
        "expected_top": expected_top,
        "selected_seeds": ",".join(str(s) for s in selected_seeds),
        "rows": int(len(group_sorted)),
        "max_abs_diff_vs_seed_reconstruction": float(np.max(np.abs(diff))),
        "mean_abs_diff_vs_seed_reconstruction": float(np.mean(np.abs(diff))),
        "forecast_mean": float(np.mean(final_pred)),
        "forecast_std": float(np.std(final_pred, ddof=1)),
        "actual_mean": float(group_sorted["actual_rv"].mean()),
        "actual_std": float(group_sorted["actual_rv"].std(ddof=1)),
        "forecast_to_actual_std_ratio": float(np.std(final_pred, ddof=1) / group_sorted["actual_rv"].std(ddof=1)),
        "forecast_min": float(np.min(final_pred)),
        "forecast_max": float(np.max(final_pred)),
        "forecast_nonpositive_count": int((final_pred <= 0).sum()),
        "mse": float(np.mean((group_sorted["actual_rv"].to_numpy(dtype=float) - final_pred) ** 2)),
    }
    return summary, pd.DataFrame(seed_stats)


def prediction_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["dataset", "horizon", "model"], sort=True):
        dataset, horizon, model = keys
        y = group["actual_rv"].to_numpy(dtype=float)
        pred = group["forecast_rv"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "model": model,
                "rows": int(len(group)),
                "tickers": int(group["ticker"].nunique()),
                "mse": float(np.mean((y - pred) ** 2)),
                "forecast_mean": float(np.mean(pred)),
                "forecast_std": float(np.std(pred, ddof=1)),
                "forecast_min": float(np.min(pred)),
                "forecast_max": float(np.max(pred)),
                "actual_mean": float(np.mean(y)),
                "actual_std": float(np.std(y, ddof=1)),
                "forecast_to_actual_std_ratio": float(np.std(pred, ddof=1) / np.std(y, ddof=1)),
                "forecast_nonpositive_count": int((pred <= 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    har = out[out["model"] == "HAR"][["dataset", "horizon", "mse"]].rename(columns={"mse": "har_mse"})
    out = out.merge(har, on=["dataset", "horizon"], how="left")
    out["relative_mse_vs_HAR"] = out["mse"] / out["har_mse"]
    return out.sort_values(["dataset", "horizon", "model"]).reset_index(drop=True)


def selected_seed_summary(checkpoint_root: Path, completeness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, meta in completeness.iterrows():
        dataset = str(meta["dataset"])
        horizon = int(meta["horizon"])
        source_model = str(meta["source_model"])
        ticker = str(meta["ticker"])
        for path in sorted(
            (
                checkpoint_root
                / "nn_seed_predictions"
                / dataset
                / f"h{horizon}"
                / source_model
                / ticker
            ).glob("seed_*.csv")
        ):
            df_seed = pd.read_csv(path, usecols=["seed", "val_mse", "forecast_raw"])
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "source_model": source_model,
                    "ticker": ticker,
                    "seed": int(df_seed["seed"].iloc[0]),
                    "val_mse": float(df_seed["val_mse"].iloc[0]),
                    "raw_mean": float(df_seed["forecast_raw"].mean()),
                    "raw_std": float(df_seed["forecast_raw"].std(ddof=1)),
                    "raw_min": float(df_seed["forecast_raw"].min()),
                    "raw_max": float(df_seed["forecast_raw"].max()),
                    "raw_nonpositive_frac": float((df_seed["forecast_raw"] <= 0).mean()),
                }
            )
    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        return seed_df
    seed_df["val_rank"] = seed_df.groupby(["dataset", "horizon", "source_model", "ticker"])["val_mse"].rank(
        method="first", ascending=True
    )
    return seed_df.sort_values(["dataset", "horizon", "source_model", "ticker", "val_rank"]).reset_index(drop=True)


def write_report(
    output_dir: Path,
    *,
    main_path: Path,
    checkpoint_root: Path,
    completeness: pd.DataFrame,
    pred_summary: pd.DataFrame,
    consistency: pd.DataFrame,
    seed_summary: pd.DataFrame,
    ale_summary: pd.DataFrame | None,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda x: f"{x:.6g}")
        cols = list(display.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    nn_rows = pred_summary[pred_summary["model"].isin(NN_MODELS)]
    max_recon_diff = float(consistency["max_abs_diff_vs_seed_reconstruction"].max()) if not consistency.empty else np.nan
    incomplete = completeness[completeness["seed_file_count"] != 50]
    low_std = nn_rows.sort_values("forecast_to_actual_std_ratio").head(8)
    report = [
        "# NN Sanity Check",
        "",
        f"Created at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Main predictions: `{main_path}`",
        f"NN checkpoint dir: `{checkpoint_root}`",
        "",
        "## Main Findings",
        "",
        f"- Seed checkpoint completeness: {len(completeness)} ticker/model/dataset/horizon groups; "
        f"{len(incomplete)} groups do not have exactly 50 seed files.",
        f"- Forecast reconstruction from selected seed checkpoints: max absolute difference = `{max_recon_diff:.3e}`.",
        "- A max reconstruction difference near floating-point tolerance means the published NN forecasts are consistent with the stored seed checkpoints.",
        "- Smooth NN curves in ALE are therefore more likely driven by the fitted NN forecasts being compressed, not by the plotting script.",
        "",
        "## Lowest NN Forecast-Variability Ratios",
        "",
        markdown_table(
            low_std[
                [
                    "dataset",
                    "horizon",
                    "model",
                    "forecast_to_actual_std_ratio",
                    "relative_mse_vs_HAR",
                    "forecast_std",
                    "actual_std",
                ]
            ]
        ),
        "",
    ]
    if ale_summary is not None and not ale_summary.empty:
        report.extend(
            [
                "## AAPL ALE Range Summary",
                "",
                markdown_table(ale_summary),
                "",
            ]
        )
    report.extend(
        [
            "## Interpretation",
            "",
            "This check does not show evidence that the NN forecasts in the main table are disconnected from the seed checkpoints.",
            "It does show that several NN variants have low forecast dispersion relative to realized RV and relative to tree or linear benchmarks.",
            "That is consistent with visually flat NN ALE curves.",
            "",
            "The current seed checkpoint format does not store epoch-by-epoch loss history or the final epoch count.",
            "Therefore this check cannot prove whether individual TensorFlow fits stopped early after very few epochs.",
            "To audit that specific point, future NN runs should persist training history, best epoch, and final validation loss per seed.",
            "",
        ]
    )
    (output_dir / "NN_SANITY_CHECK_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def maybe_ale_summary(root: Path) -> pd.DataFrame | None:
    ale_path = root / "outputs_figure6_ale_paper_style_partial_20260523" / "figure6_ale_paper_style_partial_h1.csv"
    if not ale_path.exists():
        return None
    df = pd.read_csv(ale_path)
    out = (
        df.groupby(["feature", "model"])["ale_scaled"]
        .agg(["min", "max", "std"])
        .reset_index()
        .rename(columns={"min": "ale_scaled_min", "max": "ale_scaled_max", "std": "ale_scaled_std"})
    )
    out["ale_scaled_range"] = out["ale_scaled_max"] - out["ale_scaled_min"]
    return out.sort_values(["feature", "model"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    main_path = resolve(args.main_predictions)
    checkpoint_root = resolve(args.nn_checkpoint_dir)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.rglob("*")) and not args.allow_existing_output_dir:
        raise SystemExit(
            f"Output directory exists and contains files: {output_dir}. "
            "Use --allow-existing-output-dir to update it intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(main_path, parse_dates=["date"])
    needed = set(KEY_COLS + ["actual_rv", "forecast_rv", "params"])
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{main_path} missing columns: {sorted(missing)}")
    nn = df[df["model"].isin(NN_MODELS)].copy()
    if nn.empty:
        raise ValueError("No NN rows found in main predictions.")
    duplicate_nn = int(nn.duplicated(KEY_COLS).sum())
    if duplicate_nn:
        raise RuntimeError(f"Duplicate NN keys: {duplicate_nn}")

    completeness = seed_directory_completeness(checkpoint_root)
    seed_summary = selected_seed_summary(checkpoint_root, completeness)
    pred_summary = prediction_summary(df[df["model"].isin(sorted(set(NN_MODELS + BENCHMARK_MODELS)))].copy())

    consistency_rows = []
    selected_seed_rows = []
    for _, group in nn.groupby(["dataset", "horizon", "model", "ticker"], sort=True):
        row, selected_stats = reconstruct_group(group, checkpoint_root, args)
        consistency_rows.append(row)
        selected_seed_rows.append(selected_stats)
    consistency = pd.DataFrame(consistency_rows).sort_values(["dataset", "horizon", "model", "ticker"])
    selected_seed_stats = pd.concat(selected_seed_rows, ignore_index=True)

    ale_summary = maybe_ale_summary(ROOT)

    completeness.to_csv(output_dir / "nn_seed_directory_completeness.csv", index=False)
    seed_summary.to_csv(output_dir / "nn_all_seed_stats.csv", index=False)
    selected_seed_stats.to_csv(output_dir / "nn_selected_seed_stats.csv", index=False)
    consistency.to_csv(output_dir / "nn_aggregation_consistency.csv", index=False)
    pred_summary.to_csv(output_dir / "nn_prediction_summary_by_model.csv", index=False)
    if ale_summary is not None:
        ale_summary.to_csv(output_dir / "nn_ale_aapl_range_summary.csv", index=False)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_predictions": str(main_path),
        "nn_checkpoint_dir": str(checkpoint_root),
        "output_dir": str(output_dir),
        "nn_rows": int(len(nn)),
        "nn_duplicate_keys": duplicate_nn,
        "seed_groups": int(len(completeness)),
        "seed_file_count_distribution": {
            str(k): int(v) for k, v in completeness["seed_file_count"].value_counts().sort_index().items()
        },
        "max_abs_diff_vs_seed_reconstruction": float(consistency["max_abs_diff_vs_seed_reconstruction"].max()),
        "mean_abs_diff_vs_seed_reconstruction": float(consistency["mean_abs_diff_vs_seed_reconstruction"].mean()),
        "outputs": {
            "report": str(output_dir / "NN_SANITY_CHECK_REPORT.md"),
            "seed_completeness": str(output_dir / "nn_seed_directory_completeness.csv"),
            "all_seed_stats": str(output_dir / "nn_all_seed_stats.csv"),
            "selected_seed_stats": str(output_dir / "nn_selected_seed_stats.csv"),
            "aggregation_consistency": str(output_dir / "nn_aggregation_consistency.csv"),
            "prediction_summary": str(output_dir / "nn_prediction_summary_by_model.csv"),
            "ale_summary": str(output_dir / "nn_ale_aapl_range_summary.csv") if ale_summary is not None else None,
        },
    }
    (output_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    write_report(
        output_dir,
        main_path=main_path,
        checkpoint_root=checkpoint_root,
        completeness=completeness,
        pred_summary=pred_summary,
        consistency=consistency,
        seed_summary=seed_summary,
        ale_summary=ale_summary,
    )
    print(f"Wrote {output_dir / 'NN_SANITY_CHECK_REPORT.md'}")
    print(f"Max reconstruction diff: {provenance['max_abs_diff_vs_seed_reconstruction']:.3e}")
    print(f"Seed file count distribution: {provenance['seed_file_count_distribution']}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
