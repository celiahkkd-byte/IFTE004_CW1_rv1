from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.evaluation import cross_sectional_summary, forecast_metrics, pairwise_relative_mse


# Default to the isolated output directory suggested in docs/H22_REPRODUCTION.md.
# The directory is created by a full h=22 rerun and is intentionally not
# committed to GitHub because it contains large daily prediction files.
DEFAULT_SOURCE = ROOT / "outputs_h22_rerun_nn50"
DEFAULT_OUTPUT = ROOT / "outputs_h22_all_ticker_model_results_regenerated"

MODEL_LABEL_MAP = {
    "NN1_single_best": "NN1_1",
    "NN2_single_best": "NN2_1",
    "NN3_single_best": "NN3_1",
    "NN4_single_best": "NN4_1",
}

MODEL_ORDER = [
    "HAR",
    "HARX",
    "LogHAR",
    "LevHAR",
    "SHAR",
    "HARQ",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "AdaptiveLasso",
    "PostLasso",
    "Bagging",
    "RandomForest",
    "GradientBoosting",
    "NN1_1",
    "NN1",
    "NN2_1",
    "NN2",
    "NN3_1",
    "NN3",
    "NN4_1",
    "NN4",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_model_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("model", "benchmark_row", "row_model", "col_model"):
        if col in out.columns:
            out[col] = out[col].replace(MODEL_LABEL_MAP)
    rename = {}
    for col in out.columns:
        new_col = MODEL_LABEL_MAP.get(col, col)
        for old, new in MODEL_LABEL_MAP.items():
            suffix = "_dm_reject_10pct_share"
            if col == f"{old}{suffix}":
                new_col = f"{new}{suffix}"
        if new_col != col:
            rename[col] = new_col
    if rename:
        out = out.rename(columns=rename)
    return out


def reorder_model_columns(df: pd.DataFrame, leading: list[str]) -> pd.DataFrame:
    ordered = [c for c in leading if c in df.columns]
    ordered.extend([m for m in MODEL_ORDER if m in df.columns and m not in ordered])
    ordered.extend([c for c in df.columns if c not in ordered])
    return df.loc[:, ordered]


def max_abs_diff(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> float | None:
    common_cols = sorted(set(left.columns).intersection(right.columns) - set(keys))
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(left[c]) and pd.api.types.is_numeric_dtype(right[c])]
    if not numeric_cols:
        return None
    l = left[keys + numeric_cols].sort_values(keys).reset_index(drop=True)
    r = right[keys + numeric_cols].sort_values(keys).reset_index(drop=True)
    if len(l) != len(r) or not l[keys].equals(r[keys]):
        return None
    diffs = []
    for col in numeric_cols:
        diffs.append(np.nanmax(np.abs(l[col].to_numpy(dtype=float) - r[col].to_numpy(dtype=float))))
    return float(np.nanmax(diffs)) if diffs else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate h=22 all-ticker/model evaluation tables from audited predictions.")
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--datasets", nargs="+", default=["MHAR", "PARTIAL_MALL"])
    ap.add_argument("--horizon", type=int, default=22)
    ap.add_argument("--allow-existing-output-dir", action="store_true")
    args = ap.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    pred_path = source_dir / "predictions" / "model_predictions.csv"
    source_tables = source_dir / "tables"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found. Re-run the h=22 forecasts first "
            "using docs/H22_REPRODUCTION.md, or pass --source-dir to an "
            "output directory containing predictions/model_predictions.csv."
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output_dir:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    predictions_dir = output_dir / "predictions"
    tables_dir = output_dir / "tables"
    docs_dir = output_dir / "docs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    pred = pred[pred["dataset"].isin(args.datasets) & (pred["horizon"].astype(int) == args.horizon)].copy()
    pred["horizon"] = pred["horizon"].astype(int)
    pred["model"] = pred["model"].replace(MODEL_LABEL_MAP)
    pred = pred.sort_values(["dataset", "horizon", "ticker", "model", "date"]).reset_index(drop=True)

    key_cols = ["ticker", "date", "dataset", "horizon", "model"]
    duplicate_keys = int(pred.duplicated(key_cols).sum())
    missing_forecasts = int(pred["forecast_rv"].isna().sum())
    zero_forecasts = int((pred["forecast_rv"] == 0).sum())

    metrics = forecast_metrics(pred)
    summary = cross_sectional_summary(metrics)
    pairwise, dm = pairwise_relative_mse(pred)

    pred.to_csv(predictions_dir / "model_predictions_h22_normalized.csv", index=False)
    metrics.to_csv(tables_dir / "forecast_metrics_by_asset.csv", index=False)
    summary.to_csv(tables_dir / "forecast_summary_cross_section.csv", index=False)
    pairwise.to_csv(tables_dir / "pairwise_relative_mse_matrix.csv", index=False)
    dm.to_csv(tables_dir / "diebold_mariano_tests.csv", index=False)

    wide = metrics.pivot_table(
        index=["dataset", "horizon", "ticker"],
        columns="model",
        values="relative_mse_vs_har",
        aggfunc="first",
    ).reset_index()
    wide = reorder_model_columns(wide, ["dataset", "horizon", "ticker"])
    wide.to_csv(tables_dir / "relative_mse_vs_har_by_ticker_wide.csv", index=False)

    summary_wide = summary.pivot_table(
        index=["dataset", "horizon"],
        columns="model",
        values="avg_rel_mse_vs_har",
        aggfunc="first",
    ).reset_index()
    summary_wide = reorder_model_columns(summary_wide, ["dataset", "horizon"])
    summary_wide.to_csv(tables_dir / "relative_mse_vs_har_summary_wide.csv", index=False)

    pairwise_diff = None
    summary_diff = None
    if (source_tables / "pairwise_relative_mse_matrix.csv").exists():
        source_pairwise = normalize_model_labels(pd.read_csv(source_tables / "pairwise_relative_mse_matrix.csv"))
        pairwise_diff = max_abs_diff(pairwise, source_pairwise, ["dataset", "horizon", "benchmark_row"])
    if (source_tables / "forecast_summary_cross_section.csv").exists():
        source_summary = normalize_model_labels(pd.read_csv(source_tables / "forecast_summary_cross_section.csv"))
        summary_diff = max_abs_diff(summary, source_summary, ["dataset", "horizon", "model", "scheme"])

    audit = {
        "source_prediction_file": str(pred_path),
        "source_prediction_sha256": sha256_file(pred_path),
        "output_dir": str(output_dir),
        "rows": int(len(pred)),
        "datasets": sorted(pred["dataset"].astype(str).unique().tolist()),
        "horizons": sorted(pred["horizon"].astype(int).unique().tolist()),
        "tickers": int(pred["ticker"].nunique()),
        "models": [m for m in MODEL_ORDER if m in set(pred["model"].astype(str))],
        "model_count": int(pred["model"].nunique()),
        "duplicate_keys": duplicate_keys,
        "missing_forecasts": missing_forecasts,
        "zero_forecasts": zero_forecasts,
        "model_label_normalization": MODEL_LABEL_MAP,
        "pairwise_direction": "cell(row i, column j) = MSE_j / MSE_i; values below one mean the column model improves on the row benchmark",
        "max_abs_diff_vs_source_pairwise_csv_after_label_normalization": pairwise_diff,
        "max_abs_diff_vs_source_summary_csv_after_label_normalization": summary_diff,
        "generated_files": [
            "predictions/model_predictions_h22_normalized.csv",
            "tables/forecast_metrics_by_asset.csv",
            "tables/forecast_summary_cross_section.csv",
            "tables/pairwise_relative_mse_matrix.csv",
            "tables/diebold_mariano_tests.csv",
            "tables/relative_mse_vs_har_by_ticker_wide.csv",
            "tables/relative_mse_vs_har_summary_wide.csv",
        ],
    }
    (output_dir / "regeneration_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
