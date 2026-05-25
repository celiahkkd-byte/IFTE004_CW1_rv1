#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rv1rep_matplotlib")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.config import ensure_dirs, load_config, project_path
from rv1rep.features import make_model_frame
from rv1rep.models import fit_sklearn_model
from rv1rep.preprocessing import Standardizer, enforce_positive_forecasts, insanity_filter
from rv1rep.split import chronological_split, subset_by_dates
from rv1rep.utils import setup_logging

logger = logging.getLogger(__name__)

NON_NN_MODELS = [
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
]

KEY_COLS = ["date", "ticker", "dataset", "horizon", "model"]
SORT_COLS = ["dataset", "horizon", "model", "ticker", "date"]


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _ticker_output_path(root: Path, dataset: str, horizon: int, model: str, ticker: str) -> Path:
    return root / "predictions" / "by_ticker" / f"{_safe_name(dataset)}__h{int(horizon)}__{_safe_name(model)}__{ticker.upper()}.csv"


def _model_output_path(root: Path, dataset: str, horizon: int, model: str) -> Path:
    return root / "predictions" / "by_model" / f"{_safe_name(dataset)}__h{int(horizon)}__{_safe_name(model)}.csv"


def _sort_predictions(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(SORT_COLS).reset_index(drop=True)


def _validate_prediction_keys(df: pd.DataFrame, context: str) -> None:
    duplicates = int(df.duplicated(KEY_COLS).sum())
    if duplicates:
        raise RuntimeError(f"Duplicate prediction keys in {context}: {duplicates}")


def _target_scaling_values(y_train: pd.Series) -> tuple[float, float]:
    y_mean = float(y_train.astype(float).mean())
    y_std = float(y_train.astype(float).std(ddof=1))
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    return y_mean, y_std


def _fit_one_asset_rolling_corrected(df_asset: pd.DataFrame, model_name: str, feature_cols: list[str], target_col: str, split, cfg: dict) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(pd.unique(df_asset["date"]))).normalize()
    train_n = len(split.train_dates)
    val_n = len(split.val_dates)
    train_val_window = train_n + val_n
    test_dates = list(split.test_dates)
    rows = []

    last_est = None
    last_scaler = None
    last_params = None
    last_n_train = 0
    last_n_val = 0
    last_refit_i = -10**9
    y_mean_train = np.nan
    y_std_train = np.nan
    target_scaling = ""
    log_bias_var = 0.0
    in_min = np.nan
    in_mean = np.nan
    categorical_cols: list[str] = []
    continuous_cols: list[str] = []

    refit_every = int(cfg["estimation"].get("ml_refit_every", 5))
    model_u = model_name.upper()
    target_is_log = model_u == "LOGHAR"
    non_tuned_models = {"HAR", "HARX", "LOGHAR", "LEVHAR", "SHAR", "HARQ", "BAGGING", "RANDOMFOREST"}
    daily_refit_models = {"HAR", "HARX", "LOGHAR", "LEVHAR", "SHAR", "HARQ", "RIDGE", "LASSO", "ELASTICNET", "ADAPTIVELASSO", "POSTLASSO"}

    for i, test_date in enumerate(test_dates):
        pos = np.where(dates == test_date)[0]
        if len(pos) == 0:
            continue
        pos = int(pos[0])
        window_dates = dates[max(0, pos - train_val_window):pos]
        if len(window_dates) < train_val_window:
            continue

        refit = (model_u in daily_refit_models) or last_est is None or (i - last_refit_i >= refit_every)
        if refit:
            if model_u in non_tuned_models:
                train_dates = window_dates
                val_dates = window_dates[:0]
            else:
                train_dates = window_dates[:train_n]
                val_dates = window_dates[train_n:]

            train = subset_by_dates(df_asset, train_dates)
            val = subset_by_dates(df_asset, val_dates)
            # Corrected robustness specification: continuous predictors are
            # standardized, while binary EA remains in its original 0/1 scale.
            scaler = Standardizer(standardize_binary_features=False).fit(train[feature_cols])
            X_train = scaler.transform(train[feature_cols])
            X_val = scaler.transform(val[feature_cols])
            categorical_cols = list(scaler.categorical_columns_ or [])
            continuous_cols = list(scaler.continuous_columns_ or [])

            y_train_raw = train[target_col].astype(float)
            y_val_raw = val[target_col].astype(float)
            if target_is_log:
                y_train_fit = y_train_raw
                y_val_fit = y_val_raw
                y_mean_train = np.nan
                y_std_train = np.nan
                target_scaling = "log_target_baseline_no_extra_standardization"
            else:
                y_mean_train, y_std_train = _target_scaling_values(y_train_raw)
                y_train_fit = (y_train_raw - y_mean_train) / y_std_train
                y_val_fit = (y_val_raw - y_mean_train) / y_std_train
                target_scaling = "standardized_y_train_inverse_transformed"

            est, params = fit_sklearn_model(
                model_name,
                X_train,
                y_train_fit,
                X_val,
                y_val_fit,
                cfg,
                random_state=cfg["project"]["random_seed"],
            )
            train_val_rv = pd.concat([train["rv"], val["rv"]]).dropna()
            in_min = float(train_val_rv.min())
            in_mean = float(train_val_rv.mean())

            if target_is_log:
                xy = pd.concat([X_train, X_val])
                yy = pd.concat([y_train_raw, y_val_raw])
                log_bias_var = float(np.var(yy - est.predict(xy), ddof=1))
            else:
                log_bias_var = 0.0

            last_est = est
            last_scaler = scaler
            last_params = params
            last_refit_i = i
            last_n_train = len(train)
            last_n_val = len(val)

        one = df_asset[df_asset["date"] == test_date]
        if one.empty:
            continue
        X_one = last_scaler.transform(one[feature_cols])
        pred_fit_scale = np.asarray(last_est.predict(X_one), dtype=float)
        if target_is_log:
            pred_raw = np.exp(pred_fit_scale + 0.5 * log_bias_var)
        else:
            pred_raw = pred_fit_scale * y_std_train + y_mean_train
        pred = enforce_positive_forecasts(pred_raw, in_min, cfg["estimation"]["negative_forecast_policy"])
        if cfg["estimation"]["insanity_filter"]["enabled"]:
            pred = insanity_filter(pred, in_mean, in_min, cfg["estimation"]["insanity_filter"]["max_multiple_of_in_sample_mean"])

        actual_rv_col = target_col.replace("target_log_rv_", "target_rv_")
        out = one[["date", "ticker", "rv", "oc_logret", "cc_logret", actual_rv_col]].copy()
        out = out.rename(columns={actual_rv_col: "actual_rv"})
        out["model"] = model_name
        out["forecast_rv"] = pred
        out["scheme"] = "rolling"
        out["n_train"] = last_n_train
        out["n_val"] = last_n_val
        out["params"] = json.dumps(
            {
                "specification": "combined_corrected_paper_style",
                "base_params": last_params,
                "target_scaling": target_scaling,
                "y_mean_train": None if not np.isfinite(y_mean_train) else float(y_mean_train),
                "y_std_train": None if not np.isfinite(y_std_train) else float(y_std_train),
                "x_standardization": "continuous_only_train_window",
                "categorical_features_not_standardized": categorical_cols,
                "continuous_features_standardized": continuous_cols,
            },
            sort_keys=True,
            default=str,
        )
        out["dataset"] = str(one.get("dataset", pd.Series([None])).iloc[0]) if "dataset" in one else None
        out["horizon"] = int(target_col.rsplit("_h", 1)[-1])
        out["features"] = ",".join(feature_cols)
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _run_one_ticker(panel_ticker: pd.DataFrame, dataset: str, horizon: int, model: str, cfg: dict, output_root: Path, force: bool) -> dict:
    ticker = str(panel_ticker["ticker"].iloc[0]).upper()
    path = _ticker_output_path(output_root, dataset, int(horizon), model, ticker)
    if path.exists() and not force:
        return {"dataset": dataset, "horizon": int(horizon), "model": model, "ticker": ticker, "status": "reused", "rows": len(pd.read_csv(path, usecols=["date"]))}

    frame, feature_cols, target_col = make_model_frame(panel_ticker, model, dataset, int(horizon))
    df_asset = frame[frame["ticker"].astype(str).str.upper() == ticker].copy()
    if df_asset.empty:
        return {"dataset": dataset, "horizon": int(horizon), "model": model, "ticker": ticker, "status": "empty", "rows": 0}
    split = chronological_split(
        df_asset["date"],
        train_frac=cfg["splitting"]["train_frac"],
        val_frac=cfg["splitting"]["val_frac"],
        fixed_train_days=cfg["splitting"].get("fixed_train_days"),
        fixed_val_days=cfg["splitting"].get("fixed_val_days"),
    )
    pred = _fit_one_asset_rolling_corrected(df_asset, model, feature_cols, target_col, split, cfg)
    if pred.empty:
        return {"dataset": dataset, "horizon": int(horizon), "model": model, "ticker": ticker, "status": "empty", "rows": 0}
    pred["dataset"] = dataset
    pred["horizon"] = int(horizon)
    _validate_prediction_keys(pred, f"{dataset}/h{horizon}/{model}/{ticker}")
    _atomic_write_csv(pred, path)
    return {"dataset": dataset, "horizon": int(horizon), "model": model, "ticker": ticker, "status": "completed", "rows": len(pred)}


def _merge_ticker_files(output_root: Path, dataset: str, horizon: int, model: str, tickers: list[str]) -> pd.DataFrame:
    parts = []
    for ticker in tickers:
        path = _ticker_output_path(output_root, dataset, int(horizon), model, ticker)
        if path.exists():
            parts.append(pd.read_csv(path, parse_dates=["date"]))
    if not parts:
        return pd.DataFrame()
    out = _sort_predictions(pd.concat(parts, ignore_index=True))
    _validate_prediction_keys(out, f"{dataset}/h{horizon}/{model}")
    _atomic_write_csv(out, _model_output_path(output_root, dataset, int(horizon), model))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected non-NN combined specification with standardized y.")
    parser.add_argument("--config", default=str(ROOT / "config/paper_core_rolling_tuned_no_refit.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--workers", "--n-jobs", dest="workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = _resolve(args.output_dir)
    if output_root.exists() and any(output_root.rglob("*")) and not args.allow_existing_output_dir:
        raise SystemExit(f"Output directory already exists and is non-empty: {output_root}")
    cfg["paths"]["output_dir"] = str(output_root)
    cfg["estimation"]["scheme"] = "rolling"
    cfg["estimation"]["refit_tuned_models_on_train_validation"] = False
    cfg["models"]["trees"]["n_jobs"] = 1
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, "output_dir") / "logs" / "14_corrected_nonnn_combined.log")

    panel = pd.read_csv(project_path(cfg, "processed_dir") / "forecasting_panel.csv", parse_dates=["date"])
    if args.tickers is None or args.tickers == ["all"]:
        tickers = sorted(panel["ticker"].astype(str).str.upper().unique())
    else:
        tickers = [t.upper() for t in args.tickers]
        panel = panel[panel["ticker"].astype(str).str.upper().isin(tickers)].copy()
    if panel.empty:
        raise SystemExit("No panel rows after ticker filter.")

    datasets = args.datasets or list(cfg["experiments"]["datasets"])
    horizons = [int(h) for h in (args.horizons or cfg["experiments"]["horizons"])]
    models = list(args.models or NON_NN_MODELS)
    for model in models:
        if model.upper().startswith("NN"):
            raise SystemExit("This script is for non-NN models only.")

    tasks = []
    ticker_key = panel["ticker"].astype(str).str.upper()
    for dataset in datasets:
        for horizon in horizons:
            for model in models:
                for ticker in tickers:
                    tasks.append((panel[ticker_key == ticker].copy(), dataset, int(horizon), model, cfg, output_root, args.force))

    logger.info("Corrected non-NN run output=%s datasets=%s horizons=%s models=%s tickers=%s workers=%s", output_root, datasets, horizons, models, tickers, args.workers)
    manifest = Parallel(n_jobs=max(1, int(args.workers)), backend="threading")(
        delayed(_run_one_ticker)(*task) for task in tasks
    )
    manifest_df = pd.DataFrame(manifest)
    _atomic_write_csv(manifest_df, output_root / "predictions" / "nonnn_ticker_manifest.csv")

    parts = []
    for dataset in datasets:
        for horizon in horizons:
            for model in models:
                merged = _merge_ticker_files(output_root, dataset, int(horizon), model, tickers)
                if not merged.empty:
                    parts.append(merged)
    if not parts:
        raise SystemExit("No non-NN predictions produced.")
    final = _sort_predictions(pd.concat(parts, ignore_index=True))
    _validate_prediction_keys(final, "non-NN combined predictions")
    _atomic_write_csv(final, output_root / "predictions" / "nonnn_model_predictions.csv")

    _atomic_write_json(
        {
            "script": Path(__file__).name,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "specification": "combined_corrected_paper_style",
            "datasets": datasets,
            "horizons": horizons,
            "models": models,
            "tickers": tickers,
            "target_scaling_non_loghar": "standardized_y_train_inverse_transformed",
            "target_scaling_loghar": "log_target_baseline_no_extra_standardization",
            "x_standardization": "continuous_only_train_window",
        },
        output_root / "run_provenance_nonnn.json",
    )
    logger.info("Wrote corrected non-NN predictions rows=%d", len(final))


if __name__ == "__main__":
    main()
