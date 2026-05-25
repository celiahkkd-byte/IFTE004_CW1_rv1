#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.config import ensure_dirs, load_config, project_path
from rv1rep.features import make_model_frame
from rv1rep.nn import _build_keras_model
from rv1rep.preprocessing import Standardizer, enforce_positive_forecasts, insanity_filter
from rv1rep.split import chronological_split, subset_by_dates
from rv1rep.utils import setup_logging

logger = logging.getLogger(__name__)


def _load_nn_runner():
    path = ROOT / "scripts" / "04_run_nn_checkpoints.py"
    spec = importlib.util.spec_from_file_location("_rv1rep_nn_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import NN checkpoint helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NN_RUNNER = _load_nn_runner()


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


def _seed_path(root: Path, dataset: str, horizon: int, model: str, ticker: str, seed: int) -> Path:
    return (
        root
        / "nn_seed_predictions"
        / "combined"
        / _safe_name(dataset)
        / f"h{int(horizon)}"
        / _safe_name(model)
        / _safe_name(ticker.upper())
        / f"seed_{int(seed)}.csv"
    )


def _model_output_path(root: Path, dataset: str, horizon: int, model: str) -> Path:
    return root / "predictions" / "by_model" / f"{_safe_name(dataset)}__h{int(horizon)}__{_safe_name(model)}.csv"


def _postprocess(pred_raw: np.ndarray, prepared: dict, cfg: dict) -> np.ndarray:
    pred = enforce_positive_forecasts(
        pred_raw,
        float(prepared["in_sample_min_rv"]),
        cfg["estimation"]["negative_forecast_policy"],
    )
    if cfg["estimation"]["insanity_filter"]["enabled"]:
        pred = insanity_filter(
            pred,
            float(prepared["in_sample_mean_rv"]),
            float(prepared["in_sample_min_rv"]),
            float(cfg["estimation"]["insanity_filter"]["max_multiple_of_in_sample_mean"]),
        )
    return np.asarray(pred, dtype=float)


def _prepare_asset_frame(panel: pd.DataFrame, cfg: dict, dataset: str, model_name: str, ticker: str, horizon: int) -> dict:
    frame, feature_cols, target_col = make_model_frame(panel, model_name, dataset, int(horizon))
    df_asset = frame[frame["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if df_asset.empty:
        raise ValueError(f"No rows for {dataset}/h{horizon}/{model_name}/{ticker}")

    split = chronological_split(
        df_asset["date"],
        train_frac=cfg["splitting"]["train_frac"],
        val_frac=cfg["splitting"]["val_frac"],
        fixed_train_days=cfg["splitting"].get("fixed_train_days"),
        fixed_val_days=cfg["splitting"].get("fixed_val_days"),
    )
    train = subset_by_dates(df_asset, split.train_dates)
    val = subset_by_dates(df_asset, split.val_dates)
    test = subset_by_dates(df_asset, split.test_dates)
    # Corrected robustness specification: continuous predictors are standardized,
    # while binary EA remains in its original 0/1 scale.
    scaler = Standardizer(standardize_binary_features=False).fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    y_train = train[target_col].astype(float)
    y_val = val[target_col].astype(float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=1))
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0

    actual_rv_col = target_col.replace("target_log_rv_", "target_rv_")
    test_base = test[["date", "ticker", "rv", "oc_logret", "cc_logret", actual_rv_col]].copy()
    test_base = test_base.rename(columns={actual_rv_col: "actual_rv"})
    in_sample_rv = pd.concat([train["rv"], val["rv"]]).dropna()

    return {
        "dataset": dataset,
        "horizon": int(horizon),
        "ticker": ticker.upper(),
        "model": model_name,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train": train,
        "val": val,
        "test": test,
        "test_base": test_base,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_raw": y_train,
        "y_val_raw": y_val,
        "y_mean_train": y_mean,
        "y_std_train": y_std,
        "y_train_std": (y_train - y_mean) / y_std,
        "y_val_std": (y_val - y_mean) / y_std,
        "in_sample_min_rv": float(in_sample_rv.min()),
        "in_sample_mean_rv": float(in_sample_rv.mean()),
        "x_standardization": "continuous_only_train_window",
        "categorical_features_not_standardized": list(scaler.categorical_columns_ or []),
        "continuous_features_standardized": list(scaler.continuous_columns_ or []),
    }


def _train_seed(prepared: dict, cfg: dict, model_name: str, seed: int) -> pd.DataFrame:
    import tensorflow as tf  # type: ignore

    nn_cfg = cfg["models"]["neural_network"]
    hidden = list(nn_cfg["architectures"][model_name])
    model = _build_keras_model(
        prepared["X_train"].shape[1],
        hidden,
        dropout=float(nn_cfg["dropout"]),
        learning_rate=float(nn_cfg.get("learning_rate", 0.001)),
        seed=int(seed),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(nn_cfg.get("patience", 100)),
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        prepared["X_train"].to_numpy(),
        prepared["y_train_std"].to_numpy(),
        validation_data=(prepared["X_val"].to_numpy(), prepared["y_val_std"].to_numpy()),
        epochs=int(nn_cfg.get("epochs", 500)),
        batch_size=int(nn_cfg.get("batch_size", 64)),
        verbose=0,
        callbacks=callbacks,
    )
    pred_val_std = model.predict(prepared["X_val"].to_numpy(), verbose=0).reshape(-1)
    pred_test_std = model.predict(prepared["X_test"].to_numpy(), verbose=0).reshape(-1)
    pred_val_raw = pred_val_std * prepared["y_std_train"] + prepared["y_mean_train"]
    pred_test_raw = pred_test_std * prepared["y_std_train"] + prepared["y_mean_train"]

    val_mse_raw = float(mean_squared_error(prepared["y_val_raw"].to_numpy(), pred_val_raw))
    val_mse_std = float(mean_squared_error(prepared["y_val_std"].to_numpy(), pred_val_std))
    epochs_run = int(len(history.history.get("loss", [])))
    best_epoch = int(np.argmin(history.history.get("val_loss", [np.nan])) + 1)

    out = prepared["test_base"].copy()
    out["forecast_raw"] = pred_test_raw
    out["seed"] = int(seed)
    out["val_mse"] = val_mse_raw
    out["val_mse_standardized"] = val_mse_std
    out["model"] = model_name
    out["dataset"] = prepared["dataset"]
    out["horizon"] = int(prepared["horizon"])
    out["features"] = ",".join(prepared["feature_cols"])
    out["scheme"] = "fixed"
    out["n_train"] = int(len(prepared["train"]))
    out["n_val"] = int(len(prepared["val"]))
    out["in_sample_min_rv"] = float(prepared["in_sample_min_rv"])
    out["in_sample_mean_rv"] = float(prepared["in_sample_mean_rv"])
    out["params"] = json.dumps(
        {
            "specification": "combined_corrected_paper_style",
            "backend": "tensorflow",
            "hidden": hidden,
            "seed": int(seed),
            "dropout": float(nn_cfg["dropout"]),
            "target_scaling": "standardized_y_train_inverse_transformed",
            "y_mean_train": float(prepared["y_mean_train"]),
            "y_std_train": float(prepared["y_std_train"]),
            "x_standardization": prepared["x_standardization"],
            "categorical_features_not_standardized": prepared["categorical_features_not_standardized"],
            "continuous_features_standardized": prepared["continuous_features_standardized"],
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
        },
        sort_keys=True,
    )
    return out


def _run_combo(panel: pd.DataFrame, cfg: dict, output_root: Path, dataset: str, horizon: int, model: str, ticker: str, seeds: list[int], force: bool) -> dict:
    prepared = None
    trained = 0
    reused = 0
    for seed in seeds:
        path = _seed_path(output_root, dataset, int(horizon), model, ticker, int(seed))
        if path.exists() and not force:
            reused += 1
            continue
        if prepared is None:
            prepared = _prepare_asset_frame(panel, cfg, dataset, model, ticker, int(horizon))
        logger.info("Training NN corrected dataset=%s h=%s model=%s ticker=%s seed=%s", dataset, horizon, model, ticker, seed)
        out = _train_seed(prepared, cfg, model, int(seed))
        _atomic_write_csv(out, path)
        trained += 1
    return {
        "dataset": dataset,
        "horizon": int(horizon),
        "model": model,
        "ticker": ticker.upper(),
        "trained": trained,
        "reused": reused,
    }


def _load_seed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "actual_rv", "forecast_raw", "seed", "val_mse", "val_mse_standardized"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _aggregate(output_root: Path, cfg: dict, datasets: list[str], horizons: list[int], models: list[str], tickers: list[str], seeds: list[int], ensemble_top: int, require_all: bool) -> pd.DataFrame:
    all_outputs = []
    for dataset in datasets:
        for horizon in horizons:
            for model_name in models:
                for label_kind in ("single", "top10"):
                    model_parts = []
                    out_model = f"{model_name}_1" if label_kind == "single" else model_name
                    for ticker in tickers:
                        available = []
                        for seed in seeds:
                            path = _seed_path(output_root, dataset, horizon, model_name, ticker, seed)
                            if path.exists():
                                df_seed = _load_seed(path)
                                available.append((seed, float(df_seed["val_mse_standardized"].iloc[0]), path))
                        if require_all and len(available) != len(seeds):
                            missing = sorted(set(seeds).difference(seed for seed, _, _ in available))
                            raise RuntimeError(f"Missing seeds for {dataset}/h{horizon}/{model_name}/{ticker}: {missing}")
                        need = 1 if label_kind == "single" else ensemble_top
                        if len(available) < need:
                            raise RuntimeError(f"Need {need} seeds for {dataset}/h{horizon}/{model_name}/{ticker}; found {len(available)}")
                        selected = sorted(available, key=lambda x: x[1])[:need]
                        seed_frames = [_load_seed(path) for _, _, path in selected]
                        base = seed_frames[0].copy()
                        for other in seed_frames[1:]:
                            if not base[["date", "ticker"]].equals(other[["date", "ticker"]]):
                                raise RuntimeError(f"Test rows differ across seeds for {dataset}/h{horizon}/{model_name}/{ticker}")
                        raw_matrix = np.vstack([f["forecast_raw"].to_numpy(dtype=float) for f in seed_frames])
                        raw_pred = raw_matrix.mean(axis=0)
                        prepared_meta = {
                            "in_sample_min_rv": float(base["in_sample_min_rv"].iloc[0]),
                            "in_sample_mean_rv": float(base["in_sample_mean_rv"].iloc[0]),
                        }
                        pred = _postprocess(raw_pred, prepared_meta, cfg)
                        params0 = json.loads(base["params"].iloc[0])
                        out = base[["date", "ticker", "rv", "oc_logret", "cc_logret", "actual_rv"]].copy()
                        out["model"] = out_model
                        out["forecast_rv"] = pred
                        out["scheme"] = "fixed"
                        out["n_train"] = int(base["n_train"].iloc[0])
                        out["n_val"] = int(base["n_val"].iloc[0])
                        out["params"] = json.dumps(
                            {
                                "specification": "combined_corrected_paper_style",
                                "base_model": model_name,
                                "model_label": out_model,
                                "dropout": float(params0["dropout"]),
                                "target_scaling": "standardized_y_train_inverse_transformed",
                                "x_standardization": params0.get("x_standardization"),
                                "categorical_features_not_standardized": params0.get("categorical_features_not_standardized", []),
                                "ensemble_top": need,
                                "selected_seeds": [int(seed) for seed, _, _ in selected],
                                "best_val_mse_standardized": float(selected[0][1]),
                                "seeds_available": len(available),
                            },
                            sort_keys=True,
                        )
                        out["dataset"] = dataset
                        out["horizon"] = int(horizon)
                        out["features"] = str(base["features"].iloc[0])
                        model_parts.append(out)
                    model_out = pd.concat(model_parts, ignore_index=True).sort_values(["ticker", "date"])
                    _atomic_write_csv(model_out, _model_output_path(output_root, dataset, horizon, out_model))
                    all_outputs.append(model_out)
    nn_out = pd.concat(all_outputs, ignore_index=True).sort_values(["dataset", "horizon", "model", "ticker", "date"])
    _atomic_write_csv(nn_out, output_root / "nn_aggregated" / "combined_nn_ensembles.csv")
    _atomic_write_csv(nn_out, output_root / "predictions" / "nn_model_predictions.csv")
    return nn_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected NN combined specification with dropout=0.2 and standardized y.")
    parser.add_argument("--config", default=str(ROOT / "config/paper_core_rolling_tuned_no_refit.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--archs", "--models", nargs="*", default=["NN1", "NN2", "NN3", "NN4"])
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--ensemble-top", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = _resolve(args.output_dir)
    if output_root.exists() and any(output_root.rglob("*")) and not args.allow_existing_output_dir:
        raise SystemExit(f"Output directory already exists and is non-empty: {output_root}")
    cfg["paths"]["output_dir"] = str(output_root)
    cfg["estimation"]["scheme"] = "fixed"
    cfg["models"]["neural_network"]["dropout"] = 0.2
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, "output_dir") / "logs" / "13_corrected_nn_combined.log")

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
    models = list(args.archs)
    base_seed = int(cfg["project"]["random_seed"]) if args.seed_start is None else int(args.seed_start)
    seed_count = int(args.seed_count if args.seed_count is not None else cfg["models"]["neural_network"].get("seeds", 50))
    seeds = [base_seed + i for i in range(seed_count)]
    ensemble_top = int(args.ensemble_top if args.ensemble_top is not None else cfg["models"]["neural_network"].get("ensemble_top", 10))
    if ensemble_top > len(seeds):
        raise SystemExit(f"ensemble_top={ensemble_top} cannot exceed seed_count={len(seeds)}")

    targets = set(panel.columns)
    for h in horizons:
        if f"target_rv_h{h}" not in targets:
            raise SystemExit(f"Missing target_rv_h{h}; rebuild forecasting_panel.csv for this horizon.")

    tasks = [
        (panel, cfg, output_root, dataset, horizon, model, ticker, seeds, args.force)
        for dataset in datasets
        for horizon in horizons
        for model in models
        for ticker in tickers
    ]
    logger.info(
        "Corrected NN combined run output=%s datasets=%s horizons=%s models=%s tickers=%s seeds=%s..%s workers=%s",
        output_root, datasets, horizons, models, tickers, seeds[0], seeds[-1], args.workers,
    )

    if not args.aggregate_only:
        results = Parallel(n_jobs=max(1, int(args.workers)), backend="threading")(
            delayed(_run_combo)(*task) for task in tasks
        )
        _atomic_write_csv(pd.DataFrame(results), output_root / "nn_aggregated" / "seed_training_manifest.csv")

    if not args.train_only:
        nn_out = _aggregate(output_root, cfg, datasets, horizons, models, tickers, seeds, ensemble_top, require_all=True)
        logger.info("Wrote corrected NN predictions rows=%d", len(nn_out))

    _atomic_write_json(
        {
            "script": Path(__file__).name,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "specification": "combined_corrected_paper_style",
            "datasets": datasets,
            "horizons": horizons,
            "models": models,
            "tickers": tickers,
            "seeds": seeds,
            "ensemble_top": ensemble_top,
            "dropout": 0.2,
            "target_scaling": "standardized_y_train_inverse_transformed",
            "x_standardization": "continuous_only_train_window",
        },
        output_root / "run_provenance_nn.json",
    )


if __name__ == "__main__":
    main()
