#!/usr/bin/env python3
"""Minimal NN target-scaling diagnostic.

This script compares the current raw-y NN checkpoints with a fresh run where the
NN target is standardized on the training sample and inverse-transformed before
validation/test evaluation. It is intentionally scoped to one
dataset/horizon/ticker/model by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rv1rep_matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.config import load_config, project_path
from rv1rep.nn import _build_keras_model
from rv1rep.preprocessing import enforce_positive_forecasts, insanity_filter
from rv1rep.utils import setup_logging


def _load_nn_runner():
    path = ROOT / "scripts" / "04_run_nn_checkpoints.py"
    spec = importlib.util.spec_from_file_location("_rv1rep_nn_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import NN runner helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NN_RUNNER = _load_nn_runner()


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal NN y-scaling diagnostic.")
    parser.add_argument("--config", default=str(ROOT / "config/paper_core_rolling_tuned_no_refit.yaml"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs_nn50_checkpointed_20260521"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs_nn_target_scaling_diagnostic_20260524"))
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model", default="NN2")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--ensemble-top", type=int, default=None)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def assert_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.rglob("*")) and not allow_existing:
        raise SystemExit(
            f"Output directory already exists and contains files: {output_dir}. "
            "Use --allow-existing-output-dir to update it intentionally."
        )


def summarize_forecast(label: str, actual: np.ndarray, forecast: np.ndarray, selected_seeds: list[int]) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    actual_std = float(np.std(actual, ddof=1))
    forecast_std = float(np.std(forecast, ddof=1))
    if actual_std > 0 and forecast_std > 0:
        corr = float(np.corrcoef(actual, forecast)[0, 1])
    else:
        corr = np.nan
    return {
        "variant": label,
        "n_test": int(len(actual)),
        "mse": float(mean_squared_error(actual, forecast)),
        "mae": float(mean_absolute_error(actual, forecast)),
        "actual_mean": float(np.mean(actual)),
        "forecast_mean": float(np.mean(forecast)),
        "actual_std": actual_std,
        "forecast_std": forecast_std,
        "forecast_to_actual_std_ratio": float(forecast_std / actual_std) if actual_std > 0 else np.nan,
        "corr_actual_forecast": corr,
        "forecast_min": float(np.min(forecast)),
        "forecast_max": float(np.max(forecast)),
        "selected_seeds": ",".join(str(s) for s in selected_seeds),
    }


def postprocess_forecast(raw_pred: np.ndarray, prepared: dict, cfg: dict) -> np.ndarray:
    pred = enforce_positive_forecasts(
        raw_pred,
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
    return pred


def load_current_raw_y_seed_results(
    *,
    checkpoint_dir: Path,
    dataset: str,
    horizon: int,
    model: str,
    ticker: str,
    seeds: list[int],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows = []
    frames = []
    for seed in seeds:
        path = NN_RUNNER._checkpoint_path(checkpoint_dir, dataset, horizon, model, ticker, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing current raw-y checkpoint: {path}")
        df = NN_RUNNER._load_seed_file(path)
        rows.append(
            {
                "variant": "current_raw_y",
                "seed": int(seed),
                "val_mse_raw_units": float(df["val_mse"].iloc[0]),
                "test_forecast_raw_mean": float(df["forecast_raw"].mean()),
                "test_forecast_raw_std": float(df["forecast_raw"].std(ddof=1)),
                "test_forecast_raw_min": float(df["forecast_raw"].min()),
                "test_forecast_raw_max": float(df["forecast_raw"].max()),
                "checkpoint_path": str(path),
            }
        )
        frames.append(df)
    return pd.DataFrame(rows), frames


def train_standardized_y_seeds(
    *,
    prepared: dict,
    cfg: dict,
    model_name: str,
    seeds: list[int],
    output_dir: Path,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    import tensorflow as tf  # type: ignore

    nn_cfg = cfg["models"]["neural_network"]
    hidden = list(nn_cfg["architectures"][model_name])
    y_train = prepared["y_train"].astype(float)
    y_val = prepared["y_val"].astype(float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=1))
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0

    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std
    rows = []
    frames = []
    seed_dir = output_dir / "seed_predictions" / "standardized_y"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for idx, seed in enumerate(seeds, start=1):
        print(f"[standardized-y] fitting seed={seed} ({idx}/{len(seeds)})", flush=True)
        model = _build_keras_model(
            prepared["X_train"].shape[1],
            hidden,
            dropout=float(nn_cfg.get("dropout", 0.8)),
            learning_rate=float(nn_cfg.get("learning_rate", 0.001)),
            seed=seed,
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
            y_train_std.to_numpy(),
            validation_data=(prepared["X_val"].to_numpy(), y_val_std.to_numpy()),
            epochs=int(nn_cfg.get("epochs", 500)),
            batch_size=int(nn_cfg.get("batch_size", 64)),
            verbose=0,
            callbacks=callbacks,
        )
        pred_val_std = model.predict(prepared["X_val"].to_numpy(), verbose=0).reshape(-1)
        pred_test_std = model.predict(prepared["X_test"].to_numpy(), verbose=0).reshape(-1)
        pred_val_raw = pred_val_std * y_std + y_mean
        pred_test_raw = pred_test_std * y_std + y_mean
        val_mse_raw = float(mean_squared_error(y_val.to_numpy(), pred_val_raw))
        val_mse_std = float(mean_squared_error(y_val_std.to_numpy(), pred_val_std))
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
        out["scheme"] = cfg["estimation"]["scheme"]
        out["n_train"] = int(len(prepared["train"]))
        out["n_val"] = int(len(prepared["val"]))
        out["params"] = json.dumps(
            {
                "backend": "tensorflow",
                "hidden": hidden,
                "seed": int(seed),
                "target_scaling": "train_y_standardized_then_inverse_transform",
                "y_mean_train": y_mean,
                "y_std_train": y_std,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
            },
            sort_keys=True,
        )
        seed_path = seed_dir / f"seed_{seed}.csv"
        atomic_write_csv(out, seed_path)
        rows.append(
            {
                "variant": "standardized_y",
                "seed": int(seed),
                "val_mse_raw_units": val_mse_raw,
                "val_mse_standardized_units": val_mse_std,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "test_forecast_raw_mean": float(np.mean(pred_test_raw)),
                "test_forecast_raw_std": float(np.std(pred_test_raw, ddof=1)),
                "test_forecast_raw_min": float(np.min(pred_test_raw)),
                "test_forecast_raw_max": float(np.max(pred_test_raw)),
                "checkpoint_path": str(seed_path),
            }
        )
        frames.append(out)
    return pd.DataFrame(rows), frames


def aggregate_top_k(
    frames: list[pd.DataFrame],
    seed_scores: pd.DataFrame,
    variant: str,
    ensemble_top: int,
    prepared: dict,
    cfg: dict,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    selected = (
        seed_scores[seed_scores["variant"] == variant]
        .sort_values(["val_mse_raw_units", "seed"])
        .head(int(ensemble_top))
        .copy()
    )
    selected_seeds = selected["seed"].astype(int).tolist()
    by_seed = {int(f["seed"].iloc[0]): f for f in frames}
    selected_frames = [by_seed[s] for s in selected_seeds]
    base = selected_frames[0][["date", "ticker", "rv", "oc_logret", "cc_logret", "actual_rv"]].copy()
    raw_matrix = np.vstack([f["forecast_raw"].to_numpy(dtype=float) for f in selected_frames])
    pred = postprocess_forecast(raw_matrix.mean(axis=0), prepared, cfg)
    out = base.copy()
    out["variant"] = f"{variant}_top{ensemble_top}"
    out["forecast_rv"] = pred
    out["selected_seeds"] = ",".join(str(s) for s in selected_seeds)
    return out, pred, selected_seeds


def write_report(path: Path, metrics: pd.DataFrame, seed_results: pd.DataFrame, args: argparse.Namespace) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda x: f"{x:.6g}")
        cols = list(display.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    top_seed_summary = (
        seed_results.sort_values(["variant", "val_mse_raw_units"])
        .groupby("variant", as_index=False)
        .head(10)
        [["variant", "seed", "val_mse_raw_units", "test_forecast_raw_std", "epochs_run"]]
        if "epochs_run" in seed_results.columns
        else seed_results.sort_values(["variant", "val_mse_raw_units"])
        .groupby("variant", as_index=False)
        .head(10)
        [["variant", "seed", "val_mse_raw_units", "test_forecast_raw_std"]]
    )
    report = [
        "# Minimal NN Target-Scaling Diagnostic",
        "",
        f"Created at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Dataset: `{args.dataset}`",
        f"Horizon: `{int(args.horizon)}`",
        f"Ticker: `{args.ticker}`",
        f"Model: `{args.model}`",
        "",
        "## Aggregate Test Metrics",
        "",
        md_table(metrics),
        "",
        "## Top Validation Seeds",
        "",
        md_table(top_seed_summary),
        "",
        "## Interpretation Rule",
        "",
        "If the standardized-y variant materially lowers test MSE and raises forecast dispersion without pathological forecasts, target scaling is likely affecting NN performance.",
        "If the metrics are similar, the current weak NN performance is less likely to be explained by target scale alone.",
        "",
    ]
    path.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    assert_output_dir(output_dir, bool(args.allow_existing_output_dir))
    setup_logging(output_dir / "logs" / "11_nn_target_scaling_diagnostic.log")

    cfg = load_config(args.config)
    checkpoint_dir = resolve(args.checkpoint_dir)
    panel = pd.read_csv(project_path(cfg, "processed_dir") / "forecasting_panel.csv", parse_dates=["date"])
    prepared = NN_RUNNER._prepare_asset_frame(panel, cfg, args.dataset, args.model, args.ticker, int(args.horizon))
    prepared["dataset"] = args.dataset
    prepared["horizon"] = int(args.horizon)

    seed_start = int(args.seed_start if args.seed_start is not None else cfg["project"]["random_seed"])
    seed_count = int(args.seed_count if args.seed_count is not None else cfg["models"]["neural_network"]["seeds"])
    seeds = list(range(seed_start, seed_start + seed_count))
    ensemble_top = int(args.ensemble_top if args.ensemble_top is not None else cfg["models"]["neural_network"]["ensemble_top"])

    raw_seed_scores, raw_seed_frames = load_current_raw_y_seed_results(
        checkpoint_dir=checkpoint_dir,
        dataset=args.dataset,
        horizon=int(args.horizon),
        model=args.model,
        ticker=args.ticker,
        seeds=seeds,
    )
    std_seed_scores, std_seed_frames = train_standardized_y_seeds(
        prepared=prepared,
        cfg=cfg,
        model_name=args.model,
        seeds=seeds,
        output_dir=output_dir,
    )
    seed_results = pd.concat([raw_seed_scores, std_seed_scores], ignore_index=True)
    atomic_write_csv(seed_results, output_dir / "tables" / "seed_results.csv")

    raw_pred_table, raw_pred, raw_selected = aggregate_top_k(
        raw_seed_frames, seed_results, "current_raw_y", ensemble_top, prepared, cfg
    )
    std_pred_table, std_pred, std_selected = aggregate_top_k(
        std_seed_frames, seed_results, "standardized_y", ensemble_top, prepared, cfg
    )
    pred_table = pd.concat([raw_pred_table, std_pred_table], ignore_index=True)
    atomic_write_csv(pred_table, output_dir / "predictions" / "diagnostic_predictions.csv")

    actual = raw_pred_table["actual_rv"].to_numpy(dtype=float)
    metrics = pd.DataFrame(
        [
            summarize_forecast(f"current_raw_y_top{ensemble_top}", actual, raw_pred, raw_selected),
            summarize_forecast(f"standardized_y_top{ensemble_top}", actual, std_pred, std_selected),
        ]
    )
    raw_mse = float(metrics.loc[metrics["variant"].str.startswith("current_raw_y"), "mse"].iloc[0])
    metrics["mse_relative_to_current_raw_y"] = metrics["mse"] / raw_mse if raw_mse > 0 else np.nan
    atomic_write_csv(metrics, output_dir / "tables" / "aggregate_metrics.csv")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "config": str(resolve(args.config)),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "dataset": args.dataset,
        "horizon": int(args.horizon),
        "ticker": args.ticker,
        "model": args.model,
        "seeds": seeds,
        "ensemble_top": ensemble_top,
        "target_scaling_variant": "train_y_standardized_then_inverse_transform",
        "outputs": {
            "seed_results": str(output_dir / "tables" / "seed_results.csv"),
            "aggregate_metrics": str(output_dir / "tables" / "aggregate_metrics.csv"),
            "predictions": str(output_dir / "predictions" / "diagnostic_predictions.csv"),
            "report": str(output_dir / "NN_TARGET_SCALING_DIAGNOSTIC.md"),
        },
    }
    atomic_write_json(provenance, output_dir / "run_provenance.json")
    write_report(output_dir / "NN_TARGET_SCALING_DIAGNOSTIC.md", metrics, seed_results, args)
    print(f"Wrote {output_dir / 'tables' / 'aggregate_metrics.csv'}")
    print(f"Wrote {output_dir / 'NN_TARGET_SCALING_DIAGNOSTIC.md'}")


if __name__ == "__main__":
    main()
