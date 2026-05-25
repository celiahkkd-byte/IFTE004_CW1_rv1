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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.config import load_config
from rv1rep.explain import accumulated_local_effect
from rv1rep.nn import _build_keras_model


DEFAULT_SOURCE_DIR = (
    ROOT
    / "outputs_variable_importance_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs_variable_importance_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_eafix_targeted_20260525"
)
DEFAULT_CONFIG = ROOT / "config" / "paper_core_rolling_tuned_no_refit.yaml"
DEFAULT_NN_CHECKPOINT_DIR = ROOT / "outputs_nn50_checkpointed_20260521"


def _load_vi_module():
    path = ROOT / "scripts" / "06c_compute_variable_importance.py"
    spec = importlib.util.spec_from_file_location("_rv1rep_vi_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import VI helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VI = _load_vi_module()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
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


def _ea_raw_importance(
    *,
    panel: pd.DataFrame,
    cfg: dict,
    model: str,
    ticker: str,
    dataset: str,
    horizon: int,
    grid_size: int,
    nn_checkpoint_dir: Path,
    nn_top: int,
) -> tuple[float, int, str, str]:
    prepared = VI.ALE_HELPERS._prepare_model_data(
        panel,
        cfg,
        dataset=dataset,
        model_name=model,
        ticker=ticker,
        horizon=horizon,
    )
    if model == "NN10_2":
        selected_seeds, selected_meta = _select_nn10_2_seeds(nn_checkpoint_dir, dataset, horizon, ticker, nn_top)
        predict_fn, fit_info = _fit_nn10_2_predictor(prepared, cfg, selected_seeds)
        fit_info["selected_seed_source"] = selected_meta
    else:
        predict_fn, fit_info = VI._fit_predictor(
            model,
            prepared,
            cfg,
            nn_checkpoint_dir,
            dataset,
            horizon,
            ticker,
            nn_top,
        )
    model_feature = VI.ALE_HELPERS._model_feature_name(model, "ea", prepared["X_in"].columns)
    if model_feature not in prepared["X_in"].columns:
        raise ValueError(f"EA feature is unavailable for model={model} ticker={ticker}")
    ale = accumulated_local_effect(predict_fn, prepared["X_in"], model_feature, grid_size=grid_size)
    if ale.empty or len(ale) < 2:
        raw = 0.0
    else:
        centered = ale["ale"].to_numpy(dtype=float) - float(ale["ale"].mean())
        raw = float(np.std(centered, ddof=1))
    return raw, int(len(ale)), model_feature, json.dumps(fit_info, sort_keys=True)


def _select_nn10_2_seeds(
    input_dir: Path,
    dataset: str,
    horizon: int,
    ticker: str,
    top_k: int,
) -> tuple[list[int], list[dict]]:
    directory = input_dir / "nn_seed_predictions" / dataset / f"h{int(horizon)}" / "NN2" / ticker
    files = sorted(directory.glob("seed_*.csv"))
    if not files:
        raise FileNotFoundError(f"No NN2 seed checkpoints found in {directory}")
    rows = []
    for path in files:
        head = pd.read_csv(path, nrows=1)
        missing = {"seed", "val_mse", "params"} - set(head.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {path}")
        rows.append(
            {
                "seed": int(head["seed"].iloc[0]),
                "val_mse": float(head["val_mse"].iloc[0]),
                "path": str(path),
                "params": str(head["params"].iloc[0]),
            }
        )
    selected = sorted(rows, key=lambda r: (r["val_mse"], r["seed"]))[: int(top_k)]
    return [int(row["seed"]) for row in selected], selected


def _fit_nn10_2_predictor(prepared: dict, cfg: dict, selected_seeds: list[int]):
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for NN10_2 VI repair.") from exc

    nn_cfg = cfg["models"]["neural_network"]
    hidden = list(nn_cfg["architectures"]["NN2"])
    models = []
    val_mse = []
    for seed in selected_seeds:
        model = _build_keras_model(
            prepared["X_train"].shape[1],
            hidden,
            dropout=float(nn_cfg.get("dropout", 0.8)),
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
        model.fit(
            prepared["X_train"].to_numpy(),
            prepared["y_train"].to_numpy(),
            validation_data=(prepared["X_val"].to_numpy(), prepared["y_val"].to_numpy()),
            epochs=int(nn_cfg.get("epochs", 500)),
            batch_size=int(nn_cfg.get("batch_size", 64)),
            verbose=0,
            callbacks=callbacks,
        )
        pred_val = model.predict(prepared["X_val"].to_numpy(), verbose=0).reshape(-1)
        val_mse.append(float(np.mean((prepared["y_val"].to_numpy() - pred_val) ** 2)))
        models.append(model)

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        arr = X[prepared["feature_cols"]].to_numpy()
        raw = np.mean([m.predict(arr, verbose=0).reshape(-1) for m in models], axis=0)
        return VI.ALE_HELPERS._postprocess_prediction(
            raw,
            cfg,
            prepared["in_sample_min_rv"],
            prepared["in_sample_mean_rv"],
        )

    fit_info = {
        "model": "NN10_2",
        "source_architecture": "NN2",
        "hidden": hidden,
        "selected_seeds": [int(seed) for seed in selected_seeds],
        "ensemble_top": int(len(selected_seeds)),
        "refit_for_vi_ea_repair": True,
        "val_mse_after_refit_mean": float(np.mean(val_mse)),
        "val_mse_after_refit_min": float(np.min(val_mse)),
        "target_col": prepared["target_col"],
        "feature_cols": prepared["feature_cols"],
        "n_train": len(prepared["train"]),
        "n_val": len(prepared["val"]),
    }
    return predict_fn, fit_info


def _repair_checkpoint(
    old: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    cfg: dict,
    model: str,
    ticker: str,
    dataset: str,
    horizon: int,
    grid_size: int,
    nn_checkpoint_dir: Path,
    nn_top: int,
) -> pd.DataFrame:
    rows = old.copy()
    ea_mask = rows["feature"].astype(str).str.lower().eq("ea")
    if ea_mask.sum() != 1:
        raise ValueError(f"Expected exactly one EA row for model={model} ticker={ticker}; got {int(ea_mask.sum())}")
    raw, n_points, model_feature, fit_info = _ea_raw_importance(
        panel=panel,
        cfg=cfg,
        model=model,
        ticker=ticker,
        dataset=dataset,
        horizon=horizon,
        grid_size=grid_size,
        nn_checkpoint_dir=nn_checkpoint_dir,
        nn_top=nn_top,
    )
    rows.loc[ea_mask, "importance_raw"] = raw
    rows.loc[ea_mask, "ale_points"] = n_points
    rows.loc[ea_mask, "grid_size"] = int(grid_size)
    rows.loc[ea_mask, "model_feature"] = model_feature
    rows.loc[ea_mask, "fit_info"] = fit_info

    total = float(rows["importance_raw"].sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Cannot normalize repaired VI model={model} ticker={ticker}; raw total={total}")
    rows["vi_normalized"] = rows["importance_raw"] / total
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Figure 7 VI tables by recomputing only binary EA ALE under the mainline h=1 setup."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--nn-checkpoint-dir", default=str(DEFAULT_NN_CHECKPOINT_DIR))
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--min-tickers", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=["HARX", "ElasticNet", "RandomForest", "NN10_2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = _resolve(args.source_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg["models"]["trees"]["n_jobs"] = int(cfg.get("models", {}).get("trees", {}).get("n_jobs", 1))
    panel = VI.ALE_HELPERS._load_panel(cfg)
    nn_checkpoint_dir = _resolve(args.nn_checkpoint_dir)
    nn_top = int(cfg["models"]["neural_network"].get("ensemble_top", 10))

    old_path = source_dir / "tables" / "variable_importance_by_ticker.csv"
    old_per_ticker = pd.read_csv(old_path)
    sub = old_per_ticker[
        (old_per_ticker["dataset"].astype(str) == args.dataset)
        & (old_per_ticker["horizon"].astype(int) == int(args.horizon))
        & (old_per_ticker["model"].astype(str).isin(args.models))
    ].copy()
    if sub.empty:
        raise ValueError(f"No source VI rows found in {old_path}")

    repaired_parts = []
    repair_rows = []
    for model in args.models:
        tickers = sorted(sub.loc[sub["model"].astype(str) == model, "ticker"].astype(str).unique())
        for ticker in tickers:
            checkpoint_path = output_dir / "checkpoints" / VI._safe_name(model) / f"{VI._safe_name(ticker)}.csv"
            if checkpoint_path.exists():
                repaired = pd.read_csv(checkpoint_path)
                repaired_parts.append(repaired)
                ea_new = repaired[repaired["feature"].astype(str).str.lower().eq("ea")].iloc[0]
                old_checkpoint = sub[(sub["model"].astype(str) == model) & (sub["ticker"].astype(str) == ticker)].copy()
                ea_old = old_checkpoint[old_checkpoint["feature"].astype(str).str.lower().eq("ea")].iloc[0]
                repair_rows.append(
                    {
                        "model": model,
                        "ticker": ticker,
                        "old_ea_importance_raw": float(ea_old["importance_raw"]),
                        "new_ea_importance_raw": float(ea_new["importance_raw"]),
                        "old_ea_vi_normalized": float(ea_old["vi_normalized"]),
                        "new_ea_vi_normalized": float(ea_new["vi_normalized"]),
                        "ea_ale_points": int(ea_new["ale_points"]),
                    }
                )
                print(f"reused {model} {ticker}: EA raw {float(ea_new['importance_raw']):.8g}")
                continue
            old_checkpoint = sub[(sub["model"].astype(str) == model) & (sub["ticker"].astype(str) == ticker)].copy()
            repaired = _repair_checkpoint(
                old_checkpoint,
                panel=panel,
                cfg=cfg,
                model=model,
                ticker=ticker,
                dataset=args.dataset,
                horizon=int(args.horizon),
                grid_size=int(args.grid_size),
                nn_checkpoint_dir=nn_checkpoint_dir,
                nn_top=nn_top,
            )
            repaired_parts.append(repaired)
            ea_old = old_checkpoint[old_checkpoint["feature"].astype(str).str.lower().eq("ea")].iloc[0]
            ea_new = repaired[repaired["feature"].astype(str).str.lower().eq("ea")].iloc[0]
            repair_rows.append(
                {
                    "model": model,
                    "ticker": ticker,
                    "old_ea_importance_raw": float(ea_old["importance_raw"]),
                    "new_ea_importance_raw": float(ea_new["importance_raw"]),
                    "old_ea_vi_normalized": float(ea_old["vi_normalized"]),
                    "new_ea_vi_normalized": float(ea_new["vi_normalized"]),
                    "ea_ale_points": int(ea_new["ale_points"]),
                }
            )
            _atomic_write_csv(repaired, checkpoint_path)
            print(f"repaired {model} {ticker}: EA raw {float(ea_new['importance_raw']):.8g}")

    per_ticker = pd.concat(repaired_parts, ignore_index=True)
    sums = per_ticker.groupby(["model", "ticker"])["vi_normalized"].sum()
    bad_sums = sums[(sums < 0.999) | (sums > 1.001)]
    if not bad_sums.empty:
        raise RuntimeError(f"Repaired per model/ticker VI sums are not one: {bad_sums.head().to_dict()}")

    agg = VI._aggregate_vi(per_ticker, min_tickers=int(args.min_tickers))
    table_path = output_dir / "tables" / "variable_importance.csv"
    per_ticker_path = output_dir / "tables" / "variable_importance_by_ticker.csv"
    repair_path = output_dir / "tables" / "ea_repair_audit.csv"
    figure_path = output_dir / "figures" / "figure7_variable_importance.png"
    _atomic_write_csv(agg, table_path)
    _atomic_write_csv(per_ticker, per_ticker_path)
    _atomic_write_csv(pd.DataFrame(repair_rows), repair_path)
    VI._plot_variable_importance(agg, figure_path, models=args.models)

    non_ea_old = sub[~sub["feature"].astype(str).str.lower().eq("ea")].copy()
    non_ea_new = per_ticker[~per_ticker["feature"].astype(str).str.lower().eq("ea")].copy()
    merged = non_ea_old.merge(
        non_ea_new,
        on=["dataset", "horizon", "ticker", "model", "feature"],
        suffixes=("_old", "_new"),
    )
    max_non_ea_raw_diff = float((merged["importance_raw_new"] - merged["importance_raw_old"]).abs().max())

    _atomic_write_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "targeted EA repair: recompute only binary EA raw ALE importance, preserve non-EA raw importances, renormalize within model/ticker",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "config": str(_resolve(args.config)),
            "nn_checkpoint_dir": str(nn_checkpoint_dir),
            "dataset": args.dataset,
            "horizon": int(args.horizon),
            "models": args.models,
            "n_tickers_by_model": {m: int(per_ticker[per_ticker["model"].astype(str).eq(m)]["ticker"].nunique()) for m in args.models},
            "max_non_ea_raw_importance_abs_diff_vs_source": max_non_ea_raw_diff,
            "outputs": {
                "tables/variable_importance.csv": str(table_path),
                "tables/variable_importance_by_ticker.csv": str(per_ticker_path),
                "tables/ea_repair_audit.csv": str(repair_path),
                "figures/figure7_variable_importance.png": str(figure_path),
            },
        },
        output_dir / "run_provenance.json",
    )
    print(f"Wrote {table_path}")
    print(f"Wrote {per_ticker_path}")
    print(f"Wrote {repair_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
