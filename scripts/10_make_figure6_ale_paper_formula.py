#!/usr/bin/env python3
"""Recompute Figure 6-style ALE with the paper's centering formula.

This script differs from the earlier compact ALE implementation in
``src/rv1rep/explain.py`` in two ways:

1. The ALE information set is the standardized training block ``Z``.
2. Centering follows paper equation (29), subtracting the average uncentered
   ALE evaluated over the training observations, not the simple average over
   displayed grid points.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rv1rep_matplotlib")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.config import load_config
from rv1rep.utils import setup_logging


def _load_ale_base():
    path = ROOT / "scripts" / "08_compute_ale_checkpointed.py"
    spec = importlib.util.spec_from_file_location("_rv1rep_ale_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ALE helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALE_BASE = _load_ale_base()

MODEL_ORDER = ["HARX", "LogHAR", "ElasticNet", "RandomForest", "NN10_2"]
MODEL_LABELS = {
    "HARX": "HAR-X",
    "LogHAR": "LogHAR",
    "ElasticNet": "EN",
    "RandomForest": "RF",
    "NN10_2": r"NN$^{10}_{2}$",
}
MODEL_STYLES = {
    "HARX": {"color": "black", "linestyle": "-", "linewidth": 1.6},
    "LogHAR": {"color": "red", "linestyle": (0, (4, 3)), "linewidth": 1.5},
    "ElasticNet": {"color": "blue", "linestyle": ":", "linewidth": 1.7},
    "RandomForest": {"color": "purple", "linestyle": "--", "linewidth": 1.5},
    "NN10_2": {"color": "#00cc00", "linestyle": (0, (1, 1)), "linewidth": 1.7},
}
FEATURE_ORDER = ["rvd", "rvw", "m1w"]
FEATURE_TITLES = {
    "rvd": "(A) RVD",
    "rvw": "(B) RVW",
    "m1w": "(C) M1W",
}
FEATURE_YLABELS = {
    "rvd": r"$f^{ALE}$ (rvd)",
    "rvw": r"$f^{ALE}$ (rvw)",
    "m1w": r"$f^{ALE}$ (m1w)",
}
FEATURE_YLABELS_SCALED = {
    "rvd": r"$10^3 \times f^{ALE}$ (rvd)",
    "rvw": r"$10^3 \times f^{ALE}$ (rvw)",
    "m1w": r"$10^3 \times f^{ALE}$ (m1w)",
}
DISPLAY_YLIMS = {
    "rvd": (-0.5, 0.5),
    "rvw": (-0.5, 0.5),
    "m1w": (-0.1, 0.1),
}
DISPLAY_YTICKS = {
    "rvd": np.arange(-0.5, 0.5001, 0.25),
    "rvw": np.arange(-0.5, 0.5001, 0.25),
    "m1w": np.arange(-0.1, 0.1001, 0.05),
}


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Figure 6 ALE with paper equation (29) centering.")
    parser.add_argument("--config", default=str(ROOT / "config/paper_core_rolling_tuned_no_refit.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs_figure6_ale_paper_formula_20260524"))
    parser.add_argument("--nn-checkpoint-dir", default=str(ROOT / "outputs_nn50_checkpointed_20260521"))
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--features", nargs="+", default=FEATURE_ORDER)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--nn-ensemble-top", type=int, default=10)
    parser.add_argument("--tree-n-jobs", type=int, default=5)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--display-scale", type=float, default=1000.0)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def assert_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.rglob("*")) and not allow_existing:
        raise SystemExit(
            f"Output directory already exists and contains files: {output_dir}. "
            "Use --allow-existing-output-dir to update it intentionally."
        )


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


def paper_formula_ale(
    model_predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    *,
    grid_size: int,
    eps: float = 1.0e-12,
) -> pd.DataFrame:
    """Estimate centered ALE using paper equations (28)-(29).

    Quantile grid intervals are used, matching the paper's equal-frequency ALE
    display convention. Centering is observation-weighted:
    sum_k T_k * tilde_f_k / T0.
    """
    if feature not in X.columns:
        raise ValueError(f"{feature} not in X")
    xj = X[feature].astype(float)
    valid_mask = xj.notna()
    X_valid = X.loc[valid_mask].copy()
    x_valid = X_valid[feature].astype(float)
    if X_valid.empty:
        return pd.DataFrame()

    quantiles = np.linspace(0.0, 1.0, int(grid_size) + 1)
    edges = np.quantile(x_valid.to_numpy(), quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return pd.DataFrame()
    edges[0] = edges[0] - float(eps)

    rows = []
    cumulative = 0.0
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        if hi <= lo:
            continue
        mask = (x_valid > lo) & (x_valid <= hi)
        tk = int(mask.sum())
        if tk == 0:
            interval_effect = 0.0
        else:
            X_lo = X_valid.loc[mask].copy()
            X_hi = X_valid.loc[mask].copy()
            X_lo[feature] = float(lo)
            X_hi[feature] = float(hi)
            interval_effect = float(np.mean(model_predict(X_hi) - model_predict(X_lo)))
        cumulative += interval_effect
        rows.append(
            {
                "bin": int(k),
                "bin_left": float(lo),
                "bin_right": float(hi),
                "x_standardized": float((lo + hi) / 2.0),
                "n_bin": int(tk),
                "interval_effect": interval_effect,
                "ale_uncentered": cumulative,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    t0 = int(out["n_bin"].sum())
    if t0 <= 0:
        raise RuntimeError(f"No observations assigned to ALE intervals for {feature}")
    center_offset = float(np.sum(out["n_bin"].to_numpy() * out["ale_uncentered"].to_numpy()) / t0)
    out["ale"] = out["ale_uncentered"] - center_offset
    out["center_offset_eq29"] = center_offset
    out["weighted_centered_mean_eq29"] = float(np.sum(out["n_bin"].to_numpy() * out["ale"].to_numpy()) / t0)
    out["n_training_observations"] = t0
    return out


def fit_predictor(model: str, prepared: dict, cfg: dict, args: argparse.Namespace) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    if model == "NN10_2":
        selected_seeds, selected_meta = ALE_BASE._select_nn_seeds_from_checkpoints(
            resolve(args.nn_checkpoint_dir),
            args.dataset,
            int(args.horizon),
            args.ticker,
            int(args.nn_ensemble_top),
        )
        predict_fn, fit_info = ALE_BASE._fit_nn10_2_predictor(prepared, cfg, selected_seeds)
        fit_info["selected_seed_source"] = selected_meta
        return predict_fn, fit_info
    return ALE_BASE._fit_sklearn_predictor(model, prepared, cfg)


def compute_all(cfg: dict, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    panel = ALE_BASE._load_panel(cfg)
    grid_size = int(args.grid_size or cfg.get("interpretability", {}).get("ale_grid_size", 100))
    cfg["models"]["trees"]["n_jobs"] = int(args.tree_n_jobs)

    parts = []
    fit_summaries = []
    for model in args.models:
        print(f"[fit] {model}", flush=True)
        prepared = ALE_BASE._prepare_model_data(
            panel,
            cfg,
            dataset=args.dataset,
            model_name=model,
            ticker=args.ticker,
            horizon=int(args.horizon),
        )
        predict_fn, fit_info = fit_predictor(model, prepared, cfg, args)
        fit_info["paper_formula_ale_information_set"] = "training_sample_Z"
        fit_summaries.append(fit_info)
        X_ale = prepared["X_train"]
        for paper_feature in args.features:
            model_feature = ALE_BASE._model_feature_name(model, paper_feature, X_ale.columns)
            if model_feature not in X_ale.columns:
                print(f"[skip] model={model} feature={paper_feature} unavailable", flush=True)
                continue
            print(f"[ale] model={model} feature={paper_feature}", flush=True)
            ale = paper_formula_ale(predict_fn, X_ale, model_feature, grid_size=grid_size)
            if ale.empty:
                continue
            ale.insert(0, "ticker", args.ticker)
            ale.insert(1, "dataset", args.dataset)
            ale.insert(2, "horizon", int(args.horizon))
            ale.insert(3, "model", model)
            ale["feature"] = paper_feature
            ale["model_feature"] = model_feature
            ale["grid_size"] = int(grid_size)
            ale["ale_sample"] = "train"
            ale["fit_info"] = json.dumps(fit_info, sort_keys=True)
            parts.append(ale)

    if not parts:
        raise RuntimeError("No ALE rows were produced.")
    table = pd.concat(parts, ignore_index=True)
    diagnostics = (
        table.groupby(["model", "feature"], as_index=False)
        .agg(
            n_bins=("bin", "count"),
            n_training_observations=("n_training_observations", "first"),
            center_offset_eq29=("center_offset_eq29", "first"),
            weighted_centered_mean_eq29=("weighted_centered_mean_eq29", "first"),
            simple_grid_mean=("ale", "mean"),
            ale_min=("ale", "min"),
            ale_max=("ale", "max"),
        )
        .sort_values(["feature", "model"])
        .reset_index(drop=True)
    )
    return table, diagnostics, fit_summaries


def draw_figure(
    plot_df: pd.DataFrame,
    output_path: Path,
    *,
    features: list[str],
    models: list[str],
    x_min: float,
    x_max: float,
    display_scale: float,
    fixed_display_axis: bool,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    draw_df = plot_df[plot_df["x_standardized"].between(float(x_min), float(x_max))].copy()
    draw_df["ale_draw"] = draw_df["ale"].astype(float) * float(display_scale)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, len(features), figsize=(12.0, 3.8), sharex=True)
    if len(features) == 1:
        axes = [axes]

    axis_rows = []
    for ax, feature in zip(axes, features):
        for model in models:
            sub = draw_df[(draw_df["feature"] == feature) & (draw_df["model"] == model)].sort_values("x_standardized")
            if sub.empty:
                continue
            ax.plot(
                sub["x_standardized"],
                sub["ale_draw"],
                label=MODEL_LABELS.get(model, model),
                **MODEL_STYLES.get(model, {"linewidth": 1.5}),
            )
        vals = draw_df.loc[draw_df["feature"] == feature, "ale_draw"].astype(float)
        if fixed_display_axis:
            ymin, ymax = DISPLAY_YLIMS.get(feature, (-0.5, 0.5))
            yticks = DISPLAY_YTICKS.get(feature, np.linspace(ymin, ymax, 5))
        else:
            bound = max(abs(float(vals.min())), abs(float(vals.max()))) if not vals.empty else 1.0
            bound = bound * 1.15 if bound > 0 else 1.0
            ymin, ymax = -bound, bound
            yticks = np.linspace(ymin, ymax, 5)
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(float(x_min), float(x_max) + 0.001, 0.5))
        ax.set_yticks(yticks)
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_xlabel(feature)
        ylabel_map = FEATURE_YLABELS if float(display_scale) == 1.0 else FEATURE_YLABELS_SCALED
        ax.set_ylabel(ylabel_map.get(feature, r"$f^{ALE}$"))
        ax.set_title(FEATURE_TITLES.get(feature, feature), loc="left", fontsize=12, fontweight="bold")
        ax.grid(True, color="#e0e0e0", linewidth=0.7, alpha=0.9)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
        axis_rows.append(
            {
                "feature": feature,
                "display_scale": float(display_scale),
                "fixed_display_axis": bool(fixed_display_axis),
                "y_min": float(ymin),
                "y_max": float(ymax),
                "data_min": float(vals.min()) if not vals.empty else np.nan,
                "data_max": float(vals.max()) if not vals.empty else np.nan,
                "points_outside_axis": int(((vals < ymin) | (vals > ymax)).sum()) if not vals.empty else 0,
            }
        )
    axes[0].legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#666666", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp.png")
    fig.savefig(tmp, dpi=300)
    plt.close(fig)
    tmp.replace(output_path)
    return pd.DataFrame(axis_rows)


def write_note(path: Path, args: argparse.Namespace, diagnostics: pd.DataFrame) -> None:
    note = f"""Figure 6 ALE based on the paper's centering formula.

Notes: This output recomputes ALE for ticker {args.ticker}, dataset {args.dataset},
and h={int(args.horizon)} using the paper's equations (28)-(29). The ALE
information set Z is the standardized training block. For each feature, interval
effects are accumulated and then centered by subtracting the training-observation
average of the uncentered ALE, sum_k T_k * tilde_f_k / T0. The raw table stores
the unscaled centered ALE in realized-variance units. The unscaled auto-axis
figure is the direct plot of f^ALE(feature). A second fixed-axis figure keeps the
paper-style axis limits without rescaling. The display-scaled figure multiplies
ALE by {float(args.display_scale):g} only for readability and labels the axis as
10^3 x f^ALE(feature), so it should not be read as the raw paper-scale value.

Firm-level IV is unavailable in the current processed panel, so the IV panel is
not reproduced and no proxy is substituted.

Centering diagnostics:
{diagnostics.to_string(index=False)}
"""
    path.write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    assert_output_dir(output_dir, bool(args.allow_existing_output_dir))
    setup_logging(output_dir / "logs" / "10_make_figure6_ale_paper_formula.log")

    cfg = load_config(args.config)
    table, diagnostics, fit_summaries = compute_all(cfg, args)
    table_path = output_dir / "tables" / "ale_table_paper_formula.csv"
    diag_path = output_dir / "tables" / "ale_centering_diagnostics.csv"
    atomic_write_csv(table, table_path)
    atomic_write_csv(diagnostics, diag_path)

    axis_display = draw_figure(
        table,
        output_dir / "figures" / "figure6_ale_paper_formula_display_h1.png",
        features=args.features,
        models=args.models,
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        display_scale=float(args.display_scale),
        fixed_display_axis=True,
    )
    axis_unscaled_fixed = draw_figure(
        table,
        output_dir / "figures" / "figure6_ale_paper_formula_unscaled_paper_axis_h1.png",
        features=args.features,
        models=args.models,
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        display_scale=1.0,
        fixed_display_axis=True,
    )
    axis_raw = draw_figure(
        table,
        output_dir / "figures" / "figure6_ale_paper_formula_raw_autoaxis_h1.png",
        features=args.features,
        models=args.models,
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        display_scale=1.0,
        fixed_display_axis=False,
    )
    atomic_write_csv(
        pd.concat([axis_display, axis_unscaled_fixed, axis_raw], ignore_index=True),
        output_dir / "tables" / "figure_axis_diagnostics.csv",
    )
    write_note(output_dir / "FIGURE6_PAPER_FORMULA_NOTE.md", args, diagnostics)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "config": str(resolve(args.config)),
        "nn_checkpoint_dir": str(resolve(args.nn_checkpoint_dir)),
        "output_dir": str(output_dir),
        "dataset": args.dataset,
        "horizon": int(args.horizon),
        "ticker": args.ticker,
        "models": args.models,
        "features": args.features,
        "grid_size": int(args.grid_size or cfg.get("interpretability", {}).get("ale_grid_size", 100)),
        "ale_information_set": "training_sample_Z",
        "centering": "paper_eq29_observation_weighted",
        "display_scale": float(args.display_scale),
        "fit_summaries": fit_summaries,
        "outputs": {
            "ale_table": str(table_path),
            "centering_diagnostics": str(diag_path),
            "display_figure": str(output_dir / "figures" / "figure6_ale_paper_formula_display_h1.png"),
            "unscaled_paper_axis_figure": str(
                output_dir / "figures" / "figure6_ale_paper_formula_unscaled_paper_axis_h1.png"
            ),
            "raw_autoaxis_figure": str(output_dir / "figures" / "figure6_ale_paper_formula_raw_autoaxis_h1.png"),
            "note": str(output_dir / "FIGURE6_PAPER_FORMULA_NOTE.md"),
        },
    }
    atomic_write_json(provenance, output_dir / "run_provenance.json")
    print(f"Wrote {table_path}")
    print(f"Wrote {output_dir / 'figures' / 'figure6_ale_paper_formula_display_h1.png'}")
    print(f"Wrote {output_dir / 'figures' / 'figure6_ale_paper_formula_raw_autoaxis_h1.png'}")


if __name__ == "__main__":
    main()
