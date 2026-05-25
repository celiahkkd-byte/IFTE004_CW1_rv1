#!/usr/bin/env python3
"""Draw Figure 6-style ALE curves with paper-like y-axis scale and ticks.

This is a plotting-only utility. It reads an existing ALE table and does not
refit any model.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rv1rep_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALE_TABLE = (
    ROOT
    / "outputs_ale_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523"
    / "tables"
    / "ale_table.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs_figure6_ale_paper_scale_20260524"

MODEL_ORDER = ["HARX", "LogHAR", "ElasticNet", "RandomForest", "NN1"]
MODEL_LABELS = {
    "HARX": "HAR-X",
    "LogHAR": "LogHAR",
    "ElasticNet": "EN",
    "RandomForest": "RF",
    "NN1": r"NN$^{1}_{1}$",
}
MODEL_STYLES = {
    "HARX": {"color": "black", "linestyle": "-", "linewidth": 1.6},
    "LogHAR": {"color": "red", "linestyle": (0, (4, 3)), "linewidth": 1.5},
    "ElasticNet": {"color": "blue", "linestyle": ":", "linewidth": 1.7},
    "RandomForest": {"color": "purple", "linestyle": "--", "linewidth": 1.5},
    "NN1": {"color": "#00cc00", "linestyle": (0, (1, 1)), "linewidth": 1.7},
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
FEATURE_YLIMS = {
    "rvd": (-0.5, 0.5),
    "rvw": (-0.5, 0.5),
    "m1w": (-0.1, 0.1),
}
FEATURE_YTICKS = {
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
    parser = argparse.ArgumentParser(description="Create paper-scale Figure 6 ALE plot from an existing ALE table.")
    parser.add_argument("--ale-table", default=str(DEFAULT_ALE_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--features", nargs="+", default=FEATURE_ORDER)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--ale-scale", type=float, default=1000.0)
    parser.add_argument("--y-axis-mode", choices=["fixed", "auto"], default="fixed")
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def load_plot_data(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ALE table not found: {path}")
    df = pd.read_csv(path)
    required = {"ticker", "dataset", "horizon", "model", "feature", "x_standardized", "ale"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df["horizon"] = df["horizon"].astype(int)
    sub = df[
        (df["ticker"].astype(str).str.upper() == args.ticker.upper())
        & (df["dataset"] == args.dataset)
        & (df["horizon"] == int(args.horizon))
        & (df["model"].isin(args.models))
        & (df["feature"].isin(args.features))
        & (df["x_standardized"].between(float(args.x_min), float(args.x_max)))
    ].copy()
    if sub.empty:
        raise ValueError("No ALE rows remain after filtering.")

    expected = {(feature, model) for feature in args.features for model in args.models}
    observed = set(zip(sub["feature"], sub["model"]))
    missing_cells = sorted(expected.difference(observed))
    if missing_cells:
        raise ValueError(f"Missing feature-model ALE cells: {missing_cells}")

    sub["ale_paper_scale"] = sub["ale"].astype(float) * float(args.ale_scale)
    return sub.sort_values(["feature", "model", "x_standardized"]).reset_index(drop=True)


def draw(plot_df: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, len(args.features), figsize=(12.0, 3.8), sharex=True)
    if len(args.features) == 1:
        axes = [axes]

    axis_rows = []
    for ax, feature in zip(axes, args.features):
        for model in args.models:
            sub = plot_df[(plot_df["feature"] == feature) & (plot_df["model"] == model)]
            ax.plot(
                sub["x_standardized"],
                sub["ale_paper_scale"],
                label=MODEL_LABELS.get(model, model),
                **MODEL_STYLES.get(model, {"linewidth": 1.5}),
            )
        vals = plot_df.loc[plot_df["feature"] == feature, "ale_paper_scale"].astype(float)
        if args.y_axis_mode == "auto":
            bound = max(abs(float(vals.min())), abs(float(vals.max())))
            bound = bound * 1.15 if bound > 0 else 1.0
            ymin, ymax = -bound, bound
            yticks = np.linspace(ymin, ymax, 5)
        else:
            ymin, ymax = FEATURE_YLIMS.get(feature, (-0.5, 0.5))
            yticks = FEATURE_YTICKS.get(feature, np.linspace(ymin, ymax, 5))
        ax.set_xlim(float(args.x_min), float(args.x_max))
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(float(args.x_min), float(args.x_max) + 0.001, 0.5))
        ax.set_yticks(yticks)
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel(FEATURE_YLABELS.get(feature, r"$f^{ALE}$"))
        ax.set_title(FEATURE_TITLES.get(feature, feature), loc="left", fontsize=12, fontweight="bold")
        ax.grid(True, color="#e0e0e0", linewidth=0.7, alpha=0.9)
        ax.tick_params(axis="both", direction="in", top=True, right=True)

        axis_rows.append(
            {
                "feature": feature,
                "y_min": float(ymin),
                "y_max": float(ymax),
                "tick_step": float(yticks[1] - yticks[0]) if len(yticks) > 1 else np.nan,
                "data_min": float(vals.min()),
                "data_max": float(vals.max()),
                "points_outside_axis": int(((vals < ymin) | (vals > ymax)).sum()),
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


def write_note(path: Path, args: argparse.Namespace, axis_df: pd.DataFrame) -> None:
    note = f"""Figure 6 ALE between explanatory variables and future volatility.

Notes: This plotting-only reproduction uses the existing ALE table at
{resolve(args.ale_table)}. No model is refit. The figure reports h={int(args.horizon)}
for ticker {args.ticker} and dataset {args.dataset}. ALE values are in realized-
variance units and are multiplied by {float(args.ale_scale):g} for display. The
y-axis mode is `{args.y_axis_mode}`. In fixed mode, the tick spacing follows the
paper-style scale: 0.25 for RVD/RVW and 0.05 for M1W. The x-axis is restricted to
[{float(args.x_min):g}, {float(args.x_max):g}] standardized units.

Axis diagnostics:
{axis_df.to_string(index=False)}
"""
    path.write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.rglob("*")) and not args.allow_existing_output_dir:
        raise SystemExit(
            f"Output directory already exists and contains files: {output_dir}. "
            "Use --allow-existing-output-dir to update it intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = resolve(args.ale_table)
    plot_df = load_plot_data(table_path, args)
    figure_path = output_dir / "figure6_ale_paper_scale_h1.png"
    data_path = output_dir / "figure6_ale_paper_scale_h1.csv"
    axis_path = output_dir / "figure6_axis_diagnostics.csv"
    note_path = output_dir / "FIGURE6_PAPER_SCALE_NOTE.md"
    provenance_path = output_dir / "run_provenance.json"

    axis_df = draw(plot_df, figure_path, args)
    plot_df.to_csv(data_path, index=False)
    axis_df.to_csv(axis_path, index=False)
    write_note(note_path, args, axis_df)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ale_table": str(table_path),
        "output_dir": str(output_dir),
        "ticker": args.ticker,
        "dataset": args.dataset,
        "horizon": int(args.horizon),
        "models": args.models,
        "features": args.features,
        "ale_scale": float(args.ale_scale),
        "y_axis_mode": args.y_axis_mode,
        "x_range": [float(args.x_min), float(args.x_max)],
        "outputs": {
            "figure": str(figure_path),
            "plot_data": str(data_path),
            "axis_diagnostics": str(axis_path),
            "note": str(note_path),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {figure_path}")
    print(f"Wrote {data_path}")
    print(f"Wrote {axis_path}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
