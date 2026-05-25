#!/usr/bin/env python3
"""Create a paper-style partial Figure 6 ALE plot from existing ALE output.

The original Figure 6 has panels for RVD, RVW, IV, and M1W for Apple. The
current strict reproduction uses PARTIAL_MALL with IV omitted, so this script
only plots the panels supported by the fitted project outputs: RVD, RVW, and
M1W.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rv1rep_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALE_TABLE = (
    ROOT
    / "outputs_ale_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523"
    / "tables"
    / "ale_table.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs_figure6_ale_paper_style_partial_20260523"

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
    "LogHAR": {"color": "red", "linestyle": "--", "linewidth": 1.5},
    "ElasticNet": {"color": "blue", "linestyle": ":", "linewidth": 1.6},
    "RandomForest": {"color": "purple", "linestyle": "--", "linewidth": 1.5},
    "NN10_2": {"color": "#00cc00", "linestyle": ":", "linewidth": 1.6},
}
FEATURE_ORDER = ["rvd", "rvw", "m1w"]
FEATURE_TITLES = {
    "rvd": "(A) RVD",
    "rvw": "(B) RVW",
    "m1w": "(C) M1W",
}
ALE_SCALE = 1e4


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style partial Figure 6 ALE plot.")
    parser.add_argument("--ale-table", default=str(DEFAULT_ALE_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def load_plot_data(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ALE table not found: {path}")
    df = pd.read_csv(path)
    required = {
        "ticker",
        "dataset",
        "horizon",
        "model",
        "feature",
        "x_standardized",
        "ale",
        "grid_size",
        "n_in_sample",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df["horizon"] = df["horizon"].astype(int)
    sub = df[
        (df["ticker"] == args.ticker)
        & (df["dataset"] == args.dataset)
        & (df["horizon"] == int(args.horizon))
        & (df["model"].isin(MODEL_ORDER))
        & (df["feature"].isin(FEATURE_ORDER))
        & (df["x_standardized"].between(float(args.x_min), float(args.x_max)))
    ].copy()
    if sub.empty:
        raise ValueError("No ALE rows remain after filtering.")

    expected = {(feature, model) for feature in FEATURE_ORDER for model in MODEL_ORDER}
    observed = set(zip(sub["feature"], sub["model"]))
    missing_cells = sorted(expected.difference(observed))
    if missing_cells:
        raise ValueError(f"Missing feature-model ALE cells: {missing_cells}")

    sub["ale_scaled"] = sub["ale"].astype(float) * ALE_SCALE
    sub["model_label"] = sub["model"].map(MODEL_LABELS)
    return sub.sort_values(["feature", "model", "x_standardized"]).reset_index(drop=True)


def draw_figure(plot_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)

    for ax, feature in zip(axes, FEATURE_ORDER):
        for model in MODEL_ORDER:
            sub = plot_df[(plot_df["feature"] == feature) & (plot_df["model"] == model)]
            ax.plot(
                sub["x_standardized"],
                sub["ale_scaled"],
                label=MODEL_LABELS[model],
                **MODEL_STYLES[model],
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(feature)
        ax.set_title(FEATURE_TITLES[feature], loc="left", fontsize=12, fontweight="bold")
        ax.grid(True, color="#e0e0e0", linewidth=0.7, alpha=0.9)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$f^{ALE}$ (RV $\times 10^4$)")
    axes[0].legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#666666", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_note(output_path: Path, args: argparse.Namespace, plot_df: pd.DataFrame) -> None:
    note = f"""Figure 6 ALE between explanatory variables and future volatility.

Notes: This figure plots ALE curves for Apple's stock price volatility using the
current reproduction outputs. The ALE curves are computed for h=1, ticker
{args.ticker}, dataset {args.dataset}, and the representative models HAR-X,
LogHAR, EN, RF, and NN2^10. The covariates are standardized, and the x-axis is
restricted to [-1, 1] standard deviations to match the paper-style display. The
y-axis reports ALE in realized-variance units scaled by 10^4 for readability.
The IV panel in the original paper is not reproduced because the strict
PARTIAL_MALL reproduction omits IV.

Rows plotted: {len(plot_df)}
Features plotted: {', '.join(FEATURE_ORDER)}
"""
    output_path.write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    table_path = resolve(args.ale_table)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.rglob("*")) and not args.allow_existing_output_dir:
        raise SystemExit(
            f"Output directory already exists and contains files: {output_dir}. "
            "Use --allow-existing-output-dir to update it intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = load_plot_data(table_path, args)
    figure_path = output_dir / "figure6_ale_paper_style_partial_h1.png"
    data_path = output_dir / "figure6_ale_paper_style_partial_h1.csv"
    note_path = output_dir / "FIGURE6_PARTIAL_NOTE.md"
    provenance_path = output_dir / "run_provenance.json"

    plot_df.to_csv(data_path, index=False)
    draw_figure(plot_df, figure_path)
    write_note(note_path, args, plot_df)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ale_table": str(table_path),
        "output_dir": str(output_dir),
        "ticker": args.ticker,
        "dataset": args.dataset,
        "horizon": int(args.horizon),
        "models": MODEL_ORDER,
        "features_plotted": FEATURE_ORDER,
        "features_not_plotted": ["iv"],
        "reason_iv_not_plotted": "IV is omitted from the current PARTIAL_MALL reproduction and absent from the ALE table.",
        "x_range": [float(args.x_min), float(args.x_max)],
        "ale_scale": ALE_SCALE,
        "n_rows": int(len(plot_df)),
        "outputs": {
            "figure": str(figure_path),
            "plot_data": str(data_path),
            "note": str(note_path),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {figure_path}")
    print(f"Wrote {data_path}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
