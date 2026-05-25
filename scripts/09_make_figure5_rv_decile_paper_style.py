#!/usr/bin/env python3
"""Create a paper-style Figure 5 RV-decile forecast accuracy plot.

The script uses the checked RV-decile MSE table produced by
scripts/06b_compute_rv_decile_mse.py and selects the same compact set of
representative models and realized-variance deciles used in the paper figure.
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

DEFAULT_TABLE = (
    ROOT
    / "outputs_rv_decile_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523"
    / "tables"
    / "rv_decile_mse.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs_figure5_rv_decile_paper_style_20260523"

MODEL_ORDER = ["HARX", "LogHAR", "ElasticNet", "RandomForest", "NN1"]
MODEL_LABELS = {
    "HARX": "HAR-X",
    "LogHAR": "LogHAR",
    "ElasticNet": "EN",
    "RandomForest": "RF",
    "NN1": r"NN$^{1}_{1}$",
}
DECILES = [1, 2, 6, 9, 10]
DECILE_LABELS = {
    1: "p in (0.0,0.1)",
    2: "p in (0.1,0.2)",
    6: "p in (0.5,0.6)",
    9: "p in (0.8,0.9)",
    10: "p in (0.9,1.0)",
}
DECILE_STYLES = {
    1: {"marker": "^", "edgecolors": "purple", "facecolors": "none", "linewidths": 2.6, "s": 135},
    2: {"marker": "+", "c": "#00e600", "linewidths": 2.6, "s": 145},
    6: {"marker": "o", "edgecolors": "#f2e600", "facecolors": "none", "linewidths": 2.4, "s": 150},
    9: {"marker": r"$\ast$", "c": "red", "linewidths": 1.6, "s": 220},
    10: {"marker": "x", "c": "blue", "linewidths": 2.4, "s": 120},
}


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style RV-decile Figure 5.")
    parser.add_argument("--rv-decile-table", default=str(DEFAULT_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def load_plot_data(path: Path, dataset: str, horizon: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"RV-decile table not found: {path}")
    df = pd.read_csv(path)
    required = {
        "dataset",
        "horizon",
        "model",
        "decile",
        "percentile_bin",
        "n_obs",
        "mse",
        "har_mse",
        "rel_mse_vs_har",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["horizon"] = df["horizon"].astype(int)
    df["decile"] = df["decile"].astype(int)
    sub = df[
        (df["dataset"] == dataset)
        & (df["horizon"] == int(horizon))
        & (df["model"].isin(MODEL_ORDER))
        & (df["decile"].isin(DECILES))
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset}, horizon={horizon}.")

    expected = {(m, d) for m in MODEL_ORDER for d in DECILES}
    observed = set(zip(sub["model"], sub["decile"]))
    missing_cells = sorted(expected.difference(observed))
    if missing_cells:
        raise ValueError(f"Missing model-decile cells: {missing_cells[:10]}")

    sub["model_label"] = sub["model"].map(MODEL_LABELS)
    sub["decile_label"] = sub["decile"].map(DECILE_LABELS)
    return sub.sort_values(["model", "decile"]).reset_index(drop=True)


def draw_figure(plot_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    x_positions = {model: i for i, model in enumerate(MODEL_ORDER)}

    for decile in DECILES:
        sub = plot_df[plot_df["decile"] == decile].set_index("model").loc[MODEL_ORDER].reset_index()
        x = [x_positions[m] for m in sub["model"]]
        y = sub["rel_mse_vs_har"].astype(float)
        ax.scatter(x, y, label=DECILE_LABELS[decile], **DECILE_STYLES[decile])

    ax.set_xlim(-0.8, len(MODEL_ORDER) - 0.2)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylabel("relative mse", fontsize=12)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.7)
    ax.tick_params(axis="both", direction="in", top=True, right=True, pad=6)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#444444",
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_path = resolve(args.rv_decile_table)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.rglob("*")) and not args.allow_existing_output_dir:
        raise SystemExit(
            f"Output directory already exists and contains files: {output_dir}. "
            "Use --allow-existing-output-dir to update it intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = load_plot_data(table_path, args.dataset, args.horizon)
    data_path = output_dir / "figure5_rv_decile_paper_style_h1.csv"
    figure_path = output_dir / "figure5_rv_decile_paper_style_h1.png"
    note_path = output_dir / "FIGURE5_NOTE.md"
    provenance_path = output_dir / "run_provenance.json"

    plot_df.to_csv(data_path, index=False)
    draw_figure(plot_df, figure_path)

    note = (
        "Note: This figure reports one-day-ahead out-of-sample forecast MSE relative "
        "to the HAR model for HAR-X, LogHAR, EN, RF, and NN2^10. The test-set "
        "observations are split into deciles according to observed realized variance. "
        "Only deciles (0.0,0.1), (0.1,0.2), (0.5,0.6), (0.8,0.9), and (0.9,1.0) "
        "are displayed, following the paper-style diagnostic figure. The dataset is "
        f"{args.dataset}, which is the IV-omitted reproduction counterpart of MALL.\n"
    )
    note_path.write_text(note, encoding="utf-8")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rv_decile_table": str(table_path),
        "dataset": args.dataset,
        "horizon": int(args.horizon),
        "models": MODEL_ORDER,
        "deciles": DECILES,
        "outputs": {
            "figure": str(figure_path),
            "plot_data": str(data_path),
            "note": str(note_path),
        },
        "n_rows": int(len(plot_df)),
        "n_obs_min": int(plot_df["n_obs"].min()),
        "n_obs_max": int(plot_df["n_obs"].max()),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {figure_path}")
    print(f"Wrote {data_path}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
