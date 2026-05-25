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
DEFAULT_INPUT = (
    ROOT
    / "outputs_variable_importance_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523"
    / "tables"
    / "variable_importance.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs_figure7_variable_importance_paper_style_h1_20260524"

MODEL_ORDER = ["HARX", "ElasticNet", "RandomForest", "NN10_2"]
MODEL_LABELS = {
    "HARX": "HAR-X",
    "ElasticNet": "EN",
    "RandomForest": "RF",
    "NN10_2": r"$\mathrm{NN}^{10}_{2}$",
}
PANEL_LABELS = ["A", "B", "C", "D"]
DATASET_LABELS = {
    "MHAR": r"$\mathcal{M}_{HAR}$",
    "MALL": r"$\mathcal{M}_{ALL}$",
    "PARTIAL_MALL": r"$\mathcal{M}_{PARTIAL\_MALL}$ (IV omitted)",
}
FEATURE_LABELS = {
    "rvd": "rvd",
    "rvw": "rvw",
    "rvm": "rvm",
    "vix": "vix",
    "m1w": "m1w",
    "us3m_diff": "us3m",
    "hsi": "hsi",
    "ads": "ads",
    "ea": "ea",
    "dvol": "dvol",
    "epu": "epu",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _assert_fresh_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and not allow_existing:
        files = [p for p in output_dir.rglob("*") if p.is_file()]
        if files:
            raise SystemExit(
                f"Output directory already contains files: {output_dir}. "
                "Use a fresh directory or pass --allow-existing-output-dir."
            )


def _load_vi(path: Path, dataset: str, horizon: int, models: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"dataset", "horizon", "model", "feature", "vi_mean", "n_tickers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in VI table: {sorted(missing)}")
    sub = df[
        (df["dataset"].astype(str) == dataset)
        & (df["horizon"].astype(int) == int(horizon))
        & (df["model"].astype(str).isin(models))
    ].copy()
    if sub.empty:
        raise ValueError(f"No VI rows for dataset={dataset}, horizon={horizon}, models={models}")
    got_models = set(sub["model"].astype(str))
    missing_models = set(models) - got_models
    if missing_models:
        raise ValueError(f"Missing requested models in VI table: {sorted(missing_models)}")
    sums = sub.groupby("model")["vi_mean"].sum()
    bad_sums = sums[(sums < 0.95) | (sums > 1.05)]
    if not bad_sums.empty:
        raise ValueError(f"Mean VI should sum to about one by model; got {bad_sums.to_dict()}")
    return sub


def _plot_paper_style(df: pd.DataFrame, output_path: Path, x_max: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.8), sharex=True)
    axes_flat = axes.ravel()
    bar_color = "#0072B2"

    for panel, ax, model in zip(PANEL_LABELS, axes_flat, MODEL_ORDER):
        sub = df[df["model"].astype(str) == model].sort_values("vi_mean", ascending=True).copy()
        sub["feature_label"] = sub["feature"].map(FEATURE_LABELS).fillna(sub["feature"])
        ax.barh(sub["feature_label"], sub["vi_mean"], color=bar_color, edgecolor="black", linewidth=0.6)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("variable importance")
        ax.set_ylabel("covariate")
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12, pad=8)
        ax.text(
            -0.18,
            1.04,
            panel,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    fig.tight_layout(h_pad=2.6, w_pad=2.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=240, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(output_path)


def _ea_summary(df: pd.DataFrame) -> dict:
    ea = df[df["feature"].astype(str).str.lower().eq("ea")]
    if ea.empty:
        return {"present": False, "all_zero": None, "by_model": {}}
    by_model = {str(row["model"]): float(row["vi_mean"]) for _, row in ea.iterrows()}
    return {
        "present": True,
        "all_zero": all(value == 0.0 for value in by_model.values()),
        "by_model": by_model,
    }


def _write_note(
    output_dir: Path,
    dataset: str,
    horizon: int,
    source: Path,
    figure_name: str,
    n_tickers: int,
    ea_info: dict,
) -> None:
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    ea_warning = ""
    if ea_info.get("all_zero") is True:
        ea_warning = (
            "\n\n**EA note:** In the input VI table, `ea` has zero VI for all plotted models. "
            "If that table was generated before the binary-feature ALE fix in "
            "`src/rv1rep/explain.py`, rerun `scripts/06c_compute_variable_importance.py` "
            "in a fresh output directory before using the final Figure 7. The plotting script "
            "does not modify VI values; it only visualizes the supplied table."
        )
    note = f"""# Figure 7. VI measure

**Title:** Figure 7. Variable-importance measure for one-day-ahead volatility forecasts.

**Notes:** This figure reports the ALE-based variable-importance (VI) measure for each feature in the {dataset_label} dataset for the HAR-X, EN, RF, and $\\mathrm{{NN}}^{{10}}_2$ models. The forecast horizon is h={horizon}. VI is computed as the standard deviation of the centered ALE function for each covariate and then normalized so that the VI values sum to one within each model and ticker. The bars report cross-sectional averages across {n_tickers} tickers and are sorted in descending order within each panel. The input table is `{source}`. Because the current replication dataset is {dataset_label}, IV is omitted from this figure; `us3m` denotes the transformed three-month Treasury bill variable stored in the panel as `us3m_diff`.{ea_warning}

Generated figure: `{figure_name}`
"""
    _atomic_write_text(note, output_dir / "FIGURE7_NOTE.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paper-style Figure 7 VI plot from an existing VI table.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-name", default=None)
    parser.add_argument("--dataset", default="PARTIAL_MALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _resolve(args.input)
    output_dir = _resolve(args.output_dir)
    _assert_fresh_output_dir(output_dir, args.allow_existing_output_dir)
    df = _load_vi(source, args.dataset, args.horizon, MODEL_ORDER)

    figure_name = args.figure_name or f"figure7_variable_importance_paper_style_h{int(args.horizon)}.png"
    figure_path = output_dir / figure_name
    _plot_paper_style(df, figure_path, x_max=float(args.x_max))

    sorted_table = (
        df.assign(feature_label=df["feature"].map(FEATURE_LABELS).fillna(df["feature"]))
        .sort_values(["model", "vi_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    table_path = output_dir / "figure7_variable_importance_plot_data.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_table.to_csv(table_path, index=False)

    n_tickers = int(df["n_tickers"].max())
    ea_info = _ea_summary(df)
    _write_note(output_dir, args.dataset, args.horizon, source, figure_path.name, n_tickers, ea_info)
    _atomic_write_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "output_dir": str(output_dir),
            "dataset": args.dataset,
            "horizon": int(args.horizon),
            "models": MODEL_ORDER,
            "n_tickers_max": n_tickers,
            "ea_vi_summary": ea_info,
            "outputs": {
                "figure": str(figure_path),
                "plot_data": str(table_path),
                "note": str(output_dir / "FIGURE7_NOTE.md"),
            },
        },
        output_dir / "run_provenance.json",
    )
    print(f"Wrote {figure_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {output_dir / 'FIGURE7_NOTE.md'}")


if __name__ == "__main__":
    main()
