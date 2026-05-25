#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rv1rep.evaluation import cross_sectional_summary, forecast_metrics, pairwise_relative_mse

KEY_COLS = ["date", "ticker", "dataset", "horizon", "model"]
SORT_COLS = ["dataset", "horizon", "model", "ticker", "date"]
DATASETS_DEFAULT = ["MHAR", "PARTIAL_MALL"]


def _load_word_module():
    path = ROOT / "scripts" / "09_make_pairwise_dm_word_tables.py"
    spec = importlib.util.spec_from_file_location("_rv1rep_word_tables", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Word-table helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORD = _load_word_module()


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


def _load_predictions(output_dir: Path) -> pd.DataFrame:
    paths = [
        output_dir / "predictions" / "nonnn_model_predictions.csv",
        output_dir / "predictions" / "nn_model_predictions.csv",
    ]
    parts = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction input: {path}")
        parts.append(pd.read_csv(path, parse_dates=["date"]))
    final = pd.concat(parts, ignore_index=True).sort_values(SORT_COLS).reset_index(drop=True)
    duplicates = int(final.duplicated(KEY_COLS).sum())
    if duplicates:
        raise RuntimeError(f"Duplicate prediction keys after combining: {duplicates}")
    return final


def _write_word_for_horizon(matrix: pd.DataFrame, dm: pd.DataFrame, output_dir: Path, datasets: list[str], horizon: int) -> None:
    doc = Document()
    WORD.set_doc_defaults(doc)
    audit_parts = []
    first = True
    table_number = 2 if int(horizon) == 1 else 3
    for dataset in datasets:
        if not first:
            doc.add_page_break()
        first = False
        WORD.add_caption(doc, table_number, dataset, int(horizon))
        audit_parts.append(WORD.build_table(doc, matrix, dm, dataset, int(horizon)))
        WORD.add_note(doc)
        table_number += 1
    output_path = output_dir / f"h{int(horizon)}_table2_style_relative_mse_dm.docx"
    doc.save(output_path)
    audit = pd.concat(audit_parts, ignore_index=True)
    _atomic_write_csv(audit, output_dir / f"h{int(horizon)}_table2_style_relative_mse_dm.formatting_audit.csv")


def _summary_text(summary: pd.DataFrame, horizons: list[int], datasets: list[str]) -> str:
    lines = [
        "# Corrected Paper-Style Rerun Summary",
        "",
        f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The table below reports cross-sectional average relative MSE versus the HAR benchmark within the same corrected rerun.",
        "",
    ]
    for horizon in horizons:
        for dataset in datasets:
            sub = summary[(summary["horizon"].astype(int) == int(horizon)) & (summary["dataset"] == dataset)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("avg_rel_mse_vs_har")
            lines.append(f"## {dataset}, h={horizon}")
            lines.append("")
            lines.append("| Model | Avg relative MSE vs HAR | Median relative MSE vs HAR | Assets |")
            lines.append("|---|---:|---:|---:|")
            for _, row in sub.iterrows():
                lines.append(
                    f"| {row['model']} | {float(row['avg_rel_mse_vs_har']):.4f} | "
                    f"{float(row['median_rel_mse_vs_har']):.4f} | {int(row['n_assets'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corrected paper-style evaluation tables.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=DATASETS_DEFAULT)
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--skip-word", action="store_true", help="Skip Word table generation, useful for smoke tests with partial model sets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    final = _load_predictions(output_dir)
    final = final[
        final["dataset"].isin(args.datasets)
        & final["horizon"].astype(int).isin([int(h) for h in args.horizons])
    ].copy()
    if final.empty:
        raise SystemExit("No predictions remain after dataset/horizon filter.")
    _atomic_write_csv(final, output_dir / "predictions" / "model_predictions.csv")

    metrics = forecast_metrics(final)
    summary = cross_sectional_summary(metrics)
    rel_matrix, dm = pairwise_relative_mse(final)
    _atomic_write_csv(metrics, evaluation_dir / "model_mse_by_ticker.csv")
    _atomic_write_csv(summary, evaluation_dir / "model_mse_summary.csv")
    _atomic_write_csv(rel_matrix, evaluation_dir / "pairwise_relative_mse_matrix.csv")
    _atomic_write_csv(dm, evaluation_dir / "diebold_mariano_tests.csv")

    for horizon in args.horizons:
        h = int(horizon)
        _atomic_write_csv(rel_matrix[rel_matrix["horizon"].astype(int) == h], evaluation_dir / f"h{h}_pairwise_relative_mse_matrix.csv")
        _atomic_write_csv(dm[dm["horizon"].astype(int) == h], evaluation_dir / f"h{h}_diebold_mariano_tests.csv")

    _atomic_write_text(_summary_text(summary, [int(h) for h in args.horizons], args.datasets), evaluation_dir / "summary.md")

    if not args.skip_word:
        for horizon in args.horizons:
            _write_word_for_horizon(rel_matrix, dm, evaluation_dir, args.datasets, int(horizon))

    _atomic_write_json(
        {
            "script": Path(__file__).name,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "datasets": args.datasets,
            "horizons": [int(h) for h in args.horizons],
            "rows": int(len(final)),
            "models": sorted(final["model"].astype(str).unique()),
        },
        output_dir / "run_provenance_evaluation.json",
    )
    print(f"Wrote corrected evaluation outputs to {evaluation_dir}")


if __name__ == "__main__":
    main()
