"""Integrate the final h=1/h=5 mainline with the transferred h=22 supplement.

This script does not rerun any forecasting model. It reads the audited final
h=1/h=5 output and the audited transferred h=22 package, normalizes h=22 NN
single-best labels to the host/mainline convention, and writes a separate
combined reporting directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

H1H5_DIR = ROOT / "outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523"
H22_PACKAGE_DIR = ROOT / "h22_all_models_with_code_and_word_FINAL_20260524"
H22_DIR = H22_PACKAGE_DIR / "outputs_final_h22_all_models_with_gb_nn_single_best_20260524"
H22_WORD = H22_PACKAGE_DIR / "h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx"

OUT_DIR = ROOT / "outputs_final_core_h1h5_h22_integrated_20260524"

NN_SINGLE_BEST_LABEL_MAP = {
    "NN1_single_best": "NN1_1",
    "NN2_single_best": "NN2_1",
    "NN3_single_best": "NN3_1",
    "NN4_single_best": "NN4_1",
}

EXPECTED_MODELS = [
    "AdaptiveLasso",
    "Bagging",
    "ElasticNet",
    "GradientBoosting",
    "HAR",
    "HARQ",
    "HARX",
    "Lasso",
    "LevHAR",
    "LogHAR",
    "NN1",
    "NN1_1",
    "NN2",
    "NN2_1",
    "NN3",
    "NN3_1",
    "NN4",
    "NN4_1",
    "PostLasso",
    "RandomForest",
    "Ridge",
    "SHAR",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def normalize_model_labels_series(s: pd.Series) -> pd.Series:
    return s.replace(NN_SINGLE_BEST_LABEL_MAP)


def normalize_pairwise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename: dict[str, str] = {}
    for old, new in NN_SINGLE_BEST_LABEL_MAP.items():
        if old in out.columns:
            rename[old] = new
        suffix = "_dm_reject_10pct_share"
        old_share = old + suffix
        new_share = new + suffix
        if old_share in out.columns:
            rename[old_share] = new_share
    out = out.rename(columns=rename)
    if "benchmark_row" in out.columns:
        out["benchmark_row"] = normalize_model_labels_series(out["benchmark_row"])
    return out


def normalize_table(df: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    out = df.copy()
    if table_name == "pairwise_relative_mse_matrix":
        return normalize_pairwise_columns(out)
    for col in ["model", "row_model", "col_model", "benchmark_row"]:
        if col in out.columns:
            out[col] = normalize_model_labels_series(out[col])
    return out


def load_and_combine_predictions() -> pd.DataFrame:
    h1h5 = read_csv(H1H5_DIR / "predictions" / "model_predictions.csv", parse_dates=["date"])
    h22 = read_csv(H22_DIR / "predictions" / "model_predictions.csv", parse_dates=["date"])
    h22 = h22.copy()
    h22["model"] = normalize_model_labels_series(h22["model"])
    combined = pd.concat([h1h5, h22], ignore_index=True)
    combined = combined.sort_values(["dataset", "horizon", "ticker", "date", "model"]).reset_index(drop=True)
    return combined


def load_and_combine_table(rel_path: str, table_name: str) -> pd.DataFrame:
    h1h5 = read_csv(H1H5_DIR / rel_path)
    h22 = normalize_table(read_csv(H22_DIR / rel_path), table_name=table_name)
    combined = pd.concat([h1h5, h22], ignore_index=True)
    sort_cols = [c for c in ["dataset", "horizon", "ticker", "model", "benchmark_row", "row_model", "col_model"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


def audit_outputs(combined_predictions: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict:
    key = ["date", "ticker", "dataset", "horizon", "model"]
    duplicate_keys = int(combined_predictions.duplicated(key).sum())
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Integrated reporting output for h=1, h=5, and h=22. Forecasts are copied from audited source outputs; no model is rerun.",
        "output_dir": str(OUT_DIR),
        "source_h1h5_dir": str(H1H5_DIR),
        "source_h22_dir": str(H22_DIR),
        "source_h22_word": str(H22_WORD),
        "label_mapping_applied_to_h22": NN_SINGLE_BEST_LABEL_MAP,
        "rows": int(len(combined_predictions)),
        "datasets": sorted(combined_predictions["dataset"].astype(str).unique()),
        "horizons": sorted(int(x) for x in combined_predictions["horizon"].unique()),
        "tickers": int(combined_predictions["ticker"].nunique()),
        "model_count": int(combined_predictions["model"].nunique()),
        "models": sorted(combined_predictions["model"].astype(str).unique()),
        "duplicate_keys": duplicate_keys,
        "missing_forecasts": int(combined_predictions["forecast_rv"].isna().sum()),
        "zero_forecasts": int((combined_predictions["forecast_rv"] == 0).sum()),
        "rows_by_horizon": {str(k): int(v) for k, v in combined_predictions["horizon"].value_counts().sort_index().items()},
        "rows_by_horizon_model": {
            f"h{int(h)}::{m}": int(n)
            for (h, m), n in combined_predictions.groupby(["horizon", "model"]).size().sort_index().items()
        },
        "table_rows": {name: int(len(df)) for name, df in tables.items()},
        "h22_source_hashes": {
            "predictions/model_predictions.csv": sha256(H22_DIR / "predictions" / "model_predictions.csv"),
            "audit_report.json": sha256(H22_DIR / "audit_report.json"),
            "h22_word": sha256(H22_WORD),
        },
    }
    expected_horizons = [1, 5, 22]
    problems = []
    if audit["horizons"] != expected_horizons:
        problems.append(f"Unexpected horizons: {audit['horizons']}")
    if audit["datasets"] != ["MHAR", "PARTIAL_MALL"]:
        problems.append(f"Unexpected datasets: {audit['datasets']}")
    if audit["model_count"] != len(EXPECTED_MODELS) or audit["models"] != sorted(EXPECTED_MODELS):
        problems.append("Combined model set does not match expected 22-model reporting set.")
    if duplicate_keys != 0:
        problems.append(f"Duplicate keys found: {duplicate_keys}")
    if audit["missing_forecasts"] != 0:
        problems.append(f"Missing forecasts found: {audit['missing_forecasts']}")
    if audit["zero_forecasts"] != 0:
        problems.append(f"Zero forecasts found: {audit['zero_forecasts']}")
    audit["problems"] = problems
    audit["passed"] = not problems
    return audit


def write_receipt(audit: dict) -> None:
    lines = [
        "# H1/H5/H22 Integrated Results Receipt",
        "",
        f"Created at UTC: `{audit['created_at_utc']}`",
        "",
        "## Scope",
        "",
        "This directory integrates the audited h=1/h=5 mainline and the transferred audited h=22 supplement for reporting. No forecasting model was rerun.",
        "",
        "## Sources",
        "",
        f"- h=1/h=5 mainline: `{H1H5_DIR.relative_to(ROOT)}/`",
        f"- h=22 transferred supplement: `{H22_DIR.relative_to(ROOT)}/`",
        f"- h=22 Word document copied from: `{H22_WORD.relative_to(ROOT)}`",
        "",
        "## Label Normalization",
        "",
        "For cross-horizon reporting consistency only, h=22 NN single-best labels were mapped as follows:",
        "",
    ]
    for old, new in NN_SINGLE_BEST_LABEL_MAP.items():
        lines.append(f"- `{old}` -> `{new}`")
    lines.extend(
        [
            "",
            "The transferred h=22 package itself is preserved unchanged.",
            "",
            "## Audit",
            "",
            f"- rows: `{audit['rows']}`",
            f"- datasets: `{audit['datasets']}`",
            f"- horizons: `{audit['horizons']}`",
            f"- tickers: `{audit['tickers']}`",
            f"- model_count: `{audit['model_count']}`",
            f"- duplicate_keys: `{audit['duplicate_keys']}`",
            f"- missing_forecasts: `{audit['missing_forecasts']}`",
            f"- zero_forecasts: `{audit['zero_forecasts']}`",
            f"- passed: `{audit['passed']}`",
            "",
            "## Table Outputs",
            "",
        ]
    )
    for name, rows in audit["table_rows"].items():
        lines.append(f"- `{name}`: `{rows}` rows")
    if audit["problems"]:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {p}" for p in audit["problems"])
    (OUT_DIR / "HOST_IMPORT_RECEIPT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "predictions").mkdir(exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "docs").mkdir(exist_ok=True)

    predictions = load_and_combine_predictions()
    predictions.to_csv(OUT_DIR / "predictions" / "model_predictions.csv", index=False)

    table_specs = {
        "forecast_summary_cross_section": "tables/forecast_summary_cross_section.csv",
        "forecast_metrics_by_asset": "tables/forecast_metrics_by_asset.csv",
        "pairwise_relative_mse_matrix": "tables/pairwise_relative_mse_matrix.csv",
        "diebold_mariano_tests": "tables/diebold_mariano_tests.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, rel in table_specs.items():
        tables[name] = load_and_combine_table(rel, name)
        tables[name].to_csv(OUT_DIR / rel, index=False)

    seed_selection = read_csv(H22_DIR / "tables" / "nn_single_best_seed_selection.csv")
    seed_selection.to_csv(OUT_DIR / "tables" / "h22_nn_single_best_seed_selection.csv", index=False)

    shutil.copy2(H22_WORD, OUT_DIR / "docs" / H22_WORD.name)
    shutil.copy2(H22_DIR / "audit_report.json", OUT_DIR / "docs" / "h22_audit_report.json")
    shutil.copy2(H22_DIR / "run_provenance.json", OUT_DIR / "docs" / "h22_run_provenance.json")

    audit = audit_outputs(predictions, tables | {"h22_nn_single_best_seed_selection": seed_selection})
    (OUT_DIR / "integration_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_receipt(audit)
    if not audit["passed"]:
        raise SystemExit("Integrated audit failed; inspect integration_audit.json")
    print(OUT_DIR)
    print(json.dumps({k: audit[k] for k in ["rows", "datasets", "horizons", "model_count", "duplicate_keys", "missing_forecasts", "zero_forecasts", "passed"]}, indent=2))


if __name__ == "__main__":
    main()
