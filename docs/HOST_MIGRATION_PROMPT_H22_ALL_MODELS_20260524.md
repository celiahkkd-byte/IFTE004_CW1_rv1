# Host Migration Prompt: Final h=22 All-Model Supplement With GB And NN Single-Best

Date prepared: 2026-05-24

This is the **current and authoritative** handoff note for the host/main machine. It supersedes earlier h=22 migration notes that referenced only 18 models or did not include the NN single-best seed columns.

## Use This Final Package

Use this exact final package:

```text
h22_all_models_with_code_and_word_FINAL_20260524.zip
```

It contains code, config, h=22 panel artifacts, final result tables, audit files, and the complete Word table document.

## Do Not Use These Superseded Artifacts For Reporting

The following earlier artifacts are valid historical intermediates, but they are **not** the final handoff target:

```text
outputs_final_h22_all_models_with_gb_20260524/
h22_relative_mse_dm_tables_20260524.docx
h22_all_models_with_gb_migration_package_20260524_220021.zip
h22_all_models_with_code_migration_package_20260524_223955.zip
```

Reason: those versions did not include the paper-style NN single-best seed columns `NN_i^1`. The final result below includes both `NN_i^1` and `NN_i^10`.

## Final Result Directory

The final audited h=22 result directory is:

```text
outputs_final_h22_all_models_with_gb_nn_single_best_20260524/
```

This directory contains the complete h=22 monthly horizon result with:

- all non-NN models,
- GradientBoosting h=22,
- NN top-10 ensemble columns, and
- NN single-best seed columns derived from existing seed checkpoints.

No model was retrained when adding the NN single-best columns. The single-best rows were derived from the saved NN50 seed checkpoints by selecting the minimum validation MSE seed for each `dataset × NN architecture × ticker`.

## Final Word Document

Use this Word document for the copy-ready paper-style tables:

```text
h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx
```

It contains two h=22 relative MSE / Diebold-Mariano tables:

- dataset `MHAR`
- dataset `PARTIAL_MALL`

The NN columns follow the paper-style layout:

```text
NN1^1, NN1^10, NN2^1, NN2^10, NN3^1, NN3^10, NN4^1, NN4^10
```

where:

- `NN_i^1` = `NNi_single_best`, selected by validation MSE among 50 saved seeds;
- `NN_i^10` = existing `NNi`, the top-10 validation ensemble from the NN50 run.

## Final Audit Values

The host-side audit should match:

```text
rows: 911,580
model_count: 22
datasets: MHAR, PARTIAL_MALL
horizons: 22 only
ticker_count: 25
duplicate_keys: 0
missing_forecasts: 0
zero_forecasts: 0
GradientBoosting rows: 41,436
NN single-best rows: 165,744
DM test rows: 24,200
forecast_metrics_by_asset rows: 1,100
forecast_summary_cross_section rows: 44
pairwise_relative_mse_matrix rows: 44
nn_single_best_seed_selection rows: 200
```

Expected model set:

```text
AdaptiveLasso
Bagging
ElasticNet
GradientBoosting
HAR
HARQ
HARX
Lasso
LevHAR
LogHAR
NN1
NN1_single_best
NN2
NN2_single_best
NN3
NN3_single_best
NN4
NN4_single_best
PostLasso
RandomForest
Ridge
SHAR
```

Note: `LevHAR` has slightly fewer rows than most models in the audited source. This was already present in the complete h=22 audit and should not be treated as corruption as long as duplicate keys and missing forecasts are both zero.

## Key File Hashes

Final all-model prediction file:

```text
outputs_final_h22_all_models_with_gb_nn_single_best_20260524/predictions/model_predictions.csv
SHA256: 2F2B69D5D4BFC0CF21277D44821466FA212ECE666F50BB78EA490FD14BECD5AC
```

Final audit report:

```text
outputs_final_h22_all_models_with_gb_nn_single_best_20260524/audit_report.json
SHA256: 8173B64D9E2227EFB8ECF355D307851B2D7E9CF04189056810C008466F46E850
```

Final Word table document:

```text
h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx
SHA256: 2EA67B0EF263649A947B9D3775AFD3B080285DBE06FF7618259EF6ADBE3ADB7B
```

## Package Contents

The final package should contain:

```text
H22_WITH_CODE_PACKAGE_README_20260524.md
HOST_MIGRATION_PROMPT_H22_ALL_MODELS_20260524.md
h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx
outputs_final_h22_all_models_with_gb_nn_single_best_20260524/
project/
```

Inside `project/`, the package should include:

```text
project/src/
project/scripts/
project/config/
project/docs/
project/tests/
project/requirements.txt
project/README.md
project/data/processed_h22_20260523/forecasting_panel.csv
project/data/processed/forecasting_panel_h22_20260523.csv
project/data/processed/forecasting_panel_h22_20260523_manifest.json
```

The package intentionally excludes virtual environments, caches, smoke outputs, raw worker outputs, and old intermediate final directories.

## Host-Side Prompt To Give Another AI

Copy the following prompt to the host-side AI:

```text
You are working on the host/main project. A completed h=22 monthly horizon supplement has been transferred from the backfill machine.

Use only the final transferred directory:

<HOST_PATH>\outputs_final_h22_all_models_with_gb_nn_single_best_20260524

Use the copy-ready Word document:

<HOST_PATH>\h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx

Strict rules:
1. Do not rerun any forecasting model.
2. Do not rerun GradientBoosting; GB h=22 is already included.
3. Do not rerun NN; NN single-best was derived from saved seed checkpoints.
4. Do not use the older 18-model h=22 directory for reporting.
5. Do not overwrite or modify existing h=1/h=5 strict main output directories.
6. Treat h=22 as an independent horizon-robustness supplement.
7. Do not generate VaR, ALE, VI, or MCS unless explicitly requested later.
8. Keep raw/transferred outputs and any host-side report integration outputs separate.

First verify:
- predictions/model_predictions.csv exists.
- predictions/nn_single_best_predictions.csv exists.
- tables/nn_single_best_seed_selection.csv exists.
- audit_report.json exists.
- run_provenance.json exists.
- rows = 911,580.
- model_count = 22.
- models include GradientBoosting.
- models include NN1_single_best, NN2_single_best, NN3_single_best, NN4_single_best.
- datasets are exactly MHAR and PARTIAL_MALL.
- horizon is exactly 22.
- ticker_count = 25.
- duplicate keys over date/ticker/dataset/horizon/model = 0.
- missing forecast_rv = 0.
- GradientBoosting rows = 41,436.
- NN single-best rows = 165,744.
- required tables exist:
  - tables/forecast_metrics_by_asset.csv
  - tables/forecast_summary_cross_section.csv
  - tables/pairwise_relative_mse_matrix.csv
  - tables/diebold_mariano_tests.csv
  - tables/nn_single_best_seed_selection.csv

After verification:
- Preserve the transferred directory intact.
- Add a short host import receipt/manifest recording source path, destination path, import time, and the audit values above.
- If updating the report, label these as h=22 monthly horizon robustness results.
- Use the existing CSV tables and the Word document directly for h=22 summary, relative MSE, per-asset metrics, and DM tests.
- Do not mix these rows into h=1/h=5 final outputs unless explicitly instructed; keep h=22 as a separate supplement.

Important methodological notes to preserve in the report:
- h=22 target is monthly realized volatility horizon.
- NN uses seeds = 50 for NN1, NN2, NN3, NN4 only.
- NN_i^1 columns are single-best seeds selected by validation MSE from seeds 42..91.
- NN_i^10 columns are top-10 validation ensembles.
- GradientBoosting was taken from the 40-grid GB backfill.
- GB was validation-tuned with train-only refit after validation selection, and ml_refit_every = 5.
- No VaR/ALE/VI/MCS outputs are part of this h=22 supplement.
```

## Verification Command For Host

Replace `<HOST_PATH>` below with the actual extracted package root on the host:

```powershell
python -c "import json, pandas as pd; from pathlib import Path; out=Path(r'<HOST_PATH>')/'outputs_final_h22_all_models_with_gb_nn_single_best_20260524'; p=out/'predictions/model_predictions.csv'; df=pd.read_csv(p, parse_dates=['date']); key=['date','ticker','dataset','horizon','model']; print('exists', p.exists()); print('rows', len(df)); print('models', sorted(df['model'].astype(str).unique())); print('model_count', df['model'].nunique()); print('datasets', sorted(df['dataset'].astype(str).unique())); print('horizons', sorted(df['horizon'].astype(int).unique())); print('tickers', df['ticker'].nunique()); print('duplicates', int(df.duplicated(key).sum())); print('missing_forecast', int(df['forecast_rv'].isna().sum())); print('zero_forecast', int((df['forecast_rv']==0).sum())); print('gb_rows', int((df['model'].astype(str)=='GradientBoosting').sum())); print('nn_single_best_rows', int(df['model'].astype(str).str.endswith('_single_best').sum())); aud=json.loads((out/'audit_report.json').read_text()); print('audit_rows', aud['rows']); print('audit_duplicate_keys', aud['duplicate_keys']); print('audit_missing_forecasts', aud['missing_forecasts']); print('audit_contains_gb', aud['contains_gradient_boosting'])"
```

Expected output should include:

```text
rows 911580
model_count 22
datasets ['MHAR', 'PARTIAL_MALL']
horizons [22]
tickers 25
duplicates 0
missing_forecast 0
zero_forecast 0
gb_rows 41436
nn_single_best_rows 165744
audit_contains_gb True
```

## Suggested Host Destination

Use a new independent destination, for example:

```text
<main_project>\supplements\h22\outputs_final_h22_all_models_with_gb_nn_single_best_20260524
```

Do not copy this into an existing h=1/h=5 output directory.

## Success Criteria On Host

The host import is successful when:

1. The final 22-model h=22 directory is present and intact.
2. The verification command matches the expected values.
3. The Word document opens and contains both h=22 tables with NN_i^1 and NN_i^10 columns.
4. The host has a receipt/manifest documenting the import.
5. The h=22 tables can be referenced independently in the report.
6. No h=1/h=5 outputs were overwritten or changed.
