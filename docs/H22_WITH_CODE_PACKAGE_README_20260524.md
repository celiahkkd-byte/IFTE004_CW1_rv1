# h=22 Final With-Code Migration Package

This is the **final** code-bearing host migration bundle for the h=22 monthly horizon supplement.

## Final Output To Use

Use:

```text
outputs_final_h22_all_models_with_gb_nn_single_best_20260524/
```

and:

```text
h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx
```

These supersede the older 18-model `outputs_final_h22_all_models_with_gb_20260524/` result and the older `h22_relative_mse_dm_tables_20260524.docx` Word file.

## What This Package Contains

- Project code:
  - `project/src/`
  - `project/scripts/`
  - `project/config/`
  - `project/docs/`
  - `project/tests/`
- Environment and project metadata:
  - `project/requirements.txt`
  - `project/README.md`
  - `project/main.py`
  - h=22 runbook, requirements, and migration notes
- h=22 data artifacts:
  - `project/data/processed_h22_20260523/forecasting_panel.csv`
  - `project/data/processed/forecasting_panel_h22_20260523.csv`
  - `project/data/processed/forecasting_panel_h22_20260523_manifest.json`
- Final h=22 all-model result with GB and NN single-best:
  - `outputs_final_h22_all_models_with_gb_nn_single_best_20260524/`
- Copy-ready Word tables:
  - `h22_relative_mse_dm_tables_with_nn_single_best_20260524.docx`

## Final Audit Values

```text
rows: 911,580
model_count: 22
datasets: MHAR, PARTIAL_MALL
horizon: 22
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

## Interpretation Of NN Columns

- `NN1`, `NN2`, `NN3`, `NN4` are the NN50 top-10 validation ensembles and correspond to `NN_i^10`.
- `NN1_single_best`, `NN2_single_best`, `NN3_single_best`, `NN4_single_best` are selected by validation MSE from the 50 saved seeds and correspond to `NN_i^1`.

No NN model was retrained to create the single-best columns.

## Excluded

- Python virtual environments
- `__pycache__` and bytecode
- `.pytest_cache`
- smoke output directories
- raw worker task output directories
- intermediate no-GB final directories
- superseded 18-model final directories
- macOS AppleDouble `._*` files

## Important

The final all-model h=22 result already includes GradientBoosting h=22. Do not rerun GB on the host unless explicitly requested.

Use `HOST_MIGRATION_PROMPT_H22_ALL_MODELS_20260524.md` for exact host-side verification instructions.

