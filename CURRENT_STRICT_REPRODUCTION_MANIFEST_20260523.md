# Current Strict Reproduction Manifest - 2026-05-23

This file freezes the current reproduction state before any further merging or model backfills. It is intended to prevent version confusion while additional components such as GradientBoosting or h=22 may be added later.

## Do Not Merge Yet

No final merged output has been created from the strict no-refit regularized-model run yet.

The current stable final directory remains:

```text
outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/
```

The new strict regularized-model output is an isolated increment:

```text
outputs_rolling_tuned_no_refit_20260522/
```

This increment must not be treated as a final result by itself. It contains only the five validation-tuned regularized models.

## Correct Configuration For Strict Regularized Models

Use:

```text
config/paper_core_rolling_tuned_no_refit.yaml
```

Key setting:

```yaml
estimation:
  refit_tuned_models_on_train_validation: false
```

Meaning:

- Candidate models are fitted on the training window.
- Validation is used only for hyperparameter selection.
- The selected training-fitted estimator is used directly for the test forecast.
- There is no final train+validation refit for these tuned models.

This setting currently applies to:

```text
Ridge
Lasso
ElasticNet
AdaptiveLasso
PostLasso
```

It does not change:

```text
HAR / HARX / LogHAR / LevHAR / SHAR / HARQ
Bagging
RandomForest
NN1-NN4
```

Important: the GradientBoosting branch still needs the same no-refit hook before GB is run under the paper-strict validation interpretation.

## Completed Strict No-Refit Increment

Output directory:

```text
outputs_rolling_tuned_no_refit_20260522/
```

Run scope:

```text
datasets: MHAR, PARTIAL_MALL
horizons: 1, 5
models: Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso
scheme: rolling
tickers: 25
```

Completed checkpoint count:

```text
20 / 20
```

Combined increment file:

```text
outputs_rolling_tuned_no_refit_20260522/predictions/model_predictions.csv
```

Validation summary:

```text
rows: 416,290
models: 5
datasets: MHAR, PARTIAL_MALL
horizons: 1, 5
tickers: 25
scheme: rolling
duplicate keys: 0
missing actual_rv: 0
missing forecast_rv: 0
GradientBoosting present: no
NN present: no
HAR-family present: no
Bagging/RandomForest present: no
```

All 20 dataset/horizon/model groups have:

```text
fit_sample: train_only_after_validation_selection
```

## What Can Be Reused Later

When building the next final strict result, reuse unchanged components from:

```text
outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/
```

Reusable from that directory:

```text
HAR
HARX
LogHAR
LevHAR
SHAR
HARQ
Bagging
RandomForest
NN1-NN4 top-10 ensemble
NN1_1-NN4_1 single-best seed
```

Replace only these five models with the strict no-refit increment:

```text
Ridge
Lasso
ElasticNet
AdaptiveLasso
PostLasso
```

Do not include GradientBoosting unless a separate GB backfill has been completed and validated.

## Next Merge Target, When Ready

When the user explicitly asks to merge, use a fresh directory name, for example:

```text
outputs_final_core_with_bagging_no_gb_harfix_nn50_tuned_no_refit_20260523/
```

After merging, recompute downstream outputs into fresh directories only:

```text
outputs_rv_decile_core_with_bagging_no_gb_harfix_nn50_tuned_no_refit_20260523/
outputs_fhs_var_core_with_bagging_no_gb_harfix_nn50_tuned_no_refit_20260523/
outputs_mcs_core_with_bagging_no_gb_harfix_nn50_tuned_no_refit_20260523/
outputs_ale_core_with_bagging_no_gb_harfix_nn50_tuned_no_refit_20260523/
outputs_variable_importance_core_with_bagging_no_gb_harfix_nn50_tuned_no_refit_20260523/
```

## GradientBoosting Backfill Rules

Do not run GB until its code path is confirmed to obey:

```text
fit_sample: train_only_after_validation_selection
```

Recommended GB backfill scope:

```text
model: GradientBoosting
datasets: MHAR, PARTIAL_MALL
horizons: 1, 5
scheme: rolling
refit frequency: ml_refit_every = 5
output: separate directory only
```

Recommended 24-grid:

```yaml
gradient_boosting:
  depths: [1, 2]
  n_estimators: [50, 100, 200, 300, 400, 500]
  learning_rates: [0.01, 0.1]
```

Suggested output names:

```text
outputs_gb_tuned_no_refit_24grid_smoke_20260523/
outputs_gb_tuned_no_refit_24grid_20260523/
```

GB should be treated as a robustness backfill unless the user explicitly decides to make it part of the final reported main version.

## h=22 Rules

h=22 is not part of the current final h=1/h=5 core result.

If h=22 is started later:

- Use a separate h=22 panel or manifest.
- Do not mutate the existing h=1/h=5 panel.
- Do not write into existing final output directories.
- Use separate worker output directories if running in parallel.
- Merge h=22 only after all worker outputs are validated.
- Do not change NN seeds, alpha grid, split rules, no-refit rules, or fixed/rolling scheme just to save time.

Suggested future naming:

```text
data/processed/forecasting_panel_h22_20260523.csv
outputs_h22_worker_*/
outputs_final_h22_no_gb_nn50_tuned_no_refit_20260523/
```

## Reporting Guidance

For the coursework report, present only the final selected version once it is merged and validated.

Do not mix tables from:

```text
outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/
```

with the strict no-refit increment unless a new final merged directory has been created.

The current strict increment should be described as a methodological alignment step:

```text
The validation-tuned regularized models were re-estimated using a paper-strict split: candidate models are fitted on the training window and validation is used only for hyperparameter selection, without a final train-validation refit.
```

