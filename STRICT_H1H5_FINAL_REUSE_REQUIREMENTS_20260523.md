# Strict h=1,5 Final and Reuse Requirements - 2026-05-23

This document defines the required next step after the GradientBoosting backfill and
the strict no-refit regularized-model increment. It is intended to prevent invalid
reuse of old post-processing outputs.

The guiding rule is:

> Reuse only when the model training rule, input data, target horizon, and output
> interpretation are unchanged. If the model set, fitted estimator, or final
> prediction table changes, recompute the dependent result.

This stage covers only horizons `h=1` and `h=5`. Do not include `h=22` in the
strict main final output or in downstream main-analysis outputs. The `h=22`
GradientBoosting staging output is handled separately.

## 1. Required Final Strict h=1,5 Merge

### Input Sources

Use the following three sources:

```text
Base reusable final:
outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/

Strict no-refit regularized-model increment:
outputs_rolling_tuned_no_refit_20260522/

GradientBoosting no-refit 40-grid backfill:
outputs_gb_tuned_no_refit_40grid_20260523/
```

### Output Directory

Write the new strict final output to a fresh directory:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

Do not overwrite any existing `outputs_*` directory.

### Merge Rules

Reuse these unchanged models from the base final:

```text
HAR
HARX
LogHAR
LevHAR
SHAR
HARQ
Bagging
RandomForest
NN1
NN2
NN3
NN4
NN1_1
NN2_1
NN3_1
NN4_1
```

Replace these five regularized models using the strict no-refit increment:

```text
Ridge
Lasso
ElasticNet
AdaptiveLasso
PostLasso
```

Append only these GradientBoosting rows:

```text
model = GradientBoosting
dataset in [MHAR, PARTIAL_MALL]
horizon in [1, 5]
scheme = rolling
```

Explicitly exclude all rows with:

```text
horizon = 22
```

### Tables to Recompute

After the merge, recompute all main evaluation tables from the new strict final
prediction table:

```text
predictions/model_predictions.csv
tables/forecast_metrics_by_asset.csv
tables/forecast_summary_cross_section.csv
tables/pairwise_relative_mse_matrix.csv
tables/diebold_mariano_tests.csv
```

### Validation Requirements

The strict final prediction file must satisfy:

- horizons are exactly `[1, 5]`
- datasets are exactly `MHAR` and `PARTIAL_MALL`
- model count is exactly 22
- duplicate key count is zero using:

```text
ticker/date/dataset/horizon/model
```

- all five regularized replacement models have params containing:

```text
fit_sample: train_only_after_validation_selection
```

- all GradientBoosting rows have params containing:

```text
fit_sample: train_only_after_validation_selection
```

- unchanged models from the base final are preserved exactly:

```text
HAR
HARX
LogHAR
LevHAR
SHAR
HARQ
Bagging
RandomForest
NN1-NN4
NN1_1-NN4_1
```

## 2. MCS Reuse Decision

### Existing Output

Existing MCS output:

```text
outputs_mcs_core_with_bagging_no_gb_harfix_nn50_20260522/
```

Existing provenance shows that it used:

```text
outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/predictions/model_predictions.csv
```

### Decision

Do not reuse old MCS results or old MCS checkpoints. MCS must be recomputed.

### Rationale

MCS is a model-set procedure. The old MCS result is invalid for the strict final
because:

- it does not include GradientBoosting
- it uses old regularized-model forecasts
- the MCS loss matrix changes when any model is added or replaced
- inclusion rates can change for every model, not only the modified models

### New Input

Use:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/predictions/model_predictions.csv
```

### New Output Directory

Write to:

```text
outputs_mcs_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

### Validation Requirements

The new MCS output must satisfy:

- datasets are `MHAR` and `PARTIAL_MALL`
- horizons are `[1, 5]`
- models are the 22 strict final models
- no `h=22` rows are present
- `mcs_inclusion_rates.csv` and `mcs_per_ticker.csv` are based on the new strict final
- provenance records the new strict final prediction file as input

## 3. ALE Figure 6 Reuse Decision

### Existing Output

Existing ALE output:

```text
outputs_ale_core_with_bagging_no_gb_harfix_nn50_20260522/
```

Current ALE scope:

```text
dataset = PARTIAL_MALL
horizon = 1
ticker = AAPL
models = HARX, LogHAR, ElasticNet, RandomForest, NN10_2
features = rvd, rvw, m1w
```

### Decision

Partially reuse ALE outputs. Reuse only model checkpoints whose fitting rule has not
changed.

Reusable ALE checkpoints:

```text
HARX
LogHAR
RandomForest
NN10_2
```

Must recompute:

```text
ElasticNet
```

### Rationale

ElasticNet is part of the strict no-refit regularized-model update. The old ALE
ElasticNet `fit_info` does not contain:

```text
fit_sample: train_only_after_validation_selection
```

Therefore, the old ElasticNet ALE curve does not match the strict final training rule.

The other ALE reference models are unchanged by the strict no-refit regularized-model
increment and the GB backfill:

- HARX remains OLS under the same data and feature rule
- LogHAR remains OLS on the log target under the same data and feature rule
- RandomForest remains unchanged
- NN10_2 remains unchanged and uses the same NN seed checkpoint source

### New Output Directory

Write to:

```text
outputs_ale_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523/
```

### Required Processing

Create a fresh ALE output directory.

Reuse or copy valid checkpoints for:

```text
HARX
LogHAR
RandomForest
NN10_2
```

Recompute ElasticNet ALE checkpoints for:

```text
rvd
rvw
m1w
```

Then rebuild:

```text
tables/ale_table.csv
figures/figure6_ale.png
run_provenance.json
```

### Validation Requirements

The new ALE output must satisfy:

- dataset is `PARTIAL_MALL`
- horizon is `1`
- ticker is `AAPL`
- models are exactly:

```text
HARX
LogHAR
ElasticNet
RandomForest
NN10_2
```

- features are exactly:

```text
rvd
rvw
m1w
```

- ElasticNet `fit_info` contains:

```text
fit_sample: train_only_after_validation_selection
```

- provenance states:

```text
HARX: reused
LogHAR: reused
RandomForest: reused
NN10_2: reused
ElasticNet: recomputed
```

## 4. Variable Importance Figure 7 Reuse Decision

### Existing Output

Existing variable-importance output:

```text
outputs_variable_importance_core_with_bagging_no_gb_harfix_nn50_20260522/
```

Current VI scope:

```text
dataset = PARTIAL_MALL
horizon = 1
models = HARX, ElasticNet, RandomForest, NN10_2
tickers = 25
```

### Decision

Partially reuse VI outputs. Reuse only model checkpoints whose fitting rule has not
changed.

Reusable VI checkpoints:

```text
HARX
RandomForest
NN10_2
```

Must recompute:

```text
ElasticNet
```

### Rationale

ElasticNet training changed under the strict no-refit update, so the old ElasticNet
VI checkpoints are no longer methodologically valid. HARX, RandomForest, and NN10_2
are unchanged and can be reused.

This reuse is especially valuable because the previous VI run spent most of its time
on model refitting, especially NN10_2. Reusing unchanged checkpoints avoids unnecessary
reruns while preserving methodological validity.

### New Output Directory

Write to:

```text
outputs_variable_importance_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523/
```

### Required Processing

Create a fresh VI output directory.

Reuse or copy valid per-ticker checkpoints for:

```text
HARX
RandomForest
NN10_2
```

Recompute all ElasticNet per-ticker checkpoints:

```text
25 tickers x ElasticNet
```

Then rebuild:

```text
tables/variable_importance.csv
tables/variable_importance_by_ticker.csv
figures/figure7_variable_importance.png
run_provenance.json
```

### Validation Requirements

The new VI output must satisfy:

- dataset is `PARTIAL_MALL`
- horizon is `1`
- models are exactly:

```text
HARX
ElasticNet
RandomForest
NN10_2
```

- no `iv` or `log_iv` feature appears unless a real firm-level IV file exists
- per model/ticker normalized VI sums to approximately 1
- ElasticNet checkpoint `fit_info` contains:

```text
fit_sample: train_only_after_validation_selection
```

- provenance states:

```text
HARX: reused
RandomForest: reused
NN10_2: reused
ElasticNet: recomputed
```

## 5. Outputs That Must Not Be Reused Directly

Do not directly reuse these as final strict outputs:

```text
outputs_mcs_core_with_bagging_no_gb_harfix_nn50_20260522/
outputs_mcs_core_with_bagging_no_gb_harfix_nn50_20260522/checkpoints/
outputs_ale_core_with_bagging_no_gb_harfix_nn50_20260522/checkpoints/.../ElasticNet/
outputs_variable_importance_core_with_bagging_no_gb_harfix_nn50_20260522/checkpoints/ElasticNet/
```

Also do not use any final main output that includes:

```text
horizon = 22
```

## 6. Recommended Execution Order

1. Generate the final strict `h=1,5` prediction table.
2. Validate the strict final:
   - no `h=22`
   - 22 models
   - duplicate key count is zero
   - strict no-refit markers are present for the five regularized models and GB
   - unchanged models are preserved exactly from the base final
3. Recompute all forecast metrics and pairwise/DM tables from the strict final.
4. Recompute MCS from the strict final in a fresh output directory.
5. Build a new ALE output:
   - reuse HARX, LogHAR, RandomForest, NN10_2
   - recompute ElasticNet
   - rebuild table and figure
6. Build a new VI output:
   - reuse HARX, RandomForest, NN10_2
   - recompute ElasticNet
   - rebuild aggregate tables and figure
7. Update report notes and gap documentation:
   - `h=1,5` strict main results complete
   - `h=22` deferred for a later unified run
   - MCS fully recomputed
   - ALE/VI partially reused only where methodologically valid

## 7. Summary Reuse Matrix

| Component | Reuse Old Result? | Required Action | Reason |
|---|---:|---|---|
| Final prediction table | No | Build new strict final | Regularized rows and GB rows changed |
| Forecast metrics | No | Recompute | Derived from final prediction table |
| Pairwise relative MSE | No | Recompute | Derived from final prediction table |
| DM tests | No | Recompute | Derived from final prediction table |
| MCS | No | Recompute from strict final | Model set and losses changed |
| ALE HARX | Yes | Reuse checkpoint | Training rule unchanged |
| ALE LogHAR | Yes | Reuse checkpoint | Training rule unchanged |
| ALE ElasticNet | No | Recompute | Strict no-refit rule changed estimator |
| ALE RandomForest | Yes | Reuse checkpoint | Training rule unchanged |
| ALE NN10_2 | Yes | Reuse checkpoint | Training rule unchanged |
| VI HARX | Yes | Reuse checkpoint | Training rule unchanged |
| VI ElasticNet | No | Recompute | Strict no-refit rule changed estimator |
| VI RandomForest | Yes | Reuse checkpoint | Training rule unchanged |
| VI NN10_2 | Yes | Reuse checkpoint | Training rule unchanged |

