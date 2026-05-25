# Strict h=1,5 Completion Manifest - 2026-05-23

This manifest records the completed strict `h=1,5` outputs after merging the
strict no-refit regularized-model increment and the GradientBoosting no-refit
backfill. It supersedes the earlier "do not merge yet" status for `h=1,5`.

`h=22` is not included in any main final output listed here.

## Completed Main Final

Output directory:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

Main prediction file:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/predictions/model_predictions.csv
```

Composition:

- HAR family: reused from `outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/`
- Bagging and RandomForest: reused from `outputs_final_core_with_bagging_no_gb_harfix_nn50_20260522/`
- NN ensemble and single-best-seed rows: reused from the NN50 final
- Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso: replaced with
  `outputs_rolling_tuned_no_refit_20260522/`
- GradientBoosting: appended from
  `outputs_gb_tuned_no_refit_40grid_20260523/`, restricted to `h=1,5`

Validation summary:

```text
rows: 1,831,652
models: 22
horizons: [1, 5]
duplicate key count: 0
regularized no-refit marker share: 1.0
GradientBoosting no-refit marker share: 1.0
NN ensemble rows from NN50: 333,032
NN single-best-seed rows from NN50: 333,032
```

Generated tables:

```text
tables/forecast_metrics_by_asset.csv
tables/forecast_summary_cross_section.csv
tables/pairwise_relative_mse_matrix.csv
tables/diebold_mariano_tests.csv
```

## Completed MCS

Output directory:

```text
outputs_mcs_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

Input:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/predictions/model_predictions.csv
```

Validation summary:

```text
models: 22
horizons: [1, 5]
mcs_inclusion_rates rows: 88
mcs_per_ticker rows: 2,200
mcs checkpoint manifest rows: 100
```

MCS was fully recomputed. Old MCS checkpoints were not reused because the model set
and loss matrix changed after adding GradientBoosting and replacing the regularized
models.

## Completed ALE Figure 6 Output

Output directory:

```text
outputs_ale_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523/
```

Scope:

```text
dataset: PARTIAL_MALL
horizon: 1
ticker: AAPL
models: HARX, LogHAR, ElasticNet, RandomForest, NN10_2
features: rvd, rvw, m1w
```

Reuse policy:

```text
HARX: reused
LogHAR: reused
RandomForest: reused
NN10_2: reused
ElasticNet: recomputed under strict no-refit
```

Validation summary:

```text
ale_table rows: 1,500
ElasticNet strict no-refit marker share: 1.0
figure6_ale.png exists: yes
```

## Completed Variable Importance Figure 7 Output

Output directory:

```text
outputs_variable_importance_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523/
```

Scope:

```text
dataset: PARTIAL_MALL
horizon: 1
models: HARX, ElasticNet, RandomForest, NN10_2
tickers: 25
features: 11 public PARTIAL_MALL predictors
```

Reuse policy:

```text
HARX: reused
RandomForest: reused
NN10_2: reused
ElasticNet: recomputed under strict no-refit
```

Validation summary:

```text
variable_importance rows: 44
variable_importance_by_ticker rows: 1,100
ElasticNet strict no-refit marker share: 1.0
figure7_variable_importance.png exists: yes
IV/log_IV leakage: no
per model/ticker normalized VI sums: valid
```

## Completed RV-Decile Figure 5 Output

Output directory:

```text
outputs_rv_decile_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

Input:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/predictions/model_predictions.csv
```

Scope:

```text
datasets: MHAR, PARTIAL_MALL
horizons: 1, 5
models: 22
RV-decile basis horizon: 1
deciles: 1 through 10
```

Generated outputs:

```text
tables/rv_decile_mse.csv
tables/rv_decile_assignments.csv
figures/figure5_rv_decile_mse.png
```

Validation summary:

```text
rv_decile_mse rows: 880
rv_decile_assignments rows: 41,648
minimum cell observations: 2,031
missing MSE values: 0
missing relative MSE values: 0
figure5_rv_decile_mse.png exists: yes
```

The decile basis is formed from observed realized variance, so the
model-expanded prediction table is collapsed to one asset-date observation
before assigning deciles. A total of 3,636 `h=5` prediction rows have no
matching `h=1` basis assignment at the sample boundary and are excluded from
the decile table; no `h=1` rows are excluded.

## Completed FHS VaR Output

Output directory:

```text
outputs_fhs_var_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1_20260523/
```

Input:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/predictions/model_predictions.csv
```

Scope:

```text
datasets: MHAR, PARTIAL_MALL
horizon: 1
models: 22
tickers: 25
alpha: 0.05
method: filtered_historical_simulation
```

Generated outputs:

```text
predictions/var_forecasts_fhs.csv
tables/var_backtest_fhs_summary.csv
tables/fhs_var_checkpoint_manifest.csv
```

Validation summary:

```text
VaR forecast rows: 916,160
VaR backtest summary rows: 1,100
FHS checkpoint manifest rows: 1,100
checkpoint statuses: completed
missing VaR forecasts: 0
missing hit indicators: 0
```

## Completed Paper-Style Word Tables

Output directory:

```text
outputs_paper_word_tables_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

Generated Word files:

```text
table2_mhar_h1_relative_mse_dm.docx
pairwise_relative_mse_dm_tables_h1h5.docx
```

Formatting audit files:

```text
table2_mhar_h1_relative_mse_dm.formatting_audit.csv
pairwise_relative_mse_dm_tables_h1h5.formatting_audit.csv
```

The Word tables use the strict `h=1,5` pairwise relative MSE and
Diebold-Mariano outputs. Formatting is recomputed from ticker-level p-values:
italic for rejection by more than 50% of stocks at 10%, bold italic at 5%, and
bold italic with underline at 1%.

## Completed Explanatory-Variable Summary Table

Output directory:

```text
outputs_paper_table1_explanatory_stats_20260523/
```

Generated files:

```text
table1_explanatory_variable_summary.docx
table1_explanatory_variable_summary.csv
table1_explanatory_variable_summary_audit.csv
run_provenance.json
```

Input:

```text
data/processed/forecasting_panel.csv
```

Validation summary:

```text
variables: 12
audit rows: 84
tickers: 25
sample: 2001-01-02 to 2017-12-29
```

RVD, RVW, and RVM are reported as annualized volatility percentages; US3M,
HSI, M1W, `$VOL`, and ADS use the transformed units described in the table
note. IV is left blank because firm-level OptionMetrics implied volatility is
not available in this reproduction.

## Completed Figure 3 Relative-MSE Boxplots

Output directory:

```text
outputs_figure3_relative_mse_boxplot_20260523/
```

Generated files:

```text
figure3_relative_mse_boxplot_h1_paper_axis.png
figure3_relative_mse_boxplot_h1_full_range.png
figure3_relative_mse_by_ticker.csv
figure3_relative_mse_summary.csv
figure3_paper_axis_clipping_audit.csv
run_provenance.json
```

Input:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/tables/forecast_metrics_by_asset.csv
```

Scope:

```text
horizon: 1
datasets: MHAR, PARTIAL_MALL
tickers: 25
MHAR plotted models: 20
PARTIAL_MALL plotted models: 21
```

The plot reports each model's ticker-level MSE relative to the HAR benchmark.
The paper-axis version uses the original Figure 3-style y-axis range
`[0.75, 1.75]`; the full-range version is also saved because this reproduction
has several high relative-MSE outliers.

## Completed Figure 4 MCS Inclusion-Rate Plot

Output directory:

```text
outputs_figure4_mcs_inclusion_rate_20260523/
```

Generated files:

```text
figure4_mcs_inclusion_rate_h1.png
figure4_mcs_inclusion_rate_h1.csv
run_provenance.json
```

Input:

```text
outputs_mcs_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/tables/mcs_inclusion_rates.csv
```

Scope:

```text
horizon: 1
datasets: MHAR, PARTIAL_MALL
models: 22
tickers per inclusion rate: 25
```

The plot reports the share of tickers for which each model is retained in the
90% Hansen-Lunde-Nason MCS. It is based only on the completed `h=1` MCS rows.

## Do Not Use As Main Final

The following directory contains a previously merged result that included
GradientBoosting `h=22` rows and should not be used as the strict main final:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_20260523/
```

For main `h=1,5` analysis, use only:

```text
outputs_final_core_with_bagging_gb_harfix_nn50_tuned_no_refit_h1h5_20260523/
```

The separate `h=22` staging directory remains:

```text
outputs_h22_staging_gb_only_20260523/
```

It is not part of the completed `h=1,5` strict main result.
