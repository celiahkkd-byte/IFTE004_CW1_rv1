# h=22 Transfer Requirements - 2026-05-23

## Purpose

This package is for running an independent h=22 monthly-horizon forecast comparison on another machine.
The h=22 run is optional report material and must not modify, merge into, or overwrite the existing h=1/h=5 strict mainline results.

## Included Scope

- Current strict code, scripts, configs, and documentation.
- Source forecasting panel: `data/processed/forecasting_panel.csv`.
- Processed realized-measure fallback files.
- Small external feature inputs and metadata needed to inspect or rebuild feature context.
- Current strict reproduction manifest.

The package intentionally excludes prior output directories, report bundles, raw intraday monthly files, caches, and old zip archives.

## h=22 Panel Requirement

Create a new panel only:

```text
data/processed/forecasting_panel_h22_20260523.csv
```

Do not overwrite:

```text
data/processed/forecasting_panel.csv
```

The h=22 targets must be:

```text
target_rv_h22 = mean(rv_{t+1}, ..., rv_{t+22})
target_log_rv_h22 = log(target_rv_h22)
```

Also write:

```text
data/processed/forecasting_panel_h22_20260523_manifest.json
```

The manifest should record source panel, target construction, horizon, row count, ticker count, ticker list, datasets, models, created time, code version, and environment versions.

## Model Scope

Do not include GradientBoosting in final h=22 analysis.

Run:

```text
HAR, HARX, LogHAR, LevHAR, SHAR, HARQ
Ridge, Lasso, ElasticNet, AdaptiveLasso, PostLasso
Bagging, RandomForest
NN1, NN2, NN3, NN4
```

Datasets:

```text
MHAR
PARTIAL_MALL
```

NN requirement:

```text
seeds = 50
ensemble_top = 10
```

## Method Rules

- HAR family: rolling, train + validation combined.
- Regularized models: rolling daily refit, train fit + validation tuning, no final train+validation refit.
- Bagging and RandomForest: rolling, train + validation combined, `ml_refit_every=5`.
- NN: fixed scheme, 50 seeds, top-10 ensemble.
- All other model parameters should match the current strict configuration unless a h=22-specific output path or horizon setting is required.

## Parallel Plan

Use staged parallelism with independent worker outputs and a single-threaded final merge.

Set these before starting each worker:

```bash
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Recommended task granularity:

```text
non-NN task = dataset x horizon=22 x model x ticker
NN task = dataset x horizon=22 x NN_model x ticker x seed
```

Recommended phases:

1. Phase 0: create h=22 panel, manifest, task manifest, and optional cache shards.
2. Phase 1: HAR family + Ridge/Lasso/AdaptiveLasso/PostLasso.
3. Phase 2: ElasticNet.
4. Phase 3: Bagging + RandomForest.
5. Phase 4: NN50 seed-level checkpoints.
6. Phase 5: single-threaded merge, audit, relative MSE, DM tests.

Recommended worker counts:

```text
10-core / 16GB: Phase 1 = 4 workers, Phase 2 = 3-4, Phase 3 = 4 with trees.n_jobs=1, Phase 4 = 2-3.
20-core / 32GB+: Phase 1 = 6-8 workers, Phase 2 = 5-6, Phase 3 = 5 with trees.n_jobs=2-4, Phase 4 = 4-6.
```

Do not run NN and RF/Bagging at the same time.
Do not run aggregate or merge concurrently.

## Output Directories

Use isolated output directories, for example:

```text
outputs_h22_phase1_worker_01/
outputs_h22_elasticnet_worker_01/
outputs_h22_trees_worker_01/
outputs_h22_nn_worker_01/
```

Final merged directory:

```text
outputs_final_h22_no_gb_nn50_20260523/
```

## Final Evaluation Scope

Required:

```text
forecast summary
per-asset metrics
relative MSE
Diebold-Mariano tests
audit report
```

Not required for h=22:

```text
VaR
ALE
VI
MCS
```

MCS may be generated only as an extra robustness output and should not be treated as a required paper-strict h=22 result.

## Final Audit

Before using the results in the report, confirm:

```text
horizon = 22
datasets = MHAR + PARTIAL_MALL
25 tickers covered
GradientBoosting excluded from final analysis
all required non-NN models complete
NN: 50 seeds for every dataset x NN model x ticker
duplicates = 0
missing forecasts = 0
failed tasks listed explicitly
```

If the audit fails, do not include h=22 in the report.
