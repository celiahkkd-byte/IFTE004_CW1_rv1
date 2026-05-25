# IFTE004 CW1 RV1 replication package

This repository contains the code, configuration files, processed modelling data,
and lightweight result tables for a public-data replication of Christensen,
Siggaard and Veliyev (2023), *A Machine Learning Approach to Volatility
Forecasting*.

The submitted mainline replication is the **25-ticker, all-model, h=1/h=5/h=22**
version. It covers both datasets used in the report:

- `MHAR`
- `PARTIAL_MALL`

The repository is designed so the marker can inspect the exact code path,
configuration choices, processed input panel, and final result tables without
downloading large intermediate caches.

## Submitted Mainline Scope

The mainline results use 25 tickers:

`AAPL, AXP, BA, CAT, CSCO, DIS, GE, GS, HD, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PFE, PG, UNH, VZ, WMT, XOM`

The forecast horizons are:

- `h=1`: one-day-ahead
- `h=5`: one-week-ahead
- `h=22`: one-month-ahead

The model universe is:

- HAR family: `HAR`, `HARX`, `LogHAR`, `LevHAR`, `SHAR`, `HARQ`
- regularized linear models: `Ridge`, `Lasso`, `ElasticNet`,
  `AdaptiveLasso`, `PostLasso`
- tree and boosting models: `Bagging`, `RandomForest`, `GradientBoosting`
- neural networks: `NN1`, `NN2`, `NN3`, `NN4`, each reported as single-best
  seed (`NN*_1`) and top-10 validation ensemble (`NN*`)

The mainline design is:

- HAR family: rolling estimation, with train and validation combined as the
  in-sample fitting window.
- Regularized linear models: rolling daily refit, trained on the training
  block, tuned on validation, and evaluated without final train+validation
  refit.
- Bagging and RandomForest: rolling estimation with train+validation combined;
  refitted every five test observations for computational tractability.
- GradientBoosting: validation-tuned no-refit, using the 40-grid specification
  described in the report.
- Neural networks: fixed train/validation/test split, 50 random seeds, with
  validation used for seed selection and top-10 ensemble construction.

## Included Result Tables

The GitHub repository includes report-facing result tables under
`results_release/`. These are the files intended for checking the submitted
tables and discussion.

### Mainline h=1/h=5/h=22 results

`results_release/mainline_h1h5h22/`

This directory contains the integrated 25-ticker mainline evaluation tables for
all three horizons:

- `pairwise_relative_mse_matrix.csv`
- `diebold_mariano_tests.csv`
- `forecast_summary_cross_section.csv`
- `forecast_metrics_by_asset.csv`
- `h22_nn_single_best_seed_selection.csv`
- `integration_audit.json`
- `h22_audit_report.json`
- `h22_run_provenance.json`

### h=22 regenerated audit tables

`results_release/mainline_h22_regenerated/`

This directory contains the regenerated 25-ticker monthly-horizon result tables
used to validate the final h=22 integration.

### Appendix D corrected checks

`results_release/appendix_d_corrected_h1h5/`

`results_release/appendix_d_corrected_h22/`

These contain the three-ticker corrected robustness checks for AAPL, JPM and
MSFT. They cover h=1, h=5 and h=22, and use:

- `Dropout(rate=0.2)` for NNs, interpreting the paper's reported dropout value
  as a keep probability.
- training-window target standardization with inverse transformation before
  MSE and DM evaluation.
- EA kept in its original binary `0/1` scale.

For the exact corrected-check rerun commands, see
`docs/APPENDIX_D_CORRECTED_CHECKS.md`.

### VaR and descriptive statistics

`results_release/var_h1/` contains the one-day-ahead relative VaR check-loss and
Diebold-Mariano tables.

`results_release/table1/` contains the explanatory-variable summary statistics
table and audit file.

## Files Excluded From GitHub

Large intermediate outputs are intentionally not committed:

- full daily prediction tables such as `outputs*/predictions/model_predictions.csv`
- NN per-seed checkpoint files
- logs, cache folders, generated Word files and generated figures
- raw Alpha Vantage intraday files

This is necessary because the full output directories are many gigabytes. The
repository instead keeps the processed modelling panel and lightweight result
tables needed to verify the report.

## Data Included

The processed modelling files are included:

- `data/processed/forecasting_panel.csv`
- `data/processed/daily_realized_measures.csv`
- `data/processed/intraday_bar_counts.csv`
- `data/processed/intraday_validation_summary.csv`

`forecasting_panel.csv` contains target columns for `h=1`, `h=5`, and `h=22`,
so the submitted mainline and h=22 checks can be reproduced from the committed
processed panel.

Small external inputs used to build `PARTIAL_MALL` are also included under
`data/external/`, including VIX, US 3-month T-bill, EPU, ADS, Hang Seng and
earnings-announcement data.

Raw Alpha Vantage intraday bars are not committed because of size. The download
scripts are included for a full rebuild if an Alpha Vantage API key is available:

```bash
python scripts/00_download_alpha_vantage_intraday.py --combine
python scripts/00_download_alpha_vantage_earnings.py --config config/default.yaml
```

## Configuration Files

Important mainline configuration files:

- `config/paper_core_rolling_tuned_no_refit.yaml`: h=1/h=5 mainline
  non-neural configuration.
- `config/paper_core_rolling_tuned_no_refit_h22_panel.yaml`: h=22 mainline
  configuration.
- `config/paper_core_rolling_gb_tuned_no_refit_40grid.yaml`: GB 40-grid
  backfill configuration.
- `config/default.yaml`: general project default, aligned to rolling non-neural
  estimation; the NN checkpoint script explicitly overrides scheme to fixed.

## Code Structure

```text
config/                 Mainline and robustness configurations
src/rv1rep/             Reproduction library
scripts/                Stage-by-stage execution scripts
data/processed/         Processed modelling panel and daily realized measures
data/external/          Small external predictor files and manifests
docs/                   Method and data-source documentation
results_release/        Lightweight submitted result tables
tests/                  Smoke tests
```

## Minimal Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Tables From Existing Outputs

The submitted result tables are already available in `results_release/`. The
scripts that generated them are included. Scripts that operate on full daily
prediction files require those files to be regenerated first, because large
`outputs*/` directories are intentionally excluded from GitHub.

Key table and audit scripts include:

- `scripts/05_evaluate_outputs_isolated.py`
- `scripts/09_make_pairwise_dm_word_tables.py`
- `scripts/09_regenerate_h22_all_ticker_model_tables.py`
- `scripts/09_integrate_h1h5_h22_results.py`
- `scripts/15_build_corrected_paper_tables.py`

For monthly-horizon details, see `docs/H22_REPRODUCTION.md`.
For Appendix D corrected-check details, see
`docs/APPENDIX_D_CORRECTED_CHECKS.md`.

Full model reruns are computationally heavy. Use isolated output directories
when rerunning any forecast script so existing submitted files are not
overwritten.
