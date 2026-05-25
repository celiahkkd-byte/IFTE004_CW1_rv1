# RV1 professional replication code package

This package is a **code-first, auditable replication framework** for Christensen, Siggaard and Veliyev, *A Machine Learning Approach to Volatility Forecasting* (Journal of Financial Econometrics, 2023). The default configuration uses the included Alpha Vantage 5-minute intraday data for a balanced 2001-2017 DJIA-like stock universe, and it also supports free external variables used to build a MALL-style feature set.

The package intentionally does **not** contain fabricated result tables. It builds the data, downloads public external series, trains models and writes results when you run it.

## Preserved baseline results

Current strict `h=1,5` results use NN50 seed checkpoints; see
`STRICT_H1H5_COMPLETION_MANIFEST_20260523.md` for the current final output
directory. The older preserved fallback directories below are retained only as
historical baselines.

The following outputs are the current validated baseline results and should be treated as
preserved fallback results unless a deliberate replacement is made:

- `outputs/`: fixed-scheme 14-model non-NN results after the AdaptiveLasso fix.
  `outputs/predictions/model_predictions.csv` has 583,008 rows, 14 models, 25 tickers,
  and the datasets `MHAR` and `PARTIAL_MALL`.
- `outputs_full_nn/`: fixed-scheme 18-model results including NN1--NN4 with 20 seeds
  and a top-10 ensemble. `outputs_full_nn/predictions/model_predictions.csv` has
  749,584 rows, 18 models, 25 tickers, and the datasets `MHAR` and `PARTIAL_MALL`.
- `data/processed/daily_realized_measures.csv`: validated daily realized measures with
  104,563 rows and 25 tickers.
- `data/processed/forecasting_panel.csv`: forecasting feature panel with 104,563 rows
  and 25 tickers.

Do not overwrite `outputs/` or `outputs_full_nn/` while testing extended replications.
The planned paper-closer reruns should write to isolated directories:

- `outputs_rolling/` for rolling non-NN models
- `outputs_nn50_checkpointed_20260521/` for fixed NN50 seed checkpoints and NN-only outputs
- `outputs_final/` for the merged final evaluation

The default `main.py`, `scripts/04_run_forecasts.py`, and `scripts/05_evaluate_outputs.py`
use `paths.output_dir: outputs` from `config/default.yaml`. Running them without an
isolated output directory may update the preserved `outputs/` fallback. For experimental
runs, use `scripts/04_run_forecasts_checkpoints.py` for model-level forecast
checkpoints, `scripts/04_run_nn_checkpoints.py` for seed-level NN checkpoints, and
`scripts/05_evaluate_outputs_isolated.py` with explicit output directories.

## What this package replicates

### High-frequency realized measures
From the included 5-minute intraday bars it constructs 78 intraday 5-minute log returns per full trading day and computes:

- daily realized variance `rv`
- positive and negative semivariance `rvp`, `rvn`
- realized quarticity `rq = n / 3 * sum(r_j^4)`
- open-to-close and close-to-close log returns
- daily dollar volume and log-difference dollar volume `dvol`
- 1-week momentum `m1w`

### Forecasting datasets

- `MHAR`: daily, weekly and monthly realized variance lags, with model-specific variables for LogHAR, LevHAR, SHAR and HARQ.
- `PARTIAL_MALL`: MHAR plus free/public variables: VIX, daily 3-month T-bill rate changes, Hang Seng squared return, US EPU, ADS, momentum and dollar-volume change. Firm-level OptionMetrics implied volatility is not included unless you supply it separately.

### Models

- HAR, HAR-X, LogHAR, LevHAR, SHAR, HARQ
- Ridge, Lasso, Elastic Net, Post-Lasso, Adaptive Lasso approximation
- Bagging, Random Forest, Gradient Boosting
- NN1--NN4 feed-forward neural networks with paper-like pyramid architectures, dropout and early stopping when TensorFlow is installed

### Evaluation

- MSE/RMSE/MAE/R2 by asset, model, dataset and horizon
- pairwise relative MSE matrices, following the paper's Table 2/Table 3 logic
- Diebold--Mariano one-sided predictive accuracy tests
- permutation importance and ALE utilities
- optional VaR backtest with quantile loss, Kupiec unconditional coverage and Christoffersen independence tests

## Quick start

1. Create an environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Use the included Alpha Vantage data. The default config points to:

```text
data/external/alpha_vantage_intraday_5min/combined_teacher_format/
```

The raw month-level AV files are also included under:

```text
data/external/alpha_vantage_intraday_5min/raw_monthly/
```

To re-download Alpha Vantage source data, provide an API key either through the
environment or through the download script's `--apikey` option. The repository includes
`.env.example` as the fill-in template:

```bash
cp .env.example .env
# edit .env and fill ALPHAVANTAGE_API_KEY
set -a; source .env; set +a

python scripts/00_download_alpha_vantage_intraday.py --combine
python scripts/00_download_alpha_vantage_earnings.py --config config/default.yaml
```

The real key is intentionally not committed. The included data files are sufficient to run
the replication without an API key.

3. Run the pipeline. Start with a compact run to validate the data and model plumbing:

```bash
python main.py --config config/default.yaml --models HAR LogHAR SHAR HARQ Ridge RandomForest GradientBoosting --skip-nn
```

4. For a closer paper-style run, add neural networks after installing TensorFlow:

```bash
python main.py --config config/default.yaml --models HAR HARX LogHAR LevHAR SHAR HARQ Ridge Lasso ElasticNet AdaptiveLasso PostLasso Bagging RandomForest GradientBoosting NN1 NN2 NN3 NN4
```

5. To run a computationally heavier rolling-window forecast:

```bash
python main.py --config config/default.yaml --scheme rolling --models HAR LogHAR SHAR HARQ RandomForest GradientBoosting --skip-nn
```

## External data sources

The downloader uses free public sources:

- VIX: FRED `VIXCLS`
- 3-month T-bill: FRED `DTB3`
- EPU: `https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv`
- ADS: Philadelphia Fed ADS current vintage Excel file
- Hang Seng Index: Yahoo Finance chart endpoint, with `yfinance` fallback
- Earnings announcement dates: Alpha Vantage `EARNINGS`

If network access is not available, manually place the files in `data/external/` with these names:

```text
vix_fred.csv
us3m_fred.csv
epu_daily.csv
ads.xlsx
hsi.csv
earnings_announcements.csv
```

The code logs whether a variable was successfully loaded. Missing external series are omitted from `PARTIAL_MALL` rather than silently fabricated.

## Important methodological choices

1. **No Parkinson proxy is used.** The data are high-frequency intraday bars; realized variance is computed directly.
2. **Full-day filtering is explicit.** Alpha Vantage timestamps are treated as bar-start labels. For the 5-minute data, each retained ticker-day must contain the exact regular-hours grid from `09:30` to `15:55`, giving 78 bars and 78 log returns. Half-days, missing-bar days, duplicated timestamps, off-grid timestamps and non-positive prices are excluded before RV is computed.
3. **No look-ahead standardization.** Feature scalers are fit on the training set only.
4. **Stock-level IV is not imputed.** The original paper uses OptionMetrics IV; this package excludes it unless the user supplies a CSV.
5. **PARTIAL_MALL is not exact MALL.** It is an economically motivated public-data extension.

## Project structure

```text
config/default.yaml              Main configuration
main.py                          End-to-end pipeline runner
scripts/                         Stage-by-stage scripts
src/rv1rep/                      Replication library
data/raw/                        Optional teacher-data location for the older 1-minute workflow
data/external/                   Included AV intraday data and downloaded public external series
data/processed/                  Created daily realized measures/features
outputs/tables/                  Evaluation tables
outputs/figures/                 Plots
outputs/predictions/             Model forecasts
```

## Suggested audit path

```bash
python scripts/01_build_realized_measures.py --config config/default.yaml
python scripts/02_download_external_data.py --config config/default.yaml
python scripts/03_build_features.py --config config/default.yaml
python scripts/04_run_forecasts.py --config config/default.yaml --models HAR RandomForest GradientBoosting --skip-nn
python scripts/05_evaluate_outputs.py --config config/default.yaml
```
