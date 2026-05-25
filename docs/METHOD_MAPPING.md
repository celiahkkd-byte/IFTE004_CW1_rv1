# Mapping from the paper to this code package

## 1. Realized variance target

Paper equation:

    RV_t = sum_{j=1}^n |Delta X_{t,j}|^2

This code:

- reads the included Alpha Vantage 5-minute OHLCV bars for the balanced 2001-2017 stock universe in `config/default.yaml`;
- treats Alpha Vantage timestamps as bar-start labels and keeps only exact regular-hours ticker-days on the `09:30, 09:35, ..., 15:55` grid;
- excludes half-days and days with missing, duplicated, off-grid, outside-hours, or non-positive-price bars;
- constructs exactly 78 five-minute log returns per retained day from the first opening price and the 78 interval closes;
- computes `rv = sum(r_5min^2)`.

## 2. HAR lineage

Implemented models and feature sets:

- HAR: `rvd`, `rvw`, `rvm`
- HAR-X: same as HAR in `MHAR`; all available MALL-style variables in `PARTIAL_MALL`
- LogHAR: `log_rvd`, `log_rvw`, `log_rvm`; in `PARTIAL_MALL`, VIX and optional stock-level IV enter as `log_vix` and `log_iv`, matching the paper. Forecasts are transformed back with Jensen bias correction.
- LevHAR: HAR plus negative return terms `rd`, `rw`, `rm`
- SHAR: `rvp`, `rvn`, `rvw`, `rvm`
- HARQ: `rvd`, `sqrt(rq)*rvd`, `rvw`, `rvm`

## 3. ML models

Implemented ML families:

- Ridge, Lasso, ElasticNet
- Post-Lasso and Adaptive Lasso approximation
- Bagging and Random Forest with 500 trees and minimum leaf size 5
- Gradient Boosting with validation tuning over depth, tree count and learning rate
- NN1--NN4 pyramid architectures with TensorFlow dropout/early stopping if TensorFlow is installed

## 4. Datasets

- `MHAR`: one-to-one comparison between HAR variables and ML algorithms using only lagged realized variance, plus model-specific HAR extension variables.
- `PARTIAL_MALL`: public-data extension with VIX, HSI, ADS, US3M, EPU, M1W, dollar-volume change, and AV earnings-announcement indicators. It excludes firm-level OptionMetrics IV unless the user supplies `data/external/firm_iv.csv`.

## 5. Evaluation

- MSE, RMSE, MAE and out-of-sample R2
- Pairwise relative MSE matrix, where cell(row i, column j) equals MSE_j / MSE_i averaged across assets
- One-sided Diebold--Mariano tests using squared-error loss
- Optional VaR diagnostics and coverage tests

## 6. Known differences from the paper

- The original paper uses 29 DJIA constituents with predecessor stitching. The default AV replication uses the 25 paper tickers for which the AV endpoint provides full 2001-2017 current-ticker coverage.
- `CVX`, `TRV`, `RTX` and `DOW` are documented in `data/external/alpha_vantage_intraday_5min/README.md` but are not in the default balanced panel because AV does not provide the full predecessor-stitching history used by the paper.
- `WBA` is present only as a downloaded audit file and is excluded from the main panel because it joined the DJIA after the 2001-2017 paper sample period.
- The original MALL includes stock-level OptionMetrics IV; this code does not fabricate IV.
- The original paper uses cleaned TAQ transaction prices; this code uses Alpha Vantage OHLCV bars, so it is a transparent public-data approximation rather than an exact TAQ reconstruction.
- Mainline non-neural forecasts use rolling estimation in the paper-core
  configurations. The NN checkpoint runner explicitly uses a fixed
  train/validation/test split, matching the paper's NN design.
