# Alpha Vantage 5-Minute Intraday Data

Downloaded for the RV1 volatility-forecasting replication.

## Source and query design

- Source: Alpha Vantage `TIME_SERIES_INTRADAY`.
- Interval: `5min`.
- Months requested: `2001-01` through `2017-12`.
- Trading hours: regular U.S. market hours only (`extended_hours=false`).
- Prices: raw/as-traded intraday OHLCV (`adjusted=false`).
- Output format requested from AV: CSV.

This is closer to the paper than the teacher package because the paper computes realized
variance from 5-minute intraday returns over 2001-2017. It is still not identical to the
paper's NYSE TAQ sample because Alpha Vantage does not reproduce the paper's full
corporate-action predecessor stitching for every DJIA constituent.

## Files

- `raw_monthly/<SYMBOL>/<YYYY-MM>.csv`: original month-level AV CSV files.
- `combined_teacher_format/<SYMBOL>.txt`: combined files in `date,time,open,high,low,close,volume` order.
- `download_manifest.csv`: one row per attempted symbol-month request.
- `availability_summary.csv`: coverage summary for every probed symbol.
- `paper_like_av_summary.csv`: curated coverage summary for the paper-like universe.
- `paper_tickers_from_article.txt`: tickers extracted from the article text.
- `av_downloaded_symbols.txt`: symbols downloaded during coverage checks.

## Coverage notes

- Full 2001-01 to 2017-12 AV coverage was obtained for 25 paper tickers:
  `AAPL, AXP, BA, CAT, CSCO, DIS, GE, GS, HD, IBM, INTC, JNJ, JPM, KO, MCD,
  MMM, MRK, MSFT, NKE, PFE, PG, UNH, VZ, WMT, XOM`.
- `CVX` is available from 2001-10 onward; 2001-01 through 2001-09 are unavailable
  under the current ticker.
- `TRV` is available from 2007-02-27 onward; the paper used predecessor ticker
  history before the Travelers ticker change.
- `RTX` is available from 2007-04-27 onward; the paper used United Technologies
  history for this constituent.
- `DOW` is not available for the 2001-2017 paper sample under the current ticker in
  the AV intraday endpoint. `DD/DOW` probes also failed for sampled months.
- `WBA` was downloaded during audit checks because AV provides full 2001-2017 data,
  but it is excluded from the main panel because Walgreens Boots Alliance joined the
  DJIA after the paper sample period, on 2018-06-26.

## Implementation note

The `combined_teacher_format` files contain 5-minute bars, not 1-minute bars. The
replication code now reads these files through `config/default.yaml` with
`bar_interval_minutes: 5` and `bars_per_full_day: 78`, so realized variance is
computed directly from the first opening price and the 78 five-minute interval closes.
Alpha Vantage timestamps are treated as bar-start labels: a retained full trading day
must have the exact `09:30, 09:35, ..., 15:55` grid. Half-days and days with missing,
duplicated, off-grid, outside-hours, or non-positive-price bars are excluded.
