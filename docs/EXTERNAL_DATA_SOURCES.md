# External public data sources used by the code

The original paper's MALL dataset includes both public market/macro variables and proprietary firm-level implied volatility. This package downloads only free/public variables and labels the extended dataset `PARTIAL_MALL`.

The public files used by the default run are cached in `data/external/`, so the
forecasting panel can be rebuilt without needing an API key.

For optional Alpha Vantage re-downloads, copy `.env.example` to `.env` and fill
`ALPHAVANTAGE_API_KEY`, or pass the key directly with the relevant script's `--apikey`
argument.

## VIX

- Series: FRED `VIXCLS`
- Endpoint used by the code: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS`
- Role: market-level implied volatility.

## US 3-month T-bill rate

- Series: FRED `DTB3`
- Endpoint: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3`
- Transformation: first difference, following the paper's nonstationarity treatment.

## Economic Policy Uncertainty

- Source: policyuncertainty.com daily policy data
- Endpoint: `https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv`
- Transformation: level, forward-filled to the equity trading calendar.

## ADS business conditions index

- Source: Federal Reserve Bank of Philadelphia ADS current vintage
- Endpoint attempted: `https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/ads/ads_index_most_current_vintage.xlsx`
- Transformation: level, forward-filled to equity trading days.

## Hang Seng Index

- Source: Yahoo Finance daily close via the Yahoo download endpoint or yfinance fallback
- Transformation: daily squared log return.

## Earnings announcement indicator

- Source: Alpha Vantage `EARNINGS` endpoint.
- Files:
  - `data/external/alpha_vantage_earnings/raw/<TICKER>.json`: raw API responses.
  - `data/external/earnings_announcements.csv`: standardized `date,ticker,ea` file read by the feature pipeline.
- Transformation: quarterly `reportedDate` is converted to a daily firm-level indicator `ea=1`. Non-trading reported dates are aligned to the next available trading day for the same ticker.

## Optional inputs not downloaded automatically

- `firm_iv.csv`: user-supplied stock-level implied volatility with columns `date,ticker,iv`.
