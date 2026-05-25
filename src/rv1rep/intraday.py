from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bar_count_table(intraday: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, df in intraday.items():
        cnt = df.groupby('date').size().rename(ticker)
        frames.append(cnt)
    return pd.concat(frames, axis=1).sort_index()


def expected_intraday_times(start: str, bars_per_day: int, bar_interval_minutes: int) -> pd.Index:
    return pd.Index(_minutes_to_hhmm(m) for m in expected_intraday_minutes(start, bars_per_day, bar_interval_minutes))


def _hhmm_to_minutes(value: str) -> int:
    hour, minute = str(value).split(':', 1)
    return int(hour) * 60 + int(minute)


def _minutes_to_hhmm(value: int) -> str:
    return f'{value // 60:02d}:{value % 60:02d}'


def expected_intraday_minutes(start: str, bars_per_day: int, bar_interval_minutes: int) -> pd.Index:
    start_minute = _hhmm_to_minutes(start)
    return pd.Index(range(start_minute, start_minute + bars_per_day * bar_interval_minutes, bar_interval_minutes))


def validate_intraday_panel(
    intraday: Dict[str, pd.DataFrame],
    *,
    trading_start: str,
    trading_end: str,
    bars_per_day: int,
    bar_interval_minutes: int,
    timestamp_label: str = 'bar_start',
) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Validate intraday bars before realized-variance construction.

    Alpha Vantage intraday CSV timestamps label the beginning of each bar. For a
    regular 5-minute U.S. trading day this means 78 bars from 09:30 through 15:55.
    We keep only ticker-days that match the expected regular-hours grid exactly.
    Half-days and days with missing, duplicated, or off-grid bars are excluded.
    """
    if timestamp_label != 'bar_start':
        raise ValueError(f"Only timestamp_label='bar_start' is currently supported, got {timestamp_label!r}")
    if bars_per_day <= 0:
        raise ValueError(f'bars_per_day must be positive, got {bars_per_day}')
    if bar_interval_minutes <= 0:
        raise ValueError(f'bar_interval_minutes must be positive, got {bar_interval_minutes}')

    expected_minutes = expected_intraday_minutes(trading_start, bars_per_day, bar_interval_minutes)
    expected_last_minute = int(expected_minutes[-1])
    expected_last = _minutes_to_hhmm(expected_last_minute)
    trading_start_minute = _hhmm_to_minutes(trading_start)
    if trading_end != expected_last:
        logger.warning(
            'Configured trading_end=%s differs from expected final bar-start timestamp=%s for %d %d-minute bars.',
            trading_end,
            expected_last,
            bars_per_day,
            bar_interval_minutes,
        )

    valid: Dict[str, pd.DataFrame] = {}
    summary_rows = []
    expected_set = set(expected_minutes)
    bars_per_return = 5 // bar_interval_minutes if 5 % bar_interval_minutes == 0 else 1
    expected_log_returns = bars_per_day // bars_per_return
    for ticker, df in intraday.items():
        df = df.sort_values('timestamp').copy()
        if df.empty:
            valid[ticker] = df.copy()
            summary_rows.append(
                {
                    'ticker': ticker,
                    'total_days': 0,
                    'kept_full_regular_days': 0,
                    'dropped_days': 0,
                    'bad_bar_count_days': 0,
                    'duplicate_timestamp_days': 0,
                    'outside_regular_hours_days': 0,
                    'off_grid_days': 0,
                    'nonpositive_price_days': 0,
                    'timestamp_label': timestamp_label,
                    'expected_first_time': trading_start,
                    'expected_last_time': expected_last,
                    'expected_bars_per_day': bars_per_day,
                    'bar_interval_minutes': bar_interval_minutes,
                    'expected_log_returns': expected_log_returns,
                }
            )
            continue

        minute_of_day = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        checks = pd.DataFrame(
            {
                'date': df['date'],
                'timestamp': df['timestamp'],
                'minute_of_day': minute_of_day,
                'in_expected_grid': minute_of_day.isin(expected_set),
                'in_regular_hours': (minute_of_day >= trading_start_minute) & (minute_of_day <= expected_last_minute),
                'positive_prices': (df[['open', 'high', 'low', 'close']] > 0).all(axis=1),
            }
        )
        grouped = checks.groupby('date', sort=True).agg(
            n_bars=('timestamp', 'size'),
            n_unique_timestamps=('timestamp', 'nunique'),
            all_in_expected_grid=('in_expected_grid', 'all'),
            all_in_regular_hours=('in_regular_hours', 'all'),
            all_positive_prices=('positive_prices', 'all'),
        )
        count_ok = grouped['n_bars'].eq(bars_per_day)
        duplicate_ok = grouped['n_unique_timestamps'].eq(grouped['n_bars'])
        hours_ok = grouped['all_in_regular_hours']
        grid_ok = count_ok & duplicate_ok & grouped['all_in_expected_grid']
        prices_ok = grouped['all_positive_prices']
        keep_dates = grouped.index[count_ok & duplicate_ok & hours_ok & grid_ok & prices_ok]

        valid[ticker] = df[df['date'].isin(keep_dates)].sort_values('timestamp').copy()
        total_days = int(len(grouped))
        kept_days = int(len(keep_dates))
        bad_count = int((~count_ok).sum())
        bad_duplicates = int((~duplicate_ok).sum())
        bad_regular_hours = int((~hours_ok).sum())
        bad_grid = int((~grid_ok).sum())
        bad_prices = int((~prices_ok).sum())
        summary_rows.append(
            {
                'ticker': ticker,
                'total_days': total_days,
                'kept_full_regular_days': kept_days,
                'dropped_days': total_days - kept_days,
                'bad_bar_count_days': bad_count,
                'duplicate_timestamp_days': bad_duplicates,
                'outside_regular_hours_days': bad_regular_hours,
                'off_grid_days': bad_grid,
                'nonpositive_price_days': bad_prices,
                'timestamp_label': timestamp_label,
                'expected_first_time': trading_start,
                'expected_last_time': expected_last,
                'expected_bars_per_day': bars_per_day,
                'bar_interval_minutes': bar_interval_minutes,
                'expected_log_returns': expected_log_returns,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values('ticker')
    logger.info(
        'Intraday validation keeps %d/%d ticker-days.',
        int(summary['kept_full_regular_days'].sum()) if not summary.empty else 0,
        int(summary['total_days'].sum()) if not summary.empty else 0,
    )
    return valid, summary


def choose_full_days(intraday: Dict[str, pd.DataFrame], bars_per_day: int = 390, require_all_tickers: bool = True) -> pd.DatetimeIndex:
    counts = bar_count_table(intraday)
    if require_all_tickers:
        mask = (counts == bars_per_day).all(axis=1)
    else:
        mask = (counts == bars_per_day).any(axis=1)
    kept = counts.index[mask]
    logger.info('Full-day filter keeps %d/%d dates. require_all_tickers=%s', len(kept), len(counts), require_all_tickers)
    return pd.DatetimeIndex(kept)


def _daily_5min_returns(day_df: pd.DataFrame, expected_bars: int = 390, bar_interval_minutes: int = 1) -> np.ndarray:
    """Construct five-minute log returns from intraday OHLCV bars.

    For one-minute data, the function samples each fifth close. For five-minute data,
    the supplied closes are already five-minute interval endpoints. In both cases, the
    first return starts from the first bar's open, matching a regular open-to-close
    intraday realized-variance construction.
    """
    day_df = day_df.sort_values('timestamp')
    if len(day_df) != expected_bars:
        raise ValueError(f'Expected {expected_bars} bars, got {len(day_df)}')
    if bar_interval_minutes <= 0:
        raise ValueError(f'bar_interval_minutes must be positive, got {bar_interval_minutes}')
    if 5 % bar_interval_minutes != 0:
        raise ValueError(f'bar_interval_minutes must divide 5 for five-minute returns, got {bar_interval_minutes}')
    first_open = float(day_df['open'].iloc[0])
    closes = day_df['close'].to_numpy(dtype=float)
    bars_per_return = 5 // bar_interval_minutes
    block_end_closes = closes[bars_per_return - 1::bars_per_return]
    expected_returns = expected_bars // bars_per_return
    if len(block_end_closes) != expected_returns:
        raise ValueError(f'Expected {expected_returns} block closes, got {len(block_end_closes)}')
    sampled = np.r_[first_open, block_end_closes]
    return np.diff(np.log(sampled))


def compute_daily_realized_measures(
    intraday: Dict[str, pd.DataFrame],
    *,
    full_days: Iterable[pd.Timestamp] | None = None,
    expected_bars: int = 390,
    bar_interval_minutes: int = 1,
    rv_scale: float = 1.0,
) -> pd.DataFrame:
    rows = []
    full_days_set = set(pd.DatetimeIndex(full_days)) if full_days is not None else None
    for ticker, df in intraday.items():
        for date, g in df.groupby('date', sort=True):
            if full_days_set is not None and date not in full_days_set:
                continue
            if len(g) != expected_bars:
                continue
            r5 = _daily_5min_returns(g, expected_bars=expected_bars, bar_interval_minutes=bar_interval_minutes)
            rv = float(np.sum(r5 ** 2) * rv_scale)
            rvp = float(np.sum(r5[r5 > 0] ** 2) * rv_scale)
            rvn = float(np.sum(r5[r5 < 0] ** 2) * rv_scale)
            n = len(r5)
            rq = float((n / 3.0) * np.sum(r5 ** 4) * (rv_scale ** 2))
            open_ = float(g['open'].iloc[0])
            close = float(g['close'].iloc[-1])
            dollar_volume = float((g['close'] * g['volume']).sum())
            rows.append({
                'date': date,
                'ticker': ticker,
                'rv': rv,
                'rvp': rvp,
                'rvn': rvn,
                'rq': rq,
                'n_intraday_returns': n,
                'open': open_,
                'close': close,
                'oc_logret': float(np.log(close / open_)),
                'dollar_volume': dollar_volume,
            })
    out = pd.DataFrame(rows).sort_values(['ticker', 'date'])
    out['cc_logret'] = out.groupby('ticker')['close'].transform(lambda x: np.log(x).diff())
    logger.info('Computed daily realized measures: %d rows, %d tickers', len(out), out['ticker'].nunique())
    return out
