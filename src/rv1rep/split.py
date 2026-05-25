from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SplitIndex:
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def chronological_split(dates, train_frac: float = 0.70, val_frac: float = 0.10, fixed_train_days=None, fixed_val_days=None) -> SplitIndex:
    dates = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(dates)))).normalize()
    n = len(dates)
    if n < 100:
        raise ValueError(f'Not enough dates for a meaningful split: {n}')
    if fixed_train_days is not None and fixed_val_days is not None:
        train_n = int(fixed_train_days)
        val_n = int(fixed_val_days)
        if train_n + val_n >= n:
            raise ValueError('fixed_train_days + fixed_val_days must be smaller than available dates')
    else:
        train_n = int(n * train_frac)
        val_n = int(n * val_frac)
    train_dates = dates[:train_n]
    val_dates = dates[train_n: train_n + val_n]
    test_dates = dates[train_n + val_n:]
    return SplitIndex(train_dates=train_dates, val_dates=val_dates, test_dates=test_dates)


def subset_by_dates(df: pd.DataFrame, dates) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates).normalize()
    return df[pd.to_datetime(df['date']).dt.normalize().isin(idx)].copy()
