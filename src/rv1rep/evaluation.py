from __future__ import annotations

import itertools
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def forecast_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(['dataset', 'horizon', 'ticker', 'model', 'scheme'], sort=True):
        e = g['actual_rv'].to_numpy() - g['forecast_rv'].to_numpy()
        y = g['actual_rv'].to_numpy()
        mse = float(np.mean(e ** 2))
        rows.append({
            'dataset': keys[0], 'horizon': keys[1], 'ticker': keys[2], 'model': keys[3], 'scheme': keys[4],
            'n_test': len(g),
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'mae': float(np.mean(np.abs(e))),
            'r2_oos_vs_mean': float(1.0 - np.sum(e ** 2) / np.sum((y - np.mean(y)) ** 2)) if len(g) > 1 else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Relative to HAR for each dataset/horizon/ticker.
        har = out[out['model'].str.upper() == 'HAR'][['dataset', 'horizon', 'ticker', 'mse']].rename(columns={'mse': 'har_mse'})
        out = out.merge(har, on=['dataset', 'horizon', 'ticker'], how='left')
        out['relative_mse_vs_har'] = out['mse'] / out['har_mse']
    return out


def cross_sectional_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return metrics.groupby(['dataset', 'horizon', 'model', 'scheme'], as_index=False).agg(
        n_assets=('ticker', 'nunique'),
        avg_mse=('mse', 'mean'),
        avg_rmse=('rmse', 'mean'),
        avg_mae=('mae', 'mean'),
        avg_rel_mse_vs_har=('relative_mse_vs_har', 'mean'),
        median_rel_mse_vs_har=('relative_mse_vs_har', 'median'),
    )


def diebold_mariano(loss_i: np.ndarray, loss_j: np.ndarray, alternative: str = 'greater', h: int = 1) -> Tuple[float, float]:
    """One-sided DM test.

    Tests H0: E(loss_i-loss_j)=0. `alternative='greater'` tests model j beats model i.
    Uses a simple Newey-West variance with lag h-1.
    """
    d = np.asarray(loss_i, dtype=float) - np.asarray(loss_j, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    mean_d = float(np.mean(d))
    lag = max(0, int(h) - 1)
    gamma0 = np.mean((d - mean_d) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        gamma = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        var += 2 * (1 - k / (lag + 1)) * gamma
    var = max(var, 1e-20)
    stat = mean_d / np.sqrt(var / n)
    if alternative == 'greater':
        p = 1 - stats.norm.cdf(stat)
    elif alternative == 'less':
        p = stats.norm.cdf(stat)
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def pairwise_relative_mse(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return paper-style pairwise relative MSE table and individual DM tests.

    Cell(row i, column j) is MSE_j / MSE_i averaged across tickers, matching the paper's reading:
    values below one indicate the column model beats the row benchmark.
    """
    matrix_rows = []
    dm_rows = []
    for (dataset, horizon), gd in pred.groupby(['dataset', 'horizon'], sort=True):
        models = sorted(gd['model'].unique())
        for row_model in models:
            row = {'dataset': dataset, 'horizon': horizon, 'benchmark_row': row_model}
            for col_model in models:
                ratios = []
                pvals = []
                for ticker, gt in gd.groupby('ticker'):
                    a = gt[gt['model'] == row_model][['date', 'actual_rv', 'forecast_rv']].rename(columns={'forecast_rv': 'f_i'})
                    b = gt[gt['model'] == col_model][['date', 'forecast_rv']].rename(columns={'forecast_rv': 'f_j'})
                    merged = a.merge(b, on='date', how='inner')
                    if merged.empty:
                        continue
                    li = (merged['actual_rv'] - merged['f_i']) ** 2
                    lj = (merged['actual_rv'] - merged['f_j']) ** 2
                    mse_i, mse_j = float(li.mean()), float(lj.mean())
                    if mse_i > 0:
                        ratios.append(mse_j / mse_i)
                    stat, p = diebold_mariano(li.to_numpy(), lj.to_numpy(), alternative='greater', h=int(horizon))
                    pvals.append(p)
                    dm_rows.append({'dataset': dataset, 'horizon': horizon, 'ticker': ticker, 'row_model': row_model, 'col_model': col_model, 'dm_stat': stat, 'p_value': p})
                row[col_model] = float(np.mean(ratios)) if ratios else np.nan
                row[f'{col_model}_dm_reject_10pct_share'] = float(np.mean(np.array(pvals) < 0.10)) if pvals else np.nan
            matrix_rows.append(row)
    return pd.DataFrame(matrix_rows), pd.DataFrame(dm_rows)
