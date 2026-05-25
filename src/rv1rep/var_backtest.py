from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats


def quantile_loss(r: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """Tick/check loss for VaR quantile q at tail probability alpha."""
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    return (alpha - (r < q).astype(float)) * (r - q)


def kupiec_uc_test(hits: np.ndarray, alpha: float) -> Tuple[float, float]:
    hits = np.asarray(hits, dtype=int)
    n = len(hits)
    x = hits.sum()
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    if phat in [0, 1]:
        phat = min(max(phat, 1e-10), 1 - 1e-10)
    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll_alt = (n - x) * np.log(1 - phat) + x * np.log(phat)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_independence_test(hits: np.ndarray) -> Tuple[float, float]:
    hits = np.asarray(hits, dtype=int)
    if len(hits) < 2:
        return np.nan, np.nan
    h0, h1 = hits[:-1], hits[1:]
    n00 = np.sum((h0 == 0) & (h1 == 0))
    n01 = np.sum((h0 == 0) & (h1 == 1))
    n10 = np.sum((h0 == 1) & (h1 == 0))
    n11 = np.sum((h0 == 1) & (h1 == 1))
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    def loglik(p, a, b):
        p = min(max(p, 1e-10), 1 - 1e-10)
        return a * np.log(1 - p) + b * np.log(p)
    ll_ind = loglik(pi0, n00, n01) + loglik(pi1, n10, n11)
    ll_null = loglik(pi, n00 + n10, n01 + n11)
    lr = -2 * (ll_null - ll_ind)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def make_var_forecasts(pred: pd.DataFrame, alpha: float = 0.05, empirical: bool = True) -> pd.DataFrame:
    """Build simple left-tail VaR forecasts from RV forecasts.

    Parametric version: VaR_t = z_alpha * sqrt(RVhat_t).
    Empirical version would ideally use train-standardized residuals. Since the prediction table only
    contains test rows, this function uses the normal version unless a pre-calibrated residual quantile is
    supplied in a future extension. It is still useful for coursework diagnostics.
    """
    z = stats.norm.ppf(alpha)
    out = pred.copy()
    out['var_forecast'] = z * np.sqrt(np.maximum(out['forecast_rv'], 0.0))
    out['hit'] = (out['oc_logret'] < out['var_forecast']).astype(int)
    out['var_loss'] = quantile_loss(out['oc_logret'].to_numpy(), out['var_forecast'].to_numpy(), alpha)
    return out


def var_backtest_summary(var_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for keys, g in var_df.groupby(['dataset', 'horizon', 'ticker', 'model'], sort=True):
        lr_uc, p_uc = kupiec_uc_test(g['hit'].to_numpy(), alpha)
        lr_ind, p_ind = christoffersen_independence_test(g['hit'].to_numpy())
        rows.append({
            'dataset': keys[0], 'horizon': keys[1], 'ticker': keys[2], 'model': keys[3],
            'n': len(g),
            'exceedance_rate': float(g['hit'].mean()),
            'mean_var_loss': float(g['var_loss'].mean()),
            'kupiec_lr': lr_uc, 'kupiec_p': p_uc,
            'christoffersen_ind_lr': lr_ind, 'christoffersen_ind_p': p_ind,
        })
    return pd.DataFrame(rows)
