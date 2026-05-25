from __future__ import annotations

from typing import Callable, Iterable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def normalized_permutation_importance(model, X: pd.DataFrame, y: pd.Series, repeats: int = 20, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=repeats, random_state=random_state, scoring='neg_mean_squared_error')
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False)
    denom = imp['importance_mean'].clip(lower=0).sum()
    imp['importance_normalized'] = imp['importance_mean'].clip(lower=0) / denom if denom > 0 else np.nan
    return imp


def accumulated_local_effect(model_predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, feature: str, grid_size: int = 40) -> pd.DataFrame:
    """First-order accumulated local effects for one feature.

    This is a compact implementation for diagnostics. It partitions the selected feature into
    quantile intervals and averages finite differences in each interval.
    """
    if feature not in X.columns:
        raise ValueError(f'{feature} not in X')
    xj = X[feature]
    x_nonmissing = xj.dropna()
    observed_values = np.unique(x_nonmissing)
    if len(observed_values) < 2:
        return pd.DataFrame({'feature': feature, 'x': [], 'ale': []})
    if len(observed_values) == 2:
        # Binary indicators have only one finite-difference interval. Returning
        # an empty ALE curve would force their VI to be exactly zero.
        lo, hi = map(float, np.sort(observed_values))
        mask = xj.notna() & (xj >= lo) & (xj <= hi)
        if not mask.any() or hi <= lo:
            return pd.DataFrame({'feature': feature, 'x': [], 'ale': []})
        X_lo = X.loc[mask].copy()
        X_hi = X.loc[mask].copy()
        X_lo[feature] = lo
        X_hi[feature] = hi
        effect = float(np.mean(model_predict(X_hi) - model_predict(X_lo)))
        ale_raw = np.array([0.0, effect], dtype=float)
        ale_centered = ale_raw - np.mean(ale_raw)
        return pd.DataFrame({'feature': feature, 'x': [lo, hi], 'ale': ale_centered})

    qs = np.unique(np.quantile(x_nonmissing, np.linspace(0, 1, grid_size + 1)))
    if len(qs) < 2:
        return pd.DataFrame({'feature': feature, 'x': [], 'ale': []})
    effects = []
    centers = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (xj >= lo) & (xj <= hi)
        if not mask.any() or hi <= lo:
            effects.append(0.0)
            centers.append((lo + hi) / 2)
            continue
        X_lo = X.loc[mask].copy()
        X_hi = X.loc[mask].copy()
        X_lo[feature] = lo
        X_hi[feature] = hi
        diff = model_predict(X_hi) - model_predict(X_lo)
        effects.append(float(np.mean(diff)))
        centers.append((lo + hi) / 2)
    ale = np.cumsum(effects)
    ale = ale - np.mean(ale)
    return pd.DataFrame({'feature': feature, 'x': centers, 'ale': ale})
