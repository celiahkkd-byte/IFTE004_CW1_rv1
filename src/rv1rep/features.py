from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _future_target(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    if horizon == 1:
        return s.shift(-1)
    if mode == 'future_average':
        # Mean of RV_{t+1},...,RV_{t+h} aligned at t.
        return s.shift(-1).rolling(horizon, min_periods=horizon).mean().shift(-(horizon - 1))
    if mode == 'future_sum':
        return s.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    if mode == 'future_endpoint':
        return s.shift(-horizon)
    raise ValueError(f'Unknown horizon target mode: {mode}')


def add_asset_features(daily: pd.DataFrame, weekly_window: int = 5, monthly_window: int = 22, eps: float = 1e-12) -> pd.DataFrame:
    df = daily.sort_values(['ticker', 'date']).copy()
    g = df.groupby('ticker', group_keys=False)
    df['rvd'] = df['rv']
    df['rvw'] = g['rv'].transform(lambda x: x.rolling(weekly_window, min_periods=weekly_window).mean())
    df['rvm'] = g['rv'].transform(lambda x: x.rolling(monthly_window, min_periods=monthly_window).mean())
    df['sqrt_rq_x_rvd'] = np.sqrt(np.maximum(df['rq'], eps)) * df['rvd']
    # Leverage variables: negative aggregated returns.
    df['rd'] = np.minimum(0.0, df['cc_logret'])
    df['rw'] = g['cc_logret'].transform(lambda x: np.minimum(0.0, x.rolling(weekly_window, min_periods=weekly_window).mean()))
    df['rm'] = g['cc_logret'].transform(lambda x: np.minimum(0.0, x.rolling(monthly_window, min_periods=monthly_window).mean()))
    # One-week momentum and dollar-volume first log difference.
    df['m1w'] = g['cc_logret'].transform(lambda x: x.rolling(weekly_window, min_periods=weekly_window).sum())
    df['dvol'] = g['dollar_volume'].transform(lambda x: np.log(np.maximum(x, eps)).diff())
    df['log_rv'] = np.log(np.maximum(df['rv'], eps))
    df['log_rvd'] = np.log(np.maximum(df['rvd'], eps))
    df['log_rvw'] = np.log(np.maximum(df['rvw'], eps))
    df['log_rvm'] = np.log(np.maximum(df['rvm'], eps))
    return df


def merge_external(asset_df: pd.DataFrame, external: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = asset_df.copy()
    out['date'] = pd.to_datetime(out['date']).dt.normalize()
    # Market/macro series are merged by date and forward-filled within dates available.
    for key, col in [('vix', 'vix'), ('epu', 'epu'), ('ads', 'ads'), ('hsi', 'hsi')]:
        ext = external.get(key, pd.DataFrame())
        if ext is not None and not ext.empty and col in ext.columns:
            tmp = ext[['date', col]].copy()
            tmp['date'] = pd.to_datetime(tmp['date']).dt.normalize()
            out = out.merge(tmp, on='date', how='left')
        else:
            logger.warning('External variable %s unavailable; omitted.', col)
    us3m = external.get('us3m', pd.DataFrame())
    if us3m is not None and not us3m.empty and 'us3m' in us3m.columns:
        tmp = us3m[['date', 'us3m']].copy()
        tmp['date'] = pd.to_datetime(tmp['date']).dt.normalize()
        tmp = tmp.sort_values('date')
        tmp['us3m'] = tmp['us3m'].ffill()
        tmp['us3m_diff'] = tmp['us3m'].diff()
        out = out.merge(tmp[['date', 'us3m_diff']], on='date', how='left')
    else:
        logger.warning('US3M unavailable; omitted.')
    # Optional stock-level IV and EA.
    iv = external.get('iv', pd.DataFrame())
    if iv is not None and not iv.empty:
        out = out.merge(iv[['date', 'ticker', 'iv']], on=['date', 'ticker'], how='left')
    earnings = external.get('earnings', pd.DataFrame())
    if earnings is not None and not earnings.empty:
        out = out.merge(earnings[['date', 'ticker', 'ea']], on=['date', 'ticker'], how='left')
        out['ea'] = out['ea'].fillna(0.0)
    else:
        out['ea'] = 0.0
    # Forward-fill macro variables on the trading calendar. Do this after merge to avoid look-ahead.
    for c in ['vix', 'epu', 'ads', 'hsi', 'us3m_diff']:
        if c in out.columns:
            # Same macro value for all tickers on a date, ffill by date first, then merge back.
            macro = out[['date', c]].drop_duplicates('date').sort_values('date').set_index('date')
            macro[c] = macro[c].ffill()
            out = out.drop(columns=[c]).merge(macro.reset_index(), on='date', how='left')
    return out


def add_targets(df: pd.DataFrame, horizons: Iterable[int], mode: str) -> pd.DataFrame:
    out = df.sort_values(['ticker', 'date']).copy()
    for h in horizons:
        out[f'target_rv_h{h}'] = out.groupby('ticker')['rv'].transform(lambda x: _future_target(x, h, mode))
        out[f'target_log_rv_h{h}'] = np.log(np.maximum(out[f'target_rv_h{h}'], 1e-12))
    return out


def feature_columns_for_model(model: str, dataset: str, columns: Iterable[str]) -> List[str]:
    cols = set(columns)
    model = model.upper()
    dataset = dataset.upper()

    # Concrete model-specific HAR features.
    if model in ['HAR', 'HARX', 'RIDGE', 'LASSO', 'ELASTICNET', 'ADAPTIVELASSO', 'POSTLASSO', 'BAGGING', 'RANDOMFOREST', 'GRADIENTBOOSTING'] or model.startswith('NN'):
        base = ['rvd', 'rvw', 'rvm']
    elif model == 'LOGHAR':
        base = ['log_rvd', 'log_rvw', 'log_rvm']
    elif model == 'LEVHAR':
        base = ['rvd', 'rvw', 'rvm', 'rd', 'rw', 'rm']
    elif model == 'SHAR':
        base = ['rvp', 'rvn', 'rvw', 'rvm']
    elif model == 'HARQ':
        base = ['rvd', 'sqrt_rq_x_rvd', 'rvw', 'rvm']
    else:
        raise ValueError(f'Unknown model {model}')

    if dataset.upper() == 'MHAR':
        return [c for c in base if c in cols]

    if model == 'HAR':
        # In the paper's MALL tables, HAR remains the basic HAR benchmark.
        # HAR-X is the extended linear model that receives the richer predictor set.
        return [c for c in base if c in cols]

    if model == 'LOGHAR':
        # Paper: in LogHAR, VIX and IV are log-transformed; other MALL add-ons keep
        # their paper transformations (e.g. differenced US3M and log-differenced $VOL).
        extra_candidates = ['log_iv', 'ea', 'm1w', 'dvol', 'log_vix', 'hsi', 'ads', 'us3m_diff', 'epu']
    else:
        # MALL-style add-ons. Omit IV if not provided. EA is included if available (it exists as 0 when no file is supplied).
        extra_candidates = ['iv', 'ea', 'm1w', 'dvol', 'vix', 'hsi', 'ads', 'us3m_diff', 'epu']
    extras = [c for c in extra_candidates if c in cols]
    return [c for c in base + extras if c in cols]


def prepare_feature_panel(
    daily: pd.DataFrame,
    external: Dict[str, pd.DataFrame],
    *,
    weekly_window: int = 5,
    monthly_window: int = 22,
    horizons: Iterable[int] = (1,),
    horizon_target_mode: str = 'future_average',
    eps: float = 1e-12,
) -> pd.DataFrame:
    df = add_asset_features(daily, weekly_window=weekly_window, monthly_window=monthly_window, eps=eps)
    df = merge_external(df, external)
    # Optional log transforms for LogHAR extended inputs.
    for c in ['vix', 'iv']:
        if c in df.columns:
            df[f'log_{c}'] = np.log(np.maximum(df[c], eps))
    df = add_targets(df, horizons, horizon_target_mode)
    df = df.sort_values(['ticker', 'date'])
    logger.info('Prepared feature panel: %d rows, columns=%d', len(df), df.shape[1])
    return df


def make_model_frame(panel: pd.DataFrame, model: str, dataset: str, horizon: int) -> Tuple[pd.DataFrame, List[str], str]:
    feature_cols = feature_columns_for_model(model, dataset, panel.columns)
    target_col = f'target_log_rv_h{horizon}' if model.upper() == 'LOGHAR' else f'target_rv_h{horizon}'
    actual_rv_col = f'target_rv_h{horizon}'
    needed = list(dict.fromkeys(['date', 'ticker', 'rv', 'oc_logret', 'cc_logret', actual_rv_col, target_col] + feature_cols))
    df = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df, feature_cols, target_col
