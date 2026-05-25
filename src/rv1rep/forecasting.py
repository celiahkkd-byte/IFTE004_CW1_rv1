from __future__ import annotations

import logging
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

from .features import make_model_frame
from .split import chronological_split, subset_by_dates
from .preprocessing import standardizer_from_config, enforce_positive_forecasts, insanity_filter
from .models import FittedModel, fit_sklearn_model

logger = logging.getLogger(__name__)


def _fit_one_asset_fixed(df_asset: pd.DataFrame, model_name: str, feature_cols: List[str], target_col: str, split, cfg: Dict) -> pd.DataFrame:
    train = subset_by_dates(df_asset, split.train_dates)
    val = subset_by_dates(df_asset, split.val_dates)
    test = subset_by_dates(df_asset, split.test_dates)
    if len(train) < 50 or len(val) < 20 or len(test) < 20:
        logger.warning('Small split for %s %s: train=%d val=%d test=%d', df_asset['ticker'].iloc[0], model_name, len(train), len(val), len(test))
    scaler = standardizer_from_config(cfg).fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    y_train = train[target_col]
    y_val = val[target_col]

    est, params = fit_sklearn_model(model_name, X_train, y_train, X_val, y_val, cfg, random_state=cfg['project']['random_seed'])

    target_is_log = model_name.upper() == 'LOGHAR'
    log_bias_var = 0.0
    if target_is_log:
        # Residual variance on train+validation for Jensen correction.
        xy = pd.concat([X_train, X_val])
        yy = pd.concat([y_train, y_val])
        resid = yy - est.predict(xy)
        log_bias_var = float(np.var(resid, ddof=1))

    raw_pred = np.asarray(est.predict(X_test), dtype=float)
    if target_is_log:
        raw_pred = np.exp(raw_pred + 0.5 * log_bias_var)

    in_sample_rv = pd.concat([train['rv'], val['rv']]).dropna()
    in_min = float(in_sample_rv.min())
    in_mean = float(in_sample_rv.mean())
    pred = enforce_positive_forecasts(raw_pred, in_min, cfg['estimation']['negative_forecast_policy'])
    if cfg['estimation']['insanity_filter']['enabled']:
        pred = insanity_filter(pred, in_mean, in_min, cfg['estimation']['insanity_filter']['max_multiple_of_in_sample_mean'])

    actual_rv_col = target_col.replace('target_log_rv_', 'target_rv_')
    out = test[['date', 'ticker', 'rv', 'oc_logret', 'cc_logret', actual_rv_col]].copy()
    out = out.rename(columns={actual_rv_col: 'actual_rv'})
    out['model'] = model_name
    out['forecast_rv'] = pred
    out['scheme'] = 'fixed'
    out['n_train'] = len(train)
    out['n_val'] = len(val)
    out['params'] = str(params)
    return out


def _fit_one_asset_rolling(df_asset: pd.DataFrame, model_name: str, feature_cols: List[str], target_col: str, split, cfg: Dict) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(pd.unique(df_asset['date']))).normalize()
    train_n = len(split.train_dates)
    val_n = len(split.val_dates)
    train_val_window = train_n + val_n
    test_dates = list(split.test_dates)
    rows = []
    last_est = None
    last_scaler = None
    last_params = None
    last_n_train = 0
    last_n_val = 0
    last_refit_i = -10**9
    refit_every = int(cfg['estimation'].get('ml_refit_every', 20))
    model_u = model_name.upper()
    non_tuned_models = {'HAR', 'HARX', 'LOGHAR', 'LEVHAR', 'SHAR', 'HARQ', 'BAGGING', 'RANDOMFOREST'}
    daily_refit_models = {'HAR', 'HARX', 'LOGHAR', 'LEVHAR', 'SHAR', 'HARQ', 'RIDGE', 'LASSO', 'ELASTICNET', 'ADAPTIVELASSO', 'POSTLASSO'}
    for i, test_date in enumerate(test_dates):
        pos = np.where(dates == test_date)[0]
        if len(pos) == 0:
            continue
        pos = int(pos[0])
        window_dates = dates[max(0, pos - train_val_window):pos]
        if len(window_dates) < train_val_window:
            continue
        refit = (model_u in daily_refit_models) or last_est is None or (i - last_refit_i >= refit_every)
        if refit:
            if model_u in non_tuned_models:
                # Non-tuned rolling models use the whole train+validation window as
                # one in-sample fit block. The empty validation slice is intentional:
                # it records n_val=0 because no hyperparameter selection is needed.
                train_dates = window_dates
                val_dates = window_dates[:0]
            else:
                train_dates = window_dates[:train_n]
                val_dates = window_dates[train_n:]
            train = subset_by_dates(df_asset, train_dates)
            val = subset_by_dates(df_asset, val_dates)
            scaler = standardizer_from_config(cfg).fit(train[feature_cols])
            X_train = scaler.transform(train[feature_cols])
            X_val = scaler.transform(val[feature_cols])
            y_train, y_val = train[target_col], val[target_col]
            est, params = fit_sklearn_model(model_name, X_train, y_train, X_val, y_val, cfg, random_state=cfg['project']['random_seed'])
            last_est, last_scaler, last_params, last_refit_i = est, scaler, params, i
            last_n_train, last_n_val = len(train), len(val)
            train_val_rv = pd.concat([train['rv'], val['rv']]).dropna()
            in_min = float(train_val_rv.min())
            in_mean = float(train_val_rv.mean())
            if model_u == 'LOGHAR':
                xy = pd.concat([X_train, X_val])
                yy = pd.concat([y_train, y_val])
                log_bias_var = float(np.var(yy - est.predict(xy), ddof=1))
            else:
                log_bias_var = 0.0
        one = df_asset[df_asset['date'] == test_date]
        if one.empty:
            continue
        X_one = last_scaler.transform(one[feature_cols])
        pred = np.asarray(last_est.predict(X_one), dtype=float)
        if model_u == 'LOGHAR':
            pred = np.exp(pred + 0.5 * log_bias_var)
        pred = enforce_positive_forecasts(pred, in_min, cfg['estimation']['negative_forecast_policy'])
        if cfg['estimation']['insanity_filter']['enabled']:
            pred = insanity_filter(pred, in_mean, in_min, cfg['estimation']['insanity_filter']['max_multiple_of_in_sample_mean'])
        actual_rv_col = target_col.replace('target_log_rv_', 'target_rv_')
        r = one[['date', 'ticker', 'rv', 'oc_logret', 'cc_logret', actual_rv_col]].copy()
        r = r.rename(columns={actual_rv_col: 'actual_rv'})
        r['model'] = model_name
        r['forecast_rv'] = pred
        r['scheme'] = 'rolling'
        r['n_train'] = last_n_train
        r['n_val'] = last_n_val
        r['params'] = str(last_params)
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_forecasts_for_dataset(panel: pd.DataFrame, dataset: str, horizon: int, model_names: Iterable[str], cfg: Dict) -> pd.DataFrame:
    all_predictions = []
    for model in model_names:
        frame, feature_cols, target_col = make_model_frame(panel, model, dataset, horizon)
        if not feature_cols:
            logger.warning('Skipping %s/%s/h%d because no feature columns are available.', dataset, model, horizon)
            continue
        logger.info('Running dataset=%s horizon=%s model=%s features=%s rows=%d', dataset, horizon, model, feature_cols, len(frame))
        for ticker, df_asset in frame.groupby('ticker', sort=True):
            split = chronological_split(
                df_asset['date'],
                train_frac=cfg['splitting']['train_frac'],
                val_frac=cfg['splitting']['val_frac'],
                fixed_train_days=cfg['splitting'].get('fixed_train_days'),
                fixed_val_days=cfg['splitting'].get('fixed_val_days'),
            )
            try:
                if cfg['estimation']['scheme'] == 'rolling':
                    preds = _fit_one_asset_rolling(df_asset, model, feature_cols, target_col, split, cfg)
                else:
                    preds = _fit_one_asset_fixed(df_asset, model, feature_cols, target_col, split, cfg)
                preds['dataset'] = dataset
                preds['horizon'] = horizon
                preds['features'] = ','.join(feature_cols)
                all_predictions.append(preds)
            except Exception as exc:
                logger.exception('Model failed: dataset=%s model=%s ticker=%s: %s', dataset, model, ticker, exc)
    if not all_predictions:
        return pd.DataFrame()
    return pd.concat(all_predictions, ignore_index=True)
