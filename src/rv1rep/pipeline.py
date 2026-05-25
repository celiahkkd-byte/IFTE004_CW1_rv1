from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

from .config import project_path, ensure_dirs
from .io import load_intraday_txt
from .intraday import choose_full_days, compute_daily_realized_measures, bar_count_table, validate_intraday_panel
from .external_data import download_and_load_external
from .features import prepare_feature_panel
from .forecasting import run_forecasts_for_dataset
from .evaluation import forecast_metrics, cross_sectional_summary, pairwise_relative_mse
from .var_backtest import make_var_forecasts, var_backtest_summary
from .plotting import plot_relative_mse, plot_realized_volatility

logger = logging.getLogger(__name__)


def build_realized_measures(cfg: dict) -> pd.DataFrame:
    raw_zip = project_path(cfg, 'raw_zip') if cfg.get('paths', {}).get('raw_zip') else None
    raw_dir = project_path(cfg, 'raw_dir')
    intraday = load_intraday_txt(raw_zip, raw_dir, cfg['assets']['tickers'])
    bars_per_full_day = int(cfg['assets'].get('bars_per_full_day', cfg['assets'].get('minutes_per_full_day', 390)))
    bar_interval_minutes = int(cfg['assets'].get('bar_interval_minutes', 1))
    intraday, validation = validate_intraday_panel(
        intraday,
        trading_start=str(cfg['assets']['trading_start']),
        trading_end=str(cfg['assets']['trading_end']),
        bars_per_day=bars_per_full_day,
        bar_interval_minutes=bar_interval_minutes,
        timestamp_label=str(cfg['assets'].get('timestamp_label', 'bar_start')),
    )
    counts = bar_count_table(intraday)
    processed = project_path(cfg, 'processed_dir')
    processed.mkdir(parents=True, exist_ok=True)
    validation.to_csv(processed / 'intraday_validation_summary.csv', index=False)
    counts.to_csv(processed / 'intraday_bar_counts.csv')
    require_full_day = bool(cfg['assets'].get('require_full_day_bars', cfg['assets'].get('require_full_390_bars', True)))
    full_days = choose_full_days(
        intraday,
        bars_per_day=bars_per_full_day,
        require_all_tickers=bool(cfg['assets']['require_all_tickers_full_day']),
    )
    daily = compute_daily_realized_measures(
        intraday,
        full_days=full_days if require_full_day else None,
        expected_bars=bars_per_full_day,
        bar_interval_minutes=bar_interval_minutes,
        rv_scale=float(cfg['assets'].get('rv_scale', 1.0)),
    )
    daily.to_csv(processed / 'daily_realized_measures.csv', index=False)
    plot_realized_volatility(daily, project_path(cfg, 'output_dir') / 'figures' / 'realized_variance.png')
    return daily


def build_features(cfg: dict, daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    processed = project_path(cfg, 'processed_dir')
    if daily is None:
        p = processed / 'daily_realized_measures.csv'
        if not p.exists():
            daily = build_realized_measures(cfg)
        else:
            daily = pd.read_csv(p, parse_dates=['date'])
    external = download_and_load_external(cfg)
    fe = cfg['feature_engineering']
    panel = prepare_feature_panel(
        daily,
        external,
        weekly_window=int(fe['weekly_window']),
        monthly_window=int(fe['monthly_window']),
        horizons=fe['horizons'],
        horizon_target_mode=fe.get('horizon_target_mode', 'future_average'),
        eps=float(fe.get('eps', 1e-12)),
    )
    panel.to_csv(processed / 'forecasting_panel.csv', index=False)
    return panel


def run_forecast_experiments(cfg: dict, panel: Optional[pd.DataFrame] = None, models: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if panel is None:
        p = project_path(cfg, 'processed_dir') / 'forecasting_panel.csv'
        if not p.exists():
            panel = build_features(cfg)
        else:
            panel = pd.read_csv(p, parse_dates=['date'])
    model_names = list(models) if models is not None else cfg['models']['enabled']
    all_preds = []
    for dataset in cfg['experiments']['datasets']:
        for horizon in cfg['experiments']['horizons']:
            pred = run_forecasts_for_dataset(panel, dataset, int(horizon), model_names, cfg)
            if not pred.empty:
                all_preds.append(pred)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    out_dir = project_path(cfg, 'output_dir')
    (out_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / 'predictions' / 'model_predictions.csv', index=False)
    return predictions


def evaluate_predictions(cfg: dict, predictions: Optional[pd.DataFrame] = None) -> dict:
    out_dir = project_path(cfg, 'output_dir')
    if predictions is None:
        p = out_dir / 'predictions' / 'model_predictions.csv'
        predictions = pd.read_csv(p, parse_dates=['date'])
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics = forecast_metrics(predictions)
    summary = cross_sectional_summary(metrics)
    rel_matrix, dm = pairwise_relative_mse(predictions)
    metrics.to_csv(tables_dir / 'forecast_metrics_by_asset.csv', index=False)
    summary.to_csv(tables_dir / 'forecast_summary_cross_section.csv', index=False)
    rel_matrix.to_csv(tables_dir / 'pairwise_relative_mse_matrix.csv', index=False)
    dm.to_csv(tables_dir / 'diebold_mariano_tests.csv', index=False)
    plot_relative_mse(summary, out_dir / 'figures' / 'relative_mse_PLACEHOLDER.png')
    result = {'metrics': metrics, 'summary': summary, 'relative_mse': rel_matrix, 'dm': dm}
    if cfg.get('var_backtest', {}).get('enabled', False):
        alpha = float(cfg['var_backtest']['alpha'])
        var_df = make_var_forecasts(predictions, alpha=alpha)
        var_summary = var_backtest_summary(var_df, alpha=alpha)
        var_df.to_csv(out_dir / 'predictions' / 'var_forecasts.csv', index=False)
        var_summary.to_csv(tables_dir / 'var_backtest_summary.csv', index=False)
        result['var'] = var_summary
    return result


def run_all(cfg: dict, models: Optional[Iterable[str]] = None) -> dict:
    ensure_dirs(cfg)
    daily = build_realized_measures(cfg)
    panel = build_features(cfg, daily)
    predictions = run_forecast_experiments(cfg, panel, models=models)
    results = evaluate_predictions(cfg, predictions)
    return {'daily': daily, 'panel': panel, 'predictions': predictions, **results}
