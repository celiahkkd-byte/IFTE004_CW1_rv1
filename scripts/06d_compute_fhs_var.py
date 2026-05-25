from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import load_config, project_path
from rv1rep.var_backtest import christoffersen_independence_test, kupiec_uc_test, quantile_loss

LOGGER = logging.getLogger(__name__)

REQUIRED_PREDICTION_COLUMNS = {
    'date',
    'ticker',
    'dataset',
    'horizon',
    'model',
    'oc_logret',
    'forecast_rv',
}

PROTECTED_OUTPUT_DIRS = {
    'outputs',
    'outputs_full_nn',
    'outputs_rolling',
    'outputs_nn30_checkpointed',
    'outputs_bagging_checkpointed',
    'outputs_bagging_checkpointed_20260520',
    'outputs_final',
    'outputs_final_core_no_bagging_gb_20260520',
    'outputs_nn1_single_seed_20260520',
    'outputs_ale_core_no_bagging_gb_20260520',
    'outputs_rv_decile_core_no_bagging_gb_20260520',
    'outputs_variable_importance_core_no_bagging_gb_20260520',
}

METHOD = 'filtered_historical_simulation'


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def safe_name(value: str) -> str:
    return str(value).replace('/', '_').replace('\\', '_').replace(' ', '_')


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def setup_file_logging(output_dir: Path) -> None:
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / '06d_fhs_var.log'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def assert_output_dir(output_dir: Path, allow_existing: bool, allow_protected: bool) -> None:
    protected = {resolve_path(p) for p in PROTECTED_OUTPUT_DIRS}
    if output_dir in protected and not allow_protected:
        raise SystemExit(
            f'Refusing to write FHS VaR outputs into protected directory: {output_dir}. '
            'Use a new isolated output directory.'
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise SystemExit(
            f'Output directory already exists and is not empty: {output_dir}. '
            'Use --allow-existing-output-dir only for checkpoint resume.'
        )


def load_predictions(path: Path, datasets: Iterable[str] | None, horizons: Iterable[int] | None, models: Iterable[str] | None) -> pd.DataFrame:
    pred = pd.read_csv(path, parse_dates=['date'])
    missing = REQUIRED_PREDICTION_COLUMNS - set(pred.columns)
    if missing:
        raise ValueError(f'Missing required prediction columns: {sorted(missing)}')
    pred['ticker'] = pred['ticker'].astype(str).str.upper()
    pred['dataset'] = pred['dataset'].astype(str)
    pred['model'] = pred['model'].astype(str)
    pred['horizon'] = pred['horizon'].astype(int)
    pred['forecast_rv'] = pd.to_numeric(pred['forecast_rv'], errors='coerce')
    pred['oc_logret'] = pd.to_numeric(pred['oc_logret'], errors='coerce')
    pred = pred.dropna(subset=['date', 'ticker', 'dataset', 'horizon', 'model', 'forecast_rv', 'oc_logret'])
    if datasets:
        pred = pred[pred['dataset'].isin(list(datasets))]
    if horizons:
        pred = pred[pred['horizon'].isin([int(h) for h in horizons])]
    if models:
        pred = pred[pred['model'].isin(list(models))]
    if pred.empty:
        raise ValueError('No prediction rows remain after filters.')
    return pred.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)


def load_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_path(cfg, 'processed_dir') / 'forecasting_panel.csv'
    if not panel_path.exists():
        raise FileNotFoundError(f'Missing forecasting panel: {panel_path}')
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    required = {'date', 'ticker', 'rv', 'oc_logret'}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f'Missing required panel columns: {sorted(missing)}')
    panel['ticker'] = panel['ticker'].astype(str).str.upper()
    panel['rv'] = pd.to_numeric(panel['rv'], errors='coerce')
    panel['oc_logret'] = pd.to_numeric(panel['oc_logret'], errors='coerce')
    return panel.dropna(subset=['date', 'ticker', 'rv', 'oc_logret'])


def empirical_residual_quantile(
    panel: pd.DataFrame,
    ticker: str,
    first_test_date: pd.Timestamp,
    alpha: float,
    min_calibration_obs: int,
) -> tuple[float, int]:
    calibration = panel[(panel['ticker'] == ticker) & (panel['date'] < first_test_date)].copy()
    calibration = calibration[(calibration['rv'] > 0) & np.isfinite(calibration['rv']) & np.isfinite(calibration['oc_logret'])]
    if len(calibration) < min_calibration_obs:
        raise ValueError(
            f'Insufficient pre-test calibration observations for ticker={ticker}: '
            f'{len(calibration)} < {min_calibration_obs}'
        )
    standardized = calibration['oc_logret'].to_numpy(dtype=float) / np.sqrt(calibration['rv'].to_numpy(dtype=float))
    standardized = standardized[np.isfinite(standardized)]
    if len(standardized) < min_calibration_obs:
        raise ValueError(
            f'Insufficient finite standardized residuals for ticker={ticker}: '
            f'{len(standardized)} < {min_calibration_obs}'
        )
    return float(np.quantile(standardized, alpha, method='linear')), int(len(standardized))


def checkpoint_path(output_dir: Path, dataset: str, horizon: int, model: str, ticker: str) -> Path:
    return (
        output_dir
        / 'checkpoints'
        / safe_name(dataset)
        / f'h{int(horizon)}'
        / safe_name(model)
        / f'{safe_name(ticker)}.csv'
    )


def compute_group_var(
    group: pd.DataFrame,
    panel: pd.DataFrame,
    alpha: float,
    min_calibration_obs: int,
) -> pd.DataFrame:
    group = group.sort_values('date').copy()
    first_test_date = pd.Timestamp(group['date'].min())
    ticker = str(group['ticker'].iloc[0]).upper()
    residual_quantile, calibration_n = empirical_residual_quantile(
        panel=panel,
        ticker=ticker,
        first_test_date=first_test_date,
        alpha=alpha,
        min_calibration_obs=min_calibration_obs,
    )
    vol = np.sqrt(np.maximum(group['forecast_rv'].to_numpy(dtype=float), 0.0))
    var_forecast = residual_quantile * vol
    returns = group['oc_logret'].to_numpy(dtype=float)
    hit = (returns < var_forecast).astype(int)
    out = group[['date', 'ticker', 'dataset', 'horizon', 'model', 'oc_logret', 'forecast_rv']].copy()
    out = out.rename(columns={'oc_logret': 'return'})
    out['var_forecast'] = var_forecast
    out['hit'] = hit
    out['var_loss'] = quantile_loss(returns, var_forecast, alpha)
    out['alpha'] = alpha
    out['method'] = METHOD
    out['calibration_source'] = 'pretest_realized_rv_standardized_returns'
    out['calibration_n'] = calibration_n
    out['residual_quantile'] = residual_quantile
    return out


def combine_checkpoints(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / 'checkpoints').glob('*/*/*/*.csv'))
    if not paths:
        raise RuntimeError(f'No FHS VaR checkpoints found in {output_dir / "checkpoints"}')
    parts = [pd.read_csv(path, parse_dates=['date']) for path in paths]
    combined = pd.concat(parts, ignore_index=True)
    key_cols = ['date', 'ticker', 'dataset', 'horizon', 'model']
    duplicates = int(combined.duplicated(key_cols).sum())
    if duplicates:
        raise RuntimeError(f'Duplicate VaR forecast keys after combining checkpoints: {duplicates}')
    return combined.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)


def summarize_var(var_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for keys, g in var_df.groupby(['dataset', 'horizon', 'ticker', 'model'], sort=True):
        g = g.sort_values('date')
        lr_uc, p_uc = kupiec_uc_test(g['hit'].to_numpy(dtype=int), alpha)
        lr_ind, p_ind = christoffersen_independence_test(g['hit'].to_numpy(dtype=int))
        rows.append({
            'dataset': keys[0],
            'horizon': int(keys[1]),
            'ticker': keys[2],
            'model': keys[3],
            'n': int(len(g)),
            'alpha': float(alpha),
            'exceedance_rate': float(g['hit'].mean()),
            'mean_var_loss': float(g['var_loss'].mean()),
            'kupiec_lr': lr_uc,
            'kupiec_p': p_uc,
            'christoffersen_ind_lr': lr_ind,
            'christoffersen_ind_p': p_ind,
            'method': METHOD,
            'calibration_source': str(g['calibration_source'].iloc[0]),
            'calibration_n': int(g['calibration_n'].iloc[0]),
            'residual_quantile': float(g['residual_quantile'].iloc[0]),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Checkpointed paper-style filtered historical simulation VaR diagnostics.')
    ap.add_argument('--config', default=str(ROOT / 'config' / 'paper_core_rolling.yaml'))
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--horizons', nargs='*', type=int, default=[1], help='Default is h=1, matching the paper VaR application.')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--min-calibration-obs', type=int, default=500)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--allow-protected-output-dir', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise SystemExit('--alpha must be between 0 and 1.')

    output_dir = resolve_path(args.output_dir)
    assert_output_dir(output_dir, args.allow_existing_output_dir, args.allow_protected_output_dir)
    setup_file_logging(output_dir)

    cfg = load_config(args.config)
    pred = load_predictions(
        resolve_path(args.predictions),
        datasets=args.datasets,
        horizons=args.horizons,
        models=args.models,
    )
    if args.tickers:
        keep = {t.upper() for t in args.tickers}
        pred = pred[pred['ticker'].isin(keep)].copy()
        if pred.empty:
            raise SystemExit(f'No prediction rows remain after ticker filter: {sorted(keep)}')
    panel = load_panel(cfg)

    LOGGER.info(
        'Task H FHS VaR output_dir=%s rows=%d alpha=%s datasets=%s horizons=%s models=%d tickers=%d',
        output_dir,
        len(pred),
        args.alpha,
        sorted(pred['dataset'].unique()),
        sorted(pred['horizon'].unique()),
        pred['model'].nunique(),
        pred['ticker'].nunique(),
    )
    atomic_write_json(
        {
            'created_at': utc_now(),
            'command': sys.argv,
            'config': str(resolve_path(args.config)),
            'predictions': str(resolve_path(args.predictions)),
            'output_dir': str(output_dir),
            'alpha': args.alpha,
            'horizons': [int(h) for h in sorted(pred['horizon'].unique())],
            'datasets': sorted(pred['dataset'].unique()),
            'models': sorted(pred['model'].unique()),
            'calibration_source': 'pretest_realized_rv_standardized_returns',
            'method': METHOD,
        },
        output_dir / 'run_provenance.json',
    )

    manifest_rows = []
    group_cols = ['dataset', 'horizon', 'model', 'ticker']
    for keys, group in pred.groupby(group_cols, sort=True):
        dataset, horizon, model, ticker = keys
        path = checkpoint_path(output_dir, str(dataset), int(horizon), str(model), str(ticker))
        if path.exists() and not args.force:
            manifest_rows.append({
                'dataset': dataset,
                'horizon': int(horizon),
                'model': model,
                'ticker': ticker,
                'status': 'reused',
                'rows': len(pd.read_csv(path, usecols=['date'])),
                'path': str(path),
            })
            continue
        if args.aggregate_only:
            manifest_rows.append({
                'dataset': dataset,
                'horizon': int(horizon),
                'model': model,
                'ticker': ticker,
                'status': 'missing',
                'rows': 0,
                'path': str(path),
            })
            continue
        LOGGER.info('Computing FHS checkpoint dataset=%s h=%s model=%s ticker=%s', dataset, horizon, model, ticker)
        var_group = compute_group_var(group, panel, args.alpha, args.min_calibration_obs)
        atomic_write_csv(var_group, path)
        manifest_rows.append({
            'dataset': dataset,
            'horizon': int(horizon),
            'model': model,
            'ticker': ticker,
            'status': 'completed',
            'rows': len(var_group),
            'path': str(path),
        })

    manifest = pd.DataFrame(manifest_rows)
    atomic_write_csv(manifest, output_dir / 'tables' / 'fhs_var_checkpoint_manifest.csv')

    var_df = combine_checkpoints(output_dir)
    summary = summarize_var(var_df, args.alpha)
    atomic_write_csv(var_df, output_dir / 'predictions' / 'var_forecasts_fhs.csv')
    atomic_write_csv(summary, output_dir / 'tables' / 'var_backtest_fhs_summary.csv')
    LOGGER.info('Wrote FHS VaR forecasts rows=%d summary_rows=%d', len(var_df), len(summary))


if __name__ == '__main__':
    main()
