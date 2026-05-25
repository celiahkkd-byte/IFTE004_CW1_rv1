from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import ensure_dirs, load_config, override_config, project_path
from rv1rep.forecasting import run_forecasts_for_dataset
from rv1rep.utils import setup_logging

logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    return value.replace('/', '_').replace('\\', '_').replace(' ', '_')


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _resolve_output_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = ROOT / root
    return root.resolve()


def _model_output_path(root: Path, dataset: str, horizon: int, model: str) -> Path:
    return root / 'predictions' / 'by_model' / f'{_safe_name(dataset)}__h{horizon}__{_safe_name(model)}.csv'


def _ticker_output_path(root: Path, dataset: str, horizon: int, model: str, ticker: str) -> Path:
    return root / 'predictions' / 'by_ticker' / f'{_safe_name(dataset)}__h{horizon}__{_safe_name(model)}__{ticker.upper()}.csv'


KEY_COLS = ['date', 'ticker', 'dataset', 'horizon', 'model']
SORT_COLS = ['dataset', 'horizon', 'model', 'ticker', 'date']


def _sort_predictions(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(SORT_COLS).reset_index(drop=True)


def _validate_prediction_keys(df: pd.DataFrame, context: str) -> None:
    duplicates = int(df.duplicated(KEY_COLS).sum())
    if duplicates:
        raise RuntimeError(f'Duplicate prediction keys in {context}: {duplicates}')


def _completed_tickers(df: pd.DataFrame, dataset: str, horizon: int, model: str) -> set[str]:
    if df.empty:
        return set()
    mask = (
        (df['dataset'].astype(str) == str(dataset))
        & (df['horizon'].astype(int) == int(horizon))
        & (df['model'].astype(str) == str(model))
    )
    return set(df.loc[mask, 'ticker'].astype(str).str.upper().unique())


def _run_ticker_task(
    panel_ticker: pd.DataFrame,
    dataset: str,
    horizon: int,
    model: str,
    cfg: dict,
    ticker_path: Path,
) -> tuple[str, int]:
    """Worker task: run forecasts for one ticker and write its checkpoint file."""
    ticker = str(panel_ticker['ticker'].iloc[0]).upper()
    pred = run_forecasts_for_dataset(panel_ticker, dataset, int(horizon), [model], cfg)
    if pred.empty:
        return ticker, 0
    _atomic_write_csv(pred, ticker_path)
    return ticker, len(pred)


def _merge_ticker_files(root: Path, dataset: str, horizon: int, model: str, tickers: list) -> pd.DataFrame:
    parts = []
    for ticker in tickers:
        p = _ticker_output_path(root, dataset, horizon, model, ticker)
        if p.exists():
            parts.append(pd.read_csv(p, parse_dates=['date']))
    if not parts:
        return pd.DataFrame()
    return _sort_predictions(pd.concat(parts, ignore_index=True))


def _run_model_checkpoint_parallel(
    panel: pd.DataFrame,
    dataset: str,
    horizon: int,
    model: str,
    cfg: dict,
    model_path: Path,
    *,
    force: bool = False,
    n_jobs: int = 5,
) -> tuple[pd.DataFrame, str]:
    output_root = model_path.parent.parent.parent
    ticker_key = panel['ticker'].astype(str).str.upper()
    expected_tickers = sorted(ticker_key.unique())

    completed: set[str] = set()
    if not force:
        for ticker in expected_tickers:
            if _ticker_output_path(output_root, dataset, horizon, model, ticker).exists():
                completed.add(ticker)

    missing = [t for t in expected_tickers if t not in completed]
    if not missing:
        logger.info('Reusing complete by-model checkpoint: %s', model_path)
        return _merge_ticker_files(output_root, dataset, horizon, model, expected_tickers), 'reused'

    if completed:
        logger.info(
            'Resuming partial checkpoint: %s completed_tickers=%d remaining_tickers=%d',
            model_path, len(completed), len(missing),
        )

    tasks = [
        (
            panel[ticker_key == ticker].copy(),
            dataset,
            int(horizon),
            model,
            cfg,
            _ticker_output_path(output_root, dataset, horizon, model, ticker),
        )
        for ticker in missing
    ]

    effective_jobs = min(max(1, int(n_jobs)), len(tasks))
    logger.info(
        'Running %d tickers in parallel: dataset=%s horizon=%s model=%s n_jobs=%d',
        len(tasks), dataset, horizon, model, effective_jobs,
    )
    results = Parallel(n_jobs=effective_jobs, backend='loky')(
        delayed(_run_ticker_task)(*task) for task in tasks
    )
    for ticker, nrows in results:
        if nrows == 0:
            logger.warning('No predictions produced for dataset=%s horizon=%s model=%s ticker=%s', dataset, horizon, model, ticker)
        else:
            logger.info('Completed ticker checkpoint: dataset=%s horizon=%s model=%s ticker=%s rows=%d', dataset, horizon, model, ticker, nrows)

    final = _merge_ticker_files(output_root, dataset, horizon, model, expected_tickers)
    if final.empty:
        return final, 'empty'
    _validate_prediction_keys(final, str(model_path))
    completed_after = _completed_tickers(final, dataset, int(horizon), model)
    status = 'resumed' if completed else 'completed'
    if set(expected_tickers) - completed_after:
        status = 'partial'
    return final, status


def _load_panel(cfg: dict, tickers: list[str] | None) -> pd.DataFrame:
    panel_path = project_path(cfg, 'processed_dir') / 'forecasting_panel.csv'
    if not panel_path.exists():
        raise FileNotFoundError(f'Missing forecasting panel: {panel_path}. Run scripts/03_build_features.py first.')
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    if tickers:
        keep = {t.upper() for t in tickers}
        panel = panel[panel['ticker'].astype(str).str.upper().isin(keep)].copy()
        if panel.empty:
            raise ValueError(f'No rows remain after ticker filter: {sorted(keep)}')
    return panel


def _combine_existing_outputs(output_root: Path, expected_paths: list[Path]) -> pd.DataFrame:
    parts = []
    for path in expected_paths:
        if path.exists():
            parts.append(pd.read_csv(path, parse_dates=['date']))
    if not parts:
        return pd.DataFrame()
    final = _sort_predictions(pd.concat(parts, ignore_index=True))
    _validate_prediction_keys(final, 'combined by-model outputs')
    return final


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Checkpointed forecast runner: writes one CSV per dataset/horizon/model before final aggregation.'
    )
    ap.add_argument('--config', default=str(ROOT / 'config/default.yaml'))
    ap.add_argument('--output-dir', required=True, help='Separate output directory, e.g. outputs_rolling')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--horizons', nargs='*', type=int, default=None)
    ap.add_argument('--tickers', nargs='*', default=None, help='Optional ticker subset for smoke tests or partial reruns.')
    ap.add_argument('--scheme', choices=['fixed', 'rolling'], default=None)
    ap.add_argument('--skip-nn', action='store_true')
    ap.add_argument('--force', action='store_true', help='Recompute by-model files that already exist.')
    ap.add_argument('--aggregate-only', action='store_true', help='Only combine existing by-model files.')
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--allow-main-output-dir', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=5, help='Number of parallel ticker workers.')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = _resolve_output_root(args.output_dir)
    protected_dirs = {_resolve_output_root('outputs'), _resolve_output_root('outputs_full_nn')}
    if output_root in protected_dirs and not args.allow_main_output_dir:
        raise SystemExit('Refusing to write checkpointed forecast outputs into preserved main result directories.')
    cfg['paths']['output_dir'] = str(output_root)
    cfg = override_config(cfg, scheme=args.scheme, models=args.models, skip_nn=args.skip_nn)
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, 'output_dir') / 'logs' / '04_forecasts_checkpoints.log')

    if output_root.exists() and not args.allow_existing_output_dir:
        logger.warning('Output directory exists; existing by-model files will be reused: %s', output_root)

    datasets = args.datasets or list(cfg['experiments']['datasets'])
    horizons = args.horizons or [int(h) for h in cfg['experiments']['horizons']]
    models = list(cfg['models']['enabled'])
    if not models:
        raise SystemExit('No models selected.')

    panel = _load_panel(cfg, args.tickers)
    logger.info(
        'Checkpointed forecast run output_dir=%s scheme=%s datasets=%s horizons=%s models=%s tickers=%d',
        output_root,
        cfg['estimation']['scheme'],
        datasets,
        horizons,
        models,
        panel['ticker'].nunique(),
    )

    expected_paths = []
    manifest_rows = []
    for dataset in datasets:
        for horizon in horizons:
            for model in models:
                path = _model_output_path(output_root, dataset, int(horizon), model)
                expected_paths.append(path)
                if args.aggregate_only:
                    manifest_rows.append({
                        'dataset': dataset,
                        'horizon': int(horizon),
                        'model': model,
                        'status': 'missing' if not path.exists() else 'reused',
                        'rows': 0 if not path.exists() else len(pd.read_csv(path, usecols=['date'])),
                        'path': str(path),
                    })
                    continue
                try:
                    logger.info('Running checkpoint dataset=%s horizon=%s model=%s', dataset, horizon, model)
                    pred, status = _run_model_checkpoint_parallel(
                        panel,
                        dataset,
                        int(horizon),
                        model,
                        cfg,
                        path,
                        force=args.force,
                        n_jobs=args.n_jobs,
                    )
                    if pred.empty:
                        logger.warning('No predictions produced for dataset=%s horizon=%s model=%s', dataset, horizon, model)
                        manifest_rows.append({
                            'dataset': dataset,
                            'horizon': int(horizon),
                            'model': model,
                            'status': 'empty',
                            'rows': 0,
                            'path': str(path),
                        })
                        continue
                    _atomic_write_csv(pred, path)
                    logger.info('Wrote by-model checkpoint: %s rows=%d', path, len(pred))
                    manifest_rows.append({
                        'dataset': dataset,
                        'horizon': int(horizon),
                        'model': model,
                        'status': status,
                        'rows': len(pred),
                        'path': str(path),
                    })
                except Exception as exc:
                    logger.exception('Failed checkpoint dataset=%s horizon=%s model=%s: %s', dataset, horizon, model, exc)
                    manifest_rows.append({
                        'dataset': dataset,
                        'horizon': int(horizon),
                        'model': model,
                        'status': 'failed',
                        'rows': 0,
                        'path': str(path),
                        'error': str(exc),
                    })

    manifest = pd.DataFrame(manifest_rows)
    _atomic_write_csv(manifest, output_root / 'predictions' / 'by_model_manifest.csv')

    final = _combine_existing_outputs(output_root, expected_paths)
    if final.empty:
        raise SystemExit('No by-model prediction files were available to combine.')
    _atomic_write_csv(final, output_root / 'predictions' / 'model_predictions.csv')
    completed = sorted(final['model'].astype(str).unique())
    logger.info('Wrote combined predictions rows=%d models=%d model_names=%s', len(final), len(completed), completed)


if __name__ == '__main__':
    main()
