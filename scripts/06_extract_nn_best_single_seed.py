from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import load_config
from rv1rep.preprocessing import enforce_positive_forecasts, insanity_filter
from rv1rep.utils import require_columns, setup_logging

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    'date',
    'ticker',
    'rv',
    'oc_logret',
    'cc_logret',
    'actual_rv',
    'model',
    'forecast_rv',
    'scheme',
    'n_train',
    'n_val',
    'params',
    'dataset',
    'horizon',
    'features',
]

SEED_REQUIRED_COLUMNS = [
    'date',
    'ticker',
    'rv',
    'oc_logret',
    'cc_logret',
    'actual_rv',
    'forecast_raw',
    'seed',
    'val_mse',
    'model',
    'dataset',
    'horizon',
    'features',
    'scheme',
    'n_train',
    'n_val',
    'params',
    'in_sample_min_rv',
    'in_sample_mean_rv',
]

PROTECTED_OUTPUT_DIRS = {
    'outputs',
    'outputs_full_nn',
    'outputs_rolling',
    'outputs_nn30_checkpointed',
    'outputs_bagging_checkpointed',
    'outputs_final',
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_writable_new_output_dir(output_dir: Path, allow_existing_empty: bool) -> None:
    protected = {_resolve(p) for p in PROTECTED_OUTPUT_DIRS}
    for p in protected:
        if output_dir == p or _is_relative_to(output_dir, p):
            raise SystemExit(f'Refusing to write derived NN1 output inside protected result directory: {output_dir}')
    if output_dir.exists():
        files = [p for p in output_dir.rglob('*') if p.is_file()]
        if files or not allow_existing_empty:
            raise SystemExit(
                f'Output directory already exists: {output_dir}. '
                'Use a fresh directory name; existing result folders are treated as read-only.'
            )


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def _path_stats(path: Path, rows: int | None = None) -> dict:
    stat = path.stat()
    out = {
        'path': str(path),
        'size_bytes': stat.st_size,
        'mtime_utc': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if rows is not None:
        out['rows'] = int(rows)
    return out


def _read_seed_header(path: Path) -> dict:
    head = pd.read_csv(path, nrows=1)
    require_columns(head, ['seed', 'val_mse', 'model', 'dataset', 'horizon', 'ticker'], str(path))
    return {
        'seed': int(head['seed'].iloc[0]),
        'val_mse': float(head['val_mse'].iloc[0]),
        'model': str(head['model'].iloc[0]),
        'dataset': str(head['dataset'].iloc[0]),
        'horizon': int(head['horizon'].iloc[0]),
        'ticker': str(head['ticker'].iloc[0]),
    }


def _seed_dir(input_root: Path, dataset: str, horizon: int, model: str, ticker: str) -> Path:
    return input_root / 'nn_seed_predictions' / dataset / f'h{int(horizon)}' / model / ticker


def _select_best_seed(seed_files: Iterable[Path]) -> tuple[Path, dict, int]:
    candidates = []
    for path in sorted(seed_files):
        meta = _read_seed_header(path)
        candidates.append((meta['val_mse'], meta['seed'], path, meta))
    if not candidates:
        raise ValueError('No seed files provided.')
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, path, meta = candidates[0]
    return path, meta, len(candidates)


def _make_single_seed_prediction(path: Path, output_model: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    require_columns(df, SEED_REQUIRED_COLUMNS, str(path))
    if set(df['scheme'].astype(str)) != {'fixed'}:
        raise ValueError(f'Seed file must be fixed-scheme only: {path}')
    if df['val_mse'].nunique(dropna=False) != 1:
        raise ValueError(f'Seed file has inconsistent val_mse values: {path}')
    seed = int(df['seed'].iloc[0])
    val_mse = float(df['val_mse'].iloc[0])
    in_min = float(df['in_sample_min_rv'].iloc[0])
    in_mean = float(df['in_sample_mean_rv'].iloc[0])
    pred = enforce_positive_forecasts(
        df['forecast_raw'].to_numpy(dtype=float),
        in_min,
        cfg['estimation']['negative_forecast_policy'],
    )
    filt_cfg = cfg['estimation'].get('insanity_filter', {})
    if bool(filt_cfg.get('enabled', False)):
        pred = insanity_filter(pred, in_mean, in_min, float(filt_cfg['max_multiple_of_in_sample_mean']))

    out = df[['date', 'ticker', 'rv', 'oc_logret', 'cc_logret', 'actual_rv']].copy()
    out['model'] = output_model
    out['forecast_rv'] = pred
    out['scheme'] = 'fixed'
    out['n_train'] = int(df['n_train'].iloc[0])
    out['n_val'] = int(df['n_val'].iloc[0])
    out['params'] = json.dumps(
        {
            'method': 'single_best_seed_by_validation_mse',
            'source_model': str(df['model'].iloc[0]),
            'selected_seed': seed,
            'selected_val_mse': val_mse,
            'source_file': str(path),
        },
        sort_keys=True,
    )
    out['dataset'] = str(df['dataset'].iloc[0])
    out['horizon'] = int(df['horizon'].iloc[0])
    out['features'] = str(df['features'].iloc[0])
    return out[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Extract paper-style NN^1 single-best-seed forecasts from checkpointed NN seed files.'
    )
    ap.add_argument('--config', default=str(ROOT / 'config/paper_core_rolling.yaml'))
    ap.add_argument('--input-dir', default='outputs_nn30_checkpointed')
    ap.add_argument('--output-dir', required=True, help='Fresh derived output directory, e.g. outputs_nn1_single_seed_20260520')
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--horizons', nargs='*', type=int, default=None)
    ap.add_argument('--models', nargs='*', default=['NN1', 'NN2', 'NN3', 'NN4'])
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--allow-existing-empty-output-dir', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    _assert_writable_new_output_dir(output_dir, args.allow_existing_empty_output_dir)
    setup_logging(output_dir / 'logs' / '06_extract_nn_best_single_seed.log')

    datasets = args.datasets or list(cfg['experiments']['datasets'])
    horizons = args.horizons or [int(h) for h in cfg['experiments']['horizons']]
    models = list(args.models)
    tickers = [t.upper() for t in (args.tickers or cfg['assets']['tickers'])]

    logger.info(
        'Extracting NN1 single-seed forecasts input_dir=%s output_dir=%s datasets=%s horizons=%s models=%s tickers=%d',
        input_dir,
        output_dir,
        datasets,
        horizons,
        models,
        len(tickers),
    )

    parts = []
    selection_rows = []
    selected_file_stats = []
    for dataset in datasets:
        for horizon in horizons:
            for model in models:
                output_model = f'{model}_1'
                for ticker in tickers:
                    directory = _seed_dir(input_dir, dataset, int(horizon), model, ticker)
                    seed_files = sorted(directory.glob('seed_*.csv'))
                    if not seed_files:
                        raise FileNotFoundError(f'Missing seed files for {dataset}/h{horizon}/{model}/{ticker}: {directory}')
                    selected_path, meta, n_candidates = _select_best_seed(seed_files)
                    pred = _make_single_seed_prediction(selected_path, output_model, cfg)
                    parts.append(pred)
                    selection_rows.append(
                        {
                            'dataset': dataset,
                            'horizon': int(horizon),
                            'source_model': model,
                            'output_model': output_model,
                            'ticker': ticker,
                            'selected_seed': meta['seed'],
                            'selected_val_mse': meta['val_mse'],
                            'n_candidate_seeds': n_candidates,
                            'selected_file': str(selected_path),
                            'rows': len(pred),
                        }
                    )
                    selected_file_stats.append(_path_stats(selected_path, rows=len(pred)))

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)
    duplicates = int(out.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum())
    if duplicates:
        raise RuntimeError(f'Duplicate NN1 prediction keys: {duplicates}')
    expected_models = {f'{m}_1' for m in models}
    got_models = set(out['model'].astype(str))
    if got_models != expected_models:
        raise RuntimeError(f'Unexpected NN1 model set: got={sorted(got_models)}, expected={sorted(expected_models)}')
    if set(out['scheme'].astype(str)) != {'fixed'}:
        raise RuntimeError(f'NN1 output must be fixed-scheme only, got {sorted(out["scheme"].unique())}')

    selection = pd.DataFrame(selection_rows).sort_values(['dataset', 'horizon', 'source_model', 'ticker'])
    _atomic_write_csv(out, output_dir / 'predictions' / 'nn1_model_predictions.csv')
    _atomic_write_csv(selection, output_dir / 'tables' / 'nn1_seed_selection.csv')
    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv,
        'config': _path_stats(_resolve(args.config)),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'datasets': datasets,
        'horizons': [int(h) for h in horizons],
        'source_models': models,
        'output_models': sorted(expected_models),
        'tickers': tickers,
        'n_selected_seed_files': len(selected_file_stats),
        'output_rows': int(len(out)),
        'selection_rows': int(len(selection)),
        'selected_seed_files': selected_file_stats,
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info('Wrote NN1 single-seed predictions rows=%d models=%s', len(out), sorted(expected_models))


if __name__ == '__main__':
    main()
