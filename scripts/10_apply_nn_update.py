from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.evaluation import cross_sectional_summary, forecast_metrics, pairwise_relative_mse
from rv1rep.utils import require_columns, setup_logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
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

NN10_MODELS = {'NN1', 'NN2', 'NN3', 'NN4'}
NN1_MODELS = {'NN1_1', 'NN2_1', 'NN3_1', 'NN4_1'}
NN_MODELS = NN10_MODELS | NN1_MODELS


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


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


def _read_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing {label}: {path}')
    df = pd.read_csv(path, parse_dates=['date'])
    require_columns(df, REQUIRED_COLUMNS, label)
    return df[REQUIRED_COLUMNS].copy()


def _assert_fresh_output_dir(path: Path) -> None:
    protected = {
        _resolve('outputs'),
        _resolve('outputs_full_nn'),
        _resolve('outputs_final_core_with_bagging_no_gb_harfix_20260521'),
    }
    if path in protected:
        raise SystemExit(f'Refusing to write into protected output directory: {path}')
    if path.exists() and any(path.rglob('*')):
        raise SystemExit(f'Output directory already exists and is not empty: {path}')


def _validate_nn_component(df: pd.DataFrame, expected_models: set[str], label: str) -> None:
    models = set(df['model'].astype(str))
    if models != expected_models:
        raise RuntimeError(f'{label} model set mismatch: got={sorted(models)}, expected={sorted(expected_models)}')
    if set(df['scheme'].astype(str)) != {'fixed'}:
        raise RuntimeError(f'{label} scheme must be fixed.')
    if set(df['dataset'].astype(str)) != {'MHAR', 'PARTIAL_MALL'}:
        raise RuntimeError(f'{label} must cover MHAR and PARTIAL_MALL.')
    if {int(h) for h in df['horizon'].unique()} != {1, 5}:
        raise RuntimeError(f'{label} must cover horizons 1 and 5.')
    if int(df.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum()):
        raise RuntimeError(f'{label} contains duplicate prediction keys.')


def _validate_key_match(source: pd.DataFrame, replacement: pd.DataFrame) -> None:
    old = source[source['model'].astype(str).isin(NN_MODELS)]
    if old.empty:
        raise RuntimeError('Source final predictions contain no NN rows to replace.')
    old_keys = set(map(tuple, old[['date', 'ticker', 'dataset', 'horizon', 'model']].astype(str).to_numpy()))
    new_keys = set(map(tuple, replacement[['date', 'ticker', 'dataset', 'horizon', 'model']].astype(str).to_numpy()))
    if old_keys != new_keys:
        raise RuntimeError('NN replacement keys do not match source NN rows.')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Replace NN^10 and NN^1 forecasts in a corrected final file and recompute evaluation tables.'
    )
    ap.add_argument('--source-final', required=True)
    ap.add_argument('--nn10-predictions', required=True)
    ap.add_argument('--nn1-predictions', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    output_dir = _resolve(args.output_dir)
    _assert_fresh_output_dir(output_dir)
    setup_logging(output_dir / 'logs' / '10_apply_nn_update.log')

    source_path = _resolve(args.source_final)
    nn10_path = _resolve(args.nn10_predictions)
    nn1_path = _resolve(args.nn1_predictions)
    logger.info('Reading source final predictions: %s', source_path)
    source = _read_predictions(source_path, 'source-final')
    logger.info('Reading NN10 predictions: %s', nn10_path)
    nn10 = _read_predictions(nn10_path, 'nn10')
    logger.info('Reading NN1 predictions: %s', nn1_path)
    nn1 = _read_predictions(nn1_path, 'nn1')

    _validate_nn_component(nn10, NN10_MODELS, 'nn10')
    _validate_nn_component(nn1, NN1_MODELS, 'nn1')
    replacement = pd.concat([nn10, nn1], ignore_index=True)
    _validate_key_match(source, replacement)

    keep = ~source['model'].astype(str).isin(NN_MODELS)
    combined = pd.concat([source.loc[keep], replacement], ignore_index=True)
    combined = combined.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)

    if len(combined) != len(source):
        raise RuntimeError(f'Row count changed unexpectedly: source={len(source)} combined={len(combined)}')
    duplicates = int(combined.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum())
    if duplicates:
        raise RuntimeError(f'Updated final predictions contain duplicate keys: {duplicates}')
    if set(combined['model'].astype(str)) != set(source['model'].astype(str)):
        raise RuntimeError('Model set changed unexpectedly after NN replacement.')

    pm_har = combined[(combined['dataset'] == 'PARTIAL_MALL') & (combined['model'] == 'HAR')]
    if set(pm_har['features'].astype(str)) != {'rvd,rvw,rvm'}:
        raise RuntimeError('PARTIAL_MALL/HAR feature fix was not preserved.')

    _atomic_write_csv(combined, output_dir / 'predictions' / 'model_predictions.csv')

    logger.info('Evaluating updated predictions.')
    metrics = forecast_metrics(combined)
    summary = cross_sectional_summary(metrics)
    rel_matrix, dm = pairwise_relative_mse(combined)
    _atomic_write_csv(metrics, output_dir / 'tables' / 'forecast_metrics_by_asset.csv')
    _atomic_write_csv(summary, output_dir / 'tables' / 'forecast_summary_cross_section.csv')
    _atomic_write_csv(rel_matrix, output_dir / 'tables' / 'pairwise_relative_mse_matrix.csv')
    _atomic_write_csv(dm, output_dir / 'tables' / 'diebold_mariano_tests.csv')

    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv,
        'source_final': str(source_path),
        'nn10_predictions': str(nn10_path),
        'nn1_predictions': str(nn1_path),
        'output_dir': str(output_dir),
        'replaced_models': sorted(NN_MODELS),
        'output_rows': int(len(combined)),
        'models': sorted(combined['model'].astype(str).unique()),
        'datasets': sorted(combined['dataset'].astype(str).unique()),
        'horizons': sorted(int(h) for h in combined['horizon'].unique()),
        'notes': [
            'Replaces only NN1-NN4 and NN1_1-NN4_1 rows.',
            'Preserves all non-NN forecasts, including the corrected PARTIAL_MALL/HAR feature fix.',
        ],
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info('Wrote NN-updated final predictions rows=%d', len(combined))


if __name__ == '__main__':
    main()
