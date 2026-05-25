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
        _resolve('outputs_final_core_with_bagging_no_gb_20260521'),
    }
    if path in protected:
        raise SystemExit(f'Refusing to write into protected output directory: {path}')
    if path.exists() and any(path.rglob('*')):
        raise SystemExit(f'Output directory already exists and is not empty: {path}')


def _validate_replacement(source: pd.DataFrame, corrected: pd.DataFrame) -> None:
    target = (source['dataset'] == 'PARTIAL_MALL') & (source['model'] == 'HAR')
    old = source[target]
    if old.empty:
        raise RuntimeError('Source final predictions contain no PARTIAL_MALL/HAR rows to replace.')

    if set(corrected['dataset'].astype(str)) != {'PARTIAL_MALL'}:
        raise RuntimeError('Corrected predictions must contain only dataset=PARTIAL_MALL.')
    if set(corrected['model'].astype(str)) != {'HAR'}:
        raise RuntimeError('Corrected predictions must contain only model=HAR.')
    if set(corrected['scheme'].astype(str)) != {'rolling'}:
        raise RuntimeError('Corrected PARTIAL_MALL/HAR predictions must be rolling.')
    if set(corrected['features'].astype(str)) != {'rvd,rvw,rvm'}:
        raise RuntimeError('Corrected PARTIAL_MALL/HAR must use only rvd,rvw,rvm.')

    old_keys = set(map(tuple, old[['date', 'ticker', 'dataset', 'horizon', 'model']].astype(str).to_numpy()))
    new_keys = set(map(tuple, corrected[['date', 'ticker', 'dataset', 'horizon', 'model']].astype(str).to_numpy()))
    if old_keys != new_keys:
        raise RuntimeError(
            'Corrected PARTIAL_MALL/HAR prediction keys do not match the source rows being replaced.'
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Apply the PARTIAL_MALL/HAR feature-set fix to a final prediction file in a fresh output directory.'
    )
    ap.add_argument('--source-final', required=True)
    ap.add_argument('--corrected-har', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    output_dir = _resolve(args.output_dir)
    _assert_fresh_output_dir(output_dir)
    setup_logging(output_dir / 'logs' / '09_apply_har_feature_fix.log')

    source_path = _resolve(args.source_final)
    corrected_path = _resolve(args.corrected_har)
    logger.info('Reading source final predictions: %s', source_path)
    source = _read_predictions(source_path, 'source-final')
    logger.info('Reading corrected PARTIAL_MALL/HAR predictions: %s', corrected_path)
    corrected = _read_predictions(corrected_path, 'corrected-har')
    _validate_replacement(source, corrected)

    target = (source['dataset'] == 'PARTIAL_MALL') & (source['model'] == 'HAR')
    replaced_rows = int(target.sum())
    combined = pd.concat([source.loc[~target], corrected], ignore_index=True)
    combined = combined.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)

    if len(combined) != len(source):
        raise RuntimeError(f'Row count changed unexpectedly: source={len(source)} combined={len(combined)}')
    duplicates = int(combined.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum())
    if duplicates:
        raise RuntimeError(f'Corrected final predictions contain duplicate keys: {duplicates}')
    if set(combined['model'].astype(str)) != set(source['model'].astype(str)):
        raise RuntimeError('Model set changed unexpectedly after applying HAR feature fix.')

    har_pm = combined[(combined['dataset'] == 'PARTIAL_MALL') & (combined['model'] == 'HAR')]
    harx_pm = combined[(combined['dataset'] == 'PARTIAL_MALL') & (combined['model'] == 'HARX')]
    if set(har_pm['features'].astype(str)) != {'rvd,rvw,rvm'}:
        raise RuntimeError('PARTIAL_MALL/HAR features are still incorrect after replacement.')
    if 'rvd,rvw,rvm' in set(harx_pm['features'].astype(str)):
        raise RuntimeError('PARTIAL_MALL/HARX unexpectedly lost extended features.')

    _atomic_write_csv(combined, output_dir / 'predictions' / 'model_predictions.csv')

    logger.info('Evaluating corrected predictions.')
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
        'corrected_har': str(corrected_path),
        'output_dir': str(output_dir),
        'replaced_rows': replaced_rows,
        'output_rows': int(len(combined)),
        'models': sorted(combined['model'].astype(str).unique()),
        'datasets': sorted(combined['dataset'].astype(str).unique()),
        'horizons': sorted(int(h) for h in combined['horizon'].unique()),
        'notes': [
            'Replaces only dataset=PARTIAL_MALL/model=HAR rows.',
            'Corrected PARTIAL_MALL/HAR uses basic HAR features rvd,rvw,rvm.',
            'PARTIAL_MALL/HARX and all other model predictions are preserved from the source final run.',
        ],
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info('Wrote corrected final predictions rows=%d replaced_rows=%d', len(combined), replaced_rows)


if __name__ == '__main__':
    main()
