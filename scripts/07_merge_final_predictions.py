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

from rv1rep.config import load_config
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

DEFAULT_NON_NN_MODELS = [
    'HAR',
    'HARX',
    'LogHAR',
    'LevHAR',
    'SHAR',
    'HARQ',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'AdaptiveLasso',
    'PostLasso',
    'RandomForest',
]

DEFAULT_NN10_MODELS = ['NN1', 'NN2', 'NN3', 'NN4']
DEFAULT_NN1_MODELS = ['NN1_1', 'NN2_1', 'NN3_1', 'NN4_1']

PROTECTED_OUTPUT_DIRS = {
    'outputs',
    'outputs_full_nn',
    'outputs_rolling',
    'outputs_nn30_checkpointed',
    'outputs_bagging_checkpointed',
    'outputs_bagging_smoke',
    'outputs_bagging_smoke_minimal',
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


def _assert_fresh_output_dir(output_dir: Path, allow_existing_empty: bool) -> None:
    protected = {_resolve(p) for p in PROTECTED_OUTPUT_DIRS}
    for protected_dir in protected:
        if output_dir == protected_dir or _is_relative_to(output_dir, protected_dir):
            raise SystemExit(f'Refusing to write final merged outputs inside protected result directory: {output_dir}')
    if output_dir.exists():
        files = [p for p in output_dir.rglob('*') if p.is_file()]
        if files or not allow_existing_empty:
            raise SystemExit(
                f'Output directory already exists: {output_dir}. '
                'Use a new versioned directory name to avoid overwriting or mixing result tables.'
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


def _path_stats(path: Path, rows: int | None = None, models: list[str] | None = None) -> dict:
    stat = path.stat()
    out = {
        'path': str(path),
        'size_bytes': stat.st_size,
        'mtime_utc': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if rows is not None:
        out['rows'] = int(rows)
    if models is not None:
        out['models'] = models
    return out


def _read_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing {label} predictions: {path}')
    df = pd.read_csv(path, parse_dates=['date'])
    require_columns(df, REQUIRED_COLUMNS, label)
    return df[REQUIRED_COLUMNS].copy()


def _validate_component(
    df: pd.DataFrame,
    *,
    label: str,
    allowed_models: set[str],
    required_scheme: str,
    require_exact_models: bool,
) -> None:
    models = set(df['model'].astype(str))
    unexpected = models - allowed_models
    if unexpected:
        raise RuntimeError(f'{label} has unexpected models: {sorted(unexpected)}')
    if require_exact_models and models != allowed_models:
        raise RuntimeError(f'{label} model set mismatch: got={sorted(models)}, expected={sorted(allowed_models)}')
    schemes = set(df['scheme'].astype(str))
    if schemes != {required_scheme}:
        raise RuntimeError(f'{label} scheme mismatch: got={sorted(schemes)}, expected={[required_scheme]}')
    dup = int(df.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum())
    if dup:
        raise RuntimeError(f'{label} contains duplicate prediction keys: {dup}')


def _validate_final(df: pd.DataFrame, expected_models: set[str], datasets: set[str], horizons: set[int]) -> None:
    models = set(df['model'].astype(str))
    if models != expected_models:
        raise RuntimeError(f'Final model set mismatch: got={sorted(models)}, expected={sorted(expected_models)}')
    got_datasets = set(df['dataset'].astype(str))
    if got_datasets != datasets:
        raise RuntimeError(f'Final dataset set mismatch: got={sorted(got_datasets)}, expected={sorted(datasets)}')
    got_horizons = {int(h) for h in df['horizon'].unique()}
    if got_horizons != horizons:
        raise RuntimeError(f'Final horizon set mismatch: got={sorted(got_horizons)}, expected={sorted(horizons)}')
    dup = int(df.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum())
    if dup:
        raise RuntimeError(f'Final merged predictions contain duplicate keys: {dup}')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Merge rolling non-NN, NN^10 ensemble, and NN^1 single-seed predictions into a fresh final output directory.'
    )
    ap.add_argument('--config', default=str(ROOT / 'config/paper_core_rolling.yaml'))
    ap.add_argument('--rolling-predictions', default='outputs_rolling/predictions/model_predictions.csv')
    ap.add_argument('--nn10-predictions', default='outputs_nn30_checkpointed/predictions/nn_model_predictions.csv')
    ap.add_argument('--nn1-predictions', required=True)
    ap.add_argument('--extra-predictions', nargs='*', default=[], help='Optional extra by-model or model_predictions CSVs, e.g. Bagging later.')
    ap.add_argument('--output-dir', required=True, help='Fresh final directory, e.g. outputs_final_core_no_bagging_gb_20260520')
    ap.add_argument('--expected-non-nn-models', nargs='*', default=DEFAULT_NON_NN_MODELS)
    ap.add_argument('--expected-nn10-models', nargs='*', default=DEFAULT_NN10_MODELS)
    ap.add_argument('--expected-nn1-models', nargs='*', default=DEFAULT_NN1_MODELS)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--horizons', nargs='*', type=int, default=None)
    ap.add_argument('--allow-existing-empty-output-dir', action='store_true')
    ap.add_argument('--skip-evaluation', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = _resolve(args.output_dir)
    _assert_fresh_output_dir(output_dir, args.allow_existing_empty_output_dir)
    setup_logging(output_dir / 'logs' / '07_merge_final_predictions.log')

    rolling_path = _resolve(args.rolling_predictions)
    nn10_path = _resolve(args.nn10_predictions)
    nn1_path = _resolve(args.nn1_predictions)
    extra_paths = [_resolve(p) for p in args.extra_predictions]

    expected_non_nn = set(args.expected_non_nn_models)
    expected_nn10 = set(args.expected_nn10_models)
    expected_nn1 = set(args.expected_nn1_models)
    expected_models = expected_non_nn | expected_nn10 | expected_nn1
    datasets = set(args.datasets or cfg['experiments']['datasets'])
    horizons = {int(h) for h in (args.horizons or cfg['experiments']['horizons'])}

    logger.info('Reading rolling predictions: %s', rolling_path)
    rolling = _read_predictions(rolling_path, 'rolling')
    logger.info('Reading NN10 predictions: %s', nn10_path)
    nn10 = _read_predictions(nn10_path, 'nn10')
    logger.info('Reading NN1 predictions: %s', nn1_path)
    nn1 = _read_predictions(nn1_path, 'nn1')
    extras = []
    for p in extra_paths:
        logger.info('Reading extra predictions: %s', p)
        extras.append(_read_predictions(p, f'extra:{p.name}'))

    _validate_component(
        rolling,
        label='rolling',
        allowed_models=expected_non_nn,
        required_scheme='rolling',
        require_exact_models=False,
    )
    _validate_component(
        nn10,
        label='nn10',
        allowed_models=expected_nn10,
        required_scheme='fixed',
        require_exact_models=True,
    )
    _validate_component(
        nn1,
        label='nn1',
        allowed_models=expected_nn1,
        required_scheme='fixed',
        require_exact_models=True,
    )
    for i, extra in enumerate(extras):
        _validate_component(
            extra,
            label=f'extra[{i}]',
            allowed_models=expected_non_nn,
            required_scheme='rolling',
            require_exact_models=False,
        )

    combined = pd.concat([rolling, *extras, nn10, nn1], ignore_index=True)
    combined = combined.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)
    _validate_final(combined, expected_models, datasets, horizons)

    _atomic_write_csv(combined, output_dir / 'predictions' / 'model_predictions.csv')
    written = {
        'predictions/model_predictions.csv': int(len(combined)),
    }

    if not args.skip_evaluation:
        logger.info('Evaluating merged predictions.')
        metrics = forecast_metrics(combined)
        summary = cross_sectional_summary(metrics)
        rel_matrix, dm = pairwise_relative_mse(combined)
        _atomic_write_csv(metrics, output_dir / 'tables' / 'forecast_metrics_by_asset.csv')
        _atomic_write_csv(summary, output_dir / 'tables' / 'forecast_summary_cross_section.csv')
        _atomic_write_csv(rel_matrix, output_dir / 'tables' / 'pairwise_relative_mse_matrix.csv')
        _atomic_write_csv(dm, output_dir / 'tables' / 'diebold_mariano_tests.csv')
        written.update(
            {
                'tables/forecast_metrics_by_asset.csv': int(len(metrics)),
                'tables/forecast_summary_cross_section.csv': int(len(summary)),
                'tables/pairwise_relative_mse_matrix.csv': int(len(rel_matrix)),
                'tables/diebold_mariano_tests.csv': int(len(dm)),
            }
        )

    inputs = []
    for label, path, frame in [('rolling', rolling_path, rolling), ('nn10', nn10_path, nn10), ('nn1', nn1_path, nn1)]:
        inputs.append(_path_stats(path, rows=len(frame), models=sorted(frame['model'].astype(str).unique())))
        inputs[-1]['label'] = label
    for i, (path, frame) in enumerate(zip(extra_paths, extras)):
        inputs.append(_path_stats(path, rows=len(frame), models=sorted(frame['model'].astype(str).unique())))
        inputs[-1]['label'] = f'extra[{i}]'

    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv,
        'config': _path_stats(_resolve(args.config)),
        'output_dir': str(output_dir),
        'expected_models': sorted(expected_models),
        'datasets': sorted(datasets),
        'horizons': sorted(horizons),
        'input_files': inputs,
        'output_rows': int(len(combined)),
        'output_models': sorted(combined['model'].astype(str).unique()),
        'written_files': written,
        'notes': [
            'This merge is side-effect free with respect to source prediction directories.',
            'Bagging/GradientBoosting are included only if supplied through extra-predictions and expected model arguments.',
        ],
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info('Wrote final merged predictions rows=%d models=%s', len(combined), sorted(expected_models))


if __name__ == '__main__':
    main()
