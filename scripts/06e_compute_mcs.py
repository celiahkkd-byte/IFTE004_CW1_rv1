from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1_mpl')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    'date',
    'ticker',
    'dataset',
    'horizon',
    'model',
    'actual_rv',
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
    'outputs_fhs_var_core_no_bagging_gb_20260520',
}

MODEL_ORDER = [
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
    'Bagging',
    'RandomForest',
    'GradientBoosting',
    'NN1',
    'NN2',
    'NN3',
    'NN4',
    'NN1_1',
    'NN2_1',
    'NN3_1',
    'NN4_1',
]


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


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / '06e_mcs.log'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def assert_output_dir(output_dir: Path, allow_existing: bool, allow_protected: bool) -> None:
    protected = {resolve_path(p) for p in PROTECTED_OUTPUT_DIRS}
    if output_dir in protected and not allow_protected:
        raise SystemExit(
            f'Refusing to write MCS outputs into protected directory: {output_dir}. '
            'Use a new isolated output directory.'
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise SystemExit(
            f'Output directory already exists and is not empty: {output_dir}. '
            'Use --allow-existing-output-dir only for checkpoint resume.'
        )


def load_predictions(path: Path, datasets: Iterable[str] | None, horizons: Iterable[int] | None, models: Iterable[str] | None) -> pd.DataFrame:
    pred = pd.read_csv(path, parse_dates=['date'])
    missing = REQUIRED_COLUMNS - set(pred.columns)
    if missing:
        raise ValueError(f'Missing required prediction columns: {sorted(missing)}')
    pred['ticker'] = pred['ticker'].astype(str).str.upper()
    pred['dataset'] = pred['dataset'].astype(str)
    pred['model'] = pred['model'].astype(str)
    pred['horizon'] = pred['horizon'].astype(int)
    pred['actual_rv'] = pd.to_numeric(pred['actual_rv'], errors='coerce')
    pred['forecast_rv'] = pd.to_numeric(pred['forecast_rv'], errors='coerce')
    pred = pred.dropna(subset=['date', 'ticker', 'dataset', 'horizon', 'model', 'actual_rv', 'forecast_rv'])
    if datasets:
        pred = pred[pred['dataset'].isin(list(datasets))]
    if horizons:
        pred = pred[pred['horizon'].isin([int(h) for h in horizons])]
    if models:
        pred = pred[pred['model'].isin(list(models))]
    if pred.empty:
        raise ValueError('No prediction rows remain after filters.')
    pred['sq_err'] = (pred['actual_rv'] - pred['forecast_rv']) ** 2
    return pred.sort_values(['dataset', 'horizon', 'ticker', 'date', 'model']).reset_index(drop=True)


def checkpoint_path(output_dir: Path, dataset: str, horizon: int, ticker: str) -> Path:
    return output_dir / 'checkpoints' / safe_name(dataset) / f'h{int(horizon)}' / f'{safe_name(ticker)}.csv'


def run_one_mcs(
    group: pd.DataFrame,
    confidence: float,
    reps: int,
    block_size: int,
    method: str,
    bootstrap: str,
    seed: int,
    min_valid_rows: int,
    duplicate_tolerance: float,
) -> pd.DataFrame:
    dataset = str(group['dataset'].iloc[0])
    horizon = int(group['horizon'].iloc[0])
    ticker = str(group['ticker'].iloc[0]).upper()
    loss = group.pivot_table(index='date', columns='model', values='sq_err', aggfunc='first').sort_index()
    loss = loss.dropna(how='any')
    if len(loss) < min_valid_rows:
        raise ValueError(f'Insufficient valid rows for dataset={dataset} h={horizon} ticker={ticker}: {len(loss)}')
    if loss.shape[1] < 2:
        raise ValueError(f'MCS requires at least two models for dataset={dataset} h={horizon} ticker={ticker}')

    representative_for: dict[str, str] = {}
    duplicate_group_size: dict[str, int] = {}
    unique_columns: list[str] = []
    for model in loss.columns:
        matched = None
        values = loss[model].to_numpy(dtype=float)
        for representative in unique_columns:
            rep_values = loss[representative].to_numpy(dtype=float)
            if np.allclose(values, rep_values, rtol=0.0, atol=float(duplicate_tolerance)):
                matched = representative
                break
        representative = matched or str(model)
        representative_for[str(model)] = representative
        if matched is None:
            unique_columns.append(str(model))
        duplicate_group_size[representative] = duplicate_group_size.get(representative, 0) + 1

    unique_loss = loss[unique_columns]
    if unique_loss.shape[1] == 1:
        included = set(unique_columns)
        pvals = {unique_columns[0]: 1.0}
    else:
        mcs = MCS(
            unique_loss,
            size=1.0 - confidence,
            reps=int(reps),
            block_size=int(block_size),
            method=method,
            bootstrap=bootstrap,
            seed=int(seed),
        )
        mcs.compute()
        included = set(mcs.included)
        pvals = mcs.pvalues['Pvalue'].to_dict()
    mean_loss = loss.mean(axis=0).to_dict()
    rows = []
    for model in loss.columns:
        representative = representative_for[str(model)]
        rows.append({
            'dataset': dataset,
            'horizon': horizon,
            'ticker': ticker,
            'model': model,
            'in_mcs': bool(representative in included),
            'mcs_pvalue': float(pvals.get(representative, np.nan)),
            'mean_loss': float(mean_loss[model]),
            'n_valid_days': int(len(loss)),
            'n_models': int(loss.shape[1]),
            'n_unique_loss_models': int(unique_loss.shape[1]),
            'mcs_representative': representative,
            'duplicate_loss_group_size': int(duplicate_group_size[representative]),
            'confidence': float(confidence),
            'reps': int(reps),
            'block_size': int(block_size),
            'method': method,
            'bootstrap': bootstrap,
        })
    return pd.DataFrame(rows)


def combine_checkpoints(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / 'checkpoints').glob('*/*/*.csv'))
    if not paths:
        raise RuntimeError(f'No MCS checkpoints found in {output_dir / "checkpoints"}')
    parts = [pd.read_csv(path) for path in paths]
    combined = pd.concat(parts, ignore_index=True)
    key_cols = ['dataset', 'horizon', 'ticker', 'model']
    duplicates = int(combined.duplicated(key_cols).sum())
    if duplicates:
        raise RuntimeError(f'Duplicate MCS keys after combining checkpoints: {duplicates}')
    return combined.sort_values(['dataset', 'horizon', 'ticker', 'model']).reset_index(drop=True)


def aggregate_inclusion(per_ticker: pd.DataFrame) -> pd.DataFrame:
    agg = (
        per_ticker.groupby(['dataset', 'horizon', 'model'], sort=True)
        .agg(
            n_tickers_included=('in_mcs', 'sum'),
            n_tickers_total=('in_mcs', 'count'),
            mean_mcs_pvalue=('mcs_pvalue', 'mean'),
            mean_loss=('mean_loss', 'mean'),
            min_valid_days=('n_valid_days', 'min'),
        )
        .reset_index()
    )
    agg['inclusion_rate'] = agg['n_tickers_included'] / agg['n_tickers_total']
    return agg.sort_values(['dataset', 'horizon', 'model']).reset_index(drop=True)


def ordered_models(models: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(models))
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered.extend(sorted(m for m in present if m not in ordered))
    return ordered


def plot_mcs(agg: pd.DataFrame, output_path: Path) -> None:
    datasets = sorted(agg['dataset'].unique())
    horizons = sorted(int(h) for h in agg['horizon'].unique())
    n_rows = len(datasets)
    n_cols = len(horizons)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(8, 5.5 * n_cols), max(4.5, 3.8 * n_rows)),
        squeeze=False,
        sharey=True,
    )
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            ax = axes[i][j]
            sub = agg[(agg['dataset'] == dataset) & (agg['horizon'].astype(int) == int(horizon))].copy()
            order = ordered_models(sub['model'])
            sub['model'] = pd.Categorical(sub['model'], categories=order, ordered=True)
            sub = sub.sort_values('model')
            ax.barh(sub['model'].astype(str), sub['inclusion_rate'], color='#2f6f8f')
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel('MCS inclusion rate')
            ax.set_title(f'{dataset}, h={horizon}')
            ax.grid(axis='x', alpha=0.25)
            ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Checkpointed Model Confidence Set computation using arch.bootstrap.MCS.')
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--confidence', type=float, default=0.90)
    ap.add_argument('--reps', type=int, default=5000)
    ap.add_argument('--block-size', type=int, default=10)
    ap.add_argument('--method', choices=['max', 'R'], default='max')
    ap.add_argument('--bootstrap', choices=['stationary', 'sb', 'circular', 'cbb', 'moving block', 'mbb'], default='stationary')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--horizons', nargs='*', type=int, default=None)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--min-valid-rows', type=int, default=200)
    ap.add_argument('--min-ticker-coverage', type=int, default=20)
    ap.add_argument('--duplicate-tolerance', type=float, default=1e-18)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--allow-protected-output-dir', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit('--confidence must be between 0 and 1.')
    output_dir = resolve_path(args.output_dir)
    assert_output_dir(output_dir, args.allow_existing_output_dir, args.allow_protected_output_dir)
    setup_logging(output_dir)

    pred = load_predictions(resolve_path(args.predictions), args.datasets, args.horizons, args.models)
    if args.tickers:
        keep = {t.upper() for t in args.tickers}
        pred = pred[pred['ticker'].isin(keep)].copy()
        if pred.empty:
            raise SystemExit(f'No prediction rows remain after ticker filter: {sorted(keep)}')

    LOGGER.info(
        'Task I MCS output_dir=%s rows=%d confidence=%.3f reps=%d block_size=%d datasets=%s horizons=%s models=%d tickers=%d',
        output_dir,
        len(pred),
        args.confidence,
        args.reps,
        args.block_size,
        sorted(pred['dataset'].unique()),
        sorted(pred['horizon'].unique()),
        pred['model'].nunique(),
        pred['ticker'].nunique(),
    )
    atomic_write_json(
        {
            'created_at': utc_now(),
            'command': sys.argv,
            'predictions': str(resolve_path(args.predictions)),
            'output_dir': str(output_dir),
            'confidence': args.confidence,
            'reps': args.reps,
            'block_size': args.block_size,
            'method': args.method,
            'bootstrap': args.bootstrap,
            'datasets': sorted(pred['dataset'].unique()),
            'horizons': [int(h) for h in sorted(pred['horizon'].unique())],
            'models': sorted(pred['model'].unique()),
        },
        output_dir / 'run_provenance.json',
    )

    manifest_rows = []
    for keys, group in pred.groupby(['dataset', 'horizon', 'ticker'], sort=True):
        dataset, horizon, ticker = keys
        path = checkpoint_path(output_dir, str(dataset), int(horizon), str(ticker))
        if path.exists() and not args.force:
            manifest_rows.append({
                'dataset': dataset,
                'horizon': int(horizon),
                'ticker': ticker,
                'status': 'reused',
                'rows': len(pd.read_csv(path, usecols=['model'])),
                'path': str(path),
            })
            continue
        if args.aggregate_only:
            manifest_rows.append({
                'dataset': dataset,
                'horizon': int(horizon),
                'ticker': ticker,
                'status': 'missing',
                'rows': 0,
                'path': str(path),
            })
            continue
        LOGGER.info('Computing MCS checkpoint dataset=%s h=%s ticker=%s', dataset, horizon, ticker)
        try:
            result = run_one_mcs(
                group,
                confidence=args.confidence,
                reps=args.reps,
                block_size=args.block_size,
                method=args.method,
                bootstrap=args.bootstrap,
                seed=args.seed + len(manifest_rows),
                min_valid_rows=args.min_valid_rows,
                duplicate_tolerance=args.duplicate_tolerance,
            )
            atomic_write_csv(result, path)
            manifest_rows.append({
                'dataset': dataset,
                'horizon': int(horizon),
                'ticker': ticker,
                'status': 'completed',
                'rows': len(result),
                'path': str(path),
            })
        except Exception as exc:
            LOGGER.exception('MCS checkpoint failed dataset=%s h=%s ticker=%s: %s', dataset, horizon, ticker, exc)
            manifest_rows.append({
                'dataset': dataset,
                'horizon': int(horizon),
                'ticker': ticker,
                'status': 'failed',
                'rows': 0,
                'path': str(path),
                'error': str(exc),
            })

    manifest = pd.DataFrame(manifest_rows)
    atomic_write_csv(manifest, output_dir / 'tables' / 'mcs_checkpoint_manifest.csv')
    failed = manifest[manifest['status'] == 'failed']
    if not failed.empty:
        raise RuntimeError(f'MCS failed for {len(failed)} dataset/horizon/ticker groups. See manifest.')

    per_ticker = combine_checkpoints(output_dir)
    agg = aggregate_inclusion(per_ticker)
    low = agg[agg['n_tickers_total'] < int(args.min_ticker_coverage)]
    if not low.empty:
        atomic_write_csv(per_ticker, output_dir / 'tables' / 'mcs_per_ticker.csv')
        atomic_write_csv(agg, output_dir / 'tables' / 'mcs_inclusion_rates.csv')
        raise RuntimeError(f'MCS ticker coverage below {args.min_ticker_coverage} for {len(low)} rows.')

    atomic_write_csv(per_ticker, output_dir / 'tables' / 'mcs_per_ticker.csv')
    atomic_write_csv(agg, output_dir / 'tables' / 'mcs_inclusion_rates.csv')
    plot_mcs(agg, output_dir / 'figures' / 'figure4_mcs.png')
    LOGGER.info('Wrote MCS per_ticker_rows=%d inclusion_rows=%d', len(per_ticker), len(agg))


if __name__ == '__main__':
    main()
