from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1rep_matplotlib')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.utils import require_columns, setup_logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'date',
    'ticker',
    'actual_rv',
    'forecast_rv',
    'model',
    'dataset',
    'horizon',
]

DEFAULT_PRIMARY_MODELS = ['HARX', 'LogHAR', 'ElasticNet', 'RandomForest', 'NN2']
DEFAULT_PRIMARY_DECILES = [1, 2, 6, 9, 10]

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


def _assert_output_dir(output_dir: Path, allow_existing: bool) -> None:
    protected = {_resolve(p) for p in PROTECTED_OUTPUT_DIRS}
    for protected_dir in protected:
        if output_dir == protected_dir or _is_relative_to(output_dir, protected_dir):
            raise SystemExit(f'Refusing to write RV-decile outputs inside protected result directory: {output_dir}')
    if output_dir.exists() and not allow_existing:
        files = [p for p in output_dir.rglob('*') if p.is_file()]
        if files:
            raise SystemExit(
                f'Output directory already exists and contains files: {output_dir}. '
                'Use a fresh versioned directory or pass --allow-existing-output-dir intentionally.'
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
        'size_bytes': int(stat.st_size),
        'mtime_utc': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if rows is not None:
        out['rows'] = int(rows)
    return out


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing predictions file: {path}')
    df = pd.read_csv(path, parse_dates=['date'])
    require_columns(df, REQUIRED_COLUMNS, 'final predictions')
    out = df[REQUIRED_COLUMNS].copy()
    out['ticker'] = out['ticker'].astype(str)
    out['model'] = out['model'].astype(str)
    out['dataset'] = out['dataset'].astype(str)
    out['horizon'] = out['horizon'].astype(int)
    out['actual_rv'] = pd.to_numeric(out['actual_rv'], errors='coerce')
    out['forecast_rv'] = pd.to_numeric(out['forecast_rv'], errors='coerce')
    missing = int(out[['actual_rv', 'forecast_rv']].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f'Predictions contain rows with missing actual_rv/forecast_rv: {missing}')
    return out


def _build_decile_assignments(pred: pd.DataFrame, basis_horizon: int, q: int) -> pd.DataFrame:
    basis_raw = pred[pred['horizon'] == int(basis_horizon)][['dataset', 'date', 'ticker', 'actual_rv']].copy()
    # The final prediction table is model-expanded, so the same observed RV appears
    # once per model. Collapse back to one observation per asset-day before forming
    # paper-style realized-variance deciles; allow only floating-point roundoff.
    spread = basis_raw.groupby(['dataset', 'date', 'ticker'])['actual_rv'].agg(['min', 'max'])
    max_spread = float((spread['max'] - spread['min']).max())
    if max_spread > 1e-16:
        raise ValueError(f'Decile basis has inconsistent actual_rv values; max spread={max_spread:.3g}')
    basis = (
        basis_raw.groupby(['dataset', 'date', 'ticker'], as_index=False)
        .agg(actual_rv=('actual_rv', 'mean'))
    )

    rows = []
    for dataset, g in basis.groupby('dataset', sort=True):
        if g['actual_rv'].nunique(dropna=True) <= 1:
            raise ValueError(f'Cannot form realized-variance deciles; actual_rv has zero variance for dataset={dataset}')
        ranked = g.copy()
        ranked['rv_decile_interval'] = pd.qcut(ranked['actual_rv'], q=q, duplicates='drop')
        codes = ranked['rv_decile_interval'].cat.codes
        if (codes < 0).any():
            raise ValueError(f'Decile assignment produced missing codes for dataset={dataset}')
        n_bins = int(ranked['rv_decile_interval'].cat.categories.size)
        ranked['decile'] = codes + 1
        ranked['percentile_bin'] = ranked['decile'].map(
            {i: f'({(i - 1) / n_bins:.3f},{i / n_bins:.3f}]' for i in range(1, n_bins + 1)}
        )
        logger.info(
            'Built deciles dataset=%s basis_horizon=%s rows=%s bins=%s',
            dataset,
            basis_horizon,
            len(ranked),
            n_bins,
        )
        rows.append(ranked[['dataset', 'date', 'ticker', 'decile', 'percentile_bin']])
    return pd.concat(rows, ignore_index=True)


def _compute_decile_mse(pred: pd.DataFrame, deciles: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    merged = pred.merge(deciles, on=['dataset', 'date', 'ticker'], how='inner', validate='many_to_one')
    if merged.empty:
        raise ValueError('No predictions matched the RV-decile basis assignment.')
    lost = len(pred) - len(merged)
    if lost:
        logger.warning('Dropped predictions with no decile assignment: %s rows', lost)

    merged['squared_error'] = (merged['actual_rv'] - merged['forecast_rv']) ** 2
    grouped = (
        merged.groupby(['dataset', 'horizon', 'model', 'decile', 'percentile_bin'], observed=True, as_index=False)
        .agg(
            n_obs=('squared_error', 'size'),
            mse=('squared_error', 'mean'),
            actual_rv_mean=('actual_rv', 'mean'),
            actual_rv_min=('actual_rv', 'min'),
            actual_rv_max=('actual_rv', 'max'),
            forecast_rv_mean=('forecast_rv', 'mean'),
        )
        .sort_values(['dataset', 'horizon', 'model', 'decile'])
        .reset_index(drop=True)
    )
    sparse = grouped[grouped['n_obs'] < int(min_obs)]
    if not sparse.empty:
        for row in sparse.itertuples(index=False):
            logger.warning(
                'Sparse RV decile cell dataset=%s horizon=%s model=%s decile=%s n_obs=%s',
                row.dataset,
                row.horizon,
                row.model,
                row.decile,
                row.n_obs,
            )

    har = grouped[grouped['model'].str.upper() == 'HAR'][
        ['dataset', 'horizon', 'decile', 'mse']
    ].rename(columns={'mse': 'har_mse'})
    if har.empty:
        raise ValueError('HAR baseline is required to compute rel_mse_vs_har, but no HAR rows were found.')
    out = grouped.merge(har, on=['dataset', 'horizon', 'decile'], how='left', validate='many_to_one')
    if out['har_mse'].isna().any():
        missing = out[out['har_mse'].isna()][['dataset', 'horizon', 'decile']].drop_duplicates()
        raise ValueError(f'Missing HAR baseline for decile cells: {missing.to_dict(orient="records")[:10]}')
    out['rel_mse_vs_har'] = out['mse'] / out['har_mse']
    return out


def _model_label(model: str) -> str:
    labels = {
        'HARX': 'HAR-X',
        'LogHAR': 'LogHAR',
        'ElasticNet': 'EN',
        'RandomForest': 'RF',
        'NN2': 'NN^10_2',
    }
    return labels.get(model, model)


def _plot_primary_figure(
    table: pd.DataFrame,
    output_path: Path,
    *,
    dataset: str,
    models: list[str],
    deciles: list[int],
) -> None:
    primary = table[
        (table['dataset'] == dataset)
        & table['model'].isin(models)
        & table['decile'].isin(deciles)
    ].copy()
    if primary.empty:
        raise ValueError(f'No rows available for primary RV-decile figure dataset={dataset}')
    missing_models = set(models) - set(primary['model'])
    if missing_models:
        raise ValueError(f'Missing primary RV-decile models in table: {sorted(missing_models)}')

    horizons = sorted(primary['horizon'].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.8 * len(horizons), 4.6), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    colors = {
        'HARX': '#4C78A8',
        'LogHAR': '#F58518',
        'ElasticNet': '#54A24B',
        'RandomForest': '#B279A2',
        'NN2': '#E45756',
    }
    x_positions = np.arange(len(deciles), dtype=float)
    x_labels = [f'D{d}' for d in deciles]

    for ax, horizon in zip(axes, horizons):
        gh = primary[primary['horizon'] == horizon]
        for model in models:
            gm = gh[gh['model'] == model].set_index('decile').reindex(deciles)
            ax.plot(
                x_positions,
                gm['rel_mse_vs_har'].to_numpy(dtype=float),
                marker='o',
                linewidth=1.8,
                markersize=5,
                label=_model_label(model),
                color=colors.get(model),
            )
        ax.axhline(1.0, color='#222222', linewidth=0.9, linestyle='--', alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Realized-variance decile')
        ax.set_title(f'{dataset}, h={int(horizon)}')
        ax.grid(axis='y', color='#d0d0d0', linewidth=0.7, alpha=0.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('MSE relative to HAR')
    axes[-1].legend(frameon=False, loc='best')
    fig.suptitle('RV-decile forecast MSE comparison')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f'{output_path.name}.tmp.png')
    fig.savefig(tmp, dpi=220, bbox_inches='tight')
    plt.close(fig)
    tmp.replace(output_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Compute paper-style realized-variance-decile MSE tables and Figure 5-style comparison.'
    )
    ap.add_argument(
        '--predictions',
        default='outputs_final_core_no_bagging_gb_20260520/predictions/model_predictions.csv',
        help='Final merged prediction CSV to analyze.',
    )
    ap.add_argument(
        '--output-dir',
        required=True,
        help='Fresh output directory for Task F artifacts, separate from final prediction inputs.',
    )
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--decile-basis-horizon', type=int, default=1)
    ap.add_argument('--decile-count', type=int, default=10)
    ap.add_argument('--min-obs-per-cell', type=int, default=10)
    ap.add_argument('--primary-dataset', default='PARTIAL_MALL')
    ap.add_argument('--primary-models', nargs='*', default=DEFAULT_PRIMARY_MODELS)
    ap.add_argument('--primary-deciles', nargs='*', type=int, default=DEFAULT_PRIMARY_DECILES)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = _resolve(args.predictions)
    output_dir = _resolve(args.output_dir)
    _assert_output_dir(output_dir, args.allow_existing_output_dir)
    setup_logging(output_dir / 'logs' / '06b_rv_decile_mse.log')

    logger.info('Reading predictions: %s', pred_path)
    pred = _read_predictions(pred_path)
    logger.info(
        'Loaded predictions rows=%s datasets=%s horizons=%s models=%s',
        len(pred),
        sorted(pred['dataset'].unique()),
        sorted(pred['horizon'].unique()),
        pred['model'].nunique(),
    )

    deciles = _build_decile_assignments(pred, args.decile_basis_horizon, args.decile_count)
    table = _compute_decile_mse(pred, deciles, args.min_obs_per_cell)

    table_path = output_dir / 'tables' / 'rv_decile_mse.csv'
    assignment_path = output_dir / 'tables' / 'rv_decile_assignments.csv'
    figure_path = output_dir / 'figures' / 'figure5_rv_decile_mse.png'
    _atomic_write_csv(table, table_path)
    _atomic_write_csv(deciles, assignment_path)
    _plot_primary_figure(
        table,
        figure_path,
        dataset=args.primary_dataset,
        models=list(args.primary_models),
        deciles=[int(d) for d in args.primary_deciles],
    )

    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'predictions': _path_stats(pred_path, rows=len(pred)),
        'outputs': {
            'tables/rv_decile_mse.csv': _path_stats(table_path, rows=len(table)),
            'tables/rv_decile_assignments.csv': _path_stats(assignment_path, rows=len(deciles)),
            'figures/figure5_rv_decile_mse.png': _path_stats(figure_path),
        },
        'parameters': {
            'decile_basis_horizon': int(args.decile_basis_horizon),
            'decile_count': int(args.decile_count),
            'min_obs_per_cell': int(args.min_obs_per_cell),
            'primary_dataset': args.primary_dataset,
            'primary_models': list(args.primary_models),
            'primary_deciles': [int(d) for d in args.primary_deciles],
        },
        'table_summary': {
            'rows': int(len(table)),
            'datasets': sorted(table['dataset'].unique().tolist()),
            'horizons': [int(h) for h in sorted(table['horizon'].unique())],
            'models': sorted(table['model'].unique().tolist()),
            'deciles': [int(d) for d in sorted(table['decile'].unique())],
        },
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info('Wrote RV-decile table rows=%s and figure=%s', len(table), figure_path)


if __name__ == '__main__':
    main()
