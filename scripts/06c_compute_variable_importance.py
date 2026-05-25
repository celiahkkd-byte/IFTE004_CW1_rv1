from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1rep_matplotlib')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import load_config
from rv1rep.explain import accumulated_local_effect
from rv1rep.utils import require_columns, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ['HARX', 'ElasticNet', 'RandomForest', 'NN10_2']
DEFAULT_DATASET = 'PARTIAL_MALL'
DEFAULT_HORIZON = 1
BANNED_IV_FEATURES = {'iv', 'log_iv'}

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
}


def _load_task_e_module() -> ModuleType:
    path = ROOT / 'scripts' / '08_compute_ale_checkpointed.py'
    spec = importlib.util.spec_from_file_location('_rv1rep_task_e_ale', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import ALE helpers from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALE_HELPERS = _load_task_e_module()


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
            raise SystemExit(f'Refusing to write variable-importance outputs inside protected result directory: {output_dir}')
    if output_dir.exists() and not allow_existing:
        files = [p for p in output_dir.rglob('*') if p.is_file()]
        if files:
            raise SystemExit(
                f'Output directory already exists and contains files: {output_dir}. '
                'Use a fresh versioned directory or pass --allow-existing-output-dir to resume checkpoints.'
            )


def _safe_name(value: str) -> str:
    return value.replace('/', '_').replace('\\', '_').replace(' ', '_')


def _checkpoint_path(output_dir: Path, model: str, ticker: str) -> Path:
    return output_dir / 'checkpoints' / _safe_name(model) / f'{_safe_name(ticker)}.csv'


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


def _available_tickers(panel: pd.DataFrame, tickers: list[str] | None) -> list[str]:
    if tickers:
        return [str(t).upper() for t in tickers]
    require_columns(panel, ['ticker'], 'forecasting panel')
    return sorted(panel['ticker'].astype(str).str.upper().unique().tolist())


def _model_frame_name(model: str) -> str:
    return 'NN2' if model == 'NN10_2' else model


def _eligible_features(feature_cols: list[str], max_features: int | None) -> list[str]:
    features = [f for f in feature_cols if f.lower() not in BANNED_IV_FEATURES]
    if max_features is not None:
        features = features[: int(max_features)]
    if not features:
        raise ValueError('No eligible features remain after IV exclusion.')
    return features


def _select_or_fail_nn_seeds(nn_checkpoint_dir: Path, dataset: str, horizon: int, ticker: str, top_k: int) -> tuple[list[int], list[dict]]:
    try:
        return ALE_HELPERS._select_nn_seeds_from_checkpoints(nn_checkpoint_dir, dataset, horizon, ticker, top_k)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            'NN10_2 variable importance requires Task B seed checkpoints in this run. '
            f'Missing or incomplete checkpoint source for ticker={ticker}: {exc}'
        ) from exc


def _fit_predictor(model: str, prepared: dict, cfg: dict, nn_checkpoint_dir: Path, dataset: str, horizon: int, ticker: str, nn_top: int):
    if model == 'NN10_2':
        selected_seeds, selected_meta = _select_or_fail_nn_seeds(nn_checkpoint_dir, dataset, horizon, ticker, nn_top)
        predict_fn, fit_info = ALE_HELPERS._fit_nn10_2_predictor(prepared, cfg, selected_seeds)
        fit_info['selected_seed_source'] = selected_meta
        return predict_fn, fit_info
    return ALE_HELPERS._fit_sklearn_predictor(model, prepared, cfg)


def _compute_model_ticker_checkpoint(
    *,
    panel: pd.DataFrame,
    cfg: dict,
    model: str,
    ticker: str,
    dataset: str,
    horizon: int,
    grid_size: int,
    nn_checkpoint_dir: Path,
    nn_top: int,
    min_in_sample: int,
    max_features: int | None,
) -> pd.DataFrame:
    prepared = ALE_HELPERS._prepare_model_data(panel, cfg, dataset=dataset, model_name=model, ticker=ticker, horizon=horizon)
    if len(prepared['X_in']) < int(min_in_sample):
        raise ValueError(f'Insufficient in-sample rows model={model} ticker={ticker}: {len(prepared["X_in"])}')

    features = _eligible_features(list(prepared['feature_cols']), max_features=max_features)
    predict_fn, fit_info = _fit_predictor(model, prepared, cfg, nn_checkpoint_dir, dataset, horizon, ticker, nn_top)

    rows = []
    for feature in features:
        model_feature = ALE_HELPERS._model_feature_name(model, feature, prepared['X_in'].columns)
        if model_feature not in prepared['X_in'].columns:
            logger.warning('Skipping unavailable feature model=%s ticker=%s feature=%s', model, ticker, feature)
            continue
        ale = accumulated_local_effect(predict_fn, prepared['X_in'], model_feature, grid_size=grid_size)
        if ale.empty or len(ale) < 2:
            raw_importance = 0.0
            n_points = int(len(ale))
        else:
            centered = ale['ale'].to_numpy(dtype=float) - float(ale['ale'].mean())
            raw_importance = float(np.std(centered, ddof=1))
            n_points = int(len(ale))
        rows.append(
            {
                'dataset': dataset,
                'horizon': int(horizon),
                'ticker': ticker,
                'model': model,
                'feature': feature,
                'model_feature': model_feature,
                'importance_raw': raw_importance,
                'ale_points': n_points,
                'grid_size': int(grid_size),
                'n_in_sample': int(len(prepared['X_in'])),
                'n_train': int(len(prepared['train'])),
                'n_val': int(len(prepared['val'])),
                'fit_info': json.dumps(fit_info, sort_keys=True),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f'No variable-importance rows produced model={model} ticker={ticker}')
    total = float(out['importance_raw'].sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f'Cannot normalize variable importance model={model} ticker={ticker}; raw total={total}')
    out['vi_normalized'] = out['importance_raw'] / total
    vi_sum = float(out['vi_normalized'].sum())
    if not 0.999 <= vi_sum <= 1.001:
        raise ValueError(f'VI normalization failed model={model} ticker={ticker}; sum={vi_sum}')
    return out[
        [
            'dataset',
            'horizon',
            'ticker',
            'model',
            'feature',
            'model_feature',
            'importance_raw',
            'vi_normalized',
            'ale_points',
            'grid_size',
            'n_in_sample',
            'n_train',
            'n_val',
            'fit_info',
        ]
    ]


def _aggregate_vi(per_ticker: pd.DataFrame, min_tickers: int) -> pd.DataFrame:
    agg = (
        per_ticker.groupby(['dataset', 'horizon', 'model', 'feature'], as_index=False)
        .agg(
            n_tickers=('ticker', 'nunique'),
            vi_mean=('vi_normalized', 'mean'),
            vi_median=('vi_normalized', 'median'),
            vi_std=('vi_normalized', 'std'),
            raw_importance_mean=('importance_raw', 'mean'),
            raw_importance_median=('importance_raw', 'median'),
        )
        .sort_values(['model', 'vi_mean'], ascending=[True, False])
        .reset_index(drop=True)
    )
    sparse = agg[agg['n_tickers'] < int(min_tickers)]
    if not sparse.empty:
        raise ValueError(
            'Some variable-importance rows have too few contributing tickers: '
            f'{sparse[["model", "feature", "n_tickers"]].to_dict(orient="records")[:10]}'
        )
    for model, g in agg.groupby('model'):
        total = float(g['vi_mean'].sum())
        if not 0.95 <= total <= 1.05:
            raise ValueError(f'Mean VI should sum to about one for model={model}; got {total}')
    return agg


def _plot_variable_importance(agg: pd.DataFrame, output_path: Path, models: list[str]) -> None:
    available = [m for m in models if m in set(agg['model'])]
    if not available:
        raise ValueError('No requested models available for variable-importance figure.')
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 5.4), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {
        'HARX': '#4C78A8',
        'ElasticNet': '#54A24B',
        'RandomForest': '#B279A2',
        'NN10_2': '#E45756',
    }
    labels = {
        'HARX': 'HAR-X',
        'ElasticNet': 'Elastic Net',
        'RandomForest': 'Random Forest',
        'NN10_2': 'NN^10_2',
    }
    for ax, model in zip(axes, available):
        sub = agg[agg['model'] == model].sort_values('vi_mean', ascending=True)
        ax.barh(sub['feature'], sub['vi_mean'], color=colors.get(model, '#4C78A8'), alpha=0.9)
        ax.set_title(labels.get(model, model))
        ax.set_xlabel('Mean normalized VI')
        ax.grid(axis='x', color='#d0d0d0', linewidth=0.7, alpha=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
    axes[0].set_ylabel('Feature')
    fig.suptitle('ALE-based variable importance across tickers')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f'{output_path.name}.tmp.png')
    fig.savefig(tmp, dpi=220, bbox_inches='tight')
    plt.close(fig)
    tmp.replace(output_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Checkpointed paper-style ALE variable-importance analysis for PARTIAL_MALL predictors.'
    )
    ap.add_argument('--config', default=str(ROOT / 'config/paper_core_rolling.yaml'))
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--nn-checkpoint-dir', default='outputs_nn30_checkpointed')
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--horizon', type=int, default=DEFAULT_HORIZON)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--grid-size', type=int, default=None)
    ap.add_argument('--nn-ensemble-top', type=int, default=None)
    ap.add_argument('--tree-n-jobs', type=int, default=1)
    ap.add_argument('--min-in-sample', type=int, default=100)
    ap.add_argument('--min-tickers', type=int, default=10)
    ap.add_argument('--max-features', type=int, default=None, help='Smoke-test helper; omit for full paper-style run.')
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--force', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = _resolve(args.output_dir)
    _assert_output_dir(output_dir, args.allow_existing_output_dir)
    setup_logging(output_dir / 'logs' / '06c_variable_importance.log')

    cfg['models']['trees']['n_jobs'] = int(args.tree_n_jobs)
    models = list(args.models or cfg.get('interpretability', {}).get('models_for_importance', DEFAULT_MODELS))
    grid_size = int(args.grid_size or cfg.get('interpretability', {}).get('ale_grid_size', 100))
    nn_top = int(args.nn_ensemble_top or cfg['models']['neural_network'].get('ensemble_top', 10))
    nn_checkpoint_dir = _resolve(args.nn_checkpoint_dir)
    panel = ALE_HELPERS._load_panel(cfg)
    tickers = _available_tickers(panel, args.tickers)

    logger.info(
        'Task G variable importance output_dir=%s dataset=%s horizon=%s models=%s tickers=%s grid_size=%s max_features=%s',
        output_dir,
        args.dataset,
        int(args.horizon),
        models,
        len(tickers),
        grid_size,
        args.max_features,
    )

    checkpoint_paths = []
    skipped = []
    for model in models:
        for ticker in tickers:
            path = _checkpoint_path(output_dir, model, ticker)
            checkpoint_paths.append(path)
            if path.exists() and not args.force:
                logger.info('Reusing VI checkpoint: %s', path)
                continue
            try:
                logger.info('Computing VI checkpoint model=%s ticker=%s', model, ticker)
                vi = _compute_model_ticker_checkpoint(
                    panel=panel,
                    cfg=cfg,
                    model=model,
                    ticker=ticker,
                    dataset=args.dataset,
                    horizon=int(args.horizon),
                    grid_size=grid_size,
                    nn_checkpoint_dir=nn_checkpoint_dir,
                    nn_top=nn_top,
                    min_in_sample=int(args.min_in_sample),
                    max_features=args.max_features,
                )
            except Exception as exc:
                logger.exception('Skipping failed VI checkpoint model=%s ticker=%s: %s', model, ticker, exc)
                skipped.append({'model': model, 'ticker': ticker, 'error': str(exc)})
                continue
            _atomic_write_csv(vi, path)
            logger.info('Wrote VI checkpoint: %s rows=%s', path, len(vi))

    parts = []
    for path in checkpoint_paths:
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        raise RuntimeError('No variable-importance checkpoints were produced or reusable.')
    per_ticker = pd.concat(parts, ignore_index=True)
    features_lower = set(per_ticker['feature'].astype(str).str.lower())
    if BANNED_IV_FEATURES & features_lower:
        raise RuntimeError(f'IV feature leaked into variable-importance output: {sorted(BANNED_IV_FEATURES & features_lower)}')
    expected_models = set(models)
    got_models = set(per_ticker['model'])
    missing_models = expected_models - got_models
    if missing_models:
        raise RuntimeError(f'Missing requested models in variable-importance checkpoints: {sorted(missing_models)}')

    sums = per_ticker.groupby(['model', 'ticker'])['vi_normalized'].sum()
    bad_sums = sums[(sums < 0.999) | (sums > 1.001)]
    if not bad_sums.empty:
        raise RuntimeError(f'Per model/ticker VI sums are not one: {bad_sums.head().to_dict()}')

    agg = _aggregate_vi(per_ticker, min_tickers=int(args.min_tickers))
    table_path = output_dir / 'tables' / 'variable_importance.csv'
    checkpoint_table_path = output_dir / 'tables' / 'variable_importance_by_ticker.csv'
    figure_path = output_dir / 'figures' / 'figure7_variable_importance.png'
    _atomic_write_csv(agg, table_path)
    _atomic_write_csv(per_ticker, checkpoint_table_path)
    _plot_variable_importance(agg, figure_path, models=models)

    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'config': _path_stats(_resolve(args.config)),
        'output_dir': str(output_dir),
        'parameters': {
            'dataset': args.dataset,
            'horizon': int(args.horizon),
            'models': models,
            'tickers': tickers,
            'grid_size': grid_size,
            'nn_ensemble_top': nn_top,
            'tree_n_jobs': int(args.tree_n_jobs),
            'min_in_sample': int(args.min_in_sample),
            'min_tickers': int(args.min_tickers),
            'max_features': args.max_features,
            'nn_checkpoint_dir': str(nn_checkpoint_dir),
        },
        'outputs': {
            'tables/variable_importance.csv': _path_stats(table_path, rows=len(agg)),
            'tables/variable_importance_by_ticker.csv': _path_stats(checkpoint_table_path, rows=len(per_ticker)),
            'figures/figure7_variable_importance.png': _path_stats(figure_path),
        },
        'checkpoint_count': int(len([p for p in checkpoint_paths if p.exists()])),
        'skipped': skipped,
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info(
        'Wrote variable-importance table rows=%s per_ticker_rows=%s figure=%s skipped=%s',
        len(agg),
        len(per_ticker),
        figure_path,
        len(skipped),
    )


if __name__ == '__main__':
    main()
