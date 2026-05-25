from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1rep_matplotlib')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import load_config, project_path
from rv1rep.explain import accumulated_local_effect
from rv1rep.features import make_model_frame
from rv1rep.models import fit_sklearn_model
from rv1rep.nn import _build_keras_model
from rv1rep.preprocessing import standardizer_from_config, enforce_positive_forecasts, insanity_filter
from rv1rep.split import chronological_split, subset_by_dates
from rv1rep.utils import require_columns, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ['HARX', 'LogHAR', 'ElasticNet', 'RandomForest', 'NN1']
DEFAULT_FEATURES = ['rvd', 'rvw', 'm1w']
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
            raise SystemExit(f'Refusing to write ALE output inside protected result directory: {output_dir}')
    if output_dir.exists() and not allow_existing:
        files = [p for p in output_dir.rglob('*') if p.is_file()]
        if files:
            raise SystemExit(
                f'Output directory already exists and contains files: {output_dir}. '
                'Use a fresh directory or pass --allow-existing-output-dir to resume ALE checkpoints.'
            )


def _safe_name(value: str) -> str:
    return value.replace('/', '_').replace('\\', '_').replace(' ', '_')


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


def _checkpoint_path(output_dir: Path, dataset: str, horizon: int, ticker: str, model: str, feature: str) -> Path:
    return (
        output_dir
        / 'checkpoints'
        / _safe_name(dataset)
        / f'h{int(horizon)}'
        / _safe_name(ticker)
        / _safe_name(model)
        / f'{_safe_name(feature)}.csv'
    )


def _load_panel(cfg: dict) -> pd.DataFrame:
    path = project_path(cfg, 'processed_dir') / 'forecasting_panel.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing forecasting panel: {path}')
    return pd.read_csv(path, parse_dates=['date'])


def _prepare_model_data(
    panel: pd.DataFrame,
    cfg: dict,
    *,
    dataset: str,
    model_name: str,
    ticker: str,
    horizon: int,
) -> dict:
    frame_model = 'NN2' if model_name == 'NN10_2' else model_name
    frame, feature_cols, target_col = make_model_frame(panel, frame_model, dataset, int(horizon))
    df_asset = frame[frame['ticker'].astype(str).str.upper() == ticker.upper()].copy()
    if df_asset.empty:
        raise ValueError(f'No rows for ALE frame: dataset={dataset} horizon={horizon} model={model_name} ticker={ticker}')
    split = chronological_split(
        df_asset['date'],
        train_frac=cfg['splitting']['train_frac'],
        val_frac=cfg['splitting']['val_frac'],
        fixed_train_days=cfg['splitting'].get('fixed_train_days'),
        fixed_val_days=cfg['splitting'].get('fixed_val_days'),
    )
    train = subset_by_dates(df_asset, split.train_dates)
    val = subset_by_dates(df_asset, split.val_dates)
    if len(train) + len(val) < 1000:
        raise RuntimeError(
            f'ALE requires at least 1000 train+validation rows; got {len(train) + len(val)} for {model_name}/{ticker}.'
        )
    scaler = standardizer_from_config(cfg).fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_in = pd.concat([X_train, X_val], ignore_index=True)
    y_train = train[target_col]
    y_val = val[target_col]
    y_in = pd.concat([y_train, y_val], ignore_index=True)
    in_sample_rv = pd.concat([train['rv'], val['rv']]).dropna()
    return {
        'frame_model': frame_model,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'train': train,
        'val': val,
        'X_train': X_train,
        'X_val': X_val,
        'X_in': X_in,
        'y_train': y_train,
        'y_val': y_val,
        'y_in': y_in,
        'in_sample_min_rv': float(in_sample_rv.min()),
        'in_sample_mean_rv': float(in_sample_rv.mean()),
    }


def _postprocess_prediction(raw_pred: np.ndarray, cfg: dict, in_min: float, in_mean: float) -> np.ndarray:
    pred = enforce_positive_forecasts(raw_pred, in_min, cfg['estimation']['negative_forecast_policy'])
    filt_cfg = cfg['estimation'].get('insanity_filter', {})
    if bool(filt_cfg.get('enabled', False)):
        pred = insanity_filter(pred, in_mean, in_min, float(filt_cfg['max_multiple_of_in_sample_mean']))
    return pred


def _fit_sklearn_predictor(model_name: str, prepared: dict, cfg: dict) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    est, params = fit_sklearn_model(
        model_name,
        prepared['X_train'],
        prepared['y_train'],
        prepared['X_val'],
        prepared['y_val'],
        cfg,
        random_state=int(cfg['project']['random_seed']),
    )
    log_bias_var = 0.0
    if model_name.upper() == 'LOGHAR':
        fitted = est.predict(prepared['X_in'])
        log_bias_var = float(np.var(prepared['y_in'].to_numpy() - fitted, ddof=1))

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(est.predict(X[prepared['feature_cols']]), dtype=float)
        if model_name.upper() == 'LOGHAR':
            raw = np.exp(raw + 0.5 * log_bias_var)
        return _postprocess_prediction(raw, cfg, prepared['in_sample_min_rv'], prepared['in_sample_mean_rv'])

    fit_info = {
        'model': model_name,
        'estimator': type(est).__name__,
        'params': params,
        'target_col': prepared['target_col'],
        'feature_cols': prepared['feature_cols'],
        'n_train': len(prepared['train']),
        'n_val': len(prepared['val']),
        'log_bias_var': log_bias_var,
    }
    return predict_fn, fit_info


def _seed_checkpoint_dir(input_dir: Path, dataset: str, horizon: int, model: str, ticker: str) -> Path:
    return input_dir / 'nn_seed_predictions' / dataset / f'h{int(horizon)}' / model / ticker


def _select_nn1_best_seed(input_dir: Path, dataset: str, horizon: int, ticker: str) -> tuple[int, dict]:
    directory = _seed_checkpoint_dir(input_dir, dataset, horizon, 'NN1', ticker)
    files = sorted(directory.glob('seed_*.csv'))
    if not files:
        raise FileNotFoundError(f'No NN1 seed checkpoints found in {directory}')
    rows = []
    for path in files:
        head = pd.read_csv(path, nrows=1)
        require_columns(head, ['seed', 'val_mse', 'params'], str(path))
        rows.append({'seed': int(head['seed'].iloc[0]), 'val_mse': float(head['val_mse'].iloc[0]), 'path': str(path)})
    best = sorted(rows, key=lambda r: (r['val_mse'], r['seed']))[0]
    return int(best['seed']), best


def _fit_nn1_predictor(prepared: dict, cfg: dict, best_seed: int) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('TensorFlow is required for NN1 ALE.') from exc

    nn_cfg = cfg['models']['neural_network']
    hidden = list(nn_cfg['architectures']['NN1'])
    logger.info('Fitting NN1 ALE seed=%s', best_seed)
    model = _build_keras_model(
        prepared['X_train'].shape[1],
        hidden,
        dropout=float(nn_cfg.get('dropout', 0.8)),
        learning_rate=float(nn_cfg.get('learning_rate', 0.001)),
        seed=int(best_seed),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=int(nn_cfg.get('patience', 100)),
            restore_best_weights=True,
        )
    ]
    model.fit(
        prepared['X_train'].to_numpy(),
        prepared['y_train'].to_numpy(),
        validation_data=(prepared['X_val'].to_numpy(), prepared['y_val'].to_numpy()),
        epochs=int(nn_cfg.get('epochs', 500)),
        batch_size=int(nn_cfg.get('batch_size', 64)),
        verbose=0,
        callbacks=callbacks,
    )
    pred_val = model.predict(prepared['X_val'].to_numpy(), verbose=0).reshape(-1)
    val_mse = float(np.mean((prepared['y_val'].to_numpy() - pred_val) ** 2))

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        arr = X[prepared['feature_cols']].to_numpy()
        raw = model.predict(arr, verbose=0).reshape(-1)
        return _postprocess_prediction(raw, cfg, prepared['in_sample_min_rv'], prepared['in_sample_mean_rv'])

    fit_info = {
        'model': 'NN1',
        'source_architecture': 'NN1',
        'hidden': hidden,
        'best_seed': int(best_seed),
        'refit_for_ale': True,
        'val_mse_after_refit': val_mse,
        'target_col': prepared['target_col'],
        'feature_cols': prepared['feature_cols'],
        'n_train': len(prepared['train']),
        'n_val': len(prepared['val']),
    }
    return predict_fn, fit_info


def _compute_ale_checkpoint(
    *,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X_in: pd.DataFrame,
    feature: str,
    grid_size: int,
    dataset: str,
    horizon: int,
    ticker: str,
    model: str,
    fit_info: dict,
) -> pd.DataFrame:
    model_feature = _model_feature_name(model, feature, X_in.columns)
    if model_feature not in X_in.columns:
        raise ValueError(f'Feature {feature} is not available for {model}; columns={list(X_in.columns)}')
    ale = accumulated_local_effect(predict_fn, X_in, model_feature, grid_size=grid_size)
    ale = ale.rename(columns={'x': 'x_standardized'})
    ale['model_feature'] = model_feature
    ale['paper_feature'] = feature
    ale.insert(0, 'ticker', ticker)
    ale.insert(1, 'dataset', dataset)
    ale.insert(2, 'horizon', int(horizon))
    ale.insert(3, 'model', model)
    ale['grid_size'] = int(grid_size)
    ale['n_in_sample'] = int(len(X_in))
    ale['fit_info'] = json.dumps(fit_info, sort_keys=True)
    ale['feature'] = feature
    return ale[
        [
            'ticker',
            'dataset',
            'horizon',
            'model',
            'feature',
            'model_feature',
            'paper_feature',
            'x_standardized',
            'ale',
            'grid_size',
            'n_in_sample',
            'fit_info',
        ]
    ]


def _model_feature_name(model: str, paper_feature: str, columns: pd.Index | list[str]) -> str:
    cols = set(columns)
    if model.upper() == 'LOGHAR':
        mapping = {'rvd': 'log_rvd', 'rvw': 'log_rvw', 'rvm': 'log_rvm'}
        candidate = mapping.get(paper_feature, paper_feature)
        if candidate in cols:
            return candidate
    return paper_feature


def _plot_ale(ale_table: pd.DataFrame, output_path: Path, models: list[str], features: list[str]) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(features), len(models), figsize=(3.2 * len(models), 2.4 * len(features)), sharex=True)
    if len(features) == 1 and len(models) == 1:
        axes = np.array([[axes]])
    elif len(features) == 1:
        axes = np.array([axes])
    elif len(models) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, feature in enumerate(features):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = ale_table[(ale_table['feature'] == feature) & (ale_table['model'] == model)].sort_values('x_standardized')
            sub = sub[(sub['x_standardized'] >= -1.0) & (sub['x_standardized'] <= 1.0)]
            if not sub.empty:
                ax.plot(sub['x_standardized'], sub['ale'], color='#1f77b4', linewidth=1.8)
                ax.axhline(0.0, color='black', linewidth=0.7, alpha=0.5)
            if i == 0:
                ax.set_title(model, fontsize=10)
            if j == 0:
                ax.set_ylabel(feature, fontsize=10)
            ax.set_xlim(-1.0, 1.0)
            ax.grid(True, linewidth=0.4, alpha=0.35)
            ax.tick_params(labelsize=8)
    fig.supxlabel('Standardized feature value', fontsize=10)
    fig.supylabel('Accumulated local effect', fontsize=10)
    fig.tight_layout()
    tmp = output_path.parent / f'{output_path.name}.tmp.png'
    fig.savefig(tmp, dpi=200)
    plt.close(fig)
    tmp.replace(output_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Checkpointed Figure 6-style ALE computation for the reproduction project.')
    ap.add_argument('--config', default=str(ROOT / 'config/paper_core_rolling.yaml'))
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--nn-checkpoint-dir', default='outputs_nn30_checkpointed')
    ap.add_argument('--dataset', default='PARTIAL_MALL')
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--ticker', default='AAPL')
    ap.add_argument('--models', nargs='*', default=DEFAULT_MODELS)
    ap.add_argument('--features', nargs='*', default=DEFAULT_FEATURES)
    ap.add_argument('--grid-size', type=int, default=None)
    ap.add_argument('--nn-ensemble-top', type=int, default=None)
    ap.add_argument('--tree-n-jobs', type=int, default=1)
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--force', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = _resolve(args.output_dir)
    _assert_output_dir(output_dir, args.allow_existing_output_dir)
    setup_logging(output_dir / 'logs' / '08_compute_ale_checkpointed.log')

    cfg['models']['trees']['n_jobs'] = int(args.tree_n_jobs)
    grid_size = int(args.grid_size or cfg['interpretability'].get('ale_grid_size', 100))
    nn_top = int(args.nn_ensemble_top or cfg['models']['neural_network'].get('ensemble_top', 10))
    nn_checkpoint_dir = _resolve(args.nn_checkpoint_dir)
    panel = _load_panel(cfg)

    logger.info(
        'Task E ALE output_dir=%s dataset=%s horizon=%s ticker=%s models=%s features=%s grid_size=%s',
        output_dir,
        args.dataset,
        args.horizon,
        args.ticker,
        args.models,
        args.features,
        grid_size,
    )

    checkpoint_paths = []
    model_fit_summaries = []
    for model in args.models:
        prepared = _prepare_model_data(panel, cfg, dataset=args.dataset, model_name=model, ticker=args.ticker, horizon=args.horizon)
        if model == 'NN1':
            best_seed, best_meta = _select_nn1_best_seed(
                nn_checkpoint_dir, args.dataset, int(args.horizon), args.ticker
            )
            predict_fn, fit_info = _fit_nn1_predictor(prepared, cfg, best_seed)
            fit_info['best_seed_source'] = best_meta
        else:
            predict_fn, fit_info = _fit_sklearn_predictor(model, prepared, cfg)
        model_fit_summaries.append(fit_info)
        for feature in args.features:
            path = _checkpoint_path(output_dir, args.dataset, int(args.horizon), args.ticker, model, feature)
            checkpoint_paths.append(path)
            if path.exists() and not args.force:
                logger.info('Reusing ALE checkpoint: %s', path)
                continue
            logger.info('Computing ALE checkpoint model=%s feature=%s', model, feature)
            ale = _compute_ale_checkpoint(
                predict_fn=predict_fn,
                X_in=prepared['X_in'],
                feature=feature,
                grid_size=grid_size,
                dataset=args.dataset,
                horizon=int(args.horizon),
                ticker=args.ticker,
                model=model,
                fit_info=fit_info,
            )
            _atomic_write_csv(ale, path)
            logger.info('Wrote ALE checkpoint: %s rows=%d', path, len(ale))

    parts = []
    missing = []
    for path in checkpoint_paths:
        if not path.exists():
            missing.append(str(path))
        else:
            parts.append(pd.read_csv(path))
    if missing:
        raise RuntimeError(f'Missing ALE checkpoints after run: {missing}')
    ale_table = pd.concat(parts, ignore_index=True)
    expected_models = set(args.models)
    expected_features = set(args.features)
    if set(ale_table['model']) != expected_models:
        raise RuntimeError(f'Unexpected ALE models: {sorted(ale_table["model"].unique())}')
    if set(ale_table['feature']) != expected_features:
        raise RuntimeError(f'Unexpected ALE features: {sorted(ale_table["feature"].unique())}')
    _atomic_write_csv(ale_table, output_dir / 'tables' / 'ale_table.csv')
    _plot_ale(ale_table, output_dir / 'figures' / 'figure6_ale.png', args.models, args.features)

    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv,
        'config': _path_stats(_resolve(args.config)),
        'output_dir': str(output_dir),
        'dataset': args.dataset,
        'horizon': int(args.horizon),
        'ticker': args.ticker,
        'models': args.models,
        'features': args.features,
        'grid_size': grid_size,
        'tree_n_jobs': int(args.tree_n_jobs),
        'nn_checkpoint_dir': str(nn_checkpoint_dir),
        'rows': int(len(ale_table)),
        'model_fit_summaries': model_fit_summaries,
        'checkpoint_files': [_path_stats(path) for path in checkpoint_paths],
    }
    _atomic_write_json(provenance, output_dir / 'run_provenance.json')
    logger.info('Wrote ALE table rows=%d and figure=%s', len(ale_table), output_dir / 'figures' / 'figure6_ale.png')


if __name__ == '__main__':
    main()
