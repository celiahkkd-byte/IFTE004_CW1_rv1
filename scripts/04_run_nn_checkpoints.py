from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rv1rep_matplotlib')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import ensure_dirs, load_config, project_path
from rv1rep.features import make_model_frame
from rv1rep.nn import _build_keras_model
from rv1rep.preprocessing import standardizer_from_config, enforce_positive_forecasts, insanity_filter
from rv1rep.split import chronological_split, subset_by_dates
from rv1rep.utils import setup_logging

logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    return value.replace('/', '_').replace('\\', '_').replace(' ', '_')


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _checkpoint_path(root: Path, dataset: str, horizon: int, model: str, ticker: str, seed: int) -> Path:
    return root / 'nn_seed_predictions' / _safe_name(dataset) / f'h{int(horizon)}' / _safe_name(model) / _safe_name(ticker) / f'seed_{seed}.csv'


def _legacy_checkpoint_path(root: Path, dataset: str, model: str, ticker: str, seed: int) -> Path:
    # Pre-horizon layout (h=1 only). Kept for backward compatibility when resuming a run
    # that began before the horizon directory split was introduced.
    return root / 'nn_seed_predictions' / _safe_name(dataset) / _safe_name(model) / _safe_name(ticker) / f'seed_{seed}.csv'


def _model_output_path(root: Path, dataset: str, horizon: int, model: str) -> Path:
    return root / 'predictions' / 'by_model' / f'{_safe_name(dataset)}__h{int(horizon)}__{_safe_name(model)}.csv'


def _resolve_output_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = ROOT / root
    return root.resolve()


def _fit_one_seed(
    *,
    model_name: str,
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, float, dict]:
    nn_cfg = cfg['models']['neural_network']
    hidden = list(nn_cfg['architectures'][model_name])
    use_tf = bool(nn_cfg.get('use_tensorflow', True))
    if use_tf:
        try:
            import tensorflow as tf  # type: ignore

            model = _build_keras_model(
                X_train.shape[1],
                hidden,
                dropout=float(nn_cfg.get('dropout', 0.8)),
                learning_rate=float(nn_cfg.get('learning_rate', 0.001)),
                seed=seed,
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=int(nn_cfg.get('patience', 100)),
                    restore_best_weights=True,
                )
            ]
            model.fit(
                X_train.to_numpy(),
                y_train.to_numpy(),
                validation_data=(X_val.to_numpy(), y_val.to_numpy()),
                epochs=int(nn_cfg.get('epochs', 500)),
                batch_size=int(nn_cfg.get('batch_size', 64)),
                verbose=0,
                callbacks=callbacks,
            )
            pred_val = model.predict(X_val.to_numpy(), verbose=0).reshape(-1)
            pred_test = model.predict(X_test.to_numpy(), verbose=0).reshape(-1)
            params = {'backend': 'tensorflow', 'hidden': hidden, 'seed': seed}
            return pred_test, float(mean_squared_error(y_val, pred_val)), params
        except Exception as exc:
            logger.warning('TensorFlow seed failed for %s seed=%s; falling back to sklearn MLP: %s', model_name, seed, exc)

    est = MLPRegressor(
        hidden_layer_sizes=tuple(hidden),
        activation='relu',
        solver='adam',
        learning_rate_init=float(nn_cfg.get('learning_rate', 0.001)),
        max_iter=int(nn_cfg.get('epochs', 500)),
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=max(10, min(50, int(nn_cfg.get('patience', 100)))),
        random_state=seed,
    )
    est.fit(X_train, y_train)
    pred_val = est.predict(X_val)
    pred_test = est.predict(X_test)
    params = {'backend': 'sklearn_fallback', 'hidden': hidden, 'seed': seed}
    return np.asarray(pred_test, dtype=float), float(mean_squared_error(y_val, pred_val)), params


def _load_seed_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    required = {'date', 'ticker', 'actual_rv', 'forecast_raw', 'seed', 'val_mse'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    return df


def _prepare_asset_frame(panel: pd.DataFrame, cfg: dict, dataset: str, model_name: str, ticker: str, horizon: int):
    frame, feature_cols, target_col = make_model_frame(panel, model_name, dataset, int(horizon))
    df_asset = frame[frame['ticker'] == ticker].copy()
    if df_asset.empty:
        raise ValueError(f'No rows for {dataset}/{model_name}/{ticker}')
    split = chronological_split(
        df_asset['date'],
        train_frac=cfg['splitting']['train_frac'],
        val_frac=cfg['splitting']['val_frac'],
        fixed_train_days=cfg['splitting'].get('fixed_train_days'),
        fixed_val_days=cfg['splitting'].get('fixed_val_days'),
    )
    train = subset_by_dates(df_asset, split.train_dates)
    val = subset_by_dates(df_asset, split.val_dates)
    test = subset_by_dates(df_asset, split.test_dates)
    scaler = standardizer_from_config(cfg).fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    y_train = train[target_col]
    y_val = val[target_col]
    actual_rv_col = target_col.replace('target_log_rv_', 'target_rv_')
    test_base = test[['date', 'ticker', 'rv', 'oc_logret', 'cc_logret', actual_rv_col]].copy()
    test_base = test_base.rename(columns={actual_rv_col: 'actual_rv'})
    in_sample_rv = pd.concat([train['rv'], val['rv']]).dropna()
    return {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'train': train,
        'val': val,
        'test': test,
        'test_base': test_base,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'in_sample_min_rv': float(in_sample_rv.min()),
        'in_sample_mean_rv': float(in_sample_rv.mean()),
    }


def train_missing_seed_checkpoints(
    *,
    panel: pd.DataFrame,
    cfg: dict,
    output_root: Path,
    datasets: list[str],
    horizons: list[int],
    models: list[str],
    tickers: list[str],
    seeds: list[int],
    force: bool,
) -> None:
    for dataset in datasets:
        for horizon in horizons:
            for model_name in models:
                for ticker in tickers:
                    prepared = None
                    for seed in seeds:
                        path = _checkpoint_path(output_root, dataset, horizon, model_name, ticker, seed)
                        if path.exists() and not force:
                            logger.info('Reusing seed checkpoint: %s', path)
                            continue
                        if int(horizon) == 1 and not force:
                            legacy = _legacy_checkpoint_path(output_root, dataset, model_name, ticker, seed)
                            if legacy.exists():
                                logger.info('Migrating legacy h=1 seed checkpoint: %s -> %s', legacy, path)
                                path.parent.mkdir(parents=True, exist_ok=True)
                                legacy.replace(path)
                                continue
                        if prepared is None:
                            prepared = _prepare_asset_frame(panel, cfg, dataset, model_name, ticker, horizon)
                        logger.info(
                            'Training NN checkpoint dataset=%s horizon=%s model=%s ticker=%s seed=%s',
                            dataset, horizon, model_name, ticker, seed,
                        )
                        pred_raw, val_mse, params = _fit_one_seed(
                            model_name=model_name,
                            seed=seed,
                            X_train=prepared['X_train'],
                            y_train=prepared['y_train'],
                            X_val=prepared['X_val'],
                            y_val=prepared['y_val'],
                            X_test=prepared['X_test'],
                            cfg=cfg,
                        )
                        out = prepared['test_base'].copy()
                        out['forecast_raw'] = pred_raw
                        out['seed'] = seed
                        out['val_mse'] = val_mse
                        out['model'] = model_name
                        out['dataset'] = dataset
                        out['horizon'] = int(horizon)
                        out['features'] = ','.join(prepared['feature_cols'])
                        out['scheme'] = cfg['estimation']['scheme']
                        out['n_train'] = len(prepared['train'])
                        out['n_val'] = len(prepared['val'])
                        out['params'] = json.dumps(params, sort_keys=True)
                        out['in_sample_min_rv'] = prepared['in_sample_min_rv']
                        out['in_sample_mean_rv'] = prepared['in_sample_mean_rv']
                        _atomic_write_csv(out, path)


def aggregate_nn_checkpoints(
    *,
    panel: pd.DataFrame,
    cfg: dict,
    output_root: Path,
    datasets: list[str],
    horizons: list[int],
    models: list[str],
    tickers: list[str],
    seeds: list[int],
    ensemble_top: int,
    require_all_seeds: bool,
) -> pd.DataFrame:
    all_model_outputs = []
    for dataset in datasets:
        for horizon in horizons:
            for model_name in models:
                model_parts = []
                for ticker in tickers:
                    available = []
                    for seed in seeds:
                        path = _checkpoint_path(output_root, dataset, horizon, model_name, ticker, seed)
                        if path.exists():
                            df_seed = _load_seed_file(path)
                            available.append((seed, float(df_seed['val_mse'].iloc[0]), path))
                    if require_all_seeds and len(available) != len(seeds):
                        missing = sorted(set(seeds).difference(seed for seed, _, _ in available))
                        raise RuntimeError(
                            f'Missing seed checkpoints for {dataset}/h{horizon}/{model_name}/{ticker}: {missing}'
                        )
                    if len(available) < ensemble_top:
                        raise RuntimeError(
                            f'Need at least ensemble_top={ensemble_top} checkpoints for '
                            f'{dataset}/h{horizon}/{model_name}/{ticker}; found {len(available)}'
                        )
                    selected = sorted(available, key=lambda x: x[1])[:ensemble_top]
                    seed_frames = [_load_seed_file(path) for _, _, path in selected]
                    base = seed_frames[0].copy()
                    for other in seed_frames[1:]:
                        if not base[['date', 'ticker']].equals(other[['date', 'ticker']]):
                            raise RuntimeError(
                                f'Test rows differ across selected seeds for {dataset}/h{horizon}/{model_name}/{ticker}'
                            )
                    raw_matrix = np.vstack([f['forecast_raw'].to_numpy(dtype=float) for f in seed_frames])
                    raw_pred = raw_matrix.mean(axis=0)
                    in_min = float(base['in_sample_min_rv'].iloc[0])
                    in_mean = float(base['in_sample_mean_rv'].iloc[0])
                    pred = enforce_positive_forecasts(raw_pred, in_min, cfg['estimation']['negative_forecast_policy'])
                    if cfg['estimation']['insanity_filter']['enabled']:
                        pred = insanity_filter(pred, in_mean, in_min, cfg['estimation']['insanity_filter']['max_multiple_of_in_sample_mean'])
                    out = base[['date', 'ticker', 'rv', 'oc_logret', 'cc_logret', 'actual_rv']].copy()
                    out['model'] = model_name
                    out['forecast_rv'] = pred
                    out['scheme'] = cfg['estimation']['scheme']
                    out['n_train'] = int(base['n_train'].iloc[0])
                    out['n_val'] = int(base['n_val'].iloc[0])
                    params = {
                        'backend': sorted(set(json.loads(f['params'].iloc[0])['backend'] for f in seed_frames)),
                        'hidden': json.loads(seed_frames[0]['params'].iloc[0])['hidden'],
                        'seeds_available': len(available),
                        'ensemble_top': ensemble_top,
                        'selected_seeds': [seed for seed, _, _ in selected],
                        'best_val_mse': min(loss for _, loss, _ in selected),
                        'checkpointed': True,
                    }
                    out['params'] = str(params)
                    out['dataset'] = dataset
                    out['horizon'] = int(horizon)
                    out['features'] = str(base['features'].iloc[0])
                    model_parts.append(out)
                model_out = pd.concat(model_parts, ignore_index=True).sort_values(['ticker', 'date'])
                model_path = _model_output_path(output_root, dataset, horizon, model_name)
                _atomic_write_csv(model_out, model_path)
                logger.info('Wrote dataset-model aggregate: %s rows=%d', model_path, len(model_out))
                all_model_outputs.append(model_out)
    nn_out = pd.concat(all_model_outputs, ignore_index=True).sort_values(['dataset', 'horizon', 'model', 'ticker', 'date'])
    _atomic_write_csv(nn_out, output_root / 'predictions' / 'nn_model_predictions.csv')
    return nn_out


def combine_with_base(output_root: Path, nn_out: pd.DataFrame, base_predictions: Path | None) -> pd.DataFrame:
    if base_predictions is None:
        final = nn_out
    else:
        base = pd.read_csv(base_predictions, parse_dates=['date'])
        base = base[~base['model'].astype(str).str.startswith('NN')].copy()
        final = pd.concat([base, nn_out], ignore_index=True)
    final = final.sort_values(['dataset', 'horizon', 'model', 'ticker', 'date']).reset_index(drop=True)
    duplicates = final.duplicated(['date', 'ticker', 'dataset', 'horizon', 'model']).sum()
    if duplicates:
        raise RuntimeError(f'Duplicate final prediction keys: {duplicates}')
    _atomic_write_csv(final, output_root / 'predictions' / 'model_predictions.csv')
    return final


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Checkpointed NN runner with per-seed reuse and per-model outputs.')
    ap.add_argument('--config', default=str(ROOT / 'config/default.yaml'))
    ap.add_argument('--output-dir', default='outputs_nn_checkpointed')
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--horizons', nargs='*', type=int, default=None,
                    help='Forecast horizons. Defaults to experiments.horizons in config.')
    ap.add_argument('--models', nargs='*', default=['NN1', 'NN2', 'NN3', 'NN4'])
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--seed-count', type=int, default=None, help='Use config random_seed + range(seed_count).')
    ap.add_argument('--seed-start', type=int, default=None, help='Explicit first seed. Defaults to config project.random_seed.')
    ap.add_argument('--seed-end', type=int, default=None, help='Inclusive explicit final seed.')
    ap.add_argument('--ensemble-top', type=int, default=None)
    ap.add_argument('--base-predictions', default='outputs/predictions/model_predictions_no_nn.csv')
    ap.add_argument('--train-only', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--force', action='store_true', help='Retrain existing seed checkpoints.')
    ap.add_argument('--allow-existing-output-dir', action='store_true')
    ap.add_argument('--allow-main-output-dir', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # The paper uses a fixed train/validation/test split for NNs. Keep this
    # explicit so paper-core configs can default to rolling for non-NN models.
    cfg['estimation']['scheme'] = 'fixed'
    output_root = _resolve_output_root(args.output_dir)
    protected_dirs = {_resolve_output_root('outputs'), _resolve_output_root('outputs_full_nn')}
    if output_root in protected_dirs and not args.allow_main_output_dir:
        raise SystemExit('Refusing to write checkpointed NN outputs into existing main result directories.')
    if output_root.exists() and not args.allow_existing_output_dir:
        logger.warning('Output directory exists; existing seed checkpoints will be reused: %s', output_root)
    cfg['paths']['output_dir'] = str(output_root)
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, 'output_dir') / 'logs' / '04_nn_checkpoints.log')

    datasets = args.datasets or list(cfg['experiments']['datasets'])
    horizons = [int(h) for h in (args.horizons or cfg['experiments']['horizons'])]
    if not horizons:
        raise SystemExit('No horizons selected.')
    models = args.models
    if not all(m.startswith('NN') for m in models):
        raise SystemExit('This checkpoint runner is only for NN models.')
    if args.tickers is None:
        panel_probe = pd.read_csv(project_path(cfg, 'processed_dir') / 'forecasting_panel.csv', usecols=['ticker'])
        tickers = sorted(panel_probe['ticker'].unique())
    else:
        tickers = args.tickers

    base_seed = int(cfg['project']['random_seed']) if args.seed_start is None else int(args.seed_start)
    if args.seed_end is not None:
        seeds = list(range(base_seed, int(args.seed_end) + 1))
    else:
        seed_count = int(args.seed_count if args.seed_count is not None else cfg['models']['neural_network'].get('seeds', 20))
        seeds = [base_seed + i for i in range(seed_count)]
    ensemble_top = int(args.ensemble_top if args.ensemble_top is not None else cfg['models']['neural_network'].get('ensemble_top', 10))
    if ensemble_top > len(seeds):
        raise SystemExit(f'ensemble_top={ensemble_top} cannot exceed available seed count={len(seeds)}')

    panel = pd.read_csv(project_path(cfg, 'processed_dir') / 'forecasting_panel.csv', parse_dates=['date'])
    available_targets = {c for c in panel.columns if c.startswith('target_rv_h') or c.startswith('target_log_rv_h')}
    for h in horizons:
        if f'target_rv_h{int(h)}' not in available_targets:
            raise SystemExit(
                f'forecasting_panel.csv has no target_rv_h{int(h)} column. '
                f'Rebuild the panel via scripts/03_build_features.py with the desired horizons.'
            )
    logger.info(
        'Checkpointed NN run output_dir=%s datasets=%s horizons=%s models=%s tickers=%d seeds=%s ensemble_top=%d',
        output_root,
        datasets,
        horizons,
        models,
        len(tickers),
        f'{seeds[0]}..{seeds[-1]}',
        ensemble_top,
    )
    if not args.aggregate_only:
        train_missing_seed_checkpoints(
            panel=panel,
            cfg=cfg,
            output_root=output_root,
            datasets=datasets,
            horizons=horizons,
            models=models,
            tickers=tickers,
            seeds=seeds,
            force=args.force,
        )
    if args.train_only:
        return
    nn_out = aggregate_nn_checkpoints(
        panel=panel,
        cfg=cfg,
        output_root=output_root,
        datasets=datasets,
        horizons=horizons,
        models=models,
        tickers=tickers,
        seeds=seeds,
        ensemble_top=ensemble_top,
        require_all_seeds=True,
    )
    base_path = Path(args.base_predictions) if args.base_predictions else None
    final = combine_with_base(output_root, nn_out, base_path)
    logger.info('Wrote checkpointed final predictions rows=%d models=%d', len(final), final['model'].nunique())


if __name__ == '__main__':
    main()
