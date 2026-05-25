from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class FittedModel:
    name: str
    model: object
    feature_cols: List[str]
    target_is_log: bool = False
    log_bias_var: float = 0.0
    selected_params: Optional[Dict] = None
    scaler: object | None = None
    in_sample_min_rv: float = np.nan
    in_sample_mean_rv: float = np.nan

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_in = X[self.feature_cols]
        if self.scaler is not None:
            X_in = self.scaler.transform(X_in)
        yhat = np.asarray(self.model.predict(X_in), dtype=float)
        if self.target_is_log:
            # Bias correction for LogHAR-style forecasts, as discussed in the paper.
            yhat = np.exp(yhat + 0.5 * self.log_bias_var)
        return yhat


class WeightedLasso:
    """Approximate adaptive lasso via feature rescaling.

    Solves min ||y - X beta||^2 + lambda * sum_j w_j |beta_j| by setting theta_j=w_j beta_j
    and fitting a standard lasso on X_j / w_j.
    """
    def __init__(self, alpha: float = 1.0, weights: Optional[np.ndarray] = None, max_iter: int = 10000, random_state: int = 42):
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator_: Optional[Lasso] = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.weights is None:
            self.weights_ = np.ones(X.shape[1])
        else:
            self.weights_ = np.asarray(self.weights, dtype=float)
            self.weights_ = np.where(self.weights_ <= 0, 1.0, self.weights_)
        X_scaled = X / self.weights_
        self.estimator_ = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.estimator_.fit(X_scaled, y)
        return self

    def predict(self, X):
        if self.estimator_ is None:
            raise RuntimeError('WeightedLasso not fitted')
        return self.estimator_.predict(np.asarray(X, dtype=float) / self.weights_)

    @property
    def coef_(self):
        if self.estimator_ is None:
            raise RuntimeError('WeightedLasso not fitted')
        return self.estimator_.coef_ / self.weights_


def _mse(est, X, y) -> float:
    return float(mean_squared_error(y, est.predict(X)))


def select_by_validation(candidates: Dict[str, object], X_train, y_train, X_val, y_val) -> Tuple[str, object, float]:
    best_key, best_est, best_loss = None, None, np.inf
    for key, est in candidates.items():
        est.fit(X_train, y_train)
        loss = _mse(est, X_val, y_val)
        if loss < best_loss:
            best_key, best_est, best_loss = key, est, loss
    assert best_key is not None and best_est is not None
    return best_key, best_est, best_loss


def build_alpha_grid(alpha_min: float, alpha_max: float, size: int) -> np.ndarray:
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max), int(size))


def _fit_gb_warm_start_chain(
    depth: int,
    lr: float,
    n_estimators_sorted: list,
    X_train,
    y_train,
    X_val,
    y_val,
    random_state: int,
) -> Tuple[str, object, float]:
    """Fit one (depth, lr) GB chain via warm_start, tracking the best n_estimators checkpoint."""
    est = GradientBoostingRegressor(
        max_depth=int(depth),
        n_estimators=int(n_estimators_sorted[0]),
        learning_rate=float(lr),
        warm_start=True,
        random_state=random_state,
    )
    best_key: str | None = None
    best_est: object | None = None
    best_loss: float = np.inf
    for n_est in n_estimators_sorted:
        est.n_estimators = int(n_est)
        est.fit(X_train, y_train)
        loss = _mse(est, X_val, y_val)
        if loss < best_loss:
            best_loss = loss
            best_key = f'depth={depth},trees={n_est},lr={lr}'
            best_est = copy.deepcopy(est)
            best_est.warm_start = False
    assert best_key is not None and best_est is not None
    return best_key, best_est, best_loss


def _refit_tuned_models_on_train_validation(cfg: Dict) -> bool:
    """Whether validation-tuned estimators are refit on train+validation.

    By default, match the paper-core replication: keep the validation block
    reserved for hyperparameter selection, then use the selected training-fitted
    candidate directly. Set the config flag to true only for an explicit legacy
    train+validation refit experiment.
    """
    return bool(cfg.get('estimation', {}).get('refit_tuned_models_on_train_validation', False))


def _maybe_refit_selected_tuned_model(est, X_train, y_train, X_val, y_val, cfg: Dict):
    if _refit_tuned_models_on_train_validation(cfg):
        est.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return est, 'train_plus_validation_after_selection'
    return est, 'train_only_after_validation_selection'


def fit_sklearn_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Dict,
    random_state: int = 42,
):
    name_u = name.upper()
    if name_u in ['HAR', 'HARX', 'LOGHAR', 'LEVHAR', 'SHAR', 'HARQ']:
        est = LinearRegression()
        est.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return est, {'method': 'OLS_train_plus_validation'}

    reg_cfg = cfg['models']['regularization']
    alpha_grid = build_alpha_grid(reg_cfg['alpha_min'], reg_cfg['alpha_max'], int(reg_cfg['alpha_grid_size']))

    if name_u == 'RIDGE':
        candidates = {f'alpha={a:.3g}': Ridge(alpha=float(a), random_state=random_state) for a in alpha_grid}
        key, est, loss = select_by_validation(candidates, X_train, y_train, X_val, y_val)
        est, fit_sample = _maybe_refit_selected_tuned_model(est, X_train, y_train, X_val, y_val, cfg)
        return est, {'selected': key, 'val_mse': loss, 'fit_sample': fit_sample}

    if name_u == 'LASSO':
        candidates = {f'alpha={a:.3g}': Lasso(alpha=float(a), max_iter=20000, random_state=random_state) for a in alpha_grid}
        key, est, loss = select_by_validation(candidates, X_train, y_train, X_val, y_val)
        est, fit_sample = _maybe_refit_selected_tuned_model(est, X_train, y_train, X_val, y_val, cfg)
        return est, {'selected': key, 'val_mse': loss, 'fit_sample': fit_sample}

    if name_u == 'ELASTICNET':
        candidates = {}
        for a in alpha_grid:
            for l1 in reg_cfg['elastic_l1_ratios']:
                # sklearn l1_ratio: 1=Lasso, 0=Ridge-like but not numerically ideal.
                l1 = float(l1)
                if l1 == 0.0:
                    l1 = 1e-6
                candidates[f'alpha={a:.3g},l1={l1:.3g}'] = ElasticNet(alpha=float(a), l1_ratio=l1, max_iter=20000, random_state=random_state)
        key, est, loss = select_by_validation(candidates, X_train, y_train, X_val, y_val)
        est, fit_sample = _maybe_refit_selected_tuned_model(est, X_train, y_train, X_val, y_val, cfg)
        return est, {'selected': key, 'val_mse': loss, 'fit_sample': fit_sample}

    if name_u == 'POSTLASSO':
        candidates = {f'alpha={a:.3g}': Lasso(alpha=float(a), max_iter=20000, random_state=random_state) for a in alpha_grid}
        key, lasso, loss = select_by_validation(candidates, X_train, y_train, X_val, y_val)
        nonzero = np.abs(lasso.coef_) > 1e-10
        if not nonzero.any():
            # Constant-only is not a sklearn LinearRegression with empty features; use original lasso.
            lasso, fit_sample = _maybe_refit_selected_tuned_model(lasso, X_train, y_train, X_val, y_val, cfg)
            return lasso, {'selected': key, 'nonzero_features': [], 'val_mse': loss, 'fit_sample': fit_sample}
        selected_cols = list(X_train.columns[nonzero])
        est = LinearRegression()
        if _refit_tuned_models_on_train_validation(cfg):
            est.fit(pd.concat([X_train[selected_cols], X_val[selected_cols]]), pd.concat([y_train, y_val]))
            fit_sample = 'train_plus_validation_after_selection'
        else:
            est.fit(X_train[selected_cols], y_train)
            fit_sample = 'train_only_after_validation_selection'
        # Wrapper retaining column subset.
        return ColumnSubsetModel(est, selected_cols), {'selected': key, 'nonzero_features': selected_cols, 'val_mse': loss, 'fit_sample': fit_sample}

    if name_u == 'ADAPTIVELASSO':
        first = LinearRegression().fit(X_train, y_train)
        weights = 1.0 / np.maximum(np.abs(first.coef_), 1e-6)
        # Predictors are already standardized upstream, but the RV target is kept
        # in raw variance units. Normalizing adaptive weights keeps the alpha grid
        # numerically comparable to the other Lasso variants while preserving the
        # relative adaptive penalty across coefficients.
        weights = weights / np.mean(weights)
        candidates = {f'alpha={a:.3g}': WeightedLasso(alpha=float(a), weights=weights, random_state=random_state) for a in alpha_grid}
        key, est, loss = select_by_validation(candidates, X_train, y_train, X_val, y_val)
        est, fit_sample = _maybe_refit_selected_tuned_model(est, X_train, y_train, X_val, y_val, cfg)
        return est, {'selected': key, 'val_mse': loss, 'weight_normalization': 'mean_one', 'fit_sample': fit_sample}

    tree_cfg = cfg['models']['trees']
    if tree_cfg['random_forest_max_features'] == 'one_third':
        max_features = max(1, X_train.shape[1] // 3)
    elif tree_cfg['random_forest_max_features'] == 'sqrt':
        max_features = 'sqrt'
    else:
        max_features = None

    if name_u == 'BAGGING':
        try:
            est = BaggingRegressor(
                estimator=DecisionTreeRegressor(min_samples_leaf=tree_cfg['min_samples_leaf'], random_state=random_state),
                n_estimators=tree_cfg['n_estimators'],
                n_jobs=tree_cfg['n_jobs'],
                random_state=random_state,
            )
        except TypeError:
            est = BaggingRegressor(
                base_estimator=DecisionTreeRegressor(min_samples_leaf=tree_cfg['min_samples_leaf'], random_state=random_state),
                n_estimators=tree_cfg['n_estimators'],
                n_jobs=tree_cfg['n_jobs'],
                random_state=random_state,
            )
        est.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return est, {'method': 'bagging_default_paper_like'}

    if name_u == 'RANDOMFOREST':
        est = RandomForestRegressor(
            n_estimators=tree_cfg['n_estimators'],
            min_samples_leaf=tree_cfg['min_samples_leaf'],
            max_features=max_features,
            n_jobs=tree_cfg['n_jobs'],
            random_state=random_state,
        )
        est.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return est, {'method': 'rf_default_paper_like', 'max_features': max_features}

    if name_u == 'GRADIENTBOOSTING':
        from joblib import Parallel, delayed
        gb_cfg = cfg['models']['gradient_boosting']
        n_estimators_sorted = sorted(int(n) for n in gb_cfg['n_estimators'])
        chains = [
            (int(depth), float(lr))
            for depth in gb_cfg['depths']
            for lr in gb_cfg['learning_rates']
        ]
        results = Parallel(n_jobs=len(chains), prefer='threads')(
            delayed(_fit_gb_warm_start_chain)(
                depth, lr, n_estimators_sorted, X_train, y_train, X_val, y_val, random_state
            )
            for depth, lr in chains
        )
        key, est, loss = min(results, key=lambda r: r[2])
        est, fit_sample = _maybe_refit_selected_tuned_model(est, X_train, y_train, X_val, y_val, cfg)
        return est, {'selected': key, 'val_mse': loss, 'fit_sample': fit_sample}

    if name_u.startswith('NN'):
        from .nn import fit_nn_ensemble
        return fit_nn_ensemble(name_u, X_train, y_train, X_val, y_val, cfg, random_state=random_state)

    raise ValueError(f'Unknown model name {name}')


class ColumnSubsetModel:
    def __init__(self, estimator, columns: List[str]):
        self.estimator = estimator
        self.columns = columns

    def predict(self, X):
        return self.estimator.predict(X[self.columns])
