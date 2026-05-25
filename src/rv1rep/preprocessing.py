from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ('ea',)


@dataclass
class Standardizer:
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES
    standardize_binary_features: bool = True
    mean_: pd.Series | None = None
    std_: pd.Series | None = None
    columns_: list[str] | None = None
    continuous_columns_: list[str] | None = None
    categorical_columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame):
        self.columns_ = list(X.columns)
        categorical = set(self.categorical_features)
        if self.standardize_binary_features:
            self.categorical_columns_ = []
            self.continuous_columns_ = list(self.columns_)
        else:
            self.categorical_columns_ = [c for c in self.columns_ if c in categorical]
            self.continuous_columns_ = [c for c in self.columns_ if c not in categorical]
        if self.continuous_columns_:
            self.mean_ = X[self.continuous_columns_].mean(axis=0)
            self.std_ = X[self.continuous_columns_].std(axis=0).replace(0.0, 1.0)
        else:
            self.mean_ = pd.Series(dtype=float)
            self.std_ = pd.Series(dtype=float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None or self.columns_ is None or self.continuous_columns_ is None:
            raise RuntimeError('Standardizer has not been fit')
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f'Missing columns for Standardizer.transform: {missing}')
        out = X[self.columns_].copy()
        if self.continuous_columns_:
            out = out.astype({c: 'float64' for c in self.continuous_columns_})
            out.loc[:, self.continuous_columns_] = (out[self.continuous_columns_] - self.mean_) / self.std_
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def standardizer_from_config(cfg: dict) -> Standardizer:
    prep = cfg.get('preprocessing', {}) if isinstance(cfg, dict) else {}
    binary_features = tuple(prep.get('binary_features', CATEGORICAL_FEATURES))
    standardize_binary = bool(prep.get('standardize_binary_features', True))
    return Standardizer(
        categorical_features=binary_features,
        standardize_binary_features=standardize_binary,
    )


def enforce_positive_forecasts(pred, in_sample_min: float, policy: str = 'in_sample_min_rv'):
    pred = np.asarray(pred, dtype=float)
    if policy == 'none':
        return pred
    if policy == 'zero':
        return np.maximum(pred, 0.0)
    if policy == 'in_sample_min_rv':
        return np.where(pred <= 0, in_sample_min, pred)
    raise ValueError(f'Unknown negative forecast policy {policy}')


def insanity_filter(pred, in_sample_mean: float, in_sample_min: float, max_multiple: float = 100.0):
    pred = np.asarray(pred, dtype=float)
    cap = max_multiple * in_sample_mean
    return np.where((~np.isfinite(pred)) | (pred <= 0) | (pred > cap), in_sample_min, pred)
