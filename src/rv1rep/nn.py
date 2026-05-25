from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)


class AveragedPredictor:
    def __init__(self, models: List[object]):
        self.models = models

    def predict(self, X):
        preds = [np.asarray(m.predict(X), dtype=float).reshape(-1) for m in self.models]
        return np.mean(preds, axis=0)


def _build_keras_model(input_dim: int, hidden_layers: List[int], dropout: float, learning_rate: float, seed: int):
    import tensorflow as tf  # type: ignore
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, kernel_initializer='glorot_normal')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')
    return model


def fit_nn_ensemble(name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cfg: Dict, random_state: int = 42):
    nn_cfg = cfg['models']['neural_network']
    hidden = list(nn_cfg['architectures'][name])
    use_tf = bool(nn_cfg.get('use_tensorflow', True))
    n_seeds = int(nn_cfg.get('seeds', 20))
    top_k = int(nn_cfg.get('ensemble_top', min(10, n_seeds)))
    seeds = [random_state + i for i in range(n_seeds)]

    if use_tf:
        try:
            import tensorflow as tf  # noqa: F401
            fitted = []
            scores = []
            for seed in seeds:
                model = _build_keras_model(
                    X_train.shape[1], hidden,
                    dropout=float(nn_cfg.get('dropout', 0.8)),
                    learning_rate=float(nn_cfg.get('learning_rate', 0.001)),
                    seed=seed,
                )
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(nn_cfg.get('patience', 100)), restore_best_weights=True)]
                model.fit(
                    X_train.to_numpy(), y_train.to_numpy(),
                    validation_data=(X_val.to_numpy(), y_val.to_numpy()),
                    epochs=int(nn_cfg.get('epochs', 500)),
                    batch_size=int(nn_cfg.get('batch_size', 64)),
                    verbose=0,
                    callbacks=callbacks,
                )
                pred_val = model.predict(X_val.to_numpy(), verbose=0).reshape(-1)
                scores.append(mean_squared_error(y_val, pred_val))
                fitted.append(KerasPredictor(model))
            order = np.argsort(scores)
            ensemble = AveragedPredictor([fitted[i] for i in order[:top_k]])
            return ensemble, {'backend': 'tensorflow', 'hidden': hidden, 'seeds': n_seeds, 'ensemble_top': top_k, 'best_val_mse': float(np.min(scores))}
        except Exception as exc:
            logger.warning('TensorFlow NN requested but failed/unavailable: %s. Falling back to sklearn MLPRegressor.', exc)

    # Fallback: no dropout, but same hidden-layer geometry. Kept for auditability when TensorFlow is unavailable.
    models = []
    scores = []
    for seed in seeds:
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
        pred = est.predict(X_val)
        scores.append(mean_squared_error(y_val, pred))
        models.append(est)
    order = np.argsort(scores)
    ensemble = AveragedPredictor([models[i] for i in order[:top_k]])
    return ensemble, {'backend': 'sklearn_fallback', 'hidden': hidden, 'seeds': n_seeds, 'ensemble_top': top_k, 'best_val_mse': float(np.min(scores))}


class KerasPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(np.asarray(X), verbose=0).reshape(-1)
