from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=handlers,
        force=True,
    )


def safe_log(x: pd.Series | np.ndarray, eps: float = 1e-12):
    return np.log(np.maximum(x, eps))


def winsorize_series(s: pd.Series, lower: float = 0.001, upper: float = 0.999) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def as_date_index(df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.normalize()
    out = out.set_index(col).sort_index()
    return out


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')
