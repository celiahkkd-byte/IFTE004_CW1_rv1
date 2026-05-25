from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    base = path.parent.parent.resolve()
    cfg.setdefault('_base_dir', str(base))
    return cfg


def project_path(cfg: Dict[str, Any], key: str) -> Path:
    base = Path(cfg.get('_base_dir', '.')).resolve()
    value = cfg['paths'][key]
    p = Path(value)
    return p if p.is_absolute() else base / p


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ['processed_dir', 'external_dir', 'output_dir']:
        project_path(cfg, key).mkdir(parents=True, exist_ok=True)
    out = project_path(cfg, 'output_dir')
    for sub in ['tables', 'figures', 'predictions', 'logs']:
        (out / sub).mkdir(parents=True, exist_ok=True)


def override_config(cfg: Dict[str, Any], *, scheme: Optional[str] = None, models: Optional[Iterable[str]] = None, skip_nn: bool = False) -> Dict[str, Any]:
    if scheme:
        cfg['estimation']['scheme'] = scheme
    if models:
        cfg['models']['enabled'] = list(models)
    if skip_nn:
        cfg['models']['enabled'] = [m for m in cfg['models']['enabled'] if not m.startswith('NN')]
    return cfg
