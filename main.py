from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import load_config, override_config, project_path, ensure_dirs
from rv1rep.utils import setup_logging
from rv1rep.pipeline import run_all


def parse_args():
    p = argparse.ArgumentParser(description='Professional RV1 replication pipeline')
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--models', nargs='*', default=None, help='Optional model subset')
    p.add_argument('--scheme', choices=['fixed', 'rolling'], default=None)
    p.add_argument('--skip-nn', action='store_true', help='Disable NN models even if enabled in config')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = override_config(cfg, scheme=args.scheme, models=args.models, skip_nn=args.skip_nn)
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, 'output_dir') / 'logs' / 'pipeline.log')
    run_all(cfg, models=cfg['models']['enabled'])


if __name__ == '__main__':
    main()
