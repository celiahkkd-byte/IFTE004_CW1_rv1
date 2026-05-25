from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import ensure_dirs, load_config, override_config, project_path
from rv1rep.pipeline import run_forecast_experiments
from rv1rep.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description='Run forecasts while writing outputs to an isolated output directory.')
    ap.add_argument('--config', default=str(ROOT / 'config/default.yaml'))
    ap.add_argument('--output-dir', required=True, help='Separate output directory, e.g. outputs_full_nn')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--scheme', choices=['fixed', 'rolling'], default=None)
    ap.add_argument('--skip-nn', action='store_true')
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['paths']['output_dir'] = args.output_dir
    cfg = override_config(cfg, scheme=args.scheme, models=args.models, skip_nn=args.skip_nn)
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, 'output_dir') / 'logs' / '04_forecasts_isolated.log')
    run_forecast_experiments(cfg, models=cfg['models']['enabled'])


if __name__ == '__main__':
    main()
