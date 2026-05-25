from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rv1rep.config import load_config, override_config, project_path, ensure_dirs
from rv1rep.utils import setup_logging
from rv1rep.pipeline import run_forecast_experiments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'config/default.yaml'))
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--scheme', choices=['fixed', 'rolling'], default=None)
    ap.add_argument('--skip-nn', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = override_config(cfg, scheme=args.scheme, models=args.models, skip_nn=args.skip_nn)
    ensure_dirs(cfg); setup_logging(project_path(cfg, 'output_dir') / 'logs' / '04_forecasts.log')
    run_forecast_experiments(cfg, models=cfg['models']['enabled'])
if __name__ == '__main__': main()
