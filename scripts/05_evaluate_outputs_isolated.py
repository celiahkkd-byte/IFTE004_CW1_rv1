from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rv1rep.config import ensure_dirs, load_config, project_path
from rv1rep.pipeline import evaluate_predictions
from rv1rep.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate predictions from an isolated output directory.')
    ap.add_argument('--config', default=str(ROOT / 'config/default.yaml'))
    ap.add_argument('--output-dir', required=True, help='Separate output directory, e.g. outputs_full_nn')
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['paths']['output_dir'] = args.output_dir
    ensure_dirs(cfg)
    setup_logging(project_path(cfg, 'output_dir') / 'logs' / '05_evaluate_isolated.log')
    evaluate_predictions(cfg)


if __name__ == '__main__':
    main()
