from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rv1rep.config import load_config, project_path, ensure_dirs
from rv1rep.utils import setup_logging
from rv1rep.pipeline import build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'config/default.yaml'))
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_dirs(cfg); setup_logging(project_path(cfg, 'output_dir') / 'logs' / '03_features.log')
    build_features(cfg)
if __name__ == '__main__': main()
