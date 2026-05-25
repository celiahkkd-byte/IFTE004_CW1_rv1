#!/usr/bin/env python3
"""One-shot launcher for Task G variable importance.

This wrapper is intentionally narrow: it starts the full Task G core run in an
isolated output directory and, when invoked by launchctl, removes its own label
on exit so macOS does not respawn the job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    project_root = Path(__file__).resolve().parents[1]
    python_bin = Path("/Users/celiawong/.pyenv/versions/3.11.8/bin/python")
    output_dir = project_root / "outputs_variable_importance_core_no_bagging_gb_20260520"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_bin),
        str(project_root / "scripts" / "06c_compute_variable_importance.py"),
        "--config",
        str(project_root / "config" / "paper_core_rolling.yaml"),
        "--output-dir",
        str(output_dir),
        "--allow-existing-output-dir",
        "--nn-checkpoint-dir",
        str(project_root / "outputs_nn30_checkpointed"),
        "--models",
        "HARX",
        "ElasticNet",
        "RandomForest",
        "NN10_2",
        "--dataset",
        "PARTIAL_MALL",
        "--horizon",
        "1",
        "--grid-size",
        "100",
        "--min-tickers",
        "20",
        "--tree-n-jobs",
        "1",
    ]

    metadata = {
        "started_at": utc_now(),
        "label": label,
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "command": command,
    }
    (log_dir / "task_g_launch_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    exit_code = 1
    try:
        exit_code = subprocess.run(command, cwd=project_root).returncode
        return exit_code
    finally:
        status = {
            "exit_status": exit_code,
            "finished_at": utc_now(),
            "label": label,
        }
        (log_dir / "task_g_exit_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if label:
            subprocess.run(
                ["/bin/launchctl", "remove", label],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


if __name__ == "__main__":
    raise SystemExit(main())
