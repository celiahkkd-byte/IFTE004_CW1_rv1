#!/usr/bin/env python3
"""One-shot launcher for Task I MCS.

The launcher is used with launchctl so the full MCS run can continue in the
background without respawning after completion.
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
    output_dir = project_root / "outputs_mcs_core_no_bagging_gb_20260520"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_bin),
        str(project_root / "scripts" / "06e_compute_mcs.py"),
        "--predictions",
        str(project_root / "outputs_final_core_no_bagging_gb_20260520" / "predictions" / "model_predictions.csv"),
        "--output-dir",
        str(output_dir),
        "--allow-existing-output-dir",
        "--confidence",
        "0.90",
        "--reps",
        "5000",
        "--block-size",
        "10",
        "--min-valid-rows",
        "200",
        "--min-ticker-coverage",
        "20",
    ]

    (log_dir / "task_i_launch_metadata.json").write_text(
        json.dumps(
            {
                "started_at": utc_now(),
                "label": label,
                "project_root": str(project_root),
                "output_dir": str(output_dir),
                "command": command,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = 1
    try:
        exit_code = subprocess.run(command, cwd=project_root).returncode
        return exit_code
    finally:
        (log_dir / "task_i_exit_status.json").write_text(
            json.dumps(
                {
                    "exit_status": exit_code,
                    "finished_at": utc_now(),
                    "label": label,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if label:
            subprocess.run(
                ["/bin/launchctl", "remove", label],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


if __name__ == "__main__":
    raise SystemExit(main())
