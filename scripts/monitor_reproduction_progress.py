from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pid_state(path: Path) -> str:
    if not path.exists():
        return "no_pid"
    raw = path.read_text().strip()
    if not raw:
        return "empty_pid"
    try:
        pid = int(raw)
    except ValueError:
        return f"bad_pid={raw}"
    try:
        os.kill(pid, 0)
    except PermissionError:
        return f"{pid}:alive_no_signal_permission"
    except ProcessLookupError:
        return f"{pid}:not_found"
    return f"{pid}:alive"


def _last_signal_line(path: Path) -> str:
    if not path.exists():
        return "missing_log"
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"read_error={exc}"
    for line in reversed(lines[-300:]):
        if "INFO" in line or "ERROR" in line or "Traceback" in line:
            return line[-500:]
    return "no_recent_signal_line"


def _snapshot() -> str:
    a_by_model = sum(1 for _ in (ROOT / "outputs_rolling/predictions/by_model").glob("*.csv"))
    b_seed_files = sum(1 for _ in (ROOT / "outputs_nn30_checkpointed/nn_seed_predictions").rglob("*.csv"))
    a_state = _pid_state(ROOT / "outputs_rolling/logs/04_forecasts_rolling_checkpoints.pid")
    b_state = _pid_state(ROOT / "outputs_nn30_checkpointed/logs/04_nn_full30.pid")
    a_last = _last_signal_line(ROOT / "outputs_rolling/logs/04_forecasts_rolling_checkpoints.log")
    b_last = _last_signal_line(ROOT / "outputs_nn30_checkpointed/logs/04_nn_full30.log")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{stamp} | A {a_state} by_model={a_by_model}/56 last={a_last} | "
        f"B {b_state} seeds={b_seed_files}/12000 last={b_last}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly progress monitor for reproduction tasks.")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--log", default="outputs_monitor/logs/hourly_progress.log")
    args = parser.parse_args()

    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_snapshot() + "\n")
        time.sleep(max(60, int(args.interval_seconds)))


if __name__ == "__main__":
    main()
