from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config_symbols(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(s).upper() for s in cfg["assets"]["tickers"]]


def build_url(symbol: str, apikey: str) -> str:
    params = {
        "function": "EARNINGS",
        "symbol": symbol,
        "apikey": apikey,
    }
    return "https://www.alphavantage.co/query?" + urlencode(params)


def fetch_earnings(symbol: str, apikey: str, timeout: int = 60) -> dict:
    with urlopen(build_url(symbol, apikey), timeout=timeout) as response:
        text = response.read().decode("utf-8-sig", errors="replace")
    return json.loads(text)


def api_status(payload: dict) -> tuple[str, str]:
    for key in ("Error Message", "Information", "Note"):
        if key in payload:
            return "api_message", str(payload[key])[:500]
    if "quarterlyEarnings" in payload:
        return "ok", "quarterly earnings"
    return "unexpected_json", json.dumps(payload)[:500]


def to_float(value: object) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_events(symbol: str, payload: dict, start_date: str, end_date: str) -> list[dict]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows: list[dict] = []
    for item in payload.get("quarterlyEarnings", []):
        reported = pd.to_datetime(item.get("reportedDate"), errors="coerce")
        if pd.isna(reported):
            continue
        reported = reported.normalize()
        if reported < start or reported > end:
            continue
        rows.append(
            {
                "date": reported.strftime("%Y-%m-%d"),
                "ticker": symbol,
                "ea": 1,
                "source": "Alpha Vantage EARNINGS",
                "reported_date": reported.strftime("%Y-%m-%d"),
                "fiscal_date_ending": item.get("fiscalDateEnding", ""),
                "reported_eps": to_float(item.get("reportedEPS")),
                "estimated_eps": to_float(item.get("estimatedEPS")),
                "surprise": to_float(item.get("surprise")),
                "surprise_percentage": to_float(item.get("surprisePercentage")),
            }
        )
    return rows


def align_to_trading_days(events: pd.DataFrame, trading_calendar_path: Path, max_lag_days: int = 7) -> pd.DataFrame:
    if events.empty or not trading_calendar_path.exists():
        return events
    calendar = pd.read_csv(trading_calendar_path, usecols=["date", "ticker"], parse_dates=["date"])
    calendar["ticker"] = calendar["ticker"].astype(str).str.upper()
    calendar["date"] = calendar["date"].dt.normalize()
    calendar = calendar.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])

    aligned = []
    for ticker, g in events.groupby("ticker", sort=True):
        dates = calendar.loc[calendar["ticker"] == ticker, "date"].to_numpy()
        if len(dates) == 0:
            aligned.append(g)
            continue
        g = g.copy()
        new_dates = []
        shifted = []
        for raw_date in pd.to_datetime(g["reported_date"], errors="coerce"):
            if pd.isna(raw_date):
                new_dates.append(pd.NaT)
                shifted.append(False)
                continue
            raw_date = raw_date.normalize()
            pos = dates.searchsorted(raw_date.to_datetime64(), side="left")
            if pos >= len(dates):
                new_dates.append(pd.NaT)
                shifted.append(False)
                continue
            candidate = pd.Timestamp(dates[pos])
            if (candidate - raw_date).days > max_lag_days:
                new_dates.append(pd.NaT)
                shifted.append(False)
            else:
                new_dates.append(candidate)
                shifted.append(candidate != raw_date)
        g["date"] = pd.to_datetime(new_dates).strftime("%Y-%m-%d")
        g["aligned_to_trading_day"] = shifted
        aligned.append(g.dropna(subset=["date"]))
    return pd.concat(aligned, ignore_index=True) if aligned else events


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="Download Alpha Vantage earnings announcement dates for the replication panel.")
    parser.add_argument("--config", default=str(root / "config/default.yaml"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start-date", default="2001-01-01")
    parser.add_argument("--end-date", default="2017-12-31")
    parser.add_argument("--output-dir", default=str(root / "data/external/alpha_vantage_earnings"))
    parser.add_argument("--external-dir", default=str(root / "data/external"))
    parser.add_argument("--trading-calendar", default=str(root / "data/processed/daily_realized_measures.csv"))
    parser.add_argument("--sleep", type=float, default=12.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--apikey", default=None, help="Alpha Vantage API key. Defaults to ALPHAVANTAGE_API_KEY.")
    args = parser.parse_args()

    apikey = args.apikey or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not apikey:
        print("Set ALPHAVANTAGE_API_KEY in the environment or pass --apikey.", file=sys.stderr)
        return 2

    symbols = [s.upper() for s in args.symbols] if args.symbols else load_config_symbols(Path(args.config))
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[dict] = []
    manifest: list[dict] = []
    downloaded_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for i, symbol in enumerate(symbols, start=1):
        raw_path = raw_dir / f"{symbol}.json"
        if raw_path.exists() and not args.overwrite:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status, message = api_status(payload)
        else:
            try:
                payload = fetch_earnings(symbol, apikey)
                status, message = api_status(payload)
                raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            except Exception as exc:
                payload = {}
                status, message = "exception", repr(exc)[:500]
        events = parse_events(symbol, payload, args.start_date, args.end_date) if status == "ok" else []
        all_events.extend(events)
        manifest.append(
            {
                "symbol": symbol,
                "status": status,
                "message": message,
                "quarterly_rows_raw": len(payload.get("quarterlyEarnings", [])) if isinstance(payload, dict) else 0,
                "sample_events": len(events),
                "raw_path": str(raw_path if raw_path.exists() else ""),
                "downloaded_at_utc": downloaded_at,
            }
        )
        print(f"{i:02d}/{len(symbols)} {symbol}: {status}, sample_events={len(events)}")
        if i < len(symbols) and args.sleep > 0:
            time.sleep(args.sleep)

    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df = align_to_trading_days(events_df, Path(args.trading_calendar))
        events_df = (
            events_df.sort_values(["ticker", "date", "fiscal_date_ending"])
            .drop_duplicates(["ticker", "date"])
            .reset_index(drop=True)
        )
    event_rows = events_df.to_dict("records") if not events_df.empty else []

    event_fields = [
        "date",
        "ticker",
        "ea",
        "source",
        "reported_date",
        "fiscal_date_ending",
        "reported_eps",
        "estimated_eps",
        "surprise",
        "surprise_percentage",
        "aligned_to_trading_day",
    ]
    write_csv(Path(args.external_dir) / "earnings_announcements.csv", event_rows, event_fields)
    write_csv(out_dir / "earnings_download_manifest.csv", manifest, list(manifest[0].keys()) if manifest else [])

    if not events_df.empty:
        summary = (
            events_df.groupby("ticker")
            .agg(events=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
            .reset_index()
        )
        summary.to_csv(out_dir / "earnings_coverage_summary.csv", index=False)
        print(f"Wrote {len(events_df)} earnings events for {events_df['ticker'].nunique()} tickers.")
    else:
        pd.DataFrame(columns=["ticker", "events", "first_date", "last_date"]).to_csv(out_dir / "earnings_coverage_summary.csv", index=False)
        print("No earnings events were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
