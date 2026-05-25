from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


PAPER_TICKERS = [
    "AAPL",
    "AXP",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "DIS",
    "DOW",
    "GE",
    "GS",
    "HD",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PFE",
    "PG",
    "RTX",
    "TRV",
    "UNH",
    "VZ",
    "WMT",
    "XOM",
]


def month_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    months: list[str] = []
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def build_url(symbol: str, month: str, interval: str, apikey: str) -> str:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": month,
        "outputsize": "full",
        "datatype": "csv",
        "adjusted": "false",
        "extended_hours": "false",
        "apikey": apikey,
    }
    return "https://www.alphavantage.co/query?" + urlencode(params)


def fetch_csv(symbol: str, month: str, interval: str, apikey: str, timeout: int = 60) -> str:
    url = build_url(symbol, month, interval, apikey)
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def classify_response(text: str) -> tuple[str, int, str]:
    stripped = text.strip()
    if not stripped:
        return "empty", 0, "empty response"
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return "json_error", 0, stripped[:300]
        for key in ("Error Message", "Information", "Note"):
            if key in payload:
                return "api_message", 0, str(payload[key])[:500]
        return "json", 0, json.dumps(payload)[:500]

    reader = csv.DictReader(stripped.splitlines())
    rows = list(reader)
    if reader.fieldnames and {"timestamp", "open", "high", "low", "close", "volume"}.issubset(reader.fieldnames):
        return "ok", len(rows), "csv ohlcv"
    return "unexpected_csv", len(rows), ",".join(reader.fieldnames or [])[:300]


def write_status(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symbol",
                "month",
                "status",
                "rows",
                "path",
                "message",
                "downloaded_at_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def read_existing_status(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {(r["symbol"], r["month"]): r for r in csv.DictReader(f)}


def is_hard_api_stop(record: dict) -> bool:
    if record["status"] != "api_message":
        return False
    message_lower = str(record["message"]).lower()
    return (
        "frequency" in message_lower
        or "rate" in message_lower
        or "burst" in message_lower
        or "limit" in message_lower
        or "premium" in message_lower
        or "thank you for using alpha vantage" in message_lower
    )


def download_one(symbol: str, month: str, interval: str, apikey: str, raw_root: Path) -> dict:
    out_path = raw_root / symbol / f"{month}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        text = fetch_csv(symbol, month, interval, apikey)
        status, nrows, message = classify_response(text)
        if status == "ok":
            out_path.write_text(text, encoding="utf-8")
        return {
            "symbol": symbol,
            "month": month,
            "status": status,
            "rows": nrows,
            "path": str(out_path if status == "ok" else ""),
            "message": message,
            "downloaded_at_utc": downloaded_at,
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "month": month,
            "status": "exception",
            "rows": 0,
            "path": "",
            "message": repr(exc)[:500],
            "downloaded_at_utc": downloaded_at,
        }


def convert_monthly_to_teacher_txt(raw_root: Path, combined_root: Path, symbol: str) -> int:
    rows: list[dict] = []
    for csv_path in sorted((raw_root / symbol).glob("*.csv")):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                rows.append(
                    {
                        "dt": ts,
                        "date": ts.strftime("%m/%d/%Y"),
                        "time": ts.strftime("%H:%M"),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                )
    rows.sort(key=lambda x: x["dt"])
    combined_root.mkdir(parents=True, exist_ok=True)
    out_path = combined_root / f"{symbol}.txt"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([row["date"], row["time"], row["open"], row["high"], row["low"], row["close"], row["volume"]])
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Alpha Vantage 5-minute intraday data for RV replication.")
    parser.add_argument("--symbols", nargs="*", default=PAPER_TICKERS)
    parser.add_argument("--start-month", default="2001-01")
    parser.add_argument("--end-month", default="2017-12")
    parser.add_argument("--interval", default="5min")
    parser.add_argument("--output-dir", default="data/external/alpha_vantage_intraday_5min")
    parser.add_argument("--sleep", type=float, default=12.5)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--continue-on-api-message", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--apikey", default=None, help="Alpha Vantage API key. Defaults to ALPHAVANTAGE_API_KEY.")
    args = parser.parse_args()

    apikey = args.apikey or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not apikey:
        print("Set ALPHAVANTAGE_API_KEY in the environment or pass --apikey.", file=sys.stderr)
        return 2

    out_root = Path(args.output_dir)
    raw_root = out_root / "raw_monthly"
    status_path = out_root / "download_manifest.csv"
    existing = read_existing_status(status_path)
    records = list(existing.values())
    record_map = {(r["symbol"], r["month"]): r for r in records}

    months = month_range(args.start_month, args.end_month)
    tasks: list[tuple[str, str]] = []
    for symbol in args.symbols:
        for month in months:
            out_path = raw_root / symbol / f"{month}.csv"
            prior = record_map.get((symbol, month))
            if not args.overwrite and prior and prior.get("status") == "ok" and out_path.exists():
                continue
            tasks.append((symbol, month))
            if args.max_requests is not None and len(tasks) >= args.max_requests:
                break
        if args.max_requests is not None and len(tasks) >= args.max_requests:
            break

    requests_done = 0
    workers = max(1, int(args.workers))
    if workers == 1:
        for symbol, month in tasks:
            record = download_one(symbol, month, args.interval, apikey, raw_root)
            record_map[(symbol, month)] = record
            records = list(record_map.values())
            write_status(status_path, records)
            requests_done += 1
            if not args.quiet or requests_done % 100 == 0:
                print(f"{symbol} {month}: {record['status']} rows={record['rows']} {record['message'][:120]}")
            if record["status"] == "api_message":
                if is_hard_api_stop(record) or not args.continue_on_api_message:
                    return 1
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(download_one, symbol, month, args.interval, apikey, raw_root): (symbol, month)
                for symbol, month in tasks
            }
            for future in as_completed(future_map):
                symbol, month = future_map[future]
                record = future.result()
                record_map[(symbol, month)] = record
                records = list(record_map.values())
                write_status(status_path, records)
                requests_done += 1
                if not args.quiet or requests_done % 100 == 0:
                    print(f"{symbol} {month}: {record['status']} rows={record['rows']} {record['message'][:120]}")
                if record["status"] == "api_message":
                    if is_hard_api_stop(record) or not args.continue_on_api_message:
                        return 1

    if args.combine:
        combined_root = out_root / "combined_teacher_format"
        for symbol in args.symbols:
            nrows = convert_monthly_to_teacher_txt(raw_root, combined_root, symbol)
            print(f"combined {symbol}: {nrows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
