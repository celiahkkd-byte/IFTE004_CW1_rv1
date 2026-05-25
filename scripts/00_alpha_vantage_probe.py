from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from urllib.parse import urlencode
from urllib.request import urlopen


def query(params: dict[str, str], apikey_arg: str | None = None) -> str:
    apikey = apikey_arg or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not apikey:
        raise SystemExit("Set ALPHAVANTAGE_API_KEY in the environment or pass --apikey.")
    params = {**params, "apikey": apikey}
    url = "https://www.alphavantage.co/query?" + urlencode(params)
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True)
    parser.add_argument("--param", action="append", default=[], help="key=value")
    parser.add_argument("--head", type=int, default=20)
    parser.add_argument("--apikey", default=None, help="Alpha Vantage API key. Defaults to ALPHAVANTAGE_API_KEY.")
    args = parser.parse_args()
    params = {"function": args.function}
    for item in args.param:
        key, value = item.split("=", 1)
        params[key] = value
    text = query(params, args.apikey)
    stripped = text.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        print(json.dumps(payload, indent=2)[:5000])
        return 0
    reader = csv.reader(stripped.splitlines())
    for i, row in enumerate(reader):
        if i >= args.head:
            break
        print(",".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
