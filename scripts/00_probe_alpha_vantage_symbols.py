from __future__ import annotations

import csv
import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen


KEYWORDS = [
    "Chevron",
    "ChevronTexaco",
    "Dow",
    "DowDuPont",
    "DuPont",
    "Raytheon",
    "United Technologies",
    "Travelers",
    "Walgreens",
]


def get(params: dict[str, str]) -> str:
    apikey = os.environ["ALPHAVANTAGE_API_KEY"]
    url = "https://www.alphavantage.co/query?" + urlencode({**params, "apikey": apikey})
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8-sig", errors="replace")


for keyword in KEYWORDS:
    text = get({"function": "SYMBOL_SEARCH", "keywords": keyword})
    print(f"\n## {keyword}")
    if text.strip().startswith("{"):
        payload = json.loads(text)
        matches = payload.get("bestMatches", [])
        for item in matches[:8]:
            print(
                item.get("1. symbol", ""),
                "|",
                item.get("2. name", ""),
                "|",
                item.get("4. region", ""),
                "|",
                item.get("8. currency", ""),
            )
    else:
        for i, row in enumerate(csv.reader(text.splitlines())):
            if i > 8:
                break
            print(row)
