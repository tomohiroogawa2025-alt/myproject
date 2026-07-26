#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import requests

SOURCES = [
    "https://rsshub.isrss.com/instagram/2/user/goofy.jp",
    "https://rsshub.app/instagram/2/user/goofy.jp",
]
OUTPUT = Path(__file__).with_name("feed.xml")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; goofy-instagram-rss/1.0)",
    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}


def looks_like_feed(text: str) -> bool:
    stripped = text.lstrip()
    return ("<rss" in stripped[:500] or "<feed" in stripped[:500]) and "goofy.jp" in text


def main() -> int:
    errors: list[str] = []
    for url in SOURCES:
        try:
            r = requests.get(url, headers=HEADERS, timeout=45)
            r.raise_for_status()
            if not looks_like_feed(r.text):
                raise RuntimeError("response was not a valid-looking feed")
            OUTPUT.write_text(r.text, encoding="utf-8")
            print(f"Updated {OUTPUT} from {url}")
            return 0
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    # Preserve the previous valid feed if Instagram/RSSHub is temporarily blocked.
    if OUTPUT.exists() and OUTPUT.stat().st_size > 100:
        print("All sources failed; preserving previous feed", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 0

    for err in errors:
        print(err, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
