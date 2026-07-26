#!/usr/bin/env python3
from __future__ import annotations

import email.utils
import html
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

SOURCE_URL = "https://niewmedia.com/news/"
OUTPUT = Path(__file__).with_name("feed.xml")
DATE_RE = re.compile(r"\b(20\d{2})\.(\d{1,2})\.(\d{1,2})(?:｜(\d{1,2}):(\d{2}))?")
USER_AGENT = "Mozilla/5.0 (compatible; niew-rss/1.0; +https://github.com/)"

@dataclass(frozen=True)
class Article:
    title: str
    url: str
    published: datetime


def clean(text: str) -> str:
    return " ".join(text.split())


def parse_date(text: str) -> datetime | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    y, mo, d = map(int, m.group(1, 2, 3))
    hh = int(m.group(4) or 0)
    mm = int(m.group(5) or 0)
    return datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)


def fetch_articles() -> list[Article]:
    r = requests.get(SOURCE_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    found: dict[str, Article] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(SOURCE_URL, a["href"])
        if not href.startswith("https://niewmedia.com/news/") or href.rstrip("/") == SOURCE_URL.rstrip("/"):
            continue

        text = clean(a.get_text(" ", strip=True))
        if not text:
            continue
        published = parse_date(text)
        if published is None:
            # Look a few levels upward for the timestamp shown in the card.
            node: Tag | None = a
            for _ in range(4):
                if node is None:
                    break
                published = parse_date(clean(node.get_text(" ", strip=True)))
                if published:
                    break
                node = node.parent if isinstance(node.parent, Tag) else None
        if published is None:
            continue

        title = DATE_RE.sub("", text)
        title = re.sub(r"#(?:MUSIC|ART|MOVIE|BOOK|STAGE|FASHION|OTHER)\b", "", title)
        title = clean(title)
        if len(title) < 4:
            continue

        article = Article(title=title, url=href, published=published)
        prev = found.get(href)
        if prev is None or len(article.title) < len(prev.title):
            found[href] = article

    articles = sorted(found.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("No recent NiEW news articles found")
    return articles[:50]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def render(items: list[Article]) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        '  <channel>',
        '    <title>NiEW 最新ニュース（unofficial）</title>',
        '    <link>https://niewmedia.com/news/</link>',
        '    <description>NiEWのNEWS新着記事から生成した非公式RSSです。</description>',
        '    <language>ja</language>',
        f'    <lastBuildDate>{email.utils.format_datetime(now)}</lastBuildDate>',
    ]
    for item in items:
        lines += [
            '    <item>',
            f'      <title>{esc(item.title)}</title>',
            f'      <link>{esc(item.url)}</link>',
            f'      <guid isPermaLink="true">{esc(item.url)}</guid>',
            f'      <pubDate>{email.utils.format_datetime(item.published)}</pubDate>',
            '    </item>',
        ]
    lines += ['  </channel>', '</rss>', '']
    return "\n".join(lines)


def main() -> int:
    try:
        items = fetch_articles()
        OUTPUT.write_text(render(items), encoding="utf-8")
        print(f"Wrote {len(items)} items")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
