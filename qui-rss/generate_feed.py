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

LIST_URL = "https://qui.tokyo/media"
SITE_URL = "https://qui.tokyo/"
OUTPUT = Path(__file__).with_name("feed.xml")
USER_AGENT = "Mozilla/5.0 (compatible; qui-rss/1.0)"
DATE_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),\s+(20\d{2})\b", re.I)
ARTICLE_PATHS = ("/news/", "/fashion/", "/film/", "/music/", "/art/", "/beauty/", "/life/", "/shopping/", "/feature/")

@dataclass(frozen=True)
class Article:
    title: str
    url: str
    published: datetime


def clean(s: str) -> str:
    return " ".join(s.split())


def nearby_date(node: Tag) -> datetime | None:
    cur: Tag | None = node
    for _ in range(6):
        if cur is None:
            break
        text = clean(cur.get_text(" ", strip=True))
        m = DATE_RE.search(text)
        if m:
            return datetime.strptime(m.group(0), "%b %d, %Y").replace(tzinfo=timezone.utc)
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return None


def anchor_title(a: Tag) -> str:
    for selector in ("h1", "h2", "h3", "h4", "p"):
        el = a.find(selector)
        if el:
            t = clean(el.get_text(" ", strip=True))
            if len(t) >= 5:
                return t
    img = a.find("img", alt=True)
    if img:
        t = clean(img.get("alt", ""))
        if len(t) >= 5:
            return t
    t = clean(a.get_text(" ", strip=True))
    t = DATE_RE.sub("", t)
    t = re.sub(r"^(NEWS|FASHION|FILM|MUSIC|ART/DESIGN|BEAUTY|LIFE/STYLE|SHOPPING|FEATURE)\s+", "", t, flags=re.I)
    return clean(t)


def fetch_articles() -> list[Article]:
    r = requests.get(LIST_URL, headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    found: dict[str, Article] = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(SITE_URL, a["href"])
        if not url.startswith(SITE_URL) or not any(p in url for p in ARTICLE_PATHS):
            continue
        if any(x in url for x in ("/category/", "/tag/", "/page/")):
            continue
        published = nearby_date(a)
        if published is None:
            continue
        title = anchor_title(a)
        if len(title) < 5:
            continue
        candidate = Article(title, url, published)
        prev = found.get(url)
        if prev is None or len(title) < len(prev.title):
            found[url] = candidate
    articles = sorted(found.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("No dated QUI media articles found")
    return articles[:50]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def render(articles: list[Article]) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        '  <channel>',
        '    <title>QUI - Fashion &amp; Culture media（unofficial）</title>',
        f'    <link>{SITE_URL}</link>',
        '    <description>QUIの最新メディア記事から生成した非公式RSSです。</description>',
        '    <language>ja</language>',
        f'    <lastBuildDate>{email.utils.format_datetime(now)}</lastBuildDate>',
    ]
    for a in articles:
        lines += [
            '    <item>',
            f'      <title>{esc(a.title)}</title>',
            f'      <link>{esc(a.url)}</link>',
            f'      <guid isPermaLink="true">{esc(a.url)}</guid>',
            f'      <pubDate>{email.utils.format_datetime(a.published)}</pubDate>',
            '    </item>',
        ]
    lines += ['  </channel>', '</rss>', '']
    return "\n".join(lines)


def main() -> int:
    try:
        articles = fetch_articles()
        OUTPUT.write_text(render(articles), encoding="utf-8")
        print(f"Wrote {len(articles)} articles")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
