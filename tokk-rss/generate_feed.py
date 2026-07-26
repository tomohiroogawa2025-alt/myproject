#!/usr/bin/env python3
from __future__ import annotations

import email.utils
import html
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

SITE_URL = "https://tokk-kansai.jp/"
OUTPUT = Path(__file__).with_name("feed.xml")
DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{2})\b")
UA = "Mozilla/5.0 (compatible; tokk-rss/1.0)"

@dataclass(frozen=True)
class Article:
    title: str
    url: str
    published: datetime


def clean(s: str) -> str:
    return " ".join(s.split())


def parse_date(text: str) -> datetime | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    yy, mm, dd = map(int, m.groups())
    return datetime(2000 + yy, mm, dd, tzinfo=timezone.utc)


def likely_article_url(url: str) -> bool:
    p = urlparse(url)
    if p.netloc not in {"tokk-kansai.jp", "www.tokk-kansai.jp"}:
        return False
    path = p.path
    blocked = ("/area/", "/category/", "/tag/", "/author/", "/contact/", "/coupon/", "/present/", "/wp-content/")
    return path not in {"/", ""} and not any(x in path for x in blocked)


def find_date_near(anchor: Tag) -> datetime | None:
    cur: Tag | None = anchor
    for _ in range(5):
        if cur is None:
            break
        dt = parse_date(clean(cur.get_text(" ", strip=True)))
        if dt:
            return dt
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return None


def anchor_title(anchor: Tag) -> str:
    img = anchor.find("img", alt=True)
    if img:
        alt = clean(img.get("alt", ""))
        if alt:
            return alt
    return clean(anchor.get_text(" ", strip=True))


def fetch_articles() -> list[Article]:
    r = requests.get(SITE_URL, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    heading = next((h for h in soup.find_all(["h2", "h3"]) if "最新記事" in clean(h.get_text(" ", strip=True))), None)
    scope = heading.parent if heading and isinstance(heading.parent, Tag) else soup

    found: dict[str, Article] = {}
    for a in scope.find_all("a", href=True):
        url = urljoin(SITE_URL, a["href"])
        if not likely_article_url(url):
            continue
        dt = find_date_near(a)
        if not dt:
            continue
        title = anchor_title(a)
        if len(title) < 6:
            continue
        found[url] = Article(title=title, url=url, published=dt)

    articles = sorted(found.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("No TOKK latest articles found")
    return articles[:50]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def render(articles: list[Article]) -> str:
    items = []
    for a in articles:
        items.extend([
            "    <item>",
            f"      <title>{esc(a.title)}</title>",
            f"      <link>{esc(a.url)}</link>",
            f"      <guid isPermaLink=\"true\">{esc(a.url)}</guid>",
            f"      <pubDate>{email.utils.format_datetime(a.published)}</pubDate>",
            "    </item>",
        ])
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        '  <channel>',
        '    <title>TOKK関西 最新記事（unofficial）</title>',
        f'    <link>{SITE_URL}</link>',
        '    <description>TOKK関西トップページの「最新記事」欄から生成した非公式RSSです。</description>',
        '    <language>ja</language>',
        f'    <lastBuildDate>{email.utils.format_datetime(datetime.now(timezone.utc))}</lastBuildDate>',
        *items,
        '  </channel>',
        '</rss>',
        '',
    ])


def main() -> int:
    try:
        articles = fetch_articles()
        OUTPUT.write_text(render(articles), encoding="utf-8")
        print(f"Wrote {len(articles)} items")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
