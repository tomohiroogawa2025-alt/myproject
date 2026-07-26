#!/usr/bin/env python3
from __future__ import annotations

import email.utils
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

SITE = "https://marzel.jp/"
LIST_URL = "https://marzel.jp/all/"
OUT = Path(__file__).with_name("feed.xml")
DATE_RE = re.compile(r"\b(20\d{2})[./-](\d{2})[./-](\d{2})\b")
UA = "Mozilla/5.0 (compatible; marzel-rss/1.0)"


def clean(s: str) -> str:
    return " ".join(s.split())


def nearby_date(a: Tag):
    node: Tag | None = a
    for _ in range(5):
        if node is None:
            break
        m = DATE_RE.search(clean(node.get_text(" ", strip=True)))
        if m:
            y, mo, d = map(int, m.groups())
            return datetime(y, mo, d, tzinfo=timezone.utc)
        node = node.parent if isinstance(node.parent, Tag) else None
    return None


def is_article_url(url: str) -> bool:
    p = urlparse(url)
    if p.netloc not in {"marzel.jp", "www.marzel.jp"}:
        return False
    path = p.path.rstrip("/")
    if not path or path in {"/all", "/topics", "/about", "/contact"}:
        return False
    blocked = ("/category/", "/tag/", "/author/", "/wp-", "/feed")
    return not any(x in path for x in blocked)


def main() -> None:
    r = requests.get(LIST_URL, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    items = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(SITE, a["href"]).split("#", 1)[0]
        if not is_article_url(url):
            continue
        dt = nearby_date(a)
        if not dt:
            continue
        title = clean(a.get_text(" ", strip=True))
        title = DATE_RE.sub("", title)
        title = re.sub(r"\b(fashion|art|music|food|culture|other)\b", "", title, flags=re.I)
        title = clean(title)
        if len(title) < 8:
            img = a.find("img", alt=True)
            title = clean(img.get("alt", "")) if img else title
        if len(title) < 8:
            continue
        prev = items.get(url)
        if prev is None or len(title) < len(prev[0]):
            items[url] = (title, dt)

    rows = sorted(((u, *v) for u, v in items.items()), key=lambda x: (x[2], x[0]), reverse=True)[:40]
    if not rows:
        raise RuntimeError("No MARZEL articles found")

    now = datetime.now(timezone.utc)
    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        '  <channel>',
        '    <title>MARZEL 新着記事（unofficial）</title>',
        f'    <link>{SITE}</link>',
        '    <description>MARZELの記事一覧から生成した非公式RSSです。</description>',
        '    <language>ja</language>',
        f'    <lastBuildDate>{email.utils.format_datetime(now)}</lastBuildDate>',
    ]
    for url, title, dt in rows:
        xml += [
            '    <item>',
            f'      <title>{html.escape(title)}</title>',
            f'      <link>{html.escape(url)}</link>',
            f'      <guid isPermaLink="true">{html.escape(url)}</guid>',
            f'      <pubDate>{email.utils.format_datetime(dt)}</pubDate>',
            '    </item>',
        ]
    xml += ['  </channel>', '</rss>', '']
    OUT.write_text("\n".join(xml), encoding="utf-8")
    print(f"Wrote {len(rows)} items; newest={rows[0][2].date()} {rows[0][1]}")


if __name__ == "__main__":
    main()
