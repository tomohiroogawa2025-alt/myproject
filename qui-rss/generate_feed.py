#!/usr/bin/env python3
from __future__ import annotations

import email.utils
import html
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# QUI sometimes returns 502 for /media from automated clients. Try several public
# listing pages, starting with the simpler category page.
LIST_URLS = [
    "https://qui.tokyo/category/news",
    "https://qui.tokyo/media",
    "https://qui.tokyo/",
]
SITE_URL = "https://qui.tokyo/"
OUTPUT = Path(__file__).with_name("feed.xml")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
DATE_PATTERNS = [
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),\s+(20\d{2})\b", re.I),
    re.compile(r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b"),
]
ARTICLE_PATHS = (
    "/news/", "/fashion/", "/film/", "/music/", "/art/", "/beauty/",
    "/life/", "/shopping/", "/feature/",
)


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    published: datetime


def clean(s: str) -> str:
    return " ".join(s.split())


def parse_date(text: str) -> datetime | None:
    text = clean(text)
    m = DATE_PATTERNS[0].search(text)
    if m:
        return datetime.strptime(m.group(0), "%b %d, %Y").replace(tzinfo=timezone.utc)
    m = DATE_PATTERNS[1].search(text)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, tzinfo=timezone.utc)
    return None


def nearby_date(node: Tag) -> datetime | None:
    # QUI cards can wrap the link quite deeply, so walk further up than before.
    cur: Tag | None = node
    for _ in range(14):
        if cur is None:
            break
        dt = parse_date(cur.get_text(" ", strip=True))
        if dt:
            return dt
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
    # Strip category/date noise that can be included in card links.
    for pat in DATE_PATTERNS:
        t = pat.sub("", t)
    t = re.sub(r"^(NEWS|FASHION|FILM|MUSIC|ART/DESIGN|BEAUTY|LIFE/STYLE|SHOPPING|FEATURE)\s+", "", t, flags=re.I)
    return clean(t)


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.7",
        "Cache-Control": "no-cache",
    })
    return s


def fetch_html_candidates() -> list[tuple[str, str]]:
    s = session()
    results: list[tuple[str, str]] = []
    errors: list[str] = []
    for url in LIST_URLS:
        try:
            r = s.get(url, timeout=45)
            if r.status_code == 200 and len(r.text) > 1000:
                results.append((url, r.text))
            else:
                errors.append(f"{url}: HTTP {r.status_code}, {len(r.text)} bytes")
        except Exception as e:
            errors.append(f"{url}: {e}")
        time.sleep(2)
    if not results:
        raise RuntimeError("QUI fetch failed: " + " | ".join(errors))
    return results


def extract_articles(page_url: str, page_html: str) -> list[Article]:
    soup = BeautifulSoup(page_html, "html.parser")
    found: dict[str, Article] = {}

    for a in soup.find_all("a", href=True):
        url = urljoin(SITE_URL, a["href"])
        if not url.startswith(SITE_URL):
            continue
        if not any(p in url for p in ARTICLE_PATHS):
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
    print(f"{page_url}: extracted {len(articles)} articles")
    return articles


def fetch_articles() -> list[Article]:
    merged: dict[str, Article] = {}
    for page_url, page_html in fetch_html_candidates():
        for article in extract_articles(page_url, page_html):
            prev = merged.get(article.url)
            if prev is None or len(article.title) < len(prev.title):
                merged[article.url] = article

    articles = sorted(merged.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("Fetched QUI pages successfully, but no dated article cards were found")
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
        print(f"Wrote {len(articles)} QUI articles")
        print(f"Newest: {articles[0].published.date()} {articles[0].title}")
        return 0
    except Exception as e:
        # A temporary 5xx from QUI should not turn the scheduled workflow red or
        # destroy the last good feed. Keep the existing feed and try again tomorrow.
        print(f"WARNING: {e}", file=sys.stderr)
        if OUTPUT.exists() and OUTPUT.stat().st_size > 100:
            print("Keeping previous feed.xml")
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
