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
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SITE_URL = "https://qui.tokyo/"
OUTPUT = Path(__file__).with_name("feed.xml")

# QUI's empty-search result page is currently much easier to parse reliably than
# /media for automated clients. It exposes rows such as:
#   New NEWS 202606/22 Mizunoが...を復刻発売
LIST_URLS = [
    "https://qui.tokyo/?cn=xuclu&s=",
    "https://qui.tokyo/?s=",
    "https://qui.tokyo/category/news",
    "https://qui.tokyo/media",
    "https://qui.tokyo/",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

MONTH_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),\s+(20\d{2})\b",
    re.I,
)
NUMERIC_DATE_RE = re.compile(r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b")
COMPACT_DATE_RE = re.compile(r"\b(20\d{2})(\d{2})/(\d{2})\b")
CATEGORY_RE = r"NEWS|FASHION|FILM|MUSIC|ART/DESIGN|BEAUTY|LIFE/STYLE|SHOPPING|FEATURE"
SEARCH_ROW_RE = re.compile(
    rf"^(?:New\s+)?(?:{CATEGORY_RE})\s+(20\d{{2}})(\d{{2}})/(\d{{2}})\s+(.+)$",
    re.I,
)

EXCLUDED_PATH_PARTS = (
    "/category/", "/post_tag/", "/tag/", "/page/", "/about", "/company",
    "/privacy", "/contact", "/store", "/news-release",
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
    m = COMPACT_DATE_RE.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, tzinfo=timezone.utc)
    m = MONTH_RE.search(text)
    if m:
        return datetime.strptime(m.group(0), "%b %d, %Y").replace(tzinfo=timezone.utc)
    m = NUMERIC_DATE_RE.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, tzinfo=timezone.utc)
    return None


def nearby_date(node: Tag) -> datetime | None:
    cur: Tag | None = node
    for _ in range(12):
        if cur is None:
            break
        dt = parse_date(cur.get_text(" ", strip=True))
        if dt:
            return dt
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return None


def normalize_title(text: str) -> tuple[str, datetime | None]:
    """Return clean QUI article title and date embedded in the row, if present."""
    text = clean(text)

    # Best case: QUI search-result row, e.g. "New NEWS 202606/22 Title".
    m = SEARCH_ROW_RE.match(text)
    if m:
        y, mo, d = map(int, m.group(1, 2, 3))
        title = clean(m.group(4))
        return title, datetime(y, mo, d, tzinfo=timezone.utc)

    # Remove common card noise while preserving punctuation inside the real title.
    text = re.sub(r"^New\s+", "", text, flags=re.I)
    text = re.sub(rf"^(?:{CATEGORY_RE})\s+", "", text, flags=re.I)
    text = COMPACT_DATE_RE.sub("", text, count=1)
    text = MONTH_RE.sub("", text, count=1)
    text = NUMERIC_DATE_RE.sub("", text, count=1)
    text = re.sub(rf"^(?:{CATEGORY_RE})\s+", "", clean(text), flags=re.I)
    return clean(text), None


def anchor_title(a: Tag) -> tuple[str, datetime | None]:
    # On QUI, the anchor's own text is often the most reliable source because
    # heading tags can contain only the category or other card metadata.
    raw = clean(a.get_text(" ", strip=True))
    title, embedded_date = normalize_title(raw)
    if len(title) >= 5:
        return title, embedded_date

    img = a.find("img", alt=True)
    if img:
        title, embedded_date = normalize_title(clean(img.get("alt", "")))
        if len(title) >= 5:
            return title, embedded_date

    return "", embedded_date


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.7",
    })
    return s


def is_article_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != "qui.tokyo":
        return False
    path = parsed.path.rstrip("/")
    if not path or any(part in path for part in EXCLUDED_PATH_PARTS):
        return False
    # Search/list/navigation endpoints are not articles.
    if path in ("", "/media"):
        return False
    return True


def fetch_pages() -> list[tuple[str, str]]:
    s = session()
    pages: list[tuple[str, str]] = []
    errors: list[str] = []
    for url in LIST_URLS:
        try:
            r = s.get(url, timeout=35)
            if r.status_code == 200 and len(r.text) > 1000:
                pages.append((url, r.text))
            else:
                errors.append(f"{url}: HTTP {r.status_code}, {len(r.text)} bytes")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
        time.sleep(1)
    if not pages:
        raise RuntimeError("QUI fetch failed: " + " | ".join(errors))
    return pages


def extract_articles(page_url: str, page_html: str) -> list[Article]:
    soup = BeautifulSoup(page_html, "html.parser")
    found: dict[str, Article] = {}

    for a in soup.find_all("a", href=True):
        url = urljoin(SITE_URL, a["href"]).split("#", 1)[0]
        if not is_article_url(url):
            continue

        title, embedded_date = anchor_title(a)
        if len(title) < 5:
            continue
        published = embedded_date or nearby_date(a)
        if published is None:
            continue

        # Reject obvious navigation labels accidentally caught as titles.
        if title.upper() in {"ALL", "LATEST", "POPULAR", "MORE", "READ MORE"}:
            continue

        candidate = Article(title=title, url=url, published=published)
        prev = found.get(url)
        # Prefer a plausible, compact title over a whole-card text blob.
        if prev is None or (5 <= len(title) < len(prev.title)):
            found[url] = candidate

    articles = sorted(found.values(), key=lambda x: (x.published, x.url), reverse=True)
    print(f"{page_url}: extracted {len(articles)} articles")
    return articles


def fetch_articles() -> list[Article]:
    merged: dict[str, Article] = {}
    for page_url, page_html in fetch_pages():
        for article in extract_articles(page_url, page_html):
            prev = merged.get(article.url)
            if prev is None or len(article.title) < len(prev.title):
                merged[article.url] = article

    articles = sorted(merged.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("Fetched QUI pages successfully, but no article rows were found")
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
        '    <description>QUIの最新記事から生成した非公式RSSです。</description>',
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
    except Exception as exc:
        # Keep the last known-good feed during temporary QUI outages.
        print(f"WARNING: {exc}", file=sys.stderr)
        if OUTPUT.exists() and "<item>" in OUTPUT.read_text(encoding="utf-8", errors="ignore"):
            print("Keeping previous feed.xml")
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
