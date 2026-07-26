#!/usr/bin/env python3
"""Generate an RSS 2.0 feed from the '新着記事' section of https://www.amakaratecho.jp/."""
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

SITE_URL = "https://www.amakaratecho.jp/"
OUTPUT = Path(__file__).with_name("feed.xml")
DATE_RE = re.compile(r"(20\d{2})[./-](\d{2})[./-](\d{2})")
USER_AGENT = "Mozilla/5.0 (compatible; amakaratecho-rss/1.0; +https://github.com/)"


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    published: datetime


def clean(text: str) -> str:
    return " ".join(text.split())


def same_site(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"amakaratecho.jp", "www.amakaratecho.jp"}


def extract_date(text: str) -> datetime | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return datetime(y, mo, d, 0, 0, tzinfo=timezone.utc)


def title_without_date(text: str) -> str:
    text = DATE_RE.sub("", text)
    return clean(text)


def find_new_articles_container(soup: BeautifulSoup) -> Tag:
    heading = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        if clean(tag.get_text(" ", strip=True)) == "新着記事":
            heading = tag
            break
    if heading is None:
        raise RuntimeError("Could not find 新着記事 heading")

    current: Tag | None = heading
    for _ in range(8):
        if current is None:
            break
        dated_links = 0
        texts = []
        for a in current.find_all("a", href=True):
            t = clean(a.get_text(" ", strip=True))
            texts.append(t)
            if DATE_RE.search(t):
                dated_links += 1
        block_text = clean(current.get_text(" ", strip=True))
        # Prefer the smallest useful block that contains the new-article cards,
        # but not the later '読まれている記事' section.
        if dated_links >= 3 and "読まれている記事" not in block_text:
            return current
        current = current.parent if isinstance(current.parent, Tag) else None

    raise RuntimeError("Could not isolate 新着記事 section")


def fetch_articles() -> list[Article]:
    response = requests.get(
        SITE_URL,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"},
        timeout=30,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    container = find_new_articles_container(soup)

    by_url: dict[str, Article] = {}
    for a in container.find_all("a", href=True):
        text = clean(a.get_text(" ", strip=True))
        published = extract_date(text)
        if published is None:
            continue

        url = urljoin(SITE_URL, a["href"])
        if not same_site(url):
            continue

        title = title_without_date(text)
        if len(title) < 4:
            continue

        by_url[url] = Article(title=title, url=url, published=published)

    articles = sorted(by_url.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("No dated article links found in 新着記事 section")
    return articles[:40]


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def render_rss(articles: list[Article]) -> str:
    now = datetime.now(timezone.utc)
    items: list[str] = []
    for article in articles:
        items.extend(
            [
                "    <item>",
                f"      <title>{esc(article.title)}</title>",
                f"      <link>{esc(article.url)}</link>",
                f"      <guid isPermaLink=\"true\">{esc(article.url)}</guid>",
                f"      <pubDate>{email.utils.format_datetime(article.published)}</pubDate>",
                "    </item>",
            ]
        )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rss version="2.0">',
            "  <channel>",
            "    <title>あまから手帖 Online 新着記事（unofficial）</title>",
            f"    <link>{SITE_URL}</link>",
            "    <description>あまから手帖 Online の「新着記事」欄から生成した非公式RSSです。</description>",
            "    <language>ja</language>",
            f"    <lastBuildDate>{email.utils.format_datetime(now)}</lastBuildDate>",
            *items,
            "  </channel>",
            "</rss>",
            "",
        ]
    )


def main() -> int:
    try:
        articles = fetch_articles()
        OUTPUT.write_text(render_rss(articles), encoding="utf-8")
        print(f"Wrote {len(articles)} articles to {OUTPUT}")
        print(f"Newest: {articles[0].published.date()} {articles[0].title}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
