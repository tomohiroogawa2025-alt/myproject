#!/usr/bin/env python3
"""Generate an RSS 2.0 feed from the latest articles on https://transit.jp/."""
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

SITE_URL = "https://transit.jp/"
OUTPUT = Path(__file__).with_name("feed.xml")
DATE_RE = re.compile(r"\b(20\d{2})[./-](\d{2})[./-](\d{2})\b")
ARTICLE_RE = re.compile(r"^https://transit\.jp/article/")
USER_AGENT = "Mozilla/5.0 (compatible; transit-rss/1.0; +https://github.com/)"


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    published: datetime


def clean(text: str) -> str:
    return " ".join(text.split())


def find_date(node: Tag) -> datetime | None:
    """Search nearby container text for TRANSIT's YYYY.MM.DD date."""
    current: Tag | None = node
    for _ in range(5):
        if current is None:
            break
        text = clean(current.get_text(" ", strip=True))
        match = DATE_RE.search(text)
        if match:
            y, m, d = map(int, match.groups())
            return datetime(y, m, d, 0, 0, tzinfo=timezone.utc)
        current = current.parent if isinstance(current.parent, Tag) else None
    return None


def title_from_anchor(anchor: Tag) -> str:
    # Prefer image alt text because TRANSIT's card anchors can contain metadata too.
    img = anchor.find("img", alt=True)
    if img and clean(img.get("alt", "")):
        return clean(img.get("alt", ""))

    text = clean(anchor.get_text(" ", strip=True))
    # Remove common card metadata from the tail when possible.
    text = DATE_RE.sub("", text)
    text = re.sub(r"\b\d+\s*min\s*read\b", "", text, flags=re.I)
    return clean(text)


def fetch_articles() -> list[Article]:
    response = requests.get(
        SITE_URL,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"},
        timeout=30,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    by_url: dict[str, Article] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(SITE_URL, anchor["href"])
        if not ARTICLE_RE.match(url):
            continue

        published = find_date(anchor)
        if published is None:
            continue

        title = title_from_anchor(anchor)
        if not title or len(title) < 4:
            continue

        candidate = Article(title=title, url=url, published=published)
        previous = by_url.get(url)
        # Duplicate cards occur on the home page. Keep the cleaner/shorter title.
        if previous is None or len(candidate.title) < len(previous.title):
            by_url[url] = candidate

    articles = sorted(by_url.values(), key=lambda x: (x.published, x.url), reverse=True)
    if not articles:
        raise RuntimeError("No dated /article/ links were found on TRANSIT top page")
    return articles[:40]


def xml_escape(value: str) -> str:
    return html.escape(value, quote=True)


def render_rss(articles: list[Article]) -> str:
    now = datetime.now(timezone.utc)
    items: list[str] = []
    for article in articles:
        items.append(
            "\n".join(
                [
                    "    <item>",
                    f"      <title>{xml_escape(article.title)}</title>",
                    f"      <link>{xml_escape(article.url)}</link>",
                    f"      <guid isPermaLink=\"true\">{xml_escape(article.url)}</guid>",
                    f"      <pubDate>{email.utils.format_datetime(article.published)}</pubDate>",
                    "    </item>",
                ]
            )
        )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rss version="2.0">',
            "  <channel>",
            "    <title>TRANSIT.jp 最新記事（unofficial）</title>",
            f"    <link>{SITE_URL}</link>",
            "    <description>TRANSIT.jp のトップページから最新記事を定期取得して生成した非公式RSSです。</description>",
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

# Trigger-safe source marker: changing this file starts the workflow, while feed.xml updates do not.
