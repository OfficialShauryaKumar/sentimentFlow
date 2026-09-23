"""
scrapers/news_scraper.py — Scrapes NewsAPI + RSS feeds for stock mentions
"""

from datetime import datetime, timezone
from typing import Optional
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import config
from scrapers.base_scraper import (
    get_logger, extract_tickers, clean_text, age_hours, recency_score
)

logger = get_logger("news")

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


# ─── Google News (per-ticker) ─────────────────────────────────────────────────

def _google_news_for(ticker: str) -> list[dict]:
    """Recent Google News headlines for one ticker (tagged to that ticker)."""
    url = ("https://news.google.com/rss/search?q="
           f"{quote_plus(ticker + ' stock')}+when:2d&hl=en-US&gl=US&ceid=US:en")
    out = []
    try:
        r = requests.get(url, timeout=8, headers=_UA)
        feed = feedparser.parse(r.content)
        import calendar
        for entry in feed.entries[:config.MAX_POSTS]:
            title = entry.get("title", "")
            pub = entry.get("published_parsed")
            pub_dt = (datetime.fromtimestamp(calendar.timegm(pub), tz=timezone.utc)
                      if pub else datetime.now(timezone.utc))
            out.append({
                "source":      "Google News",
                "source_type": "rss",
                "ticker":      ticker,
                "title":       clean_text(title[:200]),
                "body":        "",
                "score":       None,
                "comments":    None,
                "traction":    round(recency_score(age_hours(pub_dt), half_life=12.0), 4),
                "url":         entry.get("link", ""),
                "created_at":  pub_dt.isoformat(),
                "raw_text":    clean_text(title[:1000]),
            })
    except Exception as e:
        logger.warning(f"Google News [{ticker}] error: {e}")
    return out


def scrape_google_news(watchlist: list[str] = None) -> list[dict]:
    watchlist = watchlist or config.WATCHLIST
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = [m for batch in ex.map(_google_news_for, watchlist) for m in batch]
    logger.info(f"Google News: {len(results)} mentions")
    return results


# ─── NewsAPI ──────────────────────────────────────────────────────────────────

def scrape_newsapi(watchlist: list[str] = None) -> list[dict]:
    """Fetch finance headlines from NewsAPI and find ticker mentions."""
    watchlist = watchlist or config.WATCHLIST

    if not config.NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI scraper.")
        return []

    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=config.NEWS_API_KEY)
    except ImportError:
        logger.error("newsapi-python not installed.")
        return []

    results = []

    # Search each ticker individually for targeted results
    for ticker in watchlist:
        try:
            response = client.get_everything(
                q=ticker,
                language="en",
                sort_by="publishedAt",
                page_size=min(config.MAX_POSTS, 20),
            )

            for article in response.get("articles", []):
                title   = article.get("title") or ""
                desc    = article.get("description") or ""
                content = article.get("content") or ""
                full_text = f"{title} {desc} {content}"

                tickers = extract_tickers(full_text, watchlist)
                if ticker not in tickers:
                    tickers.insert(0, ticker)

                # Parse publish date
                pub_str = article.get("publishedAt", "")
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                hours_old = age_hours(pub_dt)
                traction  = recency_score(hours_old, half_life=6.0)

                for t in tickers:
                    results.append({
                        "source":      article.get("source", {}).get("name", "NewsAPI"),
                        "source_type": "news",
                        "ticker":      t,
                        "title":       clean_text(title[:200]),
                        "body":        clean_text(desc[:500]),
                        "score":       None,
                        "comments":    None,
                        "traction":    round(traction, 4),
                        "url":         article.get("url", ""),
                        "created_at":  pub_dt.isoformat(),
                        "raw_text":    clean_text(full_text[:1000]),
                    })

            logger.info(f"NewsAPI: {ticker} → {len(response.get('articles', []))} articles")

        except Exception as e:
            logger.error(f"NewsAPI error for {ticker}: {e}")
            continue

    return results


# ─── RSS Feeds ────────────────────────────────────────────────────────────────

def scrape_rss(watchlist: list[str] = None) -> list[dict]:
    """Parse RSS feeds and extract stock mentions."""
    watchlist = watchlist or config.WATCHLIST
    results   = []

    for feed_name, feed_url in config.RSS_FEEDS.items():
        try:
            # Fetch with a timeout — feedparser.parse(url) can hang forever.
            _resp = requests.get(feed_url, timeout=8, headers=_UA)
            feed = feedparser.parse(_resp.content)

            for entry in feed.entries[:config.MAX_POSTS]:
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                full_text = f"{title} {summary}"

                tickers = extract_tickers(full_text, watchlist)
                if not tickers:
                    continue

                # Parse date
                pub_struct = entry.get("published_parsed")
                if pub_struct:
                    import calendar
                    pub_dt = datetime.fromtimestamp(
                        calendar.timegm(pub_struct), tz=timezone.utc
                    )
                else:
                    pub_dt = datetime.now(timezone.utc)

                hours_old = age_hours(pub_dt)
                traction  = recency_score(hours_old, half_life=8.0)

                for ticker in tickers:
                    results.append({
                        "source":      feed_name,
                        "source_type": "rss",
                        "ticker":      ticker,
                        "title":       clean_text(title[:200]),
                        "body":        clean_text(summary[:500]),
                        "score":       None,
                        "comments":    None,
                        "traction":    round(traction, 4),
                        "url":         entry.get("link", ""),
                        "created_at":  pub_dt.isoformat(),
                        "raw_text":    clean_text(full_text[:1000]),
                    })

            logger.info(f"RSS [{feed_name}]: {len([r for r in results if r['source'] == feed_name])} mentions")

        except Exception as e:
            logger.error(f"RSS [{feed_name}] error: {e}")
            continue

    return results


# ─── Combined ─────────────────────────────────────────────────────────────────

def scrape_news(watchlist: list[str] = None) -> list[dict]:
    """Run both NewsAPI and RSS scrapers and merge results."""
    newsapi_results = scrape_newsapi(watchlist)
    rss_results     = scrape_rss(watchlist)
    google_results  = scrape_google_news(watchlist)
    return newsapi_results + rss_results + google_results
