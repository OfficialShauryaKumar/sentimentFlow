"""
scrapers/reddit_scraper.py — Scrapes Reddit for stock mentions
"""

import math
from datetime import datetime, timezone
from typing import Optional

import praw
import config
from scrapers.base_scraper import get_logger, extract_tickers, clean_text

logger = get_logger("reddit")


def _build_client() -> Optional[praw.Reddit]:
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        logger.warning("Reddit credentials not set — skipping Reddit scraper.")
        return None
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        check_for_async=False,
    )


def scrape_reddit(watchlist: list[str] = None) -> list[dict]:
    """
    Scrape configured subreddits and return a list of mention records.

    Returns:
        List of dicts with keys:
            source, ticker, title, body, score, comments,
            traction, url, created_at, raw_text
    """
    watchlist = watchlist or config.WATCHLIST
    reddit = _build_client()
    if reddit is None:
        return []

    results = []

    for sub_name in config.REDDIT_SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            posts = getattr(subreddit, config.REDDIT_SORT)(
                limit=config.REDDIT_LIMIT,
                time_filter=config.REDDIT_TIME if config.REDDIT_SORT == "top" else "all",
            )

            for post in posts:
                full_text = f"{post.title} {post.selftext}"
                tickers = extract_tickers(full_text, watchlist)
                if not tickers:
                    continue

                # Traction = log-scale of engagement
                upvotes  = max(post.score, 1)
                comments = max(post.num_comments, 1)
                traction = (
                    config.WEIGHT_REDDIT_UPVOTES  * math.log1p(upvotes) +
                    config.WEIGHT_REDDIT_COMMENTS * math.log1p(comments)
                ) / 10.0  # normalise to ~0-1 range

                created_at = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)

                for ticker in tickers:
                    results.append({
                        "source":     f"r/{sub_name}",
                        "source_type": "reddit",
                        "ticker":     ticker,
                        "title":      post.title[:200],
                        "body":       clean_text(post.selftext[:500]),
                        "score":      post.score,
                        "comments":   post.num_comments,
                        "traction":   round(traction, 4),
                        "url":        f"https://reddit.com{post.permalink}",
                        "created_at": created_at.isoformat(),
                        "raw_text":   clean_text(full_text[:1000]),
                    })

            logger.info(f"r/{sub_name}: scraped {len([r for r in results if r['source'] == f'r/{sub_name}'])} mentions")

        except Exception as e:
            logger.error(f"r/{sub_name} error: {e}")
            continue

    return results
