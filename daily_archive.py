"""
daily_archive.py — Standalone daily sentiment snapshot

Runs the full SentimentFlow pipeline (scrape -> analyze -> archive) once
and exits. Designed to be invoked from cron so the dashboard doesn't have
to be running for the archive to accumulate.

USAGE
    python daily_archive.py

CRON (every weekday at 6pm, with logging):
    0 18 * * 1-5 cd /Users/shauryapersonal/Downloads/sf-project && \\
        /Users/shauryapersonal/Downloads/sf-project/venv/bin/python \\
        daily_archive.py >> data/archive.log 2>&1
"""

import logging
import sys

from app import _archive_snapshot
from analysis import build_recommendations
from scrapers import scrape_news, scrape_reddit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("daily_archive")


def main() -> int:
    log.info("Daily archive starting")
    try:
        reddit = scrape_reddit()
    except Exception as e:
        log.warning(f"Reddit scrape failed: {e}")
        reddit = []
    try:
        news = scrape_news()
    except Exception as e:
        log.warning(f"News scrape failed: {e}")
        news = []

    all_m = reddit + news
    log.info(f"Total mentions: {len(all_m)} (reddit={len(reddit)}, news={len(news)})")
    if not all_m:
        log.error("Zero mentions — nothing to archive. Check API keys / quotas.")
        return 1

    recs = build_recommendations(all_m)
    log.info(f"Built {len(recs)} recommendations")
    _archive_snapshot(recs)
    log.info("Daily archive complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
