"""
scrapers/base_scraper.py — Shared utilities for all scrapers
"""

import re
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def extract_tickers(text: str, watchlist: list[str]) -> list[str]:
    """
    Find ticker mentions in a block of text.
    Matches $AAPL, AAPL, or (AAPL) style references.
    """
    found = []
    text_upper = text.upper()
    for ticker in watchlist:
        pattern = rf"(?<![A-Z])\$?{re.escape(ticker)}(?![A-Z])"
        if re.search(pattern, text_upper):
            found.append(ticker)
    return list(set(found))


def clean_text(text: str) -> str:
    """Strip URLs, special chars, and excess whitespace."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s$.,!?%\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def age_hours(dt: datetime) -> float:
    """Return how many hours ago a datetime was."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 3600


def recency_score(hours_old: float, half_life: float = 12.0) -> float:
    """
    Exponential decay score. At 0h → 1.0, at half_life hours → 0.5.
    """
    import math
    return math.exp(-0.693 * hours_old / half_life)
