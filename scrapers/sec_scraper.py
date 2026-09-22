"""
scrapers/sec_scraper.py — SEC EDGAR filing scraper
Fetches 8-K, 10-Q, and 10-K filings for watchlist tickers using the free EDGAR API.
No API key required.
"""

import re
import time
import logging
import requests
from datetime import datetime, timezone, timedelta

import config
from analysis.sentiment import score_sentiment, classify_sentiment

logger = logging.getLogger("sec_scraper")

EDGAR_BASE      = "https://data.sec.gov"
EDGAR_ARCHIVE   = "https://www.sec.gov/Archives/edgar/data"
TICKER_MAP_URL  = "https://www.sec.gov/files/company_tickers.json"
HEADERS         = {"User-Agent": "SentimentFlow/1.0 contact@sentimentflow.app"}

# Filing types to fetch
TARGET_FORMS    = {"8-K", "10-K", "10-Q"}
# How many days back to look
LOOKBACK_DAYS   = 90


# ── CIK lookup ────────────────────────────────────────────────────────────────

_ticker_cik_cache: dict[str, str] = {}


def _load_ticker_map() -> dict[str, str]:
    """Download the EDGAR ticker→CIK map once and cache it."""
    global _ticker_cik_cache
    if _ticker_cik_cache:
        return _ticker_cik_cache
    try:
        r = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Format: { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
        mapping = {}
        for entry in data.values():
            mapping[entry["ticker"].upper()] = str(entry["cik_str"])
        _ticker_cik_cache = mapping
        logger.info(f"Loaded {len(mapping)} ticker→CIK mappings from EDGAR.")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load ticker map: {e}")
        return {}


def get_cik(ticker: str) -> str | None:
    mapping = _load_ticker_map()
    return mapping.get(ticker.upper())


# ── Filing fetch ──────────────────────────────────────────────────────────────

def _get_submissions(cik: str) -> dict:
    """Fetch the submissions JSON for a company (contains all filings metadata)."""
    padded = cik.zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{padded}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _accession_url(cik: str, accession: str) -> str:
    """Build the EDGAR filing index URL."""
    acc_dashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_dashes}/{accession}-index.htm"


def _fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str:
    """Download and strip HTML/XBRL from a filing document. Returns plain text."""
    acc_no_dash = accession.replace("-", "")
    url = f"{EDGAR_ARCHIVE}/{cik}/{acc_no_dash}/{primary_doc}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        text = r.text
        # Strip XML/HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Truncate to 8000 chars for sentiment (keep intro which has the key info)
        return text[:8000]
    except Exception as e:
        logger.warning(f"Could not fetch filing text {url}: {e}")
        return ""


# ── Main scrape function ───────────────────────────────────────────────────────

def scrape_sec_filings(tickers: list[str] = None) -> list[dict]:
    """
    Fetch recent 8-K, 10-K, 10-Q filings for all tickers.
    Returns a list of filing dicts with sentiment scores attached.
    """
    if tickers is None:
        tickers = config.WATCHLIST

    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    results = []

    for ticker in tickers:
        cik = get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}, skipping.")
            continue

        try:
            subs = _get_submissions(cik)
        except Exception as e:
            logger.error(f"Could not fetch submissions for {ticker} (CIK {cik}): {e}")
            time.sleep(0.5)
            continue

        recent = subs.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        dates       = recent.get("filingDate", [])
        accessions  = recent.get("accessionNumber", [])
        documents   = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        for i, form in enumerate(forms):
            if form not in TARGET_FORMS:
                continue
            filed_date = dates[i] if i < len(dates) else ""
            if filed_date < cutoff_str:
                continue  # too old

            accession  = accessions[i] if i < len(accessions) else ""
            primary_doc = documents[i] if i < len(documents) else ""
            description = descriptions[i] if i < len(descriptions) else ""

            logger.info(f"Fetching {form} for {ticker} filed {filed_date}...")
            text = _fetch_filing_text(cik, accession.replace("-", ""), accession) if primary_doc else ""

            # Run sentiment on the filing text
            sent_score = score_sentiment(text) if text else 0.0
            sentiment  = {"score": sent_score, "label": classify_sentiment(sent_score)}

            filing = {
                "ticker":          ticker,
                "cik":             cik,
                "form_type":       form,
                "accession":       accession,
                "filed_date":      filed_date,
                "title":           description or f"{form} filing",
                "summary":         text[:500] if text else "",
                "sentiment_score": sentiment.get("score", 0.0),
                "sentiment_label": sentiment.get("label", "NEUTRAL"),
                "url":             _accession_url(cik, accession),
                "fetched_at":      datetime.now(timezone.utc).isoformat(),
            }
            results.append(filing)

            # Respect EDGAR rate limits (10 req/sec max, stay well under)
            time.sleep(0.2)

        time.sleep(0.3)

    logger.info(f"SEC scraper: fetched {len(results)} filings for {len(tickers)} tickers.")
    return results
