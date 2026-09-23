"""
analysis/price_client.py — Free stock/ETF/index price fetching via stooq.com

Why stooq instead of Yahoo Finance / yfinance:
  Yahoo Finance aggressively rate-limits and blocks cloud hosting provider
  IP ranges (Render, Railway, Heroku, etc.). stooq.com is a Polish financial
  data portal that provides free OHLCV CSV downloads with no API key, no
  rate-limit headers, and no known cloud-IP blocks.

API format: https://stooq.com/q/d/l/?s=<symbol>&i=d&d1=YYYYMMDD&d2=YYYYMMDD
Returns: CSV with columns Date,Open,High,Low,Close,Volume

Usage:
    from analysis.price_client import fetch_history, get_latest_price
    df    = fetch_history("SPY", days=90)   # returns pd.DataFrame
    price = get_latest_price("AAPL")        # returns float or None
"""

import logging
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("price_client")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Maps Yahoo Finance / common symbol → stooq symbol
# US stocks/ETFs: just append ".us" (handled by default)
# Indices and special tickers need explicit mapping
_SYMBOL_MAP: dict[str, str] = {
    # Broad market ETFs
    "SPY":      "spy.us",
    "QQQ":      "qqq.us",
    "DIA":      "dia.us",
    "IWM":      "iwm.us",
    # Sector ETFs
    "XLK":      "xlk.us",
    "XLF":      "xlf.us",
    "XLV":      "xlv.us",
    "XLE":      "xle.us",
    "XLY":      "xly.us",
    "XLU":      "xlu.us",
    "XLI":      "xli.us",
    "XLB":      "xlb.us",
    # Commodities / macro ETFs
    "GLD":      "gld.us",
    "USO":      "uso.us",
    # Volatility / rates / dollar
    "^VIX":     "^vix",
    "^TNX":     "10ust.b",    # 10-year US Treasury yield
    "DX-Y.NYB": "dxy",        # US Dollar Index
    # S&P 500 index (fallback to SPY)
    "^GSPC":    "^spx",
    "^NDX":     "^ndx",
    "^DJI":     "^dji",
}

# Simple in-process cache: (symbol, date) → DataFrame
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 3600  # seconds (1 hour)


def _to_stooq(symbol: str) -> str:
    if symbol in _SYMBOL_MAP:
        return _SYMBOL_MAP[symbol]
    # Default: treat as US stock/ETF
    return f"{symbol.lower()}.us"


def fetch_history(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV history from stooq.com.

    Parameters
    ----------
    symbol : str
        Yahoo Finance-style symbol (e.g. "AAPL", "SPY", "^VIX")
    days : int
        How many calendar days back to fetch (default 120 ≈ 4 months,
        enough for SMA-50 on 63 trading days)

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume]
    or None if the fetch failed or no data was returned.
    """
    # Check cache first
    cache_key = f"{symbol}:{days}"
    if cache_key in _cache:
        ts, df = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return df

    stooq_sym = _to_stooq(symbol)
    end   = datetime.now()
    start = end - timedelta(days=days)

    url = (
        f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}"
    )

    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
        r.raise_for_status()

        text = r.text.strip()
        # stooq returns "No data" or an HTML error page when symbol isn't found
        if not text or len(text) < 30 or text.startswith("<") or "No data" in text:
            logger.debug(f"No stooq data for {symbol} (mapped to {stooq_sym})")
            return None

        df = pd.read_csv(StringIO(text), parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date")

        # Drop rows where Close is 0 or NaN (stooq pads with zeros sometimes)
        df = df[df["Close"] > 0].dropna(subset=["Close"])

        if df.empty:
            logger.debug(f"Empty data for {symbol} after cleaning")
            return None

        _cache[cache_key] = (time.time(), df)
        logger.debug(f"stooq: {symbol} → {len(df)} rows, last close {float(df['Close'].iloc[-1]):.2f}")
        return df

    except Exception as e:
        logger.warning(f"stooq fetch failed for {symbol} ({stooq_sym}): {e}")
        return None


def get_latest_price(symbol: str) -> Optional[float]:
    """
    Return the most recent available closing price for `symbol`.
    Uses a short lookback (10 days) for speed.
    Returns float or None.
    """
    df = fetch_history(symbol, days=10)
    if df is None or df.empty:
        return None
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
