"""
analysis/price_client.py — Daily price history that works from Render.

Order of sources:
  1. Yahoo v8 chart JSON (no cookie/crumb). Skipped for 1 hour after any 429,
     since Yahoo rate-limits cloud IPs.
  2. Nasdaq public API — stocks and ETFs.
  3. FRED (St. Louis Fed) CSV — VIX, 10-year yield, dollar index.

    from analysis.price_client import fetch_history
    df = fetch_history("SPY", days=120)   # DataFrame [Open, High, Low, Close, Volume] or None
"""

import logging
import threading
import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("price_client")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_TIMEOUT = 8

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 1800  # 30 minutes

_yahoo_blocked_until = 0.0
_YAHOO_COOLDOWN = 3600

# Symbols served from FRED when Yahoo is unavailable
_FRED = {"^VIX": "VIXCLS", "^TNX": "DGS10", "DX-Y.NYB": "DTWEXBGS"}


def _finish(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.astype(float)
    df.index.name = "Date"
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    for c in ("Open", "High", "Low"):
        df[c] = df[c].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df if not df.empty else None


def _yahoo(symbol: str) -> Optional[pd.DataFrame]:
    global _yahoo_blocked_until
    if time.time() < _yahoo_blocked_until:
        return None
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                         params={"range": "1y", "interval": "1d"},
                         headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        if r.status_code == 429:
            _yahoo_blocked_until = time.time() + _YAHOO_COOLDOWN
            logger.warning("Yahoo 429 — skipping Yahoo for 1 hour")
            return None
        if r.status_code != 200:
            return None
        res = ((r.json().get("chart") or {}).get("result") or [None])[0]
        if not res or not res.get("timestamp"):
            return None
        q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        df = pd.DataFrame({k.capitalize(): q.get(k) for k in ("open", "high", "low", "close", "volume")},
                          index=pd.to_datetime(res["timestamp"], unit="s").normalize())
        return _finish(df)
    except Exception as e:
        logger.debug(f"Yahoo failed for {symbol}: {e}")
        return None


def _num(v):
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _nasdaq(symbol: str) -> Optional[pd.DataFrame]:
    if symbol.startswith("^") or "." in symbol:
        return None
    start = (pd.Timestamp.now() - pd.Timedelta(days=380)).strftime("%Y-%m-%d")
    headers = {"User-Agent": _UA, "Accept": "application/json, text/plain, */*",
               "Accept-Language": "en-US,en;q=0.9",
               "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    for ac in ("etf", "stocks"):
        try:
            r = requests.get(f"https://api.nasdaq.com/api/quote/{symbol}/historical",
                             params={"assetclass": ac, "fromdate": start, "limit": 400},
                             headers=headers, timeout=_TIMEOUT)
            if r.status_code != 200:
                continue
            rows = ((((r.json() or {}).get("data") or {}).get("tradesTable") or {}).get("rows")) or []
            if not rows:
                continue
            df = pd.DataFrame({
                "Open":   [_num(x.get("open")) for x in rows],
                "High":   [_num(x.get("high")) for x in rows],
                "Low":    [_num(x.get("low")) for x in rows],
                "Close":  [_num(x.get("close")) for x in rows],
                "Volume": [_num(x.get("volume")) for x in rows],
            }, index=pd.to_datetime([x.get("date") for x in rows], format="%m/%d/%Y", errors="coerce"))
            df = _finish(df)
            if df is not None:
                return df
        except Exception as e:
            logger.debug(f"Nasdaq ({ac}) failed for {symbol}: {e}")
    return None


def _fred(symbol: str) -> Optional[pd.DataFrame]:
    series = _FRED.get(symbol)
    if not series:
        return None
    start = (pd.Timestamp.now() - pd.Timedelta(days=380)).strftime("%Y-%m-%d")
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": series, "cosd": start},
                         headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        raw = pd.read_csv(StringIO(r.text), na_values=["."])
        raw.columns = ["Date", "Close"]
        s = pd.to_numeric(raw["Close"], errors="coerce")
        df = pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 0.0})
        df.index = pd.to_datetime(raw["Date"], errors="coerce")
        return _finish(df)
    except Exception as e:
        logger.debug(f"FRED failed for {symbol}: {e}")
        return None


def fetch_history(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """Daily OHLCV for roughly the last `days` calendar days, or None."""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(symbol)
    if hit and now - hit[0] < _CACHE_TTL:
        df = hit[1]
    else:
        df = _yahoo(symbol)
        if df is None:
            df = _fred(symbol) if symbol in _FRED else _nasdaq(symbol)
        if df is None:
            logger.warning(f"No price data for {symbol} from any source")
            return None
        with _cache_lock:
            _cache[symbol] = (now, df)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    out = df[df.index >= cutoff]
    return out if not out.empty else df.tail(1)
