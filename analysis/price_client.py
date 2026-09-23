"""
analysis/price_client.py — Price data via Yahoo's public v8 chart JSON API.

Why not the yfinance library: yfinance first hits Yahoo's cookie/crumb
endpoints, which rate-limit cloud IPs (Render) hard. The plain v8 chart
endpoint needs no cookie/crumb and returns OHLCV JSON directly.
(stooq.com was tried previously but now answers "Access denied".)

Usage:
    from analysis.price_client import fetch_history, get_latest_price
    df    = fetch_history("SPY", days=90)   # pd.DataFrame [Open, High, Low, Close, Volume]
    price = get_latest_price("AAPL")        # float or None
"""

import logging
import threading
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("price_client")

_HOSTS = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}
_TIMEOUT = 6

_session = requests.Session()
_session.headers.update(_HEADERS)

# Cache: symbol -> (timestamp, full 1y DataFrame). One request per symbol.
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 900  # 15 minutes

last_error: dict[str, str] = {}   # symbol -> last error (for /api/debug/price)


def _range_for(days: int) -> str:
    if days <= 370:
        return "1y"
    if days <= 740:
        return "2y"
    return "5y"


def _download(symbol: str, rng: str) -> Optional[pd.DataFrame]:
    for host in _HOSTS:
        url = f"{host}/v8/finance/chart/{symbol}"
        try:
            r = _session.get(url, params={"range": rng, "interval": "1d"}, timeout=_TIMEOUT)
            if r.status_code != 200:
                last_error[symbol] = f"{host} HTTP {r.status_code}: {r.text[:120]}"
                continue
            res = (r.json().get("chart") or {}).get("result") or []
            if not res:
                last_error[symbol] = f"{host}: empty result"
                continue
            res = res[0]
            ts = res.get("timestamp") or []
            q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
            if not ts or not q.get("close"):
                last_error[symbol] = f"{host}: no timestamps/close"
                continue
            df = pd.DataFrame({
                "Open":   q.get("open"),
                "High":   q.get("high"),
                "Low":    q.get("low"),
                "Close":  q.get("close"),
                "Volume": q.get("volume"),
            }, index=pd.to_datetime(ts, unit="s").normalize())
            df.index.name = "Date"
            df = df.dropna(subset=["Close"])
            df = df[df["Close"] > 0]
            df["Volume"] = df["Volume"].fillna(0)
            df = df[~df.index.duplicated(keep="last")]
            if df.empty:
                last_error[symbol] = f"{host}: empty after cleaning"
                continue
            last_error.pop(symbol, None)
            return df
        except Exception as e:
            last_error[symbol] = f"{host}: {type(e).__name__}: {e}"
            continue
    logger.warning(f"Price fetch failed for {symbol}: {last_error.get(symbol)}")
    return None


def fetch_history(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """Daily OHLCV for the last `days` calendar days, or None on failure."""
    rng = _range_for(days)
    key = f"{symbol}:{rng}"
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        df = hit[1]
    else:
        df = _download(symbol, rng)
        if df is None:
            return None
        with _cache_lock:
            _cache[key] = (now, df)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    out = df[df.index >= cutoff]
    return out if not out.empty else df.tail(1)


def get_latest_price(symbol: str) -> Optional[float]:
    df = fetch_history(symbol, days=10)
    if df is None or df.empty:
        return None
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
