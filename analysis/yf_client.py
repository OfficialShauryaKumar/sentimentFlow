"""
analysis/yf_client.py — Throttled yfinance client to avoid 429 rate limits

Why this exists: yfinance's default behavior is to hit Yahoo's API with no
User-Agent and no throttling. Yahoo aggressively rate-limits these requests
(returns HTTP 429 "Too Many Requests"), so any pipeline that scans a
watchlist of 10+ tickers in rapid succession will get blocked, and every
fetch_history / fetch_quote / t.info call returns junk.

This module provides:
  - A shared requests.Session with a proper desktop User-Agent
  - A min-interval throttle so we never burst more than 1 call / 2 seconds
  - get_ticker(symbol) — returns a yf.Ticker bound to the throttled session
  - call_with_retry(fn) — exponential backoff on 429 errors

Usage:
    from analysis.yf_client import get_ticker, call_with_retry
    t    = get_ticker("AAPL")
    hist = call_with_retry(lambda: t.history(period="6mo"))
"""

import logging
import os
import threading
import time
from typing import Callable, TypeVar

import requests

try:
    import yfinance as yf
    _yf_ok = True
except ImportError:
    _yf_ok = False

logger = logging.getLogger("yf_client")

# ─── Tunables ────────────────────────────────────────────────────────────────
# Min seconds between any two HTTP calls yfinance makes through this session.
MIN_INTERVAL_SEC = 1.0

# Set SKIP_YFINANCE=true in .env to bypass yfinance entirely (useful when
# Yahoo has rate-limited your IP and recovery requires waiting hours).
SKIP_YFINANCE = os.getenv("SKIP_YFINANCE", "false").lower() == "true"

# Circuit breaker: after this many consecutive rate-limit errors in a single
# run, stop trying yfinance entirely. Saves you from the ~10-minute hang
# yfinance's own internal retry loop produces when Yahoo blocks your IP.
CIRCUIT_BREAKER_THRESHOLD = 2

# Retries removed: yfinance has its own internal retry that already wastes
# ~8s per failure. Adding our own retries on top just compounds the wait.
MAX_RETRIES = 0

# (Kept for backward compat; unused with MAX_RETRIES=0.)
BASE_BACKOFF_SEC = 5.0

# A modern desktop User-Agent. Yahoo throttles unknown / default UAs much
# harder than browser-like UAs.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ─── Throttled session ───────────────────────────────────────────────────────

class _ThrottledSession(requests.Session):
    """A requests.Session that enforces a minimum interval between calls."""

    def __init__(self, min_interval: float = MIN_INTERVAL_SEC):
        super().__init__()
        self.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        self._min_interval = min_interval
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def request(self, method, url, **kwargs):
        with self._lock:
            elapsed = time.time() - self._last_call_time
            if elapsed < self._min_interval:
                wait = self._min_interval - elapsed
                time.sleep(wait)
            self._last_call_time = time.time()
        return super().request(method, url, **kwargs)


# Single shared session — every yfinance call in the app reuses this.
_session = _ThrottledSession()

# Circuit-breaker state. Once tripped, stays open for the rest of the run.
_failure_count = 0
_circuit_open = False
_circuit_lock = threading.Lock()


def _trip_circuit_if_needed() -> None:
    global _circuit_open
    with _circuit_lock:
        if _failure_count >= CIRCUIT_BREAKER_THRESHOLD and not _circuit_open:
            _circuit_open = True
            logger.warning(
                "yfinance circuit breaker TRIPPED after %d rate-limit errors — "
                "skipping remaining yfinance calls. Set SKIP_YFINANCE=true in .env "
                "to suppress these warnings on next run.", _failure_count,
            )


def is_circuit_open() -> bool:
    return _circuit_open or SKIP_YFINANCE


# ─── Public API ──────────────────────────────────────────────────────────────

def get_ticker(symbol: str):
    """Return a yf.Ticker bound to the throttled shared session, or None
    if yfinance is unavailable / disabled / the circuit breaker is open."""
    if not _yf_ok or is_circuit_open():
        return None
    return yf.Ticker(symbol, session=_session)


T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = MAX_RETRIES,
    base_backoff: float = BASE_BACKOFF_SEC,
) -> T:
    """
    Call `fn()` once and trip the circuit breaker on rate-limit errors.

    Retries are disabled by default (MAX_RETRIES=0) because yfinance has its
    own internal retry that already wastes ~8s per failed call. The circuit
    breaker guarantees we don't waste 10+ minutes per run when Yahoo has
    blocked the IP — after a few failures we stop trying entirely.
    """
    global _failure_count
    if is_circuit_open():
        return None  # type: ignore[return-value]

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            with _circuit_lock:
                _failure_count = 0  # reset on success
            return result
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = (
                "429" in msg
                or "too many requests" in msg
                or "rate limit" in msg
                or "rate-limited" in msg
                or "expecting value" in msg  # yahoo returns junk HTML when blocked
            )
            if is_rate_limit:
                with _circuit_lock:
                    _failure_count += 1
                _trip_circuit_if_needed()
            if is_rate_limit and attempt < max_retries:
                delay = base_backoff * (2 ** attempt)
                logger.warning(
                    "yfinance rate-limited (attempt %d/%d) — sleeping %.1fs",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
                last_err = e
                continue
            # Either not a rate-limit error, or we're out of retries
            raise
    if last_err:
        raise last_err
    raise RuntimeError("call_with_retry exited unexpectedly")
