"""
analysis/market_hours.py — US Market hours awareness

NYSE / NASDAQ trading schedule (Eastern Time):
  Pre-market:   4:00 AM  – 9:30 AM  ET
  Regular:      9:30 AM  – 4:00 PM  ET  (primary session)
  After-hours:  4:00 PM  – 8:00 PM  ET
  Closed:       8:00 PM  – 4:00 AM  ET + all weekends

All signals are tagged with market context so the dashboard
can communicate urgency correctly:
  - MARKET_OPEN      → signal is actionable right now
  - PRE_MARKET       → signal is for today's open (less liquidity)
  - AFTER_HOURS      → signal is for tomorrow's open
  - MARKET_CLOSED    → weekend / holiday, signal is for next open
"""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional

ET = ZoneInfo("America/New_York")

# US market holidays 2024-2025 (NYSE observed dates)
_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17",
    "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
}

MARKET_OPEN_TIME     = time(9, 30)
MARKET_CLOSE_TIME    = time(16, 0)
PRE_MARKET_START     = time(4, 0)
AFTER_HOURS_END      = time(20, 0)


def now_et() -> datetime:
    """Current time in US Eastern timezone."""
    return datetime.now(ET)


def is_holiday(dt: datetime) -> bool:
    return dt.strftime("%Y-%m-%d") in _HOLIDAYS


def is_trading_day(dt: Optional[datetime] = None) -> bool:
    """True if dt is a weekday that is not a US market holiday."""
    dt = dt or now_et()
    return dt.weekday() < 5 and not is_holiday(dt)


def market_status(dt: Optional[datetime] = None) -> dict:
    """
    Returns full market status dict:
      status:        OPEN | PRE_MARKET | AFTER_HOURS | CLOSED
      label:         Human-readable string
      is_open:       bool - regular session
      can_trade:     bool - any session (pre/post/regular)
      next_open:     datetime of next regular session open (ET)
      next_close:    datetime of next regular session close (ET)
      seconds_to_open:  int
      seconds_to_close: int
      session_pct:   float 0-100, how far through the regular session
    """
    dt = dt or now_et()
    t  = dt.time()

    today_is_trading = is_trading_day(dt)

    if today_is_trading:
        if MARKET_OPEN_TIME <= t < MARKET_CLOSE_TIME:
            status    = "OPEN"
            label     = "Market Open"
            is_open   = True
            can_trade = True
        elif PRE_MARKET_START <= t < MARKET_OPEN_TIME:
            status    = "PRE_MARKET"
            label     = "Pre-Market"
            is_open   = False
            can_trade = True
        elif MARKET_CLOSE_TIME <= t < AFTER_HOURS_END:
            status    = "AFTER_HOURS"
            label     = "After Hours"
            is_open   = False
            can_trade = True
        else:
            status    = "CLOSED"
            label     = "Closed (overnight)"
            is_open   = False
            can_trade = False
    else:
        status    = "CLOSED"
        label     = "Closed (weekend)" if dt.weekday() >= 5 else "Closed (holiday)"
        is_open   = False
        can_trade = False

    # Next regular open
    next_open  = _next_open(dt)
    next_close = _next_close(dt)

    sec_to_open  = int((next_open  - dt).total_seconds()) if dt < next_open  else 0
    sec_to_close = int((next_close - dt).total_seconds()) if dt < next_close else 0

    # Session progress (only meaningful when market is open)
    session_pct = 0.0
    if is_open:
        open_secs  = _time_to_seconds(MARKET_OPEN_TIME)
        close_secs = _time_to_seconds(MARKET_CLOSE_TIME)
        now_secs   = _time_to_seconds(t)
        total = close_secs - open_secs
        elapsed = now_secs - open_secs
        session_pct = round(min(100, max(0, elapsed / total * 100)), 1)

    return {
        "status":           status,
        "label":            label,
        "is_open":          is_open,
        "can_trade":        can_trade,
        "timestamp_et":     dt.strftime("%Y-%m-%d %H:%M:%S ET"),
        "next_open":        next_open.strftime("%Y-%m-%d %H:%M ET"),
        "next_close":       next_close.strftime("%Y-%m-%d %H:%M ET") if next_close else None,
        "seconds_to_open":  sec_to_open,
        "seconds_to_close": sec_to_close,
        "session_pct":      session_pct,
    }


def signal_urgency(status: str) -> dict:
    """
    Map market status → signal urgency / timing advice.
    Used to add context to trade recommendations.
    """
    mapping = {
        "OPEN": {
            "urgency":  "immediate",
            "color":    "green",
            "advice":   "Market is open — signal is actionable right now.",
            "caution":  None,
        },
        "PRE_MARKET": {
            "urgency":  "today",
            "color":    "amber",
            "advice":   "Pre-market: lower liquidity, wider spreads. Wait for 9:30 AM ET open for better fills.",
            "caution":  "Pre-market prices can gap significantly at the open.",
        },
        "AFTER_HOURS": {
            "urgency":  "tomorrow",
            "color":    "amber",
            "advice":   "After-hours: signal targets tomorrow's open. Set limit orders now or wait for 9:30 AM ET.",
            "caution":  "Overnight news can change the picture — check before placing orders.",
        },
        "CLOSED": {
            "urgency":  "next_session",
            "color":    "gray",
            "advice":   "Market is closed. Signal is for next trading session open.",
            "caution":  "Review signals again before market opens — conditions may change.",
        },
    }
    return mapping.get(status, mapping["CLOSED"])


# ── Private helpers ──────────────────────────────────────────────────────────

def _time_to_seconds(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def _next_open(dt: datetime) -> datetime:
    """Next regular session open at or after dt."""
    candidate = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    if dt.time() >= MARKET_OPEN_TIME or not is_trading_day(dt):
        candidate += timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate.replace(hour=9, minute=30, second=0, microsecond=0)


def _next_close(dt: datetime) -> Optional[datetime]:
    """Next regular session close at or after dt."""
    if is_trading_day(dt) and dt.time() < MARKET_CLOSE_TIME:
        return dt.replace(hour=16, minute=0, second=0, microsecond=0)
    candidate = dt + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate.replace(hour=16, minute=0, second=0, microsecond=0)
