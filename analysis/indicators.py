"""
analysis/indicators.py — Real price-based technical indicators

Fetches 6 months of OHLCV history from yfinance and computes:
  - RSI (14-day)
  - MACD (12/26/9)
  - Bollinger Bands (20-day, 2σ)
  - SMA 20 / 50 / 200 crossover signals
  - Volume spike detection
  - ATR (14-day) for position sizing
  - Support / resistance from recent swing points
  - 52-week range position
  - Trend direction (up/down/sideways)
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("indicators")

try:
    import yfinance as yf
    import pandas as pd
    _yf_ok = True
except ImportError:
    _yf_ok = False
    logger.warning("yfinance/pandas not installed — technical indicators unavailable.")

# Throttled yfinance client (avoids 429 rate-limit errors).
from analysis.yf_client import get_ticker, call_with_retry


# ─── Data fetch ──────────────────────────────────────────────────────────────

def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional["pd.DataFrame"]:
    """Fetch OHLCV history. Returns DataFrame or None on failure."""
    if not _yf_ok:
        return None
    try:
        t = get_ticker(ticker)
        if t is None:
            return None  # yfinance disabled or circuit breaker open
        hist = call_with_retry(lambda: t.history(period=period, interval=interval))
        if hist is None or hist.empty or len(hist) < 20:
            return None
        return hist
    except Exception as e:
        logger.debug(f"History fetch failed for {ticker}: {e}")
        return None


def fetch_quote(ticker: str) -> Optional[dict]:
    """Fetch current price, volume, and basic info."""
    if not _yf_ok:
        return None
    try:
        t = get_ticker(ticker)
        if t is None:
            return None  # yfinance disabled or circuit breaker open
        # fast_info is lazy — touching one field forces the network call;
        # wrap it so we retry on 429.
        fi = call_with_retry(lambda: t.fast_info)
        if fi is None:
            return None
        # Force eager fetch of the fields we need (also retried).
        snapshot = call_with_retry(lambda: {
            "last_price":                 fi.last_price,
            "previous_close":             fi.previous_close,
            "three_month_average_volume": fi.three_month_average_volume,
            "year_high":                  fi.year_high,
            "year_low":                   fi.year_low,
            "market_cap":                 fi.market_cap,
            "currency":                   fi.currency,
        })
        last  = snapshot["last_price"]
        prev  = snapshot["previous_close"]
        return {
            "price":          round(float(last), 2)  if last else None,
            "prev_close":     round(float(prev), 2)  if prev else None,
            "change_pct":     round((last - prev) / prev * 100, 2)
                              if last and prev else None,
            "volume":         int(snapshot["three_month_average_volume"]) if snapshot["three_month_average_volume"] else None,
            "week52_high":    round(float(snapshot["year_high"]), 2)  if snapshot["year_high"]  else None,
            "week52_low":     round(float(snapshot["year_low"]), 2)   if snapshot["year_low"]   else None,
            "market_cap":     int(snapshot["market_cap"])             if snapshot["market_cap"] else None,
            "currency":       snapshot["currency"] or "USD",
        }
    except Exception as e:
        logger.debug(f"Quote fetch failed for {ticker}: {e}")
        return None


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(close: "pd.Series", period: int = 14) -> float:
    """
    Relative Strength Index (RSI).
    <30 = oversold (potential buy), >70 = overbought (potential sell).
    """
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, float("nan"))
    rsi_s  = 100 - (100 / (1 + rs))
    return round(float(rsi_s.iloc[-1]), 1)


def rsi_signal(rsi_val: float) -> str:
    if rsi_val >= 80:  return "strongly overbought"
    if rsi_val >= 70:  return "overbought"
    if rsi_val >= 60:  return "approaching overbought"
    if rsi_val <= 20:  return "strongly oversold"
    if rsi_val <= 30:  return "oversold"
    if rsi_val <= 40:  return "approaching oversold"
    return "neutral"


# ─── MACD ─────────────────────────────────────────────────────────────────────

def macd(close: "pd.Series", fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    MACD (Moving Average Convergence/Divergence).
    macd_line > signal_line = bullish momentum.
    Histogram > 0 and rising = strong bullish momentum.
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line

    current_hist = float(histogram.iloc[-1])
    prev_hist    = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

    crossover = None
    if float(macd_line.iloc[-2]) < float(sig_line.iloc[-2]) and \
       float(macd_line.iloc[-1]) >= float(sig_line.iloc[-1]):
        crossover = "bullish_crossover"    # MACD just crossed above signal
    elif float(macd_line.iloc[-2]) > float(sig_line.iloc[-2]) and \
         float(macd_line.iloc[-1]) <= float(sig_line.iloc[-1]):
        crossover = "bearish_crossover"    # MACD just crossed below signal

    return {
        "macd_line":      round(float(macd_line.iloc[-1]), 4),
        "signal_line":    round(float(sig_line.iloc[-1]), 4),
        "histogram":      round(current_hist, 4),
        "histogram_prev": round(prev_hist, 4),
        "bullish":        float(macd_line.iloc[-1]) > float(sig_line.iloc[-1]),
        "momentum_rising":current_hist > prev_hist,
        "crossover":      crossover,
    }


# ─── Bollinger Bands ─────────────────────────────────────────────────────────

def bollinger(close: "pd.Series", period: int = 20, std: float = 2.0) -> dict:
    """
    Bollinger Bands.
    Price near lower band = potentially oversold.
    Price near upper band = potentially overbought.
    Squeeze (narrow bands) = breakout likely soon.
    """
    sma   = close.rolling(period).mean()
    std_d = close.rolling(period).std()
    upper = sma + std * std_d
    lower = sma - std * std_d

    c     = float(close.iloc[-1])
    u     = float(upper.iloc[-1])
    l     = float(lower.iloc[-1])
    m     = float(sma.iloc[-1])
    bw    = (u - l) / m if m > 0 else 0    # bandwidth — low = squeeze

    # Where is price within the bands? 0 = at lower, 1 = at upper
    pct_b = (c - l) / (u - l) if (u - l) > 0 else 0.5

    # Detect squeeze: current bandwidth < 80th percentile of last 100 days
    bws   = ((upper - lower) / sma).dropna().tail(100)
    squeeze = bool(bw < float(bws.quantile(0.20))) if len(bws) >= 10 else False

    return {
        "upper": round(u, 2), "middle": round(m, 2), "lower": round(l, 2),
        "pct_b": round(pct_b, 3),
        "bandwidth": round(bw, 4),
        "squeeze": squeeze,
        "price_vs_bands": (
            "above upper band" if c > u else
            "below lower band" if c < l else
            "near upper" if pct_b > 0.8 else
            "near lower" if pct_b < 0.2 else
            "mid-range"
        ),
    }


# ─── Moving Average Crossovers ────────────────────────────────────────────────

def moving_averages(close: "pd.Series") -> dict:
    """
    SMA 20, 50, 200 day moving averages.
    Golden cross (SMA50 > SMA200) = long-term bull trend.
    Death cross (SMA50 < SMA200) = long-term bear trend.
    Price > SMA200 = above long-term trend (bullish context).
    """
    price = float(close.iloc[-1])

    def sma(n):
        if len(close) >= n:
            return round(float(close.rolling(n).mean().iloc[-1]), 2)
        return None

    s20, s50, s200 = sma(20), sma(50), sma(200)

    golden_cross = (s50 is not None and s200 is not None and s50 > s200)
    death_cross  = (s50 is not None and s200 is not None and s50 < s200)

    above_sma20  = s20  is not None and price > s20
    above_sma50  = s50  is not None and price > s50
    above_sma200 = s200 is not None and price > s200

    # Detect recent golden/death cross (within last 10 days)
    recent_cross = None
    if s50 and s200:
        if len(close) >= 210:
            s50_prev  = float(close.rolling(50).mean().iloc[-10])
            s200_prev = float(close.rolling(200).mean().iloc[-10])
            if s50 > s200 and s50_prev <= s200_prev:
                recent_cross = "golden_cross"   # bullish long-term signal
            elif s50 < s200 and s50_prev >= s200_prev:
                recent_cross = "death_cross"    # bearish long-term signal

    return {
        "sma20":        s20,
        "sma50":        s50,
        "sma200":       s200,
        "golden_cross": golden_cross,
        "death_cross":  death_cross,
        "above_sma20":  above_sma20,
        "above_sma50":  above_sma50,
        "above_sma200": above_sma200,
        "recent_cross": recent_cross,
    }


# ─── Volume Analysis ─────────────────────────────────────────────────────────

def volume_analysis(hist: "pd.DataFrame") -> dict:
    """
    Compare today's volume to the 20-day average.
    Volume spike on up day = bullish confirmation.
    Volume spike on down day = bearish confirmation.
    """
    vol   = hist["Volume"]
    close = hist["Close"]
    avg20 = float(vol.rolling(20).mean().iloc[-1])
    today = float(vol.iloc[-1])
    ratio = today / avg20 if avg20 > 0 else 1.0

    price_up = float(close.iloc[-1]) >= float(close.iloc[-2])

    return {
        "volume_today":    int(today),
        "volume_avg20":    int(avg20),
        "volume_ratio":    round(ratio, 2),
        "spike":           ratio >= 1.5,
        "spike_direction": "up" if price_up else "down",
        "label": (
            "High-volume up day (bullish confirmation)"  if ratio >= 1.5 and price_up  else
            "High-volume down day (bearish confirmation)" if ratio >= 1.5 and not price_up else
            "Below-average volume (weak conviction)"     if ratio < 0.7 else
            "Normal volume"
        ),
    }


# ─── ATR ─────────────────────────────────────────────────────────────────────

def atr(hist: "pd.DataFrame", period: int = 14) -> float:
    """Average True Range — measures volatility for stop-loss sizing."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev  = close.shift(1)
    tr    = (high - low).abs().combine((high - prev).abs(), max).combine((low - prev).abs(), max)
    return round(float(tr.rolling(period).mean().iloc[-1]), 2)


# ─── Trend Detection ──────────────────────────────────────────────────────────

def trend(close: "pd.Series", lookback: int = 20) -> dict:
    """
    Linear regression slope over `lookback` days.
    Returns trend direction and angle.
    """
    y   = close.tail(lookback).values
    x   = np.arange(len(y))
    if len(y) < 4:
        return {"direction": "unknown", "slope_pct": 0.0, "strength": "weak"}
    coeffs = np.polyfit(x, y, 1)
    slope  = float(coeffs[0])
    pct_per_day = slope / float(y[0]) * 100 if y[0] > 0 else 0

    if pct_per_day > 0.5:
        direction, strength = "uptrend", "strong" if pct_per_day > 1.5 else "moderate"
    elif pct_per_day < -0.5:
        direction, strength = "downtrend", "strong" if pct_per_day < -1.5 else "moderate"
    else:
        direction, strength = "sideways", "weak"

    return {
        "direction":  direction,
        "slope_pct":  round(pct_per_day, 3),
        "strength":   strength,
    }


# ─── Support / Resistance ─────────────────────────────────────────────────────

def support_resistance(hist: "pd.DataFrame", window: int = 10) -> dict:
    """
    Find the nearest support and resistance levels from recent swing highs/lows.
    Uses rolling local min/max over the last 60 days.
    """
    close  = hist["Close"].tail(60)
    price  = float(close.iloc[-1])
    highs  = hist["High"].tail(60)
    lows   = hist["Low"].tail(60)

    # Local maxima = resistance candidates
    local_max = []
    for i in range(window, len(highs) - window):
        v = float(highs.iloc[i])
        if v == float(highs.iloc[i - window:i + window].max()):
            local_max.append(v)

    # Local minima = support candidates
    local_min = []
    for i in range(window, len(lows) - window):
        v = float(lows.iloc[i])
        if v == float(lows.iloc[i - window:i + window].min()):
            local_min.append(v)

    resistance = min((v for v in local_max if v > price), default=None)
    support    = max((v for v in local_min if v < price), default=None)

    return {
        "support":    round(support, 2)    if support    else None,
        "resistance": round(resistance, 2) if resistance else None,
    }


# ─── Master function ──────────────────────────────────────────────────────────

def compute_all(ticker: str) -> Optional[dict]:
    """
    Compute all technical indicators for a ticker.
    Returns a single dict with all indicators, or None if data unavailable.
    """
    hist  = fetch_history(ticker)
    quote = fetch_quote(ticker)

    if hist is None or quote is None:
        return None

    close = hist["Close"]

    try:
        rsi_val   = rsi(close)
        macd_val  = macd(close)
        boll      = bollinger(close)
        mas       = moving_averages(close)
        vol       = volume_analysis(hist)
        atr_val   = atr(hist)
        tr        = trend(close)
        sr        = support_resistance(hist)

        # Overall technical score: -1 (very bearish) to +1 (very bullish)
        score = _technical_score(rsi_val, macd_val, boll, mas, vol, tr)

        return {
            "ticker":      ticker,
            "price":       quote.get("price"),
            "change_pct":  quote.get("change_pct"),
            "week52_high": quote.get("week52_high"),
            "week52_low":  quote.get("week52_low"),
            "market_cap":  quote.get("market_cap"),
            "rsi":         rsi_val,
            "rsi_signal":  rsi_signal(rsi_val),
            "macd":        macd_val,
            "bollinger":   boll,
            "moving_avgs": mas,
            "volume":      vol,
            "atr":         atr_val,
            "trend":       tr,
            "support_resistance": sr,
            "technical_score":    score,
        }

    except Exception as e:
        logger.error(f"Indicator computation failed for {ticker}: {e}")
        return None


def _technical_score(rsi_val, macd_val, boll, mas, vol, tr) -> float:
    """
    Aggregate technical indicators into a single score: -1.0 to +1.0.
    Used for blending with sentiment to produce final predictions.
    """
    pts = 0.0
    total_weight = 0.0

    # RSI (weight 0.20)
    w = 0.20
    if rsi_val <= 30:   pts += w        # oversold = bullish
    elif rsi_val >= 70: pts -= w        # overbought = bearish
    else:               pts += w * (50 - rsi_val) / 50 * 0.5
    total_weight += w

    # MACD (weight 0.25)
    w = 0.25
    if macd_val["bullish"] and macd_val["momentum_rising"]:  pts += w
    elif not macd_val["bullish"] and not macd_val["momentum_rising"]: pts -= w
    elif macd_val["bullish"]: pts += w * 0.5
    else: pts -= w * 0.5
    if macd_val["crossover"] == "bullish_crossover": pts += 0.05
    elif macd_val["crossover"] == "bearish_crossover": pts -= 0.05
    total_weight += w

    # Bollinger (weight 0.15)
    w = 0.15
    pb = boll["pct_b"]
    pts += w * (0.5 - pb) * 2   # near lower band = bullish, near upper = bearish
    total_weight += w

    # Moving averages (weight 0.25)
    w = 0.25
    ma_score = 0
    if mas["above_sma200"]: ma_score += 1
    else:                    ma_score -= 1
    if mas["above_sma50"]:   ma_score += 0.6
    else:                    ma_score -= 0.6
    if mas["above_sma20"]:   ma_score += 0.4
    else:                    ma_score -= 0.4
    if mas["golden_cross"]:  ma_score += 0.5
    elif mas["death_cross"]: ma_score -= 0.5
    pts += w * max(-1, min(1, ma_score / 2.5))
    total_weight += w

    # Trend (weight 0.15)
    w = 0.15
    sp = tr["slope_pct"]
    pts += w * max(-1, min(1, sp / 1.5))
    total_weight += w

    return round(pts / total_weight if total_weight > 0 else 0, 4)
