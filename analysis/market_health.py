"""
analysis/market_health.py — Overall market health & trajectory engine

Fetches live data for:
  - S&P 500 (SPY), Nasdaq (QQQ), Dow (DIA), Russell 2000 (IWM)
  - VIX (volatility / fear index)
  - 10-year Treasury yield (^TNX)
  - DXY dollar index (DX-Y.NYB)
  - Gold (GLD), Oil (USO)

Produces:
  - health_score: 0–100
  - trajectory:   Bullish / Neutral / Bearish (+ strength)
  - regime:       Risk-On / Risk-Off / Transitioning
  - key_drivers:  Plain-English reasons
  - sector_snapshot: which sectors are leading / lagging
  - warnings:     Active risk flags (VIX spike, yield inversion, etc.)
"""

import logging
import time
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("market_health")

try:
    import yfinance as yf
    import numpy as np
    _ok = True
except ImportError:
    _ok = False
    logger.warning("yfinance not installed — market health unavailable.")

_DELAY = 1.0   # seconds between Yahoo Finance calls


def _fetch(ticker: str) -> Optional[dict]:
    """Fetch quote + recent history for a single ticker."""
    if not _ok:
        return None
    try:
        time.sleep(_DELAY)
        t    = yf.Ticker(ticker)
        fi   = t.fast_info
        hist = t.history(period="3mo", interval="1d")

        if hist.empty:
            return None

        price      = float(fi.last_price)      if fi.last_price      else None
        prev_close = float(fi.previous_close)  if fi.previous_close  else None
        change_pct = round((price - prev_close) / prev_close * 100, 2) if price and prev_close else 0

        close = hist["Close"]

        # 1-day, 5-day, 20-day, 50-day returns
        def pct_return(days):
            if len(close) > days:
                return round((float(close.iloc[-1]) - float(close.iloc[-days])) / float(close.iloc[-days]) * 100, 2)
            return None

        # 20-day and 50-day SMA
        sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None
        sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi_val = round(float((100 - 100 / (1 + rs)).iloc[-1]), 1) if not loss.empty else 50

        # 20-day volatility (annualised)
        daily_ret  = close.pct_change().dropna()
        volatility = round(float(daily_ret.tail(20).std()) * (252 ** 0.5) * 100, 1) if len(daily_ret) >= 20 else None

        return {
            "price":       round(price, 2) if price else None,
            "change_pct":  change_pct,
            "ret_1d":      change_pct,
            "ret_5d":      pct_return(5),
            "ret_20d":     pct_return(20),
            "ret_50d":     pct_return(50),
            "sma20":       sma20,
            "sma50":       sma50,
            "above_sma20": price > sma20 if price and sma20 else None,
            "above_sma50": price > sma50 if price and sma50 else None,
            "rsi":         rsi_val,
            "volatility":  volatility,
        }
    except Exception as e:
        logger.debug(f"Fetch failed for {ticker}: {e}")
        return None


def fetch_market_health() -> dict:
    """
    Fetch all market indicators and compute overall health score,
    trajectory, regime, and plain-English reasoning.
    """
    logger.info("Fetching market health indicators…")

    # ── Core indices ──────────────────────────────────────────────────────────
    spy  = _fetch("SPY")    # S&P 500
    qqq  = _fetch("QQQ")    # Nasdaq 100
    dia  = _fetch("DIA")    # Dow Jones
    iwm  = _fetch("IWM")    # Russell 2000 (small-cap, risk appetite)

    # ── Fear & volatility ─────────────────────────────────────────────────────
    vix  = _fetch("^VIX")   # CBOE Volatility Index

    # ── Macro / rates ─────────────────────────────────────────────────────────
    tnx  = _fetch("^TNX")   # 10-year Treasury yield
    dxy  = _fetch("DX-Y.NYB") # US Dollar index

    # ── Risk assets / commodities ─────────────────────────────────────────────
    gld  = _fetch("GLD")    # Gold (safe haven)
    uso  = _fetch("USO")    # Oil

    # ── Sector ETFs ───────────────────────────────────────────────────────────
    sectors = {
        "Technology":    _fetch("XLK"),
        "Financials":    _fetch("XLF"),
        "Healthcare":    _fetch("XLV"),
        "Energy":        _fetch("XLE"),
        "Consumer Disc": _fetch("XLY"),
        "Utilities":     _fetch("XLU"),
        "Industrials":   _fetch("XLI"),
        "Materials":     _fetch("XLB"),
    }

    # ── Compute health score ──────────────────────────────────────────────────
    score, total_w = 0.0, 0.0

    def add(val, weight):
        nonlocal score, total_w
        score += val * weight
        total_w += weight

    # SPY momentum (heavy weight — it IS the market)
    if spy:
        if spy["above_sma50"]:   add(0.7,  0.25)
        else:                    add(-0.7, 0.25)
        if spy["ret_20d"] is not None:
            add(max(-1, min(1, spy["ret_20d"] / 8)), 0.15)
        if spy["rsi"]:
            rsi_score = (50 - spy["rsi"]) / 50 * -0.5   # overbought = slight negative
            add(rsi_score, 0.05)

    # Nasdaq breadth
    if qqq:
        if qqq["above_sma50"]:  add(0.6, 0.12)
        else:                   add(-0.6, 0.12)
        if qqq["ret_20d"] is not None:
            add(max(-1, min(1, qqq["ret_20d"] / 10)), 0.08)

    # Small-cap (IWM) — risk appetite indicator
    if iwm and spy:
        # IWM outperforming SPY = risk-on
        iwm_vs_spy = (iwm.get("ret_5d") or 0) - (spy.get("ret_5d") or 0)
        add(max(-1, min(1, iwm_vs_spy / 3)), 0.10)

    # VIX — fear gauge (inverted: high VIX = bad)
    if vix and vix["price"]:
        v = vix["price"]
        if v < 15:    add(0.8,  0.15)    # very calm
        elif v < 20:  add(0.4,  0.15)    # normal
        elif v < 25:  add(0.0,  0.15)    # elevated
        elif v < 30:  add(-0.5, 0.15)    # fearful
        else:         add(-1.0, 0.15)    # extreme fear

    # 10-year yield (moderate = good, very high = headwind for stocks)
    if tnx and tnx["price"]:
        y = tnx["price"]
        if y < 3.5:   add(0.5,  0.08)
        elif y < 4.5: add(0.1,  0.08)
        elif y < 5.5: add(-0.4, 0.08)
        else:         add(-0.8, 0.08)

    # Dollar (DXY) — strong dollar = headwind for stocks & commodities
    if dxy and dxy["ret_20d"] is not None:
        dxy_score = -max(-1, min(1, dxy["ret_20d"] / 5))
        add(dxy_score, 0.05)

    final_score = round((score / total_w * 50 + 50) if total_w > 0 else 50, 1)
    final_score = max(0, min(100, final_score))

    # ── Trajectory ────────────────────────────────────────────────────────────
    if final_score >= 70:   trajectory, traj_strength = "Bullish",  "Strong"
    elif final_score >= 58: trajectory, traj_strength = "Bullish",  "Moderate"
    elif final_score >= 48: trajectory, traj_strength = "Neutral",  "Mixed"
    elif final_score >= 38: trajectory, traj_strength = "Bearish",  "Moderate"
    else:                   trajectory, traj_strength = "Bearish",  "Strong"

    # ── Market regime ─────────────────────────────────────────────────────────
    vix_val = vix["price"] if vix and vix["price"] else 20
    iwm_leading = iwm and spy and (iwm.get("ret_5d") or 0) > (spy.get("ret_5d") or 0)
    gld_rising  = gld and (gld.get("ret_5d") or 0) > 1.0

    if vix_val < 20 and iwm_leading and not gld_rising:
        regime = "Risk-On"
        regime_desc = "Investors are favouring growth and risk assets"
    elif vix_val > 25 or gld_rising:
        regime = "Risk-Off"
        regime_desc = "Investors are rotating to safe havens (bonds, gold, cash)"
    else:
        regime = "Transitioning"
        regime_desc = "Mixed signals — no clear risk-on or risk-off consensus"

    # ── Key drivers ───────────────────────────────────────────────────────────
    drivers = []

    if spy:
        trend = "above" if spy["above_sma50"] else "below"
        drivers.append(
            f"S&P 500 is {trend} its 50-day moving average "
            f"({'bullish structure' if spy['above_sma50'] else 'bearish structure'}) "
            f"with a {spy.get('ret_20d', 0):+.1f}% 20-day return."
        )

    if vix and vix["price"]:
        v = vix["price"]
        vix_label = (
            "very low — investors are complacent"   if v < 15 else
            "normal — healthy market conditions"     if v < 20 else
            "elevated — some investor anxiety"       if v < 25 else
            "high — significant fear in the market"  if v < 30 else
            "extreme — panic-level fear"
        )
        drivers.append(f"VIX (fear index) is at {v:.1f} — {vix_label}.")

    if tnx and tnx["price"]:
        y = tnx["price"]
        rate_impact = (
            "supportive for equities" if y < 4.0 else
            "neutral to slightly restrictive" if y < 5.0 else
            "a headwind for stock valuations"
        )
        drivers.append(f"10-year Treasury yield is at {y:.2f}% — {rate_impact}.")

    if iwm and spy:
        perf_diff = (iwm.get("ret_5d") or 0) - (spy.get("ret_5d") or 0)
        if abs(perf_diff) > 0.5:
            leader = "small-cap stocks (IWM)" if perf_diff > 0 else "large-cap stocks (S&P 500)"
            signal = "risk-on appetite" if perf_diff > 0 else "defensive rotation"
            drivers.append(
                f"{leader.capitalize()} are outperforming by {abs(perf_diff):.1f}% this week, "
                f"suggesting {signal}."
            )

    if dxy and dxy["ret_5d"] is not None:
        d = dxy["ret_5d"]
        if abs(d) > 0.5:
            direction = "strengthening" if d > 0 else "weakening"
            impact = "a headwind for US multinationals and commodities" if d > 0 else "a tailwind for global equities and commodities"
            drivers.append(f"The US Dollar is {direction} ({d:+.1f}% this week) — {impact}.")

    if gld:
        g = gld.get("ret_5d") or 0
        if abs(g) > 1.0:
            direction = "rising" if g > 0 else "falling"
            signal = "flight to safety" if g > 0 else "risk appetite returning"
            drivers.append(f"Gold is {direction} ({g:+.1f}% this week), suggesting {signal}.")

    if uso:
        o = uso.get("ret_5d") or 0
        if abs(o) > 2.0:
            direction = "rising" if o > 0 else "falling"
            drivers.append(
                f"Oil prices are {direction} sharply ({o:+.1f}% this week) — "
                f"{'inflationary pressure' if o > 0 else 'easing commodity pressure'}."
            )

    # ── Warnings ──────────────────────────────────────────────────────────────
    warnings = []

    if vix and vix["price"] and vix["price"] > 25:
        warnings.append(f"VIX above 25 ({vix['price']:.1f}) — elevated volatility, consider reducing position sizes.")

    if tnx and tnx["price"] and tnx["price"] > 5.0:
        warnings.append(f"10-year yield above 5% ({tnx['price']:.2f}%) — high rates compress stock valuations.")

    if spy and spy["rsi"] and spy["rsi"] > 75:
        warnings.append(f"S&P 500 RSI at {spy['rsi']} — overbought, near-term pullback risk elevated.")

    if spy and spy["rsi"] and spy["rsi"] < 30:
        warnings.append(f"S&P 500 RSI at {spy['rsi']} — oversold, potential bounce but confirm before buying.")

    if spy and not spy["above_sma50"] and qqq and not qqq["above_sma50"]:
        warnings.append("Both S&P 500 and Nasdaq are below their 50-day averages — broad market in downtrend.")

    if spy and spy.get("ret_20d") is not None and spy["ret_20d"] < -10:
        warnings.append(f"S&P 500 is down {spy['ret_20d']:.1f}% over 20 days — correction territory.")

    # ── Sector leaders & laggards ─────────────────────────────────────────────
    sector_data = []
    for name, data in sectors.items():
        if data and data.get("ret_5d") is not None:
            sector_data.append({
                "name":       name,
                "change_pct": data["change_pct"],
                "ret_5d":     data["ret_5d"],
                "ret_20d":    data.get("ret_20d"),
                "rsi":        data.get("rsi"),
                "price":      data.get("price"),
            })

    sector_data.sort(key=lambda x: x["ret_5d"], reverse=True)
    leaders  = sector_data[:3]
    laggards = sector_data[-3:][::-1] if len(sector_data) >= 3 else []

    return {
        "health_score":   final_score,
        "trajectory":     trajectory,
        "traj_strength":  traj_strength,
        "regime":         regime,
        "regime_desc":    regime_desc,
        "key_drivers":    drivers,
        "warnings":       warnings,
        "sector_leaders": leaders,
        "sector_laggards":laggards,
        "all_sectors":    sector_data,
        "indices": {
            "spy": spy,
            "qqq": qqq,
            "dia": dia,
            "iwm": iwm,
        },
        "macro": {
            "vix":  vix,
            "tnx":  tnx,
            "dxy":  dxy,
            "gld":  gld,
            "uso":  uso,
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
