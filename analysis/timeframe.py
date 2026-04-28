"""
analysis/timeframe.py — Short-term and long-term prediction engine

SHORT-TERM (1–10 days):
  - Driven by: sentiment momentum (70%) + technical indicators (30%)
  - Uses RSI, MACD, volume spikes, Bollinger position, short squeeze
  - Entry/exit levels set at 1× ATR stop, 2.5× ATR target
  - Best for: momentum trades, news-driven plays, options speculation

LONG-TERM (1–6 months):
  - Driven by: fundamentals (75%) + sentiment trend (25%)
  - Uses P/E, revenue growth, analyst targets, trend direction, SMA 200
  - Entry/exit levels set at 2× ATR stop, 4× ATR target
  - Best for: position trading, growth investing, covered calls

Each prediction returns:
  - signal:         STRONG BUY / BUY / HOLD / SELL / STRONG SELL
  - action:         Specific trade instruction (Long, Short, Hold, etc.)
  - confidence:     0–100%
  - entry:          Suggested entry price
  - stop_loss:      Where to cut losses
  - take_profit_1:  Primary target
  - take_profit_2:  Extended target
  - position_size:  How many shares / $ amount based on portfolio settings
  - rationale:      Plain-English explanation
  - key_factors:    Bullet-point reasons
  - options_plays:  Relevant options strategies with explanations
  - time_horizon:   Expected holding period
  - risk_level:     Low / Medium / High / Very High
"""

import math
import logging
from typing import Optional
import config
from analysis.fundamentals import analyst_label

logger = logging.getLogger("timeframe")


# ─── Signal helpers ───────────────────────────────────────────────────────────

def _score_to_signal(score: float) -> str:
    if score >= 0.55:  return "STRONG BUY"
    if score >= 0.25:  return "BUY"
    if score >= -0.25: return "HOLD"
    if score >= -0.55: return "SELL"
    return "STRONG SELL"


def _score_to_confidence(score: float) -> int:
    return min(99, int(abs(score) * 95 + 30))


def _risk_level(atr_pct: float, short_interest: Optional[float]) -> str:
    """ATR as % of price gives a sense of daily volatility."""
    if atr_pct > 5 or (short_interest and short_interest > 20):
        return "Very High"
    if atr_pct > 3:
        return "High"
    if atr_pct > 1.5:
        return "Medium"
    return "Low"


# ─── Position sizing ──────────────────────────────────────────────────────────

def _position_size(price: float, stop_loss: float, score: float) -> dict:
    if price <= 0 or stop_loss <= 0:
        return {"shares": 0, "dollar_amount": 0, "pct_of_portfolio": 0,
                "max_risk": 0, "risk_per_share": 0}

    risk_per_share = abs(price - stop_loss)
    if risk_per_share <= 0:
        return {"shares": 0, "dollar_amount": 0, "pct_of_portfolio": 0,
                "max_risk": 0, "risk_per_share": 0}

    # Dollar amount we're willing to risk per trade
    max_risk = config.PORTFOLIO_SIZE * (config.RISK_PER_TRADE_PCT / 100)

    # Shares = max risk / risk per share
    raw_shares = max_risk / risk_per_share

    # Scale by conviction (high conviction = full size, low = half size)
    conviction = min(abs(score) * 1.5, 1.0)
    shares = max(1, int(raw_shares * conviction))

    # Hard cap: no more than MAX_POSITION_PCT of portfolio
    max_dollars = config.PORTFOLIO_SIZE * (config.MAX_POSITION_PCT / 100)
    if shares * price > max_dollars:
        shares = max(1, int(max_dollars / price))

    dollar_amount = round(shares * price, 2)
    pct = round(dollar_amount / config.PORTFOLIO_SIZE * 100, 2)

    return {
        "shares":          shares,
        "dollar_amount":   dollar_amount,
        "pct_of_portfolio":pct,
        "max_risk":        round(max_risk, 2),
        "risk_per_share":  round(risk_per_share, 2),
    }


# ─── Options strategy selector ────────────────────────────────────────────────

_OPTIONS_INFO = {
    "Buy Calls":
        "Buy call options at-the-money or slightly out-of-the-money. "
        "Profit if the stock rises before expiry. Risk = premium paid only. "
        "Best when you expect a quick move up. Choose expiry 2–4 weeks out.",
    "Buy Puts":
        "Buy put options to profit from a price decline. Risk = premium paid. "
        "Acts as a hedge on existing long positions OR as a directional short bet. "
        "Choose expiry 2–4 weeks out for short-term, 3+ months for longer plays.",
    "Buy Shares":
        "Direct stock ownership. No expiry, no leverage. "
        "The simplest way to go long. Use the position size shown — never bet more than your risk tolerance.",
    "Short Shares":
        "Borrow shares from your broker and sell them. Buy back cheaper to pocket the difference. "
        "IMPORTANT: Loss is theoretically unlimited if the stock rallies. Always use a stop-loss.",
    "Sell Covered Call":
        "If you own 100 shares, sell a call option above the current price. "
        "Collect premium income immediately. Caps your upside at the strike price. "
        "Great for generating income while waiting for a larger move.",
    "Cash-Secured Put":
        "Sell a put option backed by cash equal to 100 × strike price. "
        "Collect premium now. If price drops below strike you buy shares at that price (which you wanted anyway). "
        "Great way to enter a position at a discount.",
    "Buy Straddle":
        "Buy a call AND a put at the same strike and expiry. "
        "Profit if price makes a BIG move either direction. "
        "Use before earnings reports or major events when you expect volatility but not sure which way.",
    "Buy Call Spread":
        "Buy a call at strike A, sell a call at higher strike B. "
        "Defined max profit and max loss. Cheaper than a naked call. "
        "Best for moderate bullish moves to a specific target.",
    "Buy Put Spread":
        "Buy a put at strike A, sell a put at lower strike B. "
        "Defined max profit and max loss. Cheaper than a naked put. "
        "Best for moderate bearish moves to a specific target.",
    "LEAPS Calls":
        "Long-dated call options (expiry 1–2 years out). "
        "Gives you upside exposure on a stock you're bullish long-term without tying up full capital. "
        "Acts like a stock substitute. Best for high-conviction long-term plays.",
}


def options_for_signal(signal: str, timeframe: str, squeeze: bool = False) -> list[dict]:
    """Return relevant options plays with their full explanations."""
    plays = []

    if timeframe == "short":
        if signal in ("STRONG BUY", "BUY"):
            names = ["Buy Calls", "Buy Shares"]
            if squeeze:
                names.insert(0, "Buy Straddle")  # pre-squeeze can go either way fast
        elif signal in ("STRONG SELL", "SELL"):
            names = ["Buy Puts", "Short Shares", "Buy Put Spread"]
        else:
            names = ["Buy Straddle", "Sell Covered Call"]
    else:  # long
        if signal in ("STRONG BUY", "BUY"):
            names = ["Buy Shares", "LEAPS Calls", "Cash-Secured Put", "Sell Covered Call"]
        elif signal in ("STRONG SELL", "SELL"):
            names = ["Buy Puts", "Buy Put Spread", "Short Shares"]
        else:
            names = ["Sell Covered Call", "Cash-Secured Put", "Buy Call Spread"]

    for name in names:
        plays.append({"name": name, "description": _OPTIONS_INFO.get(name, "")})
    return plays


# ─── Short-term prediction ────────────────────────────────────────────────────

def short_term_prediction(
    sentiment_score: float,
    technicals: Optional[dict],
    ticker: str,
) -> dict:
    """
    Build 1–10 day trading prediction.

    Inputs:
        sentiment_score: composite sentiment (-1 to +1) from social/news scraper
        technicals:      dict from indicators.compute_all()
        ticker:          stock symbol
    """
    # ── Blend scores ──────────────────────────────────────────────────────
    tech_score = technicals["technical_score"] if technicals else 0.0

    composite = (
        config.SHORT_WEIGHT_SENTIMENT   * sentiment_score +
        config.SHORT_WEIGHT_FUNDAMENTAL * tech_score
    )

    signal     = _score_to_signal(composite)
    confidence = _score_to_confidence(composite)

    # ── Price data ────────────────────────────────────────────────────────
    price  = technicals.get("price") if technicals else None
    atr_v  = technicals.get("atr")   if technicals else None

    # ── Entry / Exit levels ───────────────────────────────────────────────
    if price and atr_v:
        sl_dist = atr_v * config.STOP_LOSS_ATR_MULT
        tp_dist = sl_dist * config.TAKE_PROFIT_RR

        if signal in ("STRONG BUY", "BUY"):
            entry        = price
            stop_loss    = round(price - sl_dist, 2)
            take_profit1 = round(price + tp_dist, 2)
            take_profit2 = round(price + tp_dist * 1.6, 2)
        elif signal in ("STRONG SELL", "SELL"):
            entry        = price
            stop_loss    = round(price + sl_dist, 2)   # SL above for shorts
            take_profit1 = round(price - tp_dist, 2)
            take_profit2 = round(price - tp_dist * 1.6, 2)
        else:
            entry        = price
            stop_loss    = round(price - sl_dist * 0.75, 2)
            take_profit1 = round(price + atr_v, 2)
            take_profit2 = round(price + atr_v * 1.5, 2)

        rr_ratio      = f"1:{config.TAKE_PROFIT_RR}"
        trailing_stop = round(atr_v / price * 100 * 1.5, 2)
        atr_pct       = round(atr_v / price * 100, 2)
    else:
        entry = stop_loss = take_profit1 = take_profit2 = None
        rr_ratio = trailing_stop = atr_pct = None

    # ── Position size ─────────────────────────────────────────────────────
    pos = _position_size(price or 0, stop_loss or 0, composite) if price and stop_loss else {}

    # ── Risk level ────────────────────────────────────────────────────────
    short_int = None
    risk      = _risk_level(atr_pct or 3, short_int)

    # ── Key factors ───────────────────────────────────────────────────────
    factors = _short_factors(sentiment_score, technicals)

    # ── Rationale ─────────────────────────────────────────────────────────
    rationale = _short_rationale(signal, sentiment_score, technicals, composite)

    # ── Options ───────────────────────────────────────────────────────────
    squeeze = technicals.get("squeeze", False) if technicals else False
    options = options_for_signal(signal, "short", squeeze)

    # ── Action label ──────────────────────────────────────────────────────
    action = action_label(signal, "short", technicals)

    return {
        "timeframe":      "short",
        "label":          "Short-Term (1–10 days)",
        "signal":         signal,
        "action":         action,
        "confidence":     confidence,
        "composite_score":round(composite, 4),
        "entry":          entry,
        "stop_loss":      stop_loss,
        "take_profit_1":  take_profit1,
        "take_profit_2":  take_profit2,
        "rr_ratio":       rr_ratio,
        "trailing_stop_pct": trailing_stop,
        "atr":            atr_v,
        "position":       pos,
        "risk_level":     risk,
        "key_factors":    factors,
        "rationale":      rationale,
        "options_plays":  options,
        "horizon_note":   "Momentum/news-driven play — use tight stops and take profits quickly.",
        "portfolio_size": config.PORTFOLIO_SIZE,
    }


# ─── Long-term prediction ─────────────────────────────────────────────────────

def long_term_prediction(
    sentiment_score: float,
    technicals: Optional[dict],
    fundamentals: Optional[dict],
    ticker: str,
) -> dict:
    """
    Build 1–6 month position trade prediction.

    Inputs:
        sentiment_score: composite sentiment from social/news
        technicals:      dict from indicators.compute_all()
        fundamentals:    dict from fundamentals.fetch_fundamentals()
        ticker:          stock symbol
    """
    # ── Blend scores ──────────────────────────────────────────────────────
    fund_score = fundamentals.get("fundamental_score", 0.0) if fundamentals else 0.0
    tech_score = technicals.get("technical_score", 0.0)     if technicals   else 0.0

    # For long-term: fundamentals + technicals together represent the "objective" side
    objective_score = 0.6 * fund_score + 0.4 * tech_score

    composite = (
        config.LONG_WEIGHT_FUNDAMENTAL * objective_score +
        config.LONG_WEIGHT_SENTIMENT   * sentiment_score
    )

    signal     = _score_to_signal(composite)
    confidence = _score_to_confidence(composite)

    # ── Price data ────────────────────────────────────────────────────────
    price  = technicals.get("price") if technicals else None
    atr_v  = technicals.get("atr")   if technicals else None

    # Long-term uses wider stops (2× ATR mult) and bigger targets (4×)
    lt_sl_mult = config.STOP_LOSS_ATR_MULT * 1.5
    lt_tp_mult = config.TAKE_PROFIT_RR * 1.6

    if price and atr_v:
        sl_dist = atr_v * lt_sl_mult
        tp_dist = sl_dist * lt_tp_mult

        if signal in ("STRONG BUY", "BUY"):
            entry        = price
            stop_loss    = round(price - sl_dist, 2)
            take_profit1 = round(price + tp_dist, 2)
            take_profit2 = round(price + tp_dist * 1.5, 2)
        elif signal in ("STRONG SELL", "SELL"):
            entry        = price
            stop_loss    = round(price + sl_dist, 2)
            take_profit1 = round(price - tp_dist, 2)
            take_profit2 = round(price - tp_dist * 1.5, 2)
        else:
            entry        = price
            stop_loss    = round(price - sl_dist, 2)
            take_profit1 = round(price + atr_v * 3, 2)
            take_profit2 = round(price + atr_v * 5, 2)

        rr_ratio      = f"1:{round(lt_tp_mult, 1)}"
        trailing_stop = round(atr_v / price * 100 * 2.0, 2)
        atr_pct       = round(atr_v / price * 100, 2)
    else:
        entry = stop_loss = take_profit1 = take_profit2 = None
        rr_ratio = trailing_stop = atr_pct = None

    # ── Position size (slightly larger for long-term — conviction plays) ──
    pos = _position_size(price or 0, stop_loss or 0, composite * 1.2) if price and stop_loss else {}

    # ── Risk level ────────────────────────────────────────────────────────
    short_int = (fundamentals.get("short_interest_pct") if fundamentals else None)
    risk      = _risk_level(atr_pct or 3, short_int)

    # ── Key factors ───────────────────────────────────────────────────────
    factors = _long_factors(sentiment_score, technicals, fundamentals)

    # ── Rationale ─────────────────────────────────────────────────────────
    rationale = _long_rationale(signal, sentiment_score, technicals, fundamentals, composite)

    # ── Options ───────────────────────────────────────────────────────────
    options = options_for_signal(signal, "long")

    # ── Action label ──────────────────────────────────────────────────────
    action = action_label(signal, "long", technicals)

    # ── Analyst target ────────────────────────────────────────────────────
    target_mean = fundamentals.get("target_mean")     if fundamentals else None
    upside_pct  = fundamentals.get("upside_pct")      if fundamentals else None
    rec_label   = analyst_label(
        fundamentals.get("recommendation", ""),
        fundamentals.get("recommendation_mean")
    ) if fundamentals else "N/A"

    return {
        "timeframe":       "long",
        "label":           "Long-Term (1–6 months)",
        "signal":          signal,
        "action":          action,
        "confidence":      confidence,
        "composite_score": round(composite, 4),
        "entry":           entry,
        "stop_loss":       stop_loss,
        "take_profit_1":   take_profit1,
        "take_profit_2":   take_profit2,
        "rr_ratio":        rr_ratio,
        "trailing_stop_pct": trailing_stop,
        "atr":             atr_v,
        "position":        pos,
        "risk_level":      risk,
        "key_factors":     factors,
        "rationale":       rationale,
        "options_plays":   options,
        "horizon_note":    "Position trade — use wider stops, allow time to play out, review weekly.",
        "portfolio_size":  config.PORTFOLIO_SIZE,
        # Long-term extras
        "analyst_target":  target_mean,
        "analyst_upside":  upside_pct,
        "analyst_consensus": rec_label,
        "fundamental_score": fund_score,
        "technical_score":   tech_score,
    }


# ─── Key factors builders ─────────────────────────────────────────────────────

def _short_factors(sentiment: float, tech: Optional[dict]) -> list[str]:
    factors = []

    if sentiment >= 0.4:   factors.append(f"Strong bullish social sentiment ({sentiment:+.2f})")
    elif sentiment >= 0.15: factors.append(f"Positive social sentiment ({sentiment:+.2f})")
    elif sentiment <= -0.4: factors.append(f"Strong bearish social sentiment ({sentiment:+.2f})")
    elif sentiment <= -0.15:factors.append(f"Negative social sentiment ({sentiment:+.2f})")
    else:                   factors.append(f"Mixed/neutral sentiment ({sentiment:+.2f})")

    if not tech:
        factors.append("No technical data available — sentiment-only signal")
        return factors

    rsi_v = tech.get("rsi")
    if rsi_v is not None:
        sig = tech.get("rsi_signal", "")
        factors.append(f"RSI {rsi_v} — {sig}")

    macd_v = tech.get("macd", {})
    if macd_v:
        if macd_v.get("crossover") == "bullish_crossover":
            factors.append("MACD bullish crossover just triggered (momentum entry signal)")
        elif macd_v.get("crossover") == "bearish_crossover":
            factors.append("MACD bearish crossover just triggered (momentum exit signal)")
        elif macd_v.get("bullish") and macd_v.get("momentum_rising"):
            factors.append("MACD above signal line with rising histogram (bullish momentum)")
        elif not macd_v.get("bullish"):
            factors.append("MACD below signal line (bearish momentum)")

    boll = tech.get("bollinger", {})
    if boll:
        pvb = boll.get("price_vs_bands", "")
        if pvb:
            factors.append(f"Price {pvb} (Bollinger Bands)")
        if boll.get("squeeze"):
            factors.append("Bollinger squeeze detected — breakout likely imminent")

    vol = tech.get("volume", {})
    if vol and vol.get("spike"):
        factors.append(f"Volume spike: {vol.get('volume_ratio', 0):.1f}× average — {vol.get('label', '')}")

    trend_d = tech.get("trend", {})
    if trend_d:
        factors.append(f"Price trend: {trend_d.get('strength','')} {trend_d.get('direction','')}")

    sr = tech.get("support_resistance", {})
    if sr.get("support") and tech.get("price"):
        dist = round((tech["price"] - sr["support"]) / tech["price"] * 100, 1)
        factors.append(f"Nearest support: ${sr['support']} ({dist}% below current price)")
    if sr.get("resistance") and tech.get("price"):
        dist = round((sr["resistance"] - tech["price"]) / tech["price"] * 100, 1)
        factors.append(f"Nearest resistance: ${sr['resistance']} ({dist}% above current price)")

    return factors[:7]   # cap at 7 factors for readability


def _long_factors(sentiment: float, tech: Optional[dict], fund: Optional[dict]) -> list[str]:
    factors = []

    if fund:
        rev_g = fund.get("rev_growth")
        earn_g = fund.get("earn_growth")
        pe_f  = fund.get("pe_forward")
        rec   = analyst_label(fund.get("recommendation", ""), fund.get("recommendation_mean"))
        up    = fund.get("upside_pct")
        pm    = fund.get("profit_margin")

        if rec and rec != "N/A":
            factors.append(f"Analyst consensus: {rec} ({fund.get('analyst_count') or '?'} analysts)")
        if up is not None:
            factors.append(f"Analyst price target: ${fund.get('target_mean') or '?'} ({up:+.1f}% upside)")
        if rev_g is not None:
            factors.append(f"Revenue growth: {rev_g:+.1f}% YoY {'(strong)' if rev_g > 20 else '(moderate)' if rev_g > 5 else '(weak/declining)'}")
        if earn_g is not None:
            factors.append(f"Earnings growth: {earn_g:+.1f}% YoY")
        if pe_f is not None and pe_f > 0:
            cheap = pe_f < 20
            factors.append(f"Forward P/E: {pe_f}× {'(attractive valuation)' if cheap else '(growth premium)'}")
        if pm is not None:
            factors.append(f"Profit margin: {pm:.1f}%")

        surprises = fund.get("recent_surprises", [])
        if surprises:
            beat_count = sum(1 for s in surprises if s > 0)
            factors.append(f"Last {len(surprises)} quarters: {beat_count}/{len(surprises)} earnings beats")

        next_earn = fund.get("next_earnings")
        if next_earn:
            factors.append(f"Next earnings date: {next_earn} — potential catalyst")

    if tech:
        mas = tech.get("moving_avgs", {})
        if mas.get("above_sma200"):
            factors.append("Price above 200-day SMA — long-term uptrend intact")
        elif not mas.get("above_sma200") and mas.get("sma200"):
            factors.append(f"Price below 200-day SMA (${mas['sma200']}) — long-term downtrend")

        if mas.get("golden_cross"):
            factors.append("Golden cross in effect (SMA50 > SMA200) — secular bull signal")
        elif mas.get("death_cross"):
            factors.append("Death cross in effect (SMA50 < SMA200) — secular bear warning")

        trend_d = tech.get("trend", {})
        if trend_d:
            factors.append(f"Price trend: {trend_d.get('strength','')} {trend_d.get('direction','')}")

    if sentiment >= 0.25:
        factors.append(f"Sustained positive sentiment from social/news sources (+{sentiment:.2f})")
    elif sentiment <= -0.25:
        factors.append(f"Sustained negative sentiment from social/news sources ({sentiment:.2f})")

    return factors[:8]


# ─── Rationale builders ───────────────────────────────────────────────────────

def _short_rationale(signal: str, sentiment: float, tech: Optional[dict], composite: float) -> str:
    direction = "bullish" if composite > 0 else "bearish"
    strength  = "strong " if abs(composite) >= 0.4 else ""

    if not tech:
        return (f"{strength.capitalize()}{'Bullish' if composite>0 else 'Bearish'} short-term sentiment signal "
                f"(score: {composite:+.3f}). No technical data available — treat with caution. "
                f"Signal is based on social media and news sentiment only.")

    rsi_v   = tech.get("rsi", 50)
    macd_d  = tech.get("macd", {})
    trend_d = tech.get("trend", {})
    boll    = tech.get("bollinger", {})

    tech_aligned = (composite > 0 and tech.get("technical_score", 0) > 0) or \
                   (composite < 0 and tech.get("technical_score", 0) < 0)

    parts = [
        f"Short-term signal is {strength}{direction} (composite: {composite:+.3f}).",
        f"Sentiment from social and news sources is {'positive' if sentiment > 0 else 'negative'} ({sentiment:+.2f}).",
    ]

    if tech_aligned:
        parts.append(f"Technical indicators {'confirm' if abs(composite) > 0.3 else 'are broadly consistent with'} this direction.")
    else:
        parts.append("Note: technical indicators are not fully aligned with sentiment — reduce position size.")

    if macd_d.get("crossover"):
        parts.append(f"MACD {macd_d['crossover'].replace('_', ' ')} is a fresh entry signal.")

    if rsi_v <= 30:
        parts.append(f"RSI {rsi_v} = oversold — historically a buying opportunity.")
    elif rsi_v >= 70:
        parts.append(f"RSI {rsi_v} = overbought — consider waiting for a pullback before entering long.")

    if boll.get("squeeze"):
        parts.append("Bollinger squeeze detected — expect a sharp directional move soon.")

    return " ".join(parts)


def _long_rationale(signal: str, sentiment: float, tech: Optional[dict],
                    fund: Optional[dict], composite: float) -> str:
    direction = "bullish" if composite > 0 else "bearish"
    strength  = "strongly " if abs(composite) >= 0.45 else ""

    parts = [f"Long-term outlook is {strength}{direction} (composite: {composite:+.3f})."]

    if fund:
        rec   = analyst_label(fund.get("recommendation", ""), fund.get("recommendation_mean"))
        up    = fund.get("upside_pct")
        rev_g = fund.get("rev_growth")
        fs    = fund.get("fundamental_score", 0)

        parts.append(
            f"Fundamental score is {'strong' if fs > 0.3 else 'weak' if fs < -0.3 else 'mixed'} ({fs:+.2f}). "
            f"Wall Street consensus: {rec}."
        )
        if up is not None:
            parts.append(f"Analyst mean price target implies {up:+.1f}% {'upside' if up > 0 else 'downside'} from current price.")
        if rev_g is not None:
            parts.append(f"Revenue is growing {'strongly' if rev_g > 20 else 'moderately' if rev_g > 5 else 'slowly/declining'} at {rev_g:.1f}% YoY.")
    else:
        parts.append("Fundamental data unavailable — long-term signal is based on technicals and sentiment only.")

    if tech:
        mas = tech.get("moving_avgs", {})
        if mas.get("above_sma200"):
            parts.append("Price is above the 200-day SMA — the long-term trend is up.")
        elif not mas.get("above_sma200"):
            parts.append("Price is below the 200-day SMA — long-term trend is down; only buy on confirmed reversal.")

    return " ".join(parts)


# ─── Action label ─────────────────────────────────────────────────────────────

def action_label(signal: str, timeframe: str, tech: Optional[dict]) -> str:
    rsi_v = tech.get("rsi", 50) if tech else 50
    squeeze = tech.get("squeeze", False) if tech else False

    if signal == "STRONG BUY":
        if squeeze and timeframe == "short":
            return "LONG — SQUEEZE PLAY"
        return "STRONG BUY — INITIATE LONG"
    if signal == "BUY":
        if rsi_v <= 35 and timeframe == "short":
            return "BUY — OVERSOLD BOUNCE"
        return "BUY — INITIATE LONG"
    if signal == "HOLD":
        return "HOLD / WAIT FOR SETUP"
    if signal == "SELL":
        return "SELL / INITIATE SHORT"
    if signal == "STRONG SELL":
        return "STRONG SELL — INITIATE SHORT"
    return "HOLD"
