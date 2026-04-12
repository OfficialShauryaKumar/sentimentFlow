"""
analysis/portfolio.py — Portfolio analysis engine

For every position you hold, this cross-references your entry cost
against live signals and generates specific recommendations:

  STOP HIT         — Price broke below entry stop. Exit now.
  SELL NOW         — Both ST + LT signals strongly bearish. Exit.
  TAKE PROFITS     — Gain target hit, signals fading. Lock in.
  SELL PARTIAL     — Trim half, let rest run (overbought + gain).
  WATCH            — Signals weakening. Monitor closely.
  ADD MORE         — High-conviction bullish, scale in.
  HOLD             — Signals intact. Stay the course.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("portfolio")

try:
    import yfinance as yf
    _yf_ok = True
except ImportError:
    _yf_ok = False


def _live_price(ticker: str) -> Optional[float]:
    if not _yf_ok:
        return None
    try:
        fi = yf.Ticker(ticker).fast_info
        return round(float(fi.last_price), 2) if fi.last_price else None
    except Exception:
        return None


def _analyze_position(holding: dict, signal_map: dict) -> dict:
    ticker   = holding["ticker"].upper()
    shares   = float(holding["shares"])
    avg_cost = float(holding["avg_cost"])
    buy_date = holding.get("buy_date", "")
    notes    = holding.get("notes", "")

    price = _live_price(ticker)
    rec   = signal_map.get(ticker, {})
    if price is None:
        price = rec.get("price")

    cost_basis     = round(avg_cost * shares, 2)
    current_value  = round(price * shares, 2)    if price else None
    unrealized_pnl = round(current_value - cost_basis, 2) if current_value else None
    unrealized_pct = round((price - avg_cost) / avg_cost * 100, 2) if price and avg_cost else None
    pnl_per_share  = round(price - avg_cost, 2) if price else None

    days_held = None
    if buy_date:
        try:
            bought = datetime.fromisoformat(buy_date.replace("Z", "+00:00"))
            if bought.tzinfo is None:
                bought = bought.replace(tzinfo=timezone.utc)
            days_held = (datetime.now(timezone.utc) - bought).days
        except Exception:
            pass

    st   = rec.get("short_term") or {}
    lt   = rec.get("long_term")  or {}
    tech = rec.get("technicals") or {}

    st_signal = st.get("signal", "HOLD")
    lt_signal = lt.get("signal", "HOLD")
    st_score  = st.get("composite_score", 0.0)
    lt_score  = lt.get("composite_score", 0.0)
    rsi       = rec.get("rsi") or tech.get("rsi")
    atr       = rec.get("atr") or tech.get("atr")
    sentiment = rec.get("composite_score", 0.0)

    # ATR-based levels from current price
    stop_loss   = round(price - atr * 2.0, 2) if price and atr else None
    take_profit = round(price + atr * 5.0, 2) if price and atr else None

    # Entry-based floors (always available)
    entry_stop = round(avg_cost * 0.93, 2)   # 7% below entry
    entry_tp   = round(avg_cost * 1.20, 2)   # 20% above entry
    stop_hit   = price is not None and price <= entry_stop
    tp_hit     = price is not None and price >= entry_tp

    action, urgency, rationale, detail = _recommend(
        ticker, shares, avg_cost, price, st_signal, lt_signal,
        st_score, lt_score, unrealized_pct, stop_hit, tp_hit, rsi, days_held, sentiment
    )

    return {
        "ticker": ticker, "shares": shares, "avg_cost": avg_cost,
        "buy_date": buy_date, "notes": notes, "days_held": days_held,
        "current_price": price, "cost_basis": cost_basis,
        "current_value": current_value, "unrealized_pnl": unrealized_pnl,
        "unrealized_pct": unrealized_pct, "pnl_per_share": pnl_per_share,
        "stop_loss": stop_loss, "take_profit": take_profit,
        "entry_stop": entry_stop, "entry_tp": entry_tp,
        "stop_hit": stop_hit, "tp_hit": tp_hit,
        "st_signal": st_signal, "lt_signal": lt_signal,
        "st_score": st_score, "lt_score": lt_score,
        "rsi": rsi, "atr": atr,
        "action": action, "urgency": urgency,
        "rationale": rationale, "detail": detail,
        "short_term": st, "long_term": lt,
        "technicals": tech, "fundamentals": rec.get("fundamentals") or {},
        "mention_count": rec.get("mention_count", 0),
        "sources": rec.get("sources", []),
        "bullish_pct": rec.get("bullish_pct", 0),
        "bearish_pct": rec.get("bearish_pct", 0),
    }


def _recommend(ticker, shares, avg_cost, price, st_signal, lt_signal,
               st_score, lt_score, unrealized_pct, stop_hit, tp_hit,
               rsi, days_held, sentiment):
    pct  = unrealized_pct or 0
    loss = pct < 0

    # 1. Stop hit
    if stop_hit and price:
        return (
            "STOP HIT — SELL",
            "HIGH",
            f"Price has fallen more than 7% below your entry of ${avg_cost:.2f}. "
            f"Your stop at ${avg_cost * 0.93:.2f} has been breached. Exit now to limit further losses.",
            f"Current loss: {pct:+.1f}% (${(price - avg_cost) * shares:.0f}). "
            f"Never let a loss run past your stop."
        )

    # 2. Both timeframes strongly bearish
    if st_signal in ("STRONG SELL", "SELL") and lt_signal in ("SELL", "STRONG SELL"):
        verb = "Lock in gains — " if pct > 5 else "Cut losses — "
        return (
            "SELL NOW",
            "HIGH",
            f"{verb}both short-term ({st_signal}) and long-term ({lt_signal}) signals are bearish. "
            f"{'You are up ' + str(abs(pct)) + '% — take the profit before sentiment reverses.' if pct > 5 else 'Exit before further downside.'}",
            f"ST score: {st_score:+.3f}. LT score: {lt_score:+.3f}. Sentiment: {sentiment:+.3f}."
        )

    # 3. Take profit target hit, signals cooling
    if tp_hit and pct >= 15:
        if st_signal in ("BUY", "STRONG BUY") and lt_signal in ("BUY", "STRONG BUY"):
            return (
                "SELL PARTIAL — TAKE PROFITS",
                "MEDIUM",
                f"You are up {pct:+.1f}% — your 20% gain target has been hit. "
                f"Signals are still bullish, so sell half to lock in profits and let the rest run with a trailing stop.",
                f"Sell {int(shares * 0.5)} shares. Keep {shares - int(shares * 0.5)} with stop at break-even."
            )
        else:
            return (
                "TAKE PROFITS — SELL",
                "MEDIUM",
                f"Up {pct:+.1f}% and signals are no longer strongly bullish. This is a clean exit — take the gain.",
                f"Realized gain if you sell: ${(price - avg_cost) * shares if price else 0:.0f}."
            )

    # 4. Short-term bearish, long-term still bullish — trim
    if st_signal in ("SELL", "STRONG SELL") and lt_signal in ("BUY", "STRONG BUY", "HOLD"):
        return (
            "WATCH — POSSIBLE TRIM",
            "MEDIUM",
            f"Short-term signal has turned bearish ({st_signal}) but long-term outlook is still {lt_signal}. "
            f"Consider trimming 25–30% of your position to reduce risk while keeping long-term exposure.",
            f"{'You are up ' + str(abs(pct)) + '% — a partial sell locks in profit.' if pct > 0 else 'Set a tighter stop on your remaining position.'}"
        )

    # 5. RSI overbought on a large gain
    if rsi and rsi >= 75 and pct >= 20:
        return (
            "SELL PARTIAL — OVERBOUGHT",
            "MEDIUM",
            f"RSI is {rsi} (overbought) and you are up {pct:+.1f}%. "
            f"RSI above 75 with a large unrealized gain historically signals a near-term pullback. "
            f"Consider trimming half the position.",
            f"Sell {int(shares * 0.5)} shares at ~${price:.2f}. "
            f"Lock in ${(price - avg_cost) * int(shares * 0.5) if price else 0:.0f} profit."
        )

    # 6. Very bullish — add more
    if st_signal == "STRONG BUY" and lt_signal in ("BUY", "STRONG BUY") and not loss:
        return (
            "ADD MORE",
            "LOW",
            f"Both timeframes are bullish (ST: {st_signal}, LT: {lt_signal}) and you are already up {pct:+.1f}%. "
            f"This is a high-conviction setup — you could scale in further.",
            f"If you add, keep the total position within your max allocation. Move your stop up to protect gains."
        )

    # 7. Bullish — hold
    if st_signal in ("BUY", "STRONG BUY") and lt_signal in ("BUY", "STRONG BUY"):
        msg = f"You are up {pct:+.1f}% — thesis intact." if pct > 0 else f"Down {abs(pct):.1f}% but signals remain bullish. Thesis unchanged."
        return (
            "HOLD",
            "LOW",
            f"Both signals bullish (ST: {st_signal}, LT: {lt_signal}). {msg} No action needed.",
            "Continue holding. Review if signal changes or price breaks your stop."
        )

    # 8. Losing + bearish — watch
    if loss and pct < -5 and st_signal in ("SELL", "STRONG SELL"):
        return (
            "WATCH — CONSIDER EXIT",
            "MEDIUM",
            f"Down {abs(pct):.1f}% and short-term signal is {st_signal}. "
            f"If price keeps weakening, exit before losses grow.",
            f"Current loss: ${(price - avg_cost) * shares if price else 0:.0f}. Is your original thesis still valid?"
        )

    # 9. Default hold
    return (
        "HOLD",
        "LOW",
        f"No strong signal to act on (ST: {st_signal}, LT: {lt_signal}). "
        f"{'Up ' + str(abs(pct)) + '% — ' if pct > 0 else ''}Hold and review when signals change.",
        "Check again after the next trading day or a new catalyst."
    )


def analyze_portfolio(holdings: list[dict], recs: list[dict]) -> dict:
    if not holdings:
        return {"positions": [], "summary": _empty_summary(),
                "analyzed_at": datetime.now(timezone.utc).isoformat()}

    signal_map = {r["ticker"]: r for r in recs}

    for h in holdings:
        t = h["ticker"].upper()
        if t not in signal_map:
            price = _live_price(t)
            signal_map[t] = {
                "ticker": t, "price": price,
                "short_term": {"signal": "HOLD", "composite_score": 0.0},
                "long_term":  {"signal": "HOLD", "composite_score": 0.0},
                "composite_score": 0.0,
                "rsi": None, "atr": None,
                "technicals": {}, "fundamentals": {},
                "mention_count": 0, "sources": [],
                "bullish_pct": 0, "bearish_pct": 0,
            }

    positions = []
    for h in holdings:
        try:
            positions.append(_analyze_position(h, signal_map))
        except Exception as e:
            logger.error(f"Position error {h.get('ticker','?')}: {e}")

    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    positions.sort(key=lambda p: (urgency_order.get(p["urgency"], 2), -(p["unrealized_pnl"] or 0)))

    return {
        "positions":   positions,
        "summary":     _portfolio_summary(positions),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def _portfolio_summary(positions: list[dict]) -> dict:
    priced       = [p for p in positions if p["current_value"] is not None]
    total_cost   = sum(p["cost_basis"]    for p in priced)
    total_value  = sum(p["current_value"] for p in priced)
    total_pnl    = round(total_value - total_cost, 2)
    total_pnl_pct = round(total_pnl / total_cost * 100, 2) if total_cost else 0

    winners = [p for p in priced if (p["unrealized_pnl"] or 0) > 0]
    losers  = [p for p in priced if (p["unrealized_pnl"] or 0) < 0]
    alerts  = [p for p in positions if p["urgency"] == "HIGH"]
    watches = [p for p in positions if p["urgency"] == "MEDIUM"]

    action_counts: dict[str, int] = {}
    for p in positions:
        a = p["action"].split(" — ")[0]
        action_counts[a] = action_counts.get(a, 0) + 1

    best  = max(priced, key=lambda p: p["unrealized_pct"] or -999) if priced else None
    worst = min(priced, key=lambda p: p["unrealized_pct"] or  999) if priced else None

    pts = 70
    for p in priced:
        pct = p.get("unrealized_pct") or 0
        if p["stop_hit"]:            pts -= 15
        elif pct < -10:              pts -= 8
        elif pct < -5:               pts -= 3
        elif pct > 20:               pts += 5
        elif pct > 10:               pts += 3
        if "SELL NOW" in p["action"]:  pts -= 10
        if "STOP HIT" in p["action"]:  pts -= 15
        if "ADD MORE" in p["action"]:  pts += 4

    return {
        "total_positions":  len(positions),
        "total_cost":       round(total_cost, 2),
        "total_value":      round(total_value, 2),
        "total_pnl":        total_pnl,
        "total_pnl_pct":    total_pnl_pct,
        "winners":          len(winners),
        "losers":           len(losers),
        "high_urgency":     len(alerts),
        "medium_urgency":   len(watches),
        "action_breakdown": action_counts,
        "best_performer":   {"ticker": best["ticker"],  "pct": best["unrealized_pct"]}  if best  else None,
        "worst_performer":  {"ticker": worst["ticker"], "pct": worst["unrealized_pct"]} if worst else None,
        "health_score":     max(0, min(100, pts)),
    }


def _empty_summary() -> dict:
    return {
        "total_positions": 0, "total_cost": 0, "total_value": 0,
        "total_pnl": 0, "total_pnl_pct": 0,
        "winners": 0, "losers": 0, "high_urgency": 0, "medium_urgency": 0,
        "action_breakdown": {}, "best_performer": None, "worst_performer": None,
        "health_score": 50,
    }
