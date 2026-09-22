"""
paper_trading/engine.py — Mock trading account engine
$100,000 starting balance. Auto-executes high-conviction signals,
queues medium-conviction signals for manual approval.

Auto-trade threshold  : conviction score >= 0.40 (4★ or 5★)
Manual approval queue : conviction score 0.25–0.39 (3★)
Ignored               : conviction score < 0.25 (1★ / 2★)
"""

import logging
from datetime import datetime, timezone

import yfinance as yf

import config
from database.db import get_connection

logger = logging.getLogger("paper_trading")

STARTING_BALANCE    = 100_000.0
AUTO_TRADE_MIN      = 0.40   # >= this → auto execute
MANUAL_QUEUE_MIN    = 0.25   # >= this → queue for approval
MAX_POSITION_PCT    = config.MAX_POSITION_PCT / 100   # e.g. 0.10
RISK_PER_TRADE_PCT  = config.RISK_PER_TRADE_PCT / 100  # e.g. 0.02


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_price(ticker: str) -> float | None:
    """Fetch current price via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if price and price > 0:
            return float(price)
        # fallback: history
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Price fetch failed for {ticker}: {e}")
    return None


def _ensure_account():
    """Make sure the account row exists."""
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM paper_account LIMIT 1").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO paper_account (cash, total_invested, updated_at) VALUES (?,?,?)",
                (STARTING_BALANCE, 0.0, _now()),
            )


def _get_cash() -> float:
    _ensure_account()
    with get_connection() as conn:
        row = conn.execute("SELECT cash FROM paper_account LIMIT 1").fetchone()
    return float(row["cash"]) if row else STARTING_BALANCE


def _update_cash(new_cash: float):
    with get_connection() as conn:
        conn.execute("UPDATE paper_account SET cash=?, updated_at=?", (new_cash, _now()))


def _get_position(ticker: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM paper_positions WHERE ticker=?", (ticker,)
        ).fetchone()
    return dict(row) if row else None


def _upsert_position(ticker: str, shares: float, avg_cost: float):
    now = _now()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT shares, avg_cost FROM paper_positions WHERE ticker=?", (ticker,)
        ).fetchone()
        if existing:
            old_shares = float(existing["shares"])
            old_cost   = float(existing["avg_cost"])
            new_shares = old_shares + shares
            new_avg    = ((old_shares * old_cost) + (shares * avg_cost)) / new_shares
            conn.execute(
                "UPDATE paper_positions SET shares=?, avg_cost=?, updated_at=? WHERE ticker=?",
                (new_shares, new_avg, now, ticker),
            )
        else:
            conn.execute(
                "INSERT INTO paper_positions (ticker, shares, avg_cost, opened_at, updated_at) VALUES (?,?,?,?,?)",
                (ticker, shares, avg_cost, now, now),
            )


def _remove_position(ticker: str, shares_sold: float) -> bool:
    pos = _get_position(ticker)
    if not pos:
        return False
    remaining = float(pos["shares"]) - shares_sold
    if remaining <= 0.001:
        with get_connection() as conn:
            conn.execute("DELETE FROM paper_positions WHERE ticker=?", (ticker,))
    else:
        with get_connection() as conn:
            conn.execute(
                "UPDATE paper_positions SET shares=?, updated_at=? WHERE ticker=?",
                (remaining, _now(), ticker),
            )
    return True


def _record_trade(ticker, action, shares, price, total_value,
                  signal_score, signal_reason, mode, status, pnl=None):
    now = _now()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO paper_trades
                (ticker, action, shares, price, total_value, signal_score,
                 signal_reason, mode, status, pnl, created_at, executed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ticker, action, shares, price, total_value,
            signal_score, signal_reason, mode, status, pnl,
            now, now if status == "EXECUTED" else None,
        ))
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


# ── Core trading logic ─────────────────────────────────────────────────────────

def _calc_shares_to_buy(price: float, cash: float) -> float:
    """
    Position sizing: risk RISK_PER_TRADE_PCT of account, capped at MAX_POSITION_PCT.
    Uses the total account value (cash + positions) as the base.
    """
    account_value  = get_account()["total_value"]
    max_by_pct     = account_value * MAX_POSITION_PCT
    risk_amount    = account_value * RISK_PER_TRADE_PCT
    # Use the smaller of the two limits, and don't exceed available cash
    invest_amount  = min(max_by_pct, risk_amount * 5, cash * 0.95)
    if invest_amount < price:
        return 0.0
    return round(invest_amount / price, 4)


def _execute_buy(ticker: str, price: float, signal_score: float,
                 signal_reason: str, mode: str) -> dict:
    cash = _get_cash()
    shares = _calc_shares_to_buy(price, cash)
    if shares <= 0:
        return {"ok": False, "error": "Insufficient funds or price too high."}

    total = round(shares * price, 2)
    if total > cash:
        return {"ok": False, "error": f"Not enough cash (need ${total:.2f}, have ${cash:.2f})."}

    trade_id = _record_trade(
        ticker, "BUY", shares, price, total,
        signal_score, signal_reason, mode, "EXECUTED",
    )
    _upsert_position(ticker, shares, price)
    _update_cash(cash - total)
    logger.info(f"[PAPER] BUY {shares} {ticker} @ ${price:.2f} = ${total:.2f} ({mode})")
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": total}


def _execute_sell(ticker: str, price: float, signal_score: float,
                  signal_reason: str, mode: str) -> dict:
    pos = _get_position(ticker)
    if not pos:
        return {"ok": False, "error": f"No position in {ticker}."}

    shares   = float(pos["shares"])
    avg_cost = float(pos["avg_cost"])
    total    = round(shares * price, 2)
    pnl      = round(total - (shares * avg_cost), 2)

    trade_id = _record_trade(
        ticker, "SELL", shares, price, total,
        signal_score, signal_reason, mode, "EXECUTED", pnl,
    )
    _remove_position(ticker, shares)
    _update_cash(_get_cash() + total)
    logger.info(f"[PAPER] SELL {shares} {ticker} @ ${price:.2f} = ${total:.2f} | P&L: ${pnl:.2f} ({mode})")
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": total, "pnl": pnl}


def _queue_trade(ticker: str, action: str, price: float, signal_score: float,
                 signal_reason: str) -> dict:
    """Add a trade to the pending approval queue."""
    cash   = _get_cash()
    shares = _calc_shares_to_buy(price, cash) if action == "BUY" else (
        float(_get_position(ticker)["shares"]) if _get_position(ticker) else 0
    )
    if shares <= 0:
        return {"ok": False, "error": "Nothing to queue."}

    total    = round(shares * price, 2)
    trade_id = _record_trade(
        ticker, action, shares, price, total,
        signal_score, signal_reason, "MANUAL", "PENDING_APPROVAL",
    )
    logger.info(f"[PAPER] QUEUED {action} {shares} {ticker} @ ${price:.2f} (awaiting approval)")
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": total, "status": "PENDING_APPROVAL"}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_account() -> dict:
    """Return account summary: cash, positions value, total, P&L."""
    _ensure_account()
    cash      = _get_cash()
    positions = get_positions()
    pos_value = 0.0
    for p in positions:
        price = _get_price(p["ticker"])
        if price:
            pos_value += price * float(p["shares"])

    # Total P&L from all completed trades
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl),0) AS total_pnl FROM paper_trades WHERE status='EXECUTED' AND action='SELL'"
        ).fetchone()
    total_pnl = float(row["total_pnl"]) if row else 0.0

    return {
        "cash":              round(cash, 2),
        "positions_value":   round(pos_value, 2),
        "total_value":       round(cash + pos_value, 2),
        "starting_balance":  STARTING_BALANCE,
        "total_return_pct":  round(((cash + pos_value - STARTING_BALANCE) / STARTING_BALANCE) * 100, 2),
        "realized_pnl":      round(total_pnl, 2),
        "unrealized_pnl":    round(pos_value - sum(float(p["shares"]) * float(p["avg_cost"]) for p in positions), 2),
        "updated_at":        _now(),
    }


def get_positions() -> list[dict]:
    """Return all open paper positions."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM paper_positions ORDER BY opened_at DESC").fetchall()
    positions = []
    for r in rows:
        p = dict(r)
        price = _get_price(p["ticker"])
        if price:
            p["current_price"] = round(price, 2)
            p["market_value"]  = round(price * float(p["shares"]), 2)
            p["unrealized_pnl"]= round((price - float(p["avg_cost"])) * float(p["shares"]), 2)
            p["pnl_pct"]       = round(((price - float(p["avg_cost"])) / float(p["avg_cost"])) * 100, 2)
        positions.append(p)
    return positions


def get_trades(limit: int = 50) -> list[dict]:
    """Return executed trade history."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status='EXECUTED' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_trades() -> list[dict]:
    """Return trades awaiting manual approval."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status='PENDING_APPROVAL' ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def approve_trade(trade_id: int) -> dict:
    """Manually approve a queued trade and execute it at current market price."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM paper_trades WHERE id=? AND status='PENDING_APPROVAL'", (trade_id,)
        ).fetchone()
    if not row:
        return {"ok": False, "error": "Trade not found or already processed."}

    trade  = dict(row)
    ticker = trade["ticker"]
    action = trade["action"]
    price  = _get_price(ticker)
    if not price:
        return {"ok": False, "error": f"Could not fetch current price for {ticker}."}

    # Execute at current price
    if action == "BUY":
        result = _execute_buy(ticker, price, trade["signal_score"], trade["signal_reason"], "MANUAL")
    else:
        result = _execute_sell(ticker, price, trade["signal_score"], trade["signal_reason"], "MANUAL")

    # Mark original queued trade as executed
    with get_connection() as conn:
        conn.execute(
            "UPDATE paper_trades SET status='EXECUTED', executed_at=? WHERE id=?",
            (_now(), trade_id),
        )
    return result


def reject_trade(trade_id: int) -> dict:
    """Reject a queued trade."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE paper_trades SET status='REJECTED', executed_at=? WHERE id=? AND status='PENDING_APPROVAL'",
            (_now(), trade_id),
        )
    if cur.rowcount == 0:
        return {"ok": False, "error": "Trade not found or already processed."}
    logger.info(f"[PAPER] Rejected trade #{trade_id}")
    return {"ok": True, "message": f"Trade #{trade_id} rejected."}


def close_position(ticker: str) -> dict:
    """Manually close an open position at market price."""
    price = _get_price(ticker)
    if not price:
        return {"ok": False, "error": f"Could not fetch price for {ticker}."}
    return _execute_sell(ticker, price, 0.0, "Manual close", "MANUAL")


def reset_account() -> dict:
    """Reset the paper account back to $100,000 (clears all trades and positions)."""
    with get_connection() as conn:
        conn.execute("UPDATE paper_account SET cash=?, updated_at=?", (STARTING_BALANCE, _now()))
        conn.execute("DELETE FROM paper_positions")
        conn.execute("DELETE FROM paper_trades")
    logger.warning("[PAPER] Account reset to $100,000.")
    return {"ok": True, "message": "Account reset.", "balance": STARTING_BALANCE}


def run_paper_trading_cycle(recommendations: list[dict]) -> dict:
    """
    Process a list of recommendation dicts from the analysis engine.
    Auto-executes high-conviction signals, queues medium-conviction for approval.
    Returns a summary of actions taken.
    """
    _ensure_account()
    auto_executed = []
    queued        = []
    skipped       = []

    for rec in recommendations:
        ticker = rec.get("ticker", "").upper()
        action = rec.get("action", "")         # BUY / SELL / HOLD / AVOID
        score  = float(rec.get("score", 0.0))
        reason = rec.get("reason", "")

        # Only trade on BUY and SELL signals
        if action not in ("BUY", "SELL"):
            skipped.append({"ticker": ticker, "reason": f"Signal is {action}, not trading."})
            continue

        # Get current price
        price = _get_price(ticker)
        if not price:
            skipped.append({"ticker": ticker, "reason": "Could not fetch price."})
            continue

        # Don't re-enter existing positions (for BUY)
        if action == "BUY" and _get_position(ticker):
            skipped.append({"ticker": ticker, "reason": "Already holding position."})
            continue

        # Don't try to sell if no position
        if action == "SELL" and not _get_position(ticker):
            skipped.append({"ticker": ticker, "reason": "No position to sell."})
            continue

        if score >= AUTO_TRADE_MIN:
            # High conviction → auto execute
            if action == "BUY":
                result = _execute_buy(ticker, price, score, reason, "AUTO")
            else:
                result = _execute_sell(ticker, price, score, reason, "AUTO")
            if result.get("ok"):
                auto_executed.append({"ticker": ticker, "action": action, **result})
            else:
                skipped.append({"ticker": ticker, "reason": result.get("error")})

        elif score >= MANUAL_QUEUE_MIN:
            # Medium conviction → queue for approval
            result = _queue_trade(ticker, action, price, score, reason)
            if result.get("ok"):
                queued.append({"ticker": ticker, "action": action, **result})
            else:
                skipped.append({"ticker": ticker, "reason": result.get("error")})

        else:
            skipped.append({"ticker": ticker, "reason": f"Score {score:.2f} below threshold."})

    return {
        "auto_executed": auto_executed,
        "queued_for_approval": queued,
        "skipped": skipped,
        "account": get_account(),
    }
