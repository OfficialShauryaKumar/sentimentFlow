"""
paper_trading/engine.py — Automatic paper-trading agent driven by SentimentFlow signals

A simulated $10,000 account, compared against simply holding the S&P 500 (SPY).
Once per trading day (after the close) run_cycle():
  1. sells any position whose short-term signal turned SELL / STRONG SELL, or whose
     price reached the stop-loss or take-profit SentimentFlow set when it was bought;
  2. buys tickers with a short-term BUY / STRONG BUY signal it doesn't already hold,
     sized like the recommender's own position sizing, scaled to the account's value;
  3. records the account's value and the S&P 500 benchmark for the day.
Long only, whole shares, no leverage. Everything is stored via database._durable(),
so it lives in Postgres on Render and in local SQLite otherwise.
"""

import math
from datetime import datetime, timezone

import config
from database.db import _durable

START_VALUE = 10_000.0
MAX_POSITION_PCT = config.MAX_POSITION_PCT / 100
BUYS = {"BUY", "STRONG BUY"}
SELLS = {"SELL", "STRONG SELL"}


def _value(cash: float, positions: dict, prices: dict) -> float:
    """Cash plus positions at today's price (last known cost if a price is missing)."""
    return cash + sum(p["shares"] * prices.get(t, p["avg_cost"]) for t, p in positions.items())


def run_cycle(recs: list[dict], spy_price: float, today: str) -> dict:
    """Trade once for `today` (YYYY-MM-DD) and record the day's value. Safe to re-run:
    a second run on the same day only refreshes the snapshot, it never trades twice."""
    prices = {r["ticker"]: float(r["price"]) for r in recs if r.get("price")}
    signal = {r["ticker"]: r.get("short_term") or {} for r in recs}
    now = f"{today}T{datetime.now(timezone.utc):%H:%M:%S}Z"  # trading day first, so the log reads by market day
    trades = []

    with _durable() as (conn, sql):
        acct = conn.execute(sql("SELECT * FROM paper_account WHERE id=1")).fetchone()
        if acct is None:
            if not spy_price:
                raise ValueError("No S&P 500 price, so the account can't be opened yet")
            conn.execute(sql("INSERT INTO paper_account (id, start_value, cash, spy_start, start_date) "
                             "VALUES (1, ?, ?, ?, ?)"), (START_VALUE, START_VALUE, spy_price, today))
            acct = conn.execute(sql("SELECT * FROM paper_account WHERE id=1")).fetchone()
        acct = dict(acct)
        cash = acct["cash"]
        positions = {p["ticker"]: dict(p) for p in conn.execute(sql("SELECT * FROM paper_positions")).fetchall()}

        def trade(ticker, action, shares, price, reason, pnl=None):
            conn.execute(sql("INSERT INTO paper_trades (created_at, ticker, action, shares, price, pnl, reason) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)"), (now, ticker, action, shares, price, pnl, reason))
            trades.append({"ticker": ticker, "action": action, "shares": shares, "price": price, "pnl": pnl, "reason": reason})

        if acct["last_run"] != today:
            # 1. Sells
            for t, p in list(positions.items()):
                price = prices.get(t)
                if not price:
                    continue  # no price today: keep holding rather than sell blind
                sig = signal.get(t, {}).get("signal", "HOLD")
                reason = (f"Signal turned {sig.lower()}" if sig in SELLS
                          else "Stop-loss reached" if p["stop_loss"] and price <= p["stop_loss"]
                          else "Take-profit reached" if p["take_profit"] and price >= p["take_profit"]
                          else None)
                if reason:
                    cash += p["shares"] * price
                    conn.execute(sql("DELETE FROM paper_positions WHERE ticker=?"), (t,))
                    del positions[t]
                    trade(t, "SELL", p["shares"], price, reason, pnl=round((price - p["avg_cost"]) * p["shares"], 2))

            # 2. Buys, most confident first, while cash lasts
            account_value = _value(cash, positions, prices)
            candidates = sorted((r for r in recs if signal[r["ticker"]].get("signal") in BUYS
                                 and r["ticker"] not in positions and r["ticker"] in prices),
                                key=lambda r: signal[r["ticker"]].get("confidence") or 0, reverse=True)
            for r in candidates:
                t, st, price = r["ticker"], signal[r["ticker"]], prices[r["ticker"]]
                sized = (st.get("position") or {}).get("dollar_amount") or 0
                budget = min(sized * account_value / config.PORTFOLIO_SIZE, account_value * MAX_POSITION_PCT, cash)
                shares = math.floor(budget / price)
                if shares < 1:
                    continue
                cash -= shares * price
                positions[t] = {"ticker": t, "shares": shares, "avg_cost": price,
                                "stop_loss": st.get("stop_loss"), "take_profit": st.get("take_profit_1")}
                conn.execute(sql("INSERT INTO paper_positions (ticker, shares, avg_cost, stop_loss, take_profit, opened_at) "
                                 "VALUES (?, ?, ?, ?, ?, ?)"),
                             (t, shares, price, st.get("stop_loss"), st.get("take_profit_1"), now))
                trade(t, "BUY", shares, price, f"{st.get('signal', 'BUY').capitalize()} signal, {st.get('confidence') or 0}% confidence")

            conn.execute(sql("UPDATE paper_account SET cash=?, last_run=? WHERE id=1"), (cash, today))

        # 3. Daily snapshot (refreshed on every run)
        equity = round(_value(cash, positions, prices), 2)
        benchmark = round(acct["start_value"] * spy_price / acct["spy_start"], 2) if spy_price else None
        conn.execute(sql("INSERT INTO paper_daily (date, equity, cash, spy_close, benchmark) VALUES (?, ?, ?, ?, ?) "
                         "ON CONFLICT(date) DO UPDATE SET equity=excluded.equity, cash=excluded.cash, "
                         "spy_close=excluded.spy_close, benchmark=excluded.benchmark"),
                     (today, equity, round(cash, 2), spy_price, benchmark))

    return {"date": today, "traded": acct["last_run"] != today, "trades": trades,
            "equity": equity, "benchmark": benchmark, "cash": round(cash, 2), "positions": len(positions)}


def get_results() -> dict:
    """Everything the dashboard shows: account, open positions, trade log, daily values."""
    with _durable() as (conn, sql):
        acct = conn.execute(sql("SELECT * FROM paper_account WHERE id=1")).fetchone()
        if acct is None:
            return {"started": False}
        daily = [dict(r) for r in conn.execute(sql("SELECT * FROM paper_daily ORDER BY date")).fetchall()]
        positions = [dict(r) for r in conn.execute(sql("SELECT * FROM paper_positions ORDER BY opened_at")).fetchall()]
        trades = [dict(r) for r in conn.execute(sql("SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT 100")).fetchall()]
        closed = [dict(r) for r in conn.execute(sql("SELECT pnl FROM paper_trades WHERE action='SELL'")).fetchall()]
    acct = dict(acct)
    last = daily[-1] if daily else {"equity": acct["start_value"], "benchmark": acct["start_value"]}
    pct = lambda v: round((v / acct["start_value"] - 1) * 100, 2) if v else None
    return {
        "started": True,
        "start_date": acct["start_date"],
        "start_value": acct["start_value"],
        "cash": round(acct["cash"], 2),
        "equity": last["equity"],
        "benchmark": last["benchmark"],
        "return_pct": pct(last["equity"]),
        "benchmark_pct": pct(last["benchmark"]),
        "realized_pnl": round(sum(t["pnl"] or 0 for t in closed), 2),
        "win_rate": round(100 * sum(1 for t in closed if (t["pnl"] or 0) > 0) / len(closed)) if closed else None,
        "positions": positions,
        "trades": trades,
        "daily": daily,
    }
