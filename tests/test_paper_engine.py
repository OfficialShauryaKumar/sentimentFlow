"""Self-check for the paper-trading agent. Run: venv/bin/python tests/test_paper_engine.py
Uses a throwaway SQLite file, never the real (Neon) database."""

import os
import sys
import tempfile

os.environ["DATABASE_URL"] = ""   # before config loads, so .env can't point this at Neon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database.db as db  # noqa: E402
db.DB_PATH = os.path.join(tempfile.mkdtemp(), "paper_test.db")
db.init_db()

from paper_trading import get_results, run_cycle  # noqa: E402


def rec(ticker, signal, price, dollars=0, stop=None, target=None, confidence=50):
    return {"ticker": ticker, "price": price, "short_term": {
        "signal": signal, "confidence": confidence, "stop_loss": stop, "take_profit_1": target,
        "position": {"dollar_amount": dollars}}}


# Day 1: buys on BUY / STRONG BUY, each capped at 10% of the $10,000 account; HOLD ignored.
day1 = [rec("AAA", "BUY", 100, 1000, stop=90, target=120),
        rec("BBB", "HOLD", 40, 1000),
        rec("CCC", "STRONG BUY", 50, 2000, stop=45, target=70, confidence=80)]
r = run_cycle(day1, spy_price=500, today="2026-11-02")
assert r["traded"] and {(t["ticker"], t["shares"]) for t in r["trades"]} == {("AAA", 10), ("CCC", 20)}, r
assert r["cash"] == 8000 and r["equity"] == 10000 and r["benchmark"] == 10000, r

# Same day again: refreshes the snapshot, never trades twice.
again = run_cycle(day1, spy_price=500, today="2026-11-02")
assert not again["traded"] and again["trades"] == [] and again["cash"] == 8000, again

# Day 2: AAA hits its stop-loss, CCC's signal turns SELL; SPY up 2%.
day2 = [rec("AAA", "HOLD", 89), rec("BBB", "HOLD", 41), rec("CCC", "SELL", 55)]
r = run_cycle(day2, spy_price=510, today="2026-11-03")
reasons = {t["ticker"]: (t["reason"], t["pnl"]) for t in r["trades"]}
assert reasons == {"AAA": ("Stop-loss reached", -110), "CCC": ("Signal turned sell", 100)}, reasons
assert r["equity"] == 9990 and r["benchmark"] == 10200 and r["positions"] == 0, r

res = get_results()
assert res["return_pct"] == -0.1 and res["benchmark_pct"] == 2.0, res
assert res["realized_pnl"] == -10 and res["win_rate"] == 50 and len(res["daily"]) == 2, res
print("paper trading engine: all checks passed")
