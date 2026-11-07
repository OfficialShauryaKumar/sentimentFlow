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

# Day 3: buy DDD; day 4 it hits take-profit while still a BUY: sold, not bought back the same run.
run_cycle([rec("DDD", "BUY", 10, 500, stop=9, target=12)], spy_price=510, today="2026-11-04")
r = run_cycle([rec("DDD", "BUY", 12.5, 500, stop=11, target=14)], spy_price=510, today="2026-11-05")
assert [(t["action"], t["ticker"]) for t in r["trades"]] == [("SELL", "DDD")] and r["positions"] == 0, r
r = run_cycle([rec("DDD", "BUY", 12.5, 500, stop=11, target=14)], spy_price=510, today="2026-11-06")
assert [(t["action"], t["ticker"]) for t in r["trades"]] == [("BUY", "DDD")], r  # next day it may buy again

# Earnings rule: no new position in a ticker reporting soon; others still bought.
r = run_cycle([rec("EEE", "STRONG BUY", 20, 500, confidence=90), rec("FFF", "BUY", 10, 500)],
              spy_price=510, today="2026-11-09", reporting_soon={"EEE"})
assert [t["ticker"] for t in r["trades"]] == ["FFF"] and r["skipped_for_earnings"] == ["EEE"], r

res = get_results()
assert res["benchmark_pct"] == 2.0 and len(res["daily"]) == 6, res
assert res["realized_pnl"] == 112.5 and res["win_rate"] == 67, res  # AAA -110, CCC +100, DDD 49 sh x +2.50
# Tax: all 2026 trades net +112.50, at the default 25% => 28.125 (rounds to 28.12), off the account value.
assert res["est_tax"] == 28.12 and res["after_tax_equity"] == 10084.38, res
# Which session a run belongs to (GitHub's scheduler often starts hours late).
from datetime import datetime  # noqa: E402
from analysis.market_hours import ET, last_closed_session  # noqa: E402
at = lambda s: datetime.fromisoformat(s).replace(tzinfo=ET)
assert last_closed_session(at("2026-11-06 17:15")) == "2026-11-06"   # Friday after the close
assert last_closed_session(at("2026-11-06 12:00")) is None           # market open: refuse
assert last_closed_session(at("2026-11-07 01:30")) == "2026-11-06"   # late Friday run lands Saturday
assert last_closed_session(at("2026-11-09 08:00")) == "2026-11-06"   # Monday before the open
assert last_closed_session(at("2026-11-27 18:00")) == "2026-11-27"   # day after Thanksgiving
assert last_closed_session(at("2026-11-26 18:00")) == "2026-11-25"   # Thanksgiving itself: market shut
print("paper trading engine: all checks passed")
