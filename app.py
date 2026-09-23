"""
app.py — Flask REST API for SentimentFlow
Includes persistent portfolio management, paper trading, and SEC filings endpoints.
"""
import csv
import logging
import os
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from database import (
    init_db, cache_get, cache_set, cache_invalidate,
    portfolio_get_all, portfolio_upsert, portfolio_get_one, portfolio_delete,
)
from database.db import sec_upsert_filing, sec_get_filings
from scrapers import scrape_reddit, scrape_news
from scrapers.sec_scraper import scrape_sec_filings
from analysis import build_recommendations
from analysis.market_health import fetch_market_health
import paper_trading as pt
from analysis.portfolio import analyze_portfolio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

# Path where daily sentiment snapshots accumulate. Each row = one (date, ticker)
# pair. After 60–90 trading days this file is the input for backtest/backtest.py.
HISTORY_PATH   = os.path.join("data", "sentiment_history.csv")
HISTORY_HEADER = ["date", "ticker", "sentiment_score", "article_count",
                  "bullish_pct", "bearish_pct", "engine"]


def _archive_snapshot(recs: list[dict]) -> None:
    """Append today's per-ticker aggregated sentiment to the history CSV.

    Deduplicates by (date, ticker) so multiple dashboard calls per day only
    produce one row per ticker per day. Quietly no-ops on errors so a write
    failure can never break the dashboard.
    """
    if not recs:
        return
    try:
        today  = datetime.now(timezone.utc).date().isoformat()
        engine = "finbert" if config.USE_FINBERT else "vader"
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

        # Load keys already written so we don't re-archive today.
        existing: set[tuple[str, str]] = set()
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    existing.add((row["date"], row["ticker"]))

        new_rows = []
        for r in recs:
            key = (today, r["ticker"])
            if key in existing:
                continue
            new_rows.append([
                today,
                r["ticker"],
                r.get("composite_score", 0.0),
                r.get("mention_count", 0),
                r.get("bullish_pct", 0.0),
                r.get("bearish_pct", 0.0),
                engine,
            ])

        if not new_rows:
            return

        write_header = not os.path.exists(HISTORY_PATH)
        with open(HISTORY_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(HISTORY_HEADER)
            w.writerows(new_rows)
        logger.info(f"Archived {len(new_rows)} sentiment rows for {today}")
    except Exception as e:
        logger.warning(f"Archive snapshot failed (non-fatal): {e}")

    # Also push to Google Sheets if configured (best-effort, never raises).
    try:
        from sheets_sync import push_snapshot
        push_snapshot(recs, engine=engine)
    except Exception as e:
        logger.warning(f"Sheets sync failed (non-fatal): {e}")


app = Flask(__name__, static_folder="dashboard")
CORS(app)
init_db()

# ── Background refresh state ──────────────────────────────────────────────────
_refresh_lock    = threading.Lock()
_refresh_running = False   # True while _run_analysis is in progress


def _run_analysis_bg():
    """Run analysis in a background thread and cache the result."""
    global _refresh_running
    try:
        data = _run_analysis()
        cache_set("analysis_v2", data)
        logger.info("Background refresh complete.")
    except Exception as e:
        logger.error(f"Background refresh failed: {e}", exc_info=True)
    finally:
        with _refresh_lock:
            _refresh_running = False


def _start_bg_refresh():
    """Start background refresh if one is not already running."""
    global _refresh_running
    with _refresh_lock:
        if _refresh_running:
            return False
        _refresh_running = True
    t = threading.Thread(target=_run_analysis_bg, daemon=True)
    t.start()
    return True


def _run_analysis() -> dict:
    logger.info("Starting full scrape + analysis…")
    reddit = scrape_reddit()
    news   = scrape_news()
    all_m  = reddit + news
    logger.info(f"Total mentions: {len(all_m)}")
    recs   = build_recommendations(all_m)
    _archive_snapshot(recs)
    return {
        "recommendations": recs,
        "total_mentions":  len(all_m),
        "reddit_mentions": len(reddit),
        "news_mentions":   len(news),
        "refreshed_at":    datetime.now(timezone.utc).isoformat(),
        "watchlist":       config.WATCHLIST,
        "engine":          "finbert" if config.USE_FINBERT else "vader",
        "portfolio_size":  config.PORTFOLIO_SIZE,
    }


# ── Serve dashboard ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


# ── Recommendations ───────────────────────────────────────────────────────────

@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    force = request.args.get("refresh", "false").lower() == "true"
    if force:
        cache_invalidate("analysis_v2")
        started = _start_bg_refresh()
        # Return immediately so Render doesn't time out.
        # Frontend will poll this endpoint until refreshing=false.
        return jsonify({
            "ok":             True,
            "refreshing":     True,
            "started":        started,
            "recommendations":[],
            "total_mentions": 0,
            "reddit_mentions":0,
            "news_mentions":  0,
            "refreshed_at":   None,
            "engine":         config.USE_FINBERT and "finbert" or "vader",
            "portfolio_size": config.PORTFOLIO_SIZE,
            "count":          0,
        })

    # Normal (non-force) read: return cached data if available, else start bg refresh
    data = cache_get("analysis_v2")
    if data is None:
        # No cache yet — kick off background refresh and tell frontend to poll
        _start_bg_refresh()
        return jsonify({
            "ok":             True,
            "refreshing":     True,
            "recommendations":[],
            "total_mentions": 0,
            "reddit_mentions":0,
            "news_mentions":  0,
            "refreshed_at":   None,
            "engine":         config.USE_FINBERT and "finbert" or "vader",
            "portfolio_size": config.PORTFOLIO_SIZE,
            "count":          0,
        })

    return jsonify({
        "ok":             True,
        "refreshing":     _refresh_running,
        "recommendations":data["recommendations"],
        "total_mentions": data["total_mentions"],
        "reddit_mentions":data["reddit_mentions"],
        "news_mentions":  data["news_mentions"],
        "refreshed_at":   data["refreshed_at"],
        "engine":         data["engine"],
        "portfolio_size": data["portfolio_size"],
        "count":          len(data["recommendations"]),
    })


@app.route("/api/refresh", methods=["POST"])
def force_refresh():
    cache_invalidate("analysis_v2")
    started = _start_bg_refresh()
    return jsonify({"ok": True, "refreshing": True, "started": started})


@app.route("/api/ticker/<ticker>", methods=["GET"])
def get_ticker(ticker: str):
    ticker = ticker.upper()
    data   = cache_get("analysis_v2")
    if data is None:
        _start_bg_refresh()
        return jsonify({"ok": False, "error": "No data yet — refresh in progress."}), 503
    match = next((r for r in data["recommendations"] if r["ticker"] == ticker), None)
    if not match:
        return jsonify({"ok": False, "error": f"{ticker} not found."}), 404
    return jsonify({"ok": True, "data": match})


# ── Portfolio — persistent holdings ──────────────────────────────────────────

@app.route("/api/portfolio/holdings", methods=["GET"])
def get_holdings():
    """Return all saved holdings from the database."""
    holdings = portfolio_get_all()
    return jsonify({"ok": True, "holdings": holdings, "count": len(holdings)})


@app.route("/api/portfolio/holdings", methods=["POST"])
def add_holding():
    """
    Add or update a holding.
    Body: { "ticker": "AAPL", "shares": 10, "avg_cost": 175.50,
            "buy_date": "2024-01-15", "notes": "Long-term hold" }
    """
    body = request.get_json(silent=True) or {}
    ticker   = (body.get("ticker") or "").upper().strip()
    shares   = body.get("shares")
    avg_cost = body.get("avg_cost")

    if not ticker:
        return jsonify({"ok": False, "error": "ticker is required"}), 400
    if shares is None or float(shares) <= 0:
        return jsonify({"ok": False, "error": "shares must be > 0"}), 400
    if avg_cost is None or float(avg_cost) <= 0:
        return jsonify({"ok": False, "error": "avg_cost must be > 0"}), 400

    row = portfolio_upsert(
        ticker=ticker,
        shares=float(shares),
        avg_cost=float(avg_cost),
        buy_date=body.get("buy_date"),
        notes=body.get("notes"),
    )
    logger.info(f"Portfolio: upserted {ticker} x{shares} @ ${avg_cost}")
    return jsonify({"ok": True, "holding": row})


@app.route("/api/portfolio/holdings/<ticker>", methods=["DELETE"])
def remove_holding(ticker: str):
    """Remove a holding from the portfolio."""
    ticker = ticker.upper()
    deleted = portfolio_delete(ticker)
    if not deleted:
        return jsonify({"ok": False, "error": f"{ticker} not found in portfolio"}), 404
    logger.info(f"Portfolio: removed {ticker}")
    return jsonify({"ok": True, "message": f"{ticker} removed from portfolio"})


@app.route("/api/portfolio/analyze", methods=["GET"])
def analyze():
    """
    Analyze all saved holdings against current signals.
    Returns per-position recommendations (SELL, HOLD, ADD, STOP HIT, etc.)
    """
    holdings = portfolio_get_all()
    if not holdings:
        return jsonify({
            "ok": True,
            "analysis": {"positions": [], "summary": {}, "analyzed_at": datetime.now(timezone.utc).isoformat()},
            "message": "No holdings saved. Add positions via POST /api/portfolio/holdings"
        })

    data = cache_get("analysis_v2")
    if data is None:
        data = _run_analysis()
        cache_set("analysis_v2", data)

    try:
        result = analyze_portfolio(holdings, data.get("recommendations", []))
        return jsonify({"ok": True, "analysis": result})
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/api/market-health", methods=["GET"])
def get_market_health():
    """
    GET /api/market-health
    Returns overall market health score, trajectory, regime, sector leaders/laggards.
    Cached for 30 minutes — market data doesn't need to refresh every run.
    """
    force = request.args.get("refresh", "false").lower() == "true"
    if force:
        cache_invalidate("market_health_v1")

    data = cache_get("market_health_v1")
    if data is None:
        try:
            data = fetch_market_health()
            # Cache for 30 min regardless of CACHE_TTL_MINUTES setting
            import json
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc).isoformat()
            from database.db import get_connection
            with get_connection() as conn:
                conn.execute(
                    """INSERT INTO scrape_cache (cache_key, data, created_at) VALUES (?,?,?)
                       ON CONFLICT(cache_key) DO UPDATE SET data=excluded.data, created_at=excluded.created_at""",
                    ("market_health_v1", json.dumps(data), now),
                )
        except Exception as e:
            logger.error(f"Market health fetch failed: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "market_health": data})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "running",
                    "cache_ttl_minutes": config.CACHE_TTL_MINUTES})


@app.route("/api/paper/auto-run", methods=["GET", "POST"])
def paper_auto_run():
    """
    Combined endpoint: refresh signals + run trading cycle in one call.
    Designed for cron-job.org or any external scheduler.
    Accepts GET or POST so it works with simple HTTP pingers too.
    """
    try:
        cache_invalidate("analysis_v2")
        data = _run_analysis()
        cache_set("analysis_v2", data)
        result = pt.run_paper_trading_cycle(data.get("recommendations", []))
        account = pt.get_account()
        return jsonify({
            "ok": True,
            "auto_executed":        len(result.get("auto_executed", [])),
            "queued_for_approval":  len(result.get("queued_for_approval", [])),
            "cash":                 account["cash"],
            "total_value":          account["total_value"],
            "total_return_pct":     account["total_return_pct"],
        })
    except Exception as e:
        logger.error(f"Auto-run failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Paper Trading ─────────────────────────────────────────────────────────────

@app.route("/api/paper/account", methods=["GET"])
def paper_account():
    """GET account summary: cash, positions value, total return, P&L."""
    try:
        return jsonify({"ok": True, "account": pt.get_account()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/paper/positions", methods=["GET"])
def paper_positions():
    """GET all open paper positions with live prices."""
    try:
        return jsonify({"ok": True, "positions": pt.get_positions()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/paper/trades", methods=["GET"])
def paper_trades():
    """GET executed trade history."""
    limit = int(request.args.get("limit", 50))
    return jsonify({"ok": True, "trades": pt.get_trades(limit)})


@app.route("/api/paper/pending", methods=["GET"])
def paper_pending():
    """GET trades awaiting manual approval."""
    return jsonify({"ok": True, "pending": pt.get_pending_trades()})


@app.route("/api/paper/approve/<int:trade_id>", methods=["POST"])
def paper_approve(trade_id: int):
    """Approve a queued trade and execute it at current market price."""
    result = pt.approve_trade(trade_id)
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/paper/reject/<int:trade_id>", methods=["POST"])
def paper_reject(trade_id: int):
    """Reject a queued trade."""
    result = pt.reject_trade(trade_id)
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/paper/close/<ticker>", methods=["POST"])
def paper_close(ticker: str):
    """Manually close an open position at market price."""
    result = pt.close_position(ticker.upper())
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/paper/run", methods=["POST"])
def paper_run():
    """
    Run a full paper trading cycle against current sentiment signals.
    Auto-executes high-conviction (>=4★), queues medium-conviction (3★).
    Uses cached signals only — won't trigger a full scrape (which can
    timeout on Render's 30s limit). Hit Refresh first to load signals.
    """
    try:
        data = cache_get("analysis_v2")
        if data is None:
            return jsonify({
                "ok": False,
                "error": "No signal data loaded yet. Click the main Refresh button first to scrape signals, then run the cycle."
            }), 400

        recs = data.get("recommendations", [])
        result = pt.run_paper_trading_cycle(recs)
        return jsonify({"ok": True, **result})
    except Exception as e:
        logger.error(f"Paper trading run failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/paper/reset", methods=["POST"])
def paper_reset():
    """Reset the paper account to $100,000. Clears all trades and positions."""
    result = pt.reset_account()
    return jsonify(result)


# ── SEC Filings ───────────────────────────────────────────────────────────────

@app.route("/api/sec/filings", methods=["GET"])
def sec_filings():
    """
    GET /api/sec/filings?ticker=AAPL&form=8-K&limit=20
    Returns stored SEC filings from the database.
    """
    ticker    = request.args.get("ticker", "").upper() or None
    form_type = request.args.get("form", "").upper() or None
    limit     = int(request.args.get("limit", 50))
    filings   = sec_get_filings(ticker=ticker, form_type=form_type, limit=limit)
    return jsonify({"ok": True, "filings": filings, "count": len(filings)})


@app.route("/api/sec/fetch", methods=["POST"])
def sec_fetch():
    """
    POST /api/sec/fetch
    Body (optional): { "tickers": ["AAPL", "MSFT"] }
    Scrapes EDGAR for recent 8-K, 10-K, 10-Q filings and stores them.
    Uses the full watchlist if no tickers provided.
    """
    body    = request.get_json(silent=True) or {}
    tickers = body.get("tickers") or config.WATCHLIST

    logger.info(f"SEC fetch triggered for {len(tickers)} tickers.")
    try:
        filings = scrape_sec_filings(tickers)
        for f in filings:
            sec_upsert_filing(f)
        return jsonify({
            "ok":      True,
            "fetched": len(filings),
            "tickers": tickers,
        })
    except Exception as e:
        logger.error(f"SEC fetch failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sec/filings/<ticker>", methods=["GET"])
def sec_filings_for_ticker(ticker: str):
    """GET all stored SEC filings for a specific ticker."""
    filings = sec_get_filings(ticker=ticker.upper())
    return jsonify({"ok": True, "ticker": ticker.upper(), "filings": filings, "count": len(filings)})


if __name__ == "__main__":
    logger.info(f"Port: {config.FLASK_PORT}")
    logger.info(f"Tracking: {len(config.WATCHLIST)} tickers")
    logger.info(f"Portfolio: ${config.PORTFOLIO_SIZE:,.0f}")
    logger.info(f"Cache TTL: {config.CACHE_TTL_MINUTES} min")
    saved = portfolio_get_all()
    if saved:
        logger.info(f"Saved holdings: {[h['ticker'] for h in saved]}")
    else:
        logger.info("No holdings saved yet — add them in the Portfolio tab")
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=True)
