"""
app.py — Flask REST API for SentimentFlow
Includes persistent portfolio management endpoints.
"""
import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from database import (
    init_db, cache_get, cache_set, cache_invalidate,
    portfolio_get_all, portfolio_upsert, portfolio_get_one, portfolio_delete,
)
from scrapers import scrape_reddit, scrape_news
from analysis import build_recommendations
from analysis.market_health import fetch_market_health
from analysis.portfolio import analyze_portfolio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

app = Flask(__name__, static_folder="dashboard")
CORS(app)
init_db()


def _run_analysis() -> dict:
    logger.info("Starting full scrape + analysis…")
    reddit = scrape_reddit()
    news   = scrape_news()
    all_m  = reddit + news
    logger.info(f"Total mentions: {len(all_m)}")
    recs   = build_recommendations(all_m)
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
    data = cache_get("analysis_v2")
    if data is None:
        data = _run_analysis()
        cache_set("analysis_v2", data)
    return jsonify({
        "ok":             True,
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
    data = _run_analysis()
    cache_set("analysis_v2", data)
    return jsonify({"ok": True, "count": len(data["recommendations"]),
                    "refreshed_at": data["refreshed_at"]})


@app.route("/api/ticker/<ticker>", methods=["GET"])
def get_ticker(ticker: str):
    ticker = ticker.upper()
    data   = cache_get("analysis_v2")
    if data is None:
        data = _run_analysis()
        cache_set("analysis_v2", data)
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
