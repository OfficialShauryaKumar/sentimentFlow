"""
app.py — Flask REST API for SentimentFlow
Includes persistent portfolio management endpoints.
"""
import csv
import hmac
import logging
import os
import re
from datetime import date, datetime, timezone
from functools import wraps

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from database import (
    init_db, cache_get, cache_set, cache_invalidate,
    portfolio_get_all, portfolio_upsert, portfolio_get_one, portfolio_delete,
    history_insert, HISTORY_COLUMNS,
)
from scrapers import scrape_reddit, scrape_news
from analysis import build_recommendations
from analysis.market_health import fetch_market_health
from analysis.market_hours import is_trading_day, now_et
from analysis.portfolio import analyze_portfolio
from analysis.price_client import fetch_history
from paper_trading import get_results, run_cycle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

# Path where daily sentiment snapshots accumulate. Each row = one (date, ticker)
# pair. After 60–90 trading days this file is the input for backtest/backtest.py.
HISTORY_PATH   = os.path.join("data", "sentiment_history.csv")
HISTORY_HEADER = HISTORY_COLUMNS


def _archive_snapshot(recs: list[dict]) -> None:
    """Append today's per-ticker aggregated sentiment to the history CSV,
    and to Postgres when DATABASE_URL is set (the copy that survives restarts).

    Deduplicates by (date, ticker) so multiple dashboard calls per day only
    produce one row per ticker per day. Quietly no-ops on errors so a write
    failure can never break the dashboard.
    """
    if not recs:
        return
    today  = datetime.now(timezone.utc).date().isoformat()
    engine = "finbert" if config.USE_FINBERT else "vader"
    today_rows = [[
        today,
        r["ticker"],
        r.get("composite_score", 0.0),
        r.get("mention_count", 0),
        r.get("bullish_pct", 0.0),
        r.get("bearish_pct", 0.0),
        engine,
    ] for r in recs]

    try:
        history_insert(today_rows)
    except Exception as e:
        logger.warning(f"Saving history to Postgres failed (non-fatal): {e}")

    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

        # Load keys already written so we don't re-archive today.
        existing: set[tuple[str, str]] = set()
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    existing.add((row["date"], row["ticker"]))

        new_rows = [row for row in today_rows if (row[0], row[1]) not in existing]
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

# NaN/Inf aren't valid JSON (the browser can't parse them) — send null instead.
import math as _math
from flask.json.provider import DefaultJSONProvider as _DJP
def _clean_nan(o):
    if isinstance(o, float):
        return None if (_math.isnan(o) or _math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    return o
class _SafeJSON(_DJP):
    def dumps(self, obj, **kw):
        return super().dumps(_clean_nan(obj), **kw)
app.json = _SafeJSON(app)

# The dashboard is served from this same origin, so cross-origin access is off
# by default. Set CORS_ORIGINS="https://a.example,https://b.example" to allow others.
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    CORS(app, origins=_cors_origins)

init_db()

if not config.ADMIN_PASSWORD:
    logger.warning("ADMIN_PASSWORD is not set: anyone who can reach this server can "
                   "change holdings and force a re-scrape. Set it on public deployments.")


# ── Owner password ────────────────────────────────────────────────────────────

def _is_owner() -> bool:
    """True when no password is configured, or the request carries the right one."""
    if not config.ADMIN_PASSWORD:
        return True
    header = request.headers.get("Authorization", "")
    supplied = header[len("Bearer "):] if header.startswith("Bearer ") else ""
    return hmac.compare_digest(supplied.encode(), config.ADMIN_PASSWORD.encode())


def _owner_required():
    return jsonify({"ok": False, "error": "Owner password required"}), 401


def owner_only(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        return view(*args, **kwargs) if _is_owner() else _owner_required()
    return wrapper


def _wants_refresh() -> bool:
    return request.args.get("refresh", "false").lower() == "true"


# ── Analysis cache ────────────────────────────────────────────────────────────

def _cache_minutes() -> int:
    if config.ADMIN_PASSWORD:
        return max(config.CACHE_TTL_MINUTES, config.PUBLIC_CACHE_MINUTES)
    return config.CACHE_TTL_MINUTES


def _get_analysis(force: bool = False) -> dict:
    """Latest analysis; re-scrapes when the cache is stale or `force` is set."""
    if force:
        cache_invalidate("analysis_v2")
    data = cache_get("analysis_v2", _cache_minutes())
    if data is None:
        data = _run_analysis()
        cache_set("analysis_v2", data)
    return data


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
    force = _wants_refresh()
    if force and not _is_owner():
        return _owner_required()
    data = _get_analysis(force)
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
@owner_only
def force_refresh():
    data = _get_analysis(force=True)
    return jsonify({"ok": True, "count": len(data["recommendations"]),
                    "refreshed_at": data["refreshed_at"]})


@app.route("/api/ticker/<ticker>", methods=["GET"])
def get_ticker(ticker: str):
    ticker = ticker.upper()
    data   = _get_analysis()
    match = next((r for r in data["recommendations"] if r["ticker"] == ticker), None)
    if not match:
        return jsonify({"ok": False, "error": f"{ticker} not found."}), 404
    return jsonify({"ok": True, "data": match})


# ── Portfolio — persistent holdings ──────────────────────────────────────────

TICKER_RE = re.compile(r"[A-Z0-9.\-]{1,10}")


def _positive_number(value):
    """Parse a JSON value as a finite number > 0, or return None."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 and n != float("inf") else None


@app.route("/api/portfolio/holdings", methods=["GET"])
def get_holdings():
    """Return all saved holdings from the database."""
    holdings = portfolio_get_all()
    return jsonify({"ok": True, "holdings": holdings, "count": len(holdings)})


@app.route("/api/portfolio/holdings", methods=["POST"])
@owner_only
def add_holding():
    """
    Add or update a holding.
    Body: { "ticker": "AAPL", "shares": 10, "avg_cost": 175.50,
            "buy_date": "2024-01-15", "notes": "Long-term hold" }
    """
    body = request.get_json(silent=True) or {}
    ticker   = str(body.get("ticker") or "").upper().strip()
    shares   = _positive_number(body.get("shares"))
    avg_cost = _positive_number(body.get("avg_cost"))
    buy_date = body.get("buy_date") or None
    notes    = body.get("notes")

    if not ticker:
        return jsonify({"ok": False, "error": "ticker is required"}), 400
    if not TICKER_RE.fullmatch(ticker):
        return jsonify({"ok": False, "error": "ticker must be 1-10 letters, digits, '.' or '-'"}), 400
    if shares is None:
        return jsonify({"ok": False, "error": "shares must be > 0"}), 400
    if avg_cost is None:
        return jsonify({"ok": False, "error": "avg_cost must be > 0"}), 400
    if buy_date is not None:
        try:
            date.fromisoformat(str(buy_date))
        except ValueError:
            return jsonify({"ok": False, "error": "buy_date must be YYYY-MM-DD"}), 400
    if notes is not None and len(str(notes)) > 500:
        return jsonify({"ok": False, "error": "notes must be 500 characters or fewer"}), 400

    row = portfolio_upsert(
        ticker=ticker,
        shares=shares,
        avg_cost=avg_cost,
        buy_date=buy_date,
        notes=notes,
    )
    logger.info(f"Portfolio: upserted {ticker} x{shares} @ ${avg_cost}")
    return jsonify({"ok": True, "holding": row})


@app.route("/api/portfolio/holdings/<ticker>", methods=["DELETE"])
@owner_only
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

    data = _get_analysis()

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
    if _wants_refresh():
        if not _is_owner():
            return _owner_required()
        cache_invalidate("market_health_v1")

    data = cache_get("market_health_v1", 30)  # 30 min regardless of CACHE_TTL_MINUTES
    if data is None:
        try:
            data = fetch_market_health()
            cache_set("market_health_v1", data)
        except Exception as e:
            logger.error(f"Market health fetch failed: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "market_health": data})


# ── Paper-trading agent ───────────────────────────────────────────────────────

@app.route("/api/paper/run", methods=["POST"])
@owner_only
def paper_run():
    """Trade once for today on fresh signals. Called each weekday after the close
    by .github/workflows/paper-trade.yml; safe to call twice (it won't trade twice)."""
    today = now_et()
    if not is_trading_day(today):
        return jsonify({"ok": True, "skipped": "Market closed today", "date": today.date().isoformat()})
    data = _get_analysis(force=True)
    spy = fetch_history("SPY", days=10)
    spy_price = float(spy["Close"].iloc[-1]) if spy is not None and not spy.empty else None
    try:
        result = run_cycle(data["recommendations"], spy_price, today.date().isoformat())
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    logger.info(f"Paper trading: {len(result['trades'])} trades, equity ${result['equity']:,.2f}")
    return jsonify({"ok": True, **result})


@app.route("/api/paper", methods=["GET"])
def paper_results():
    return jsonify({"ok": True, **get_results()})


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
    # The Werkzeug debugger can run arbitrary code, so it is opt-in and, when on,
    # only reachable from this machine. Production runs under gunicorn (Procfile).
    debug = os.getenv("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1" if debug else "0.0.0.0", port=config.FLASK_PORT, debug=debug)
