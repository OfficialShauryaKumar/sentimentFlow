"""
database/db.py — SQLite layer: cache + persistent portfolio holdings
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
import config

logger = logging.getLogger("db")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sentimentflow.db")


def _ensure_data_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scrape_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key   TEXT    UNIQUE NOT NULL,
                data        TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mentions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                source          TEXT,
                source_type     TEXT,
                title           TEXT,
                url             TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                traction        REAL,
                created_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL UNIQUE,
                shares      REAL    NOT NULL,
                avg_cost    REAL    NOT NULL,
                buy_date    TEXT,
                notes       TEXT,
                added_at    TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_mentions_ticker  ON mentions(ticker);
            CREATE INDEX IF NOT EXISTS idx_mentions_created ON mentions(created_at);
            CREATE INDEX IF NOT EXISTS idx_holdings_ticker  ON portfolio_holdings(ticker);

            -- ── Paper trading ────────────────────────────────────────────────

            CREATE TABLE IF NOT EXISTS paper_account (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cash            REAL    NOT NULL DEFAULT 100000.0,
                total_invested  REAL    NOT NULL DEFAULT 0.0,
                updated_at      TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL UNIQUE,
                shares      REAL    NOT NULL,
                avg_cost    REAL    NOT NULL,
                opened_at   TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                shares          REAL    NOT NULL,
                price           REAL    NOT NULL,
                total_value     REAL    NOT NULL,
                signal_score    REAL,
                signal_reason   TEXT,
                mode            TEXT    NOT NULL,
                status          TEXT    NOT NULL,
                pnl             REAL,
                created_at      TEXT    NOT NULL,
                executed_at     TEXT
            );

            -- ── SEC filings ──────────────────────────────────────────────────

            CREATE TABLE IF NOT EXISTS sec_filings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                cik             TEXT    NOT NULL,
                form_type       TEXT    NOT NULL,
                accession       TEXT    NOT NULL UNIQUE,
                filed_date      TEXT    NOT NULL,
                title           TEXT,
                summary         TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                url             TEXT,
                fetched_at      TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker  ON paper_trades(ticker);
            CREATE INDEX IF NOT EXISTS idx_paper_trades_status  ON paper_trades(status);
            CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker   ON sec_filings(ticker);
            CREATE INDEX IF NOT EXISTS idx_sec_filings_form     ON sec_filings(form_type);
        """)
    logger.info("Database initialised.")


# ─── Scrape cache ─────────────────────────────────────────────────────────────

def cache_get(key: str):
    if config.CACHE_TTL_MINUTES == 0:
        return None   # always-fresh mode
    ttl    = timedelta(minutes=config.CACHE_TTL_MINUTES)
    cutoff = (datetime.now(timezone.utc) - ttl).isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT data FROM scrape_cache WHERE cache_key=? AND created_at>?",
            (key, cutoff),
        ).fetchone()
    if row:
        logger.debug(f"Cache HIT: {key}")
        return json.loads(row["data"])
    logger.debug(f"Cache MISS: {key}")
    return None


def cache_set(key: str, data):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO scrape_cache (cache_key, data, created_at) VALUES (?,?,?)
               ON CONFLICT(cache_key) DO UPDATE SET data=excluded.data, created_at=excluded.created_at""",
            (key, json.dumps(data), now),
        )
    logger.debug(f"Cache SET: {key}")


def cache_invalidate(key: str = None):
    with get_connection() as conn:
        if key:
            conn.execute("DELETE FROM scrape_cache WHERE cache_key=?", (key,))
        else:
            conn.execute("DELETE FROM scrape_cache")
    logger.info(f"Cache invalidated: {key or 'ALL'}")


# ─── Portfolio CRUD ───────────────────────────────────────────────────────────

def portfolio_get_all() -> list[dict]:
    """Return every saved holding, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio_holdings ORDER BY added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def portfolio_upsert(ticker: str, shares: float, avg_cost: float,
                     buy_date: str = None, notes: str = None) -> dict:
    """Insert or update a position. Returns the saved row."""
    now    = datetime.now(timezone.utc).isoformat()
    ticker = ticker.upper().strip()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO portfolio_holdings (ticker, shares, avg_cost, buy_date, notes, added_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                shares=excluded.shares,
                avg_cost=excluded.avg_cost,
                buy_date=COALESCE(excluded.buy_date, buy_date),
                notes=COALESCE(excluded.notes, notes),
                updated_at=excluded.updated_at
        """, (ticker, shares, avg_cost, buy_date, notes, now, now))
    return portfolio_get_one(ticker)


def portfolio_get_one(ticker: str) -> dict | None:
    ticker = ticker.upper().strip()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM portfolio_holdings WHERE ticker=?", (ticker,)
        ).fetchone()
    return dict(row) if row else None


def portfolio_delete(ticker: str) -> bool:
    ticker = ticker.upper().strip()
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM portfolio_holdings WHERE ticker=?", (ticker,))
    return cur.rowcount > 0


# ─── SEC Filings ──────────────────────────────────────────────────────────────

def sec_upsert_filing(filing: dict):
    """Insert a SEC filing, ignore if accession already exists."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO sec_filings
                (ticker, cik, form_type, accession, filed_date, title,
                 summary, sentiment_score, sentiment_label, url, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            filing["ticker"], filing["cik"], filing["form_type"],
            filing["accession"], filing["filed_date"], filing.get("title"),
            filing.get("summary"), filing.get("sentiment_score"),
            filing.get("sentiment_label"), filing.get("url"), filing["fetched_at"],
        ))


def sec_get_filings(ticker: str = None, form_type: str = None, limit: int = 50) -> list[dict]:
    """Fetch stored SEC filings, optionally filtered by ticker and/or form type."""
    query  = "SELECT * FROM sec_filings WHERE 1=1"
    params = []
    if ticker:
        query += " AND ticker=?"
        params.append(ticker.upper())
    if form_type:
        query += " AND form_type=?"
        params.append(form_type.upper())
    query += " ORDER BY filed_date DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]
