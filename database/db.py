"""
database/db.py — SQLite layer: cache + persistent portfolio holdings

With DATABASE_URL set (e.g. a free Neon Postgres), portfolio holdings and the
daily sentiment history are kept in Postgres instead, so they survive hosts
whose disk is wiped on restart (Render's free tier). The scrape cache always
stays in local SQLite: losing it only costs one re-scrape.
"""

import os
import json
import sqlite3
import logging
from contextlib import contextmanager
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


USE_POSTGRES = bool(config.DATABASE_URL)


@contextmanager
def _durable():
    """Connection for holdings and history: Postgres when configured, else SQLite.

    Yields (conn, sql) where sql() adapts '?' placeholders to the backend.
    Both connections commit when the block exits without an error.
    """
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(config.DATABASE_URL, row_factory=dict_row, connect_timeout=15) as conn:
            yield conn, lambda q: q.replace("?", "%s")
    else:
        with get_connection() as conn:
            yield conn, lambda q: q


_PG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        id          BIGSERIAL PRIMARY KEY,
        ticker      TEXT             NOT NULL UNIQUE,
        shares      DOUBLE PRECISION NOT NULL,
        avg_cost    DOUBLE PRECISION NOT NULL,
        buy_date    TEXT,
        notes       TEXT,
        added_at    TEXT             NOT NULL,
        updated_at  TEXT             NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sentiment_history (
        date            TEXT NOT NULL,
        ticker          TEXT NOT NULL,
        sentiment_score DOUBLE PRECISION,
        article_count   INTEGER,
        bullish_pct     DOUBLE PRECISION,
        bearish_pct     DOUBLE PRECISION,
        engine          TEXT,
        PRIMARY KEY (date, ticker)
    );
"""


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
        """)
    logger.info("Database initialised.")

    if USE_POSTGRES:
        # Never let an unreachable database stop the app from starting;
        # holdings requests will report the error instead.
        try:
            with _durable() as (conn, _):
                conn.execute(_PG_SCHEMA)
            logger.info("Postgres ready for holdings and sentiment history.")
        except Exception as e:
            logger.error(f"Postgres setup failed: {e}")


# ─── Scrape cache ─────────────────────────────────────────────────────────────

def cache_get(key: str, ttl_minutes: int = None):
    if ttl_minutes is None:
        ttl_minutes = config.CACHE_TTL_MINUTES
    if ttl_minutes == 0:
        return None   # always-fresh mode
    ttl    = timedelta(minutes=ttl_minutes)
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
    with _durable() as (conn, sql):
        rows = conn.execute(
            sql("SELECT * FROM portfolio_holdings ORDER BY added_at DESC")
        ).fetchall()
    return [dict(r) for r in rows]


def portfolio_upsert(ticker: str, shares: float, avg_cost: float,
                     buy_date: str = None, notes: str = None) -> dict:
    """Insert or update a position. Returns the saved row."""
    now    = datetime.now(timezone.utc).isoformat()
    ticker = ticker.upper().strip()
    with _durable() as (conn, sql):
        conn.execute(sql("""
            INSERT INTO portfolio_holdings (ticker, shares, avg_cost, buy_date, notes, added_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                shares=excluded.shares,
                avg_cost=excluded.avg_cost,
                buy_date=COALESCE(excluded.buy_date, portfolio_holdings.buy_date),
                notes=COALESCE(excluded.notes, portfolio_holdings.notes),
                updated_at=excluded.updated_at
        """), (ticker, shares, avg_cost, buy_date, notes, now, now))
    return portfolio_get_one(ticker)


def portfolio_get_one(ticker: str) -> dict | None:
    ticker = ticker.upper().strip()
    with _durable() as (conn, sql):
        row = conn.execute(
            sql("SELECT * FROM portfolio_holdings WHERE ticker=?"), (ticker,)
        ).fetchone()
    return dict(row) if row else None


def portfolio_delete(ticker: str) -> bool:
    ticker = ticker.upper().strip()
    with _durable() as (conn, sql):
        cur = conn.execute(sql("DELETE FROM portfolio_holdings WHERE ticker=?"), (ticker,))
        deleted = cur.rowcount > 0
    return deleted


# ─── Sentiment history (Postgres only; locally it lives in a CSV) ────────────

HISTORY_COLUMNS = ["date", "ticker", "sentiment_score", "article_count",
                   "bullish_pct", "bearish_pct", "engine"]


def history_insert(rows: list[list]) -> int:
    """Store daily snapshot rows, skipping (date, ticker) pairs already saved."""
    if not USE_POSTGRES or not rows:
        return 0
    with _durable() as (conn, sql):
        with conn.cursor() as cur:
            cur.executemany(sql(
                f"INSERT INTO sentiment_history ({', '.join(HISTORY_COLUMNS)}) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT (date, ticker) DO NOTHING"
            ), rows)
            return cur.rowcount


def history_all() -> list[dict]:
    """Every stored snapshot row, oldest first (Postgres only)."""
    if not USE_POSTGRES:
        return []
    with _durable() as (conn, sql):
        return [dict(r) for r in conn.execute(sql(
            f"SELECT {', '.join(HISTORY_COLUMNS)} FROM sentiment_history ORDER BY date, ticker"
        )).fetchall()]
