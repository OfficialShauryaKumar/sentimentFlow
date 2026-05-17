"""
sheets_sync.py — Push daily sentiment snapshots to a Google Sheet

Best-effort. If the sheet isn't configured (no GOOGLE_SHEET_ID env var, no
credentials file), this is a silent no-op. If the push fails for any
reason — network, auth, quota — it logs a warning but never raises.
The local CSV archive is the source of truth; Sheets is just a live view.

Setup: see SHEETS_SETUP.md at the repo root.
"""

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("sheets_sync")

SHEET_ID    = os.getenv("GOOGLE_SHEET_ID", "").strip()
CREDS_PATH  = os.getenv("GOOGLE_CREDS_PATH", "credentials/google_service_account.json")
SHEET_HEADER = ["date", "ticker", "sentiment_score", "article_count",
                "bullish_pct", "bearish_pct", "engine"]

_worksheet_cache = None  # cached so we don't re-auth on every call


def _get_worksheet():
    """Return the gspread worksheet, or None if Sheets isn't configured."""
    global _worksheet_cache
    if _worksheet_cache is not None:
        return _worksheet_cache
    if not SHEET_ID:
        return None
    if not os.path.exists(CREDS_PATH):
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
        client = gspread.authorize(creds)
        _worksheet_cache = client.open_by_key(SHEET_ID).sheet1
        return _worksheet_cache
    except Exception as e:
        logger.warning(f"Could not connect to Google Sheets: {e}")
        return None


def push_snapshot(recs: list[dict], engine: str = "vader") -> None:
    """Append today's per-ticker rows to the Google Sheet.

    Deduplicates against existing rows so multiple calls per day (dashboard
    page-loads + nightly cron) only produce one row per (date, ticker).
    Silently no-ops if Sheets is not configured.
    """
    if not recs:
        return
    sheet = _get_worksheet()
    if sheet is None:
        return  # not configured — that's fine

    try:
        today = datetime.now(timezone.utc).date().isoformat()

        # Pull existing rows once to build the dedup set.
        all_rows = sheet.get_all_values()
        if not all_rows:
            sheet.append_row(SHEET_HEADER)
            existing: set[tuple[str, str]] = set()
        elif all_rows[0] != SHEET_HEADER:
            # Header exists but doesn't match our schema — leave alone, just dedup.
            existing = {(r[0], r[1]) for r in all_rows[1:] if len(r) >= 2}
        else:
            existing = {(r[0], r[1]) for r in all_rows[1:] if len(r) >= 2}

        new_rows = []
        for r in recs:
            if (today, r["ticker"]) in existing:
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

        if new_rows:
            sheet.append_rows(new_rows, value_input_option="USER_ENTERED")
            logger.info(f"Pushed {len(new_rows)} rows to Google Sheets")
    except Exception as e:
        logger.warning(f"Google Sheets push failed (non-fatal): {e}")
