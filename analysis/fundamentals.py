"""
analysis/fundamentals.py — Real fundamental data via yfinance

Fetches:
  - Valuation: P/E (TTM + Forward), P/S, P/B, EV/EBITDA
  - Growth:    Revenue YoY, Earnings YoY, EPS trend
  - Quality:   Profit margin, ROE, debt/equity, current ratio
  - Analyst:   Consensus rating, price target (mean, high, low),
               number of analysts, % upside to target
  - Ownership: Institutional %, insider %
  - Earnings:  Next earnings date, recent earnings surprises
  
Produces a fundamental_score: -1.0 (very poor) to +1.0 (very strong)
Used to weight long-term predictions.
"""

import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("fundamentals")

try:
    import yfinance as yf
    _yf_ok = True
except ImportError:
    _yf_ok = False
    logger.warning("yfinance not installed — fundamental data unavailable.")


def fetch_fundamentals(ticker: str) -> Optional[dict]:
    """
    Fetch all fundamental data for a ticker.
    Returns enriched dict or None if unavailable.
    """
    if not _yf_ok:
        return None

    try:
        t    = yf.Ticker(ticker)
        info = t.info
        if not info or "symbol" not in info:
            return None

        # ── Valuation ──────────────────────────────────────────────────────
        pe_ttm      = _safe(info, "trailingPE")
        pe_forward  = _safe(info, "forwardPE")
        ps_ratio    = _safe(info, "priceToSalesTrailing12Months")
        pb_ratio    = _safe(info, "priceToBook")
        ev_ebitda   = _safe(info, "enterpriseToEbitda")

        # ── Growth ─────────────────────────────────────────────────────────
        rev_growth  = _pct(info, "revenueGrowth")         # YoY revenue growth
        earn_growth = _pct(info, "earningsGrowth")        # YoY earnings growth
        earn_qtr    = _pct(info, "earningsQuarterlyGrowth")
        eps_ttm     = _safe(info, "trailingEps")
        eps_fwd     = _safe(info, "forwardEps")
        eps_growth  = round((eps_fwd - eps_ttm) / abs(eps_ttm) * 100, 1) \
                      if eps_ttm and eps_fwd and eps_ttm != 0 else None

        # ── Quality / Profitability ─────────────────────────────────────────
        profit_margin = _pct(info, "profitMargins")
        oper_margin   = _pct(info, "operatingMargins")
        roe           = _pct(info, "returnOnEquity")
        roa           = _pct(info, "returnOnAssets")
        debt_equity   = _safe(info, "debtToEquity")
        current_ratio = _safe(info, "currentRatio")
        free_cf       = _safe(info, "freeCashflow")

        # ── Analyst Consensus ──────────────────────────────────────────────
        analyst_count     = _safe(info, "numberOfAnalystOpinions")
        target_mean       = _safe(info, "targetMeanPrice")
        target_high       = _safe(info, "targetHighPrice")
        target_low        = _safe(info, "targetLowPrice")
        recommendation    = info.get("recommendationKey", "").upper()   # STRONG_BUY / BUY / HOLD / SELL
        recommendation_mean = _safe(info, "recommendationMean")         # 1=Strong Buy .. 5=Sell

        current_price = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice")
        upside_pct = round((target_mean - current_price) / current_price * 100, 1) \
                     if target_mean and current_price and current_price > 0 else None

        # ── Ownership ──────────────────────────────────────────────────────
        inst_pct    = _pct(info, "heldPercentInstitutions")
        insider_pct = _pct(info, "heldPercentInsiders")
        short_pct   = _pct(info, "shortPercentOfFloat")        # short interest

        # ── Earnings calendar ──────────────────────────────────────────────
        next_earnings = None
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                dates = cal.get("Earnings Date")
                if dates is not None and len(dates) > 0:
                    next_earnings = str(dates.iloc[0])[:10]
        except Exception:
            pass

        # ── Recent earnings surprises ──────────────────────────────────────
        surprises = []
        try:
            hist = t.earnings_history
            if hist is not None and not hist.empty:
                for _, row in hist.tail(4).iterrows():
                    if "surprisePercent" in row and row["surprisePercent"] is not None:
                        surprises.append(round(float(row["surprisePercent"]) * 100, 1))
        except Exception:
            pass

        avg_surprise = round(sum(surprises) / len(surprises), 1) if surprises else None

        # ── Compute fundamental score ──────────────────────────────────────
        fscore = _fundamental_score(
            pe_forward=pe_forward, pe_ttm=pe_ttm,
            rev_growth=rev_growth, earn_growth=earn_growth,
            profit_margin=profit_margin, roe=roe,
            debt_equity=debt_equity, current_ratio=current_ratio,
            upside_pct=upside_pct, recommendation_mean=recommendation_mean,
            avg_surprise=avg_surprise,
        )

        return {
            "ticker": ticker,
            # Valuation
            "pe_ttm":       _round(pe_ttm),
            "pe_forward":   _round(pe_forward),
            "ps_ratio":     _round(ps_ratio),
            "pb_ratio":     _round(pb_ratio),
            "ev_ebitda":    _round(ev_ebitda),
            # Growth
            "rev_growth":   rev_growth,
            "earn_growth":  earn_growth,
            "earn_qtr_growth": earn_qtr,
            "eps_ttm":      _round(eps_ttm),
            "eps_forward":  _round(eps_fwd),
            "eps_growth_pct": eps_growth,
            # Quality
            "profit_margin":  profit_margin,
            "oper_margin":    oper_margin,
            "roe":            roe,
            "roa":            roa,
            "debt_equity":    _round(debt_equity),
            "current_ratio":  _round(current_ratio),
            "free_cashflow":  free_cf,
            # Analysts
            "analyst_count":      analyst_count,
            "target_mean":        _round(target_mean),
            "target_high":        _round(target_high),
            "target_low":         _round(target_low),
            "recommendation":     recommendation,
            "recommendation_mean":_round(recommendation_mean),
            "upside_pct":         upside_pct,
            # Ownership
            "institutional_pct": inst_pct,
            "insider_pct":       insider_pct,
            "short_interest_pct":short_pct,
            # Earnings
            "next_earnings":  next_earnings,
            "recent_surprises": surprises,
            "avg_surprise":   avg_surprise,
            # Score
            "fundamental_score": fscore,
        }

    except Exception as e:
        logger.error(f"Fundamental fetch failed for {ticker}: {e}")
        return None


def _safe(info: dict, key: str) -> Optional[float]:
    v = info.get(key)
    if v is None or v == "N/A" or v == "":
        return None
    try:
        f = float(v)
        return f if not (f != f) else None   # NaN check
    except (TypeError, ValueError):
        return None


def _pct(info: dict, key: str) -> Optional[float]:
    v = _safe(info, key)
    return round(v * 100, 1) if v is not None else None


def _round(v, decimals: int = 2) -> Optional[float]:
    return round(v, decimals) if v is not None else None


def _fundamental_score(
    pe_forward, pe_ttm, rev_growth, earn_growth,
    profit_margin, roe, debt_equity, current_ratio,
    upside_pct, recommendation_mean, avg_surprise
) -> float:
    """
    Score fundamentals from -1.0 (very weak) to +1.0 (very strong).
    Each factor is scored independently then weighted.
    """
    pts = 0.0
    total_w = 0.0

    def add(score, weight):
        nonlocal pts, total_w
        pts += score * weight
        total_w += weight

    # Forward P/E — lower is cheaper (for profitable companies)
    if pe_forward is not None and pe_forward > 0:
        if pe_forward < 15:    add(0.8,  0.15)
        elif pe_forward < 25:  add(0.4,  0.15)
        elif pe_forward < 40:  add(0.0,  0.15)
        elif pe_forward < 60:  add(-0.3, 0.15)
        else:                  add(-0.6, 0.15)

    # Revenue growth
    if rev_growth is not None:
        if rev_growth >= 30:   add(1.0,  0.15)
        elif rev_growth >= 15: add(0.6,  0.15)
        elif rev_growth >= 5:  add(0.2,  0.15)
        elif rev_growth >= 0:  add(-0.1, 0.15)
        else:                  add(-0.7, 0.15)

    # Earnings growth
    if earn_growth is not None:
        if earn_growth >= 25:  add(1.0,  0.12)
        elif earn_growth >= 10:add(0.5,  0.12)
        elif earn_growth >= 0: add(0.1,  0.12)
        else:                  add(-0.6, 0.12)

    # Profit margin
    if profit_margin is not None:
        if profit_margin >= 20:  add(0.8,  0.10)
        elif profit_margin >= 10:add(0.4,  0.10)
        elif profit_margin >= 0: add(0.0,  0.10)
        else:                    add(-0.8, 0.10)

    # ROE
    if roe is not None:
        if roe >= 20:    add(0.8,  0.10)
        elif roe >= 10:  add(0.4,  0.10)
        elif roe >= 0:   add(0.0,  0.10)
        else:            add(-0.5, 0.10)

    # Debt/Equity (lower is safer)
    if debt_equity is not None:
        if debt_equity < 30:    add(0.6,  0.08)
        elif debt_equity < 80:  add(0.2,  0.08)
        elif debt_equity < 150: add(-0.2, 0.08)
        else:                   add(-0.6, 0.08)

    # Analyst consensus (1=Strong Buy, 5=Sell)
    if recommendation_mean is not None:
        score = (3 - recommendation_mean) / 2   # maps 1→1.0, 3→0.0, 5→-1.0
        add(max(-1, min(1, score)), 0.20)

    # Upside to analyst price target
    if upside_pct is not None:
        if upside_pct >= 30:    add(0.9,  0.10)
        elif upside_pct >= 15:  add(0.5,  0.10)
        elif upside_pct >= 0:   add(0.1,  0.10)
        elif upside_pct >= -10: add(-0.3, 0.10)
        else:                   add(-0.7, 0.10)

    # Earnings surprise history (positive = beats expectations)
    if avg_surprise is not None:
        if avg_surprise >= 10:  add(0.8,  0.08)
        elif avg_surprise >= 3: add(0.4,  0.08)
        elif avg_surprise >= 0: add(0.1,  0.08)
        else:                   add(-0.6, 0.08)

    return round(pts / total_w if total_w > 0 else 0.0, 4)


def analyst_label(recommendation: str, recommendation_mean: Optional[float]) -> str:
    """Human-readable analyst consensus label."""
    if recommendation in ("STRONG_BUY", "STRONGBUY"):
        return "Strong Buy"
    if recommendation == "BUY":
        return "Buy"
    if recommendation == "HOLD":
        return "Hold"
    if recommendation in ("SELL", "UNDERPERFORM"):
        return "Sell / Underperform"
    if recommendation in ("STRONG_SELL", "STRONGSELL"):
        return "Strong Sell"
    # Fallback to numeric
    if recommendation_mean is not None:
        if recommendation_mean <= 1.5: return "Strong Buy"
        if recommendation_mean <= 2.5: return "Buy"
        if recommendation_mean <= 3.5: return "Hold"
        if recommendation_mean <= 4.5: return "Underperform"
        return "Sell"
    return "No Consensus"
