"""
analysis/recommender.py — Master pipeline
Sentiment → Indicators → Fundamentals → Short/Long predictions
"""

import logging
from collections import defaultdict
import config
from analysis.sentiment import score_sentiment, classify_sentiment
from analysis.indicators import compute_all as compute_technicals
from analysis.fundamentals import fetch_fundamentals
from analysis.timeframe import short_term_prediction, long_term_prediction

logger = logging.getLogger("recommender")


def enrich_with_sentiment(mentions: list[dict]) -> list[dict]:
    for m in mentions:
        text = m.get("raw_text") or m.get("title", "")
        score = score_sentiment(text)
        m["sentiment_score"] = score
        m["sentiment_label"] = classify_sentiment(score)
        m["weighted_score"]  = round(score * m.get("traction", 1.0), 4)
    return mentions


def aggregate_by_ticker(mentions: list[dict]) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for m in mentions:
        groups[m["ticker"]].append(m)

    aggregated = {}
    for ticker, items in groups.items():
        if not items:
            continue

        n          = len(items)
        sentiments = [i["sentiment_score"] for i in items]
        tractions  = [i["traction"] for i in items]
        weighted   = [i["weighted_score"] for i in items]
        labels     = [i["sentiment_label"] for i in items]

        bullish_n = labels.count("bullish")
        bearish_n = labels.count("bearish")
        neutral_n = labels.count("neutral")

        avg_sent   = sum(sentiments) / n
        total_tr   = sum(tractions)
        avg_w      = sum(weighted) / n

        composite = round(
            0.6 * avg_w + 0.4 * (avg_sent * min(total_tr / 5, 1.0)), 4
        )

        # legacy signal for top-level card
        if composite >= config.SENTIMENT_BUY:
            signal = "BUY"
        elif composite <= config.SENTIMENT_AVOID:
            signal = "AVOID"
        else:
            signal = "NEUTRAL"

        sources = list({i["source"] for i in items})
        top_m   = sorted(items, key=lambda x: abs(x["weighted_score"]), reverse=True)[:5]

        aggregated[ticker] = {
            "ticker":          ticker,
            "mention_count":   n,
            "avg_sentiment":   round(avg_sent, 4),
            "total_traction":  round(total_tr, 4),
            "composite_score": composite,
            "signal":          signal,
            "bullish_pct":     round(bullish_n / n * 100, 1),
            "bearish_pct":     round(bearish_n / n * 100, 1),
            "neutral_pct":     round(neutral_n / n * 100, 1),
            "sources":         sources,
            "top_mentions":    [
                {
                    "source":          i["source"],
                    "source_type":     i.get("source_type", ""),
                    "title":           i.get("title", ""),
                    "sentiment_score": i["sentiment_score"],
                    "sentiment_label": i["sentiment_label"],
                    "traction":        i["traction"],
                    "url":             i.get("url", ""),
                    "created_at":      i.get("created_at", ""),
                }
                for i in top_m
            ],
        }

    return aggregated


def build_recommendations(mentions: list[dict]) -> list[dict]:
    """
    Full pipeline:
      1. Score sentiment on all mentions
      2. Aggregate by ticker
      3. For each ticker: fetch technical indicators + fundamentals
      4. Build short-term AND long-term predictions
      5. Sort and return
    """
    if not mentions:
        logger.warning("No mentions to process.")
        return []

    logger.info(f"Scoring {len(mentions)} mentions…")
    mentions = enrich_with_sentiment(mentions)

    logger.info("Aggregating by ticker…")
    by_ticker = aggregate_by_ticker(mentions)

    filtered = {k: v for k, v in by_ticker.items()
                if v["total_traction"] >= config.MIN_TRACTION}

    logger.info(f"Fetching data for {len(filtered)} tickers…")
    results = []

    for ticker, data in filtered.items():
        sent_score = data["composite_score"]

        # ── Technical indicators (price history, RSI, MACD, etc.) ──────────
        tech = None
        try:
            tech = compute_technicals(ticker)
            if tech:
                data["price"]      = tech.get("price")
                data["change_pct"] = tech.get("change_pct")
                data["week52_high"]= tech.get("week52_high")
                data["week52_low"] = tech.get("week52_low")
                data["market_cap"] = tech.get("market_cap")
                data["rsi"]        = tech.get("rsi")
                data["atr"]        = tech.get("atr")
                data["trend"]      = tech.get("trend")
                data["technicals"] = tech
        except Exception as e:
            logger.warning(f"Technical error for {ticker}: {e}")

        if not tech:
            data.update({"price": None, "change_pct": None,
                         "week52_high": None, "week52_low": None,
                         "market_cap": None, "rsi": None, "atr": None})

        # ── Fundamentals ────────────────────────────────────────────────────
        fund = None
        try:
            fund = fetch_fundamentals(ticker)
            if fund:
                data["fundamentals"] = fund
        except Exception as e:
            logger.warning(f"Fundamental error for {ticker}: {e}")

        # ── Short-term prediction ────────────────────────────────────────────
        try:
            data["short_term"] = short_term_prediction(sent_score, tech, ticker)
        except Exception as e:
            logger.warning(f"Short-term prediction error for {ticker}: {e}")
            data["short_term"] = None

        # ── Long-term prediction ─────────────────────────────────────────────
        try:
            data["long_term"] = long_term_prediction(sent_score, tech, fund, ticker)
        except Exception as e:
            logger.warning(f"Long-term prediction error for {ticker}: {e}")
            data["long_term"] = None

        results.append(data)

    # Sort: strong buy first, then buy, neutral, sell, strong sell
    def sort_key(r):
        st = r.get("short_term") or {}
        lt = r.get("long_term")  or {}
        sig_order = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG SELL": 4}
        st_sig = sig_order.get(st.get("signal", "HOLD"), 2)
        return (st_sig, -(r["composite_score"]))

    results.sort(key=sort_key)

    logger.info(f"Built {len(results)} recommendations.")
    return results
