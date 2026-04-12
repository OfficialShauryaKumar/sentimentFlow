"""
analysis/sentiment.py — Sentiment scoring for stock mentions

Supports two engines:
  - VADER   (default, fast, no GPU needed)
  - FinBERT (optional, finance-tuned transformer)
"""

import logging
import config

logger = logging.getLogger("sentiment")

# ─── VADER (default) ──────────────────────────────────────────────────────────

_vader = None

def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        # Inject finance-specific lexicon boosts
        _finance_lexicon = {
            "moon": 2.5, "mooning": 2.5, "bull": 1.5, "bullish": 2.0,
            "rocket": 2.0, "pump": 1.5, "green": 0.8, "squeeze": 1.5,
            "yolo": 1.0, "calls": 0.5, "puts": -0.5, "bear": -1.5,
            "bearish": -2.0, "crash": -2.5, "dump": -2.0, "red": -0.8,
            "bankruptcy": -3.0, "fraud": -3.0, "short": -0.8,
            "rip": -2.0, "rug": -2.5, "scam": -2.5, "dip": -0.5,
            "buy": 1.0, "sell": -1.0, "hold": 0.3, "strong": 1.2,
            "weak": -1.2, "overvalued": -1.5, "undervalued": 1.5,
            "earnings": 0.5, "miss": -1.5, "beat": 1.5, "guidance": 0.3,
        }
        _vader.lexicon.update(_finance_lexicon)
    return _vader


def score_vader(text: str) -> float:
    """Return compound VADER score: -1.0 (very negative) to +1.0 (very positive)."""
    if not text or not text.strip():
        return 0.0
    analyzer = _get_vader()
    scores   = analyzer.polarity_scores(text)
    return round(scores["compound"], 4)


# ─── FinBERT (optional) ───────────────────────────────────────────────────────

_finbert_pipeline = None

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
            logger.info("FinBERT loaded.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return None
    return _finbert_pipeline


def score_finbert(text: str) -> float:
    """
    Return sentiment score using FinBERT.
    Maps: positive→+score, negative→-score, neutral→0
    """
    if not text or not text.strip():
        return 0.0
    pipe = _get_finbert()
    if pipe is None:
        return score_vader(text)   # fallback
    try:
        # FinBERT max input is 512 tokens
        result = pipe(text[:512], truncation=True)[0]
        label  = result["label"].lower()
        conf   = result["score"]
        if label == "positive":
            return round(conf, 4)
        elif label == "negative":
            return round(-conf, 4)
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"FinBERT inference error: {e} — falling back to VADER")
        return score_vader(text)


# ─── Unified interface ────────────────────────────────────────────────────────

def score_sentiment(text: str) -> float:
    """
    Score sentiment using the engine configured in config.USE_FINBERT.
    Returns a float in [-1.0, 1.0].
    """
    if config.USE_FINBERT:
        return score_finbert(text)
    return score_vader(text)


def classify_sentiment(score: float) -> str:
    """Convert numeric score to human-readable label."""
    if score >= config.SENTIMENT_BUY:
        return "bullish"
    elif score <= config.SENTIMENT_AVOID:
        return "bearish"
    else:
        return "neutral"
