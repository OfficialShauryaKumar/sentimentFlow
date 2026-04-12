"""
config.py — Central configuration for Stock Sentiment Analyzer
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "StockSentimentBot/1.0")
NEWS_API_KEY         = os.getenv("NEWS_API_KEY", "")

# ─── App ─────────────────────────────────────────────────────────────────────

FLASK_PORT        = int(os.getenv("FLASK_PORT", 5000))

# CACHE_TTL_MINUTES=0 means "always re-scrape on every run"
# Set to 30+ to cache between runs (saves API quota)
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", 0))

USE_FINBERT       = os.getenv("USE_FINBERT", "false").lower() == "true"
MAX_POSTS         = int(os.getenv("MAX_POSTS_PER_SOURCE", 25))
MIN_TRACTION      = float(os.getenv("MIN_TRACTION_SCORE", 0.05))

# ─── Tracked Tickers ─────────────────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "AMD", "PLTR", "SOFI",
    "RIVN", "COIN", "RBLX", "SNAP", "UBER",
    "NET", "CRWD", "DKNG", "GME", "AMC",
]

# ─── Reddit Sources ───────────────────────────────────────────────────────────

REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
    "pennystocks",
    "ValueInvesting",
]

REDDIT_SORT  = "hot"
REDDIT_TIME  = "day"
REDDIT_LIMIT = MAX_POSTS

# ─── News Sources (RSS) ───────────────────────────────────────────────────────

RSS_FEEDS = {
    "Reuters Business":  "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch":       "http://feeds.marketwatch.com/marketwatch/topstories/",
    "Yahoo Finance":     "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "Seeking Alpha":     "https://seekingalpha.com/market_currents.xml",
    "Investopedia":      "https://www.investopedia.com/feeds/rss.aspx",
    "The Motley Fool":   "https://www.fool.com/feeds/index.aspx",
}

# ─── Sentiment Thresholds ─────────────────────────────────────────────────────

SENTIMENT_BUY   =  0.20
SENTIMENT_AVOID = -0.20

# ─── Scoring Weights ─────────────────────────────────────────────────────────

WEIGHT_REDDIT_UPVOTES  = 0.4
WEIGHT_REDDIT_COMMENTS = 0.3
WEIGHT_NEWS_RECENCY    = 0.3

# ─── Portfolio / Risk Settings ────────────────────────────────────────────────

PORTFOLIO_SIZE     = float(os.getenv("PORTFOLIO_SIZE", 10000))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", 2.0))
MAX_POSITION_PCT   = float(os.getenv("MAX_POSITION_PCT", 10.0))
STOP_LOSS_ATR_MULT = float(os.getenv("STOP_LOSS_ATR_MULT", 2.0))
TAKE_PROFIT_RR     = float(os.getenv("TAKE_PROFIT_RR", 2.5))

# ─── Conviction Thresholds ────────────────────────────────────────────────────

CONVICTION_5_STAR = 0.55
CONVICTION_4_STAR = 0.40
CONVICTION_3_STAR = 0.25
CONVICTION_2_STAR = 0.10

# ─── Timeframe Settings ───────────────────────────────────────────────────────
# Short-term: 1–10 days  (driven by sentiment + price momentum)
# Long-term:  1–12 months (driven by fundamentals + trend)

SHORT_TERM_DAYS = 10
LONG_TERM_DAYS  = 180

# Weights for blending sentiment vs fundamentals per timeframe
SHORT_WEIGHT_SENTIMENT    = 0.70   # sentiment dominates short-term
SHORT_WEIGHT_FUNDAMENTAL  = 0.30
LONG_WEIGHT_SENTIMENT     = 0.25   # fundamentals dominate long-term
LONG_WEIGHT_FUNDAMENTAL   = 0.75
