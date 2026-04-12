# 📈 Stock Sentiment Analyzer

A real-time stock recommendation engine powered by sentiment analysis across Reddit, news outlets, and RSS feeds.

---

## Features

- 🔍 **Multi-source scraping** — Reddit (PRAW), NewsAPI, and RSS feeds
- 🧠 **Dual sentiment engines** — VADER (fast) + FinBERT (accurate, optional)
- 📊 **Traction scoring** — Weighted by upvotes, comments, article recency
- 🚀 **Flask REST API** — JSON endpoints ready for any frontend
- 🖥️ **Live dashboard** — Beautiful single-page app with auto-refresh
- 🗃️ **SQLite caching** — Avoids redundant API calls

---

## Project Structure

```
stock-sentiment-analyzer/
├── app.py                  # Flask API server
├── config.py               # All settings & thresholds
├── requirements.txt
├── .env.example
├── scrapers/
│   ├── reddit_scraper.py   # PRAW-based Reddit scraper
│   ├── news_scraper.py     # NewsAPI + RSS scraper
│   └── base_scraper.py     # Shared utilities
├── analysis/
│   ├── sentiment.py        # VADER + optional FinBERT
│   └── recommender.py      # Scoring & ranking logic
├── database/
│   └── db.py               # SQLite cache layer
├── dashboard/
│   └── index.html          # Frontend dashboard
└── data/                   # Auto-created at runtime
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/stock-sentiment-analyzer.git
cd stock-sentiment-analyzer
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Where to get it |
|---|---|
| `REDDIT_CLIENT_ID` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| `REDDIT_CLIENT_SECRET` | Same page |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) (free tier) |

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000** — the dashboard loads automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/recommendations` | Top stock picks with scores |
| GET | `/api/sentiment/<ticker>` | Detailed sentiment for one stock |
| GET | `/api/sources` | Recent scraped articles/posts |
| POST | `/api/refresh` | Force re-scrape all sources |

---

## Sentiment Scoring

Each mention is scored on:
- **Sentiment** (-1.0 to +1.0) via VADER
- **Traction** — Reddit: `log(upvotes + comments)`, News: recency decay
- **Final score** = `sentiment × traction_weight`

Stocks with score > **0.3** are flagged as **BUY**, < **-0.3** as **AVOID**.

---

## Optional: FinBERT (Better Accuracy)

For financial-domain sentiment analysis, install the extras:

```bash
pip install transformers torch
```

Then set in `.env`:
```
USE_FINBERT=true
```

> ⚠️ FinBERT requires ~500MB RAM and is slower. Recommended for batch analysis.

---

## Deployment

This project is ready for deployment on **Railway**, **Render**, or **Heroku**:

```bash
# Procfile is included
web: gunicorn app:app
```

---

## License

MIT
