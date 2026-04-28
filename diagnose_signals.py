"""
diagnose_signals.py — Drop into the sf-project root and run with:
    python diagnose_signals.py

Prints the actual distribution of composite scores across your watchlist,
so you can confirm WHY everything resolves to HOLD before applying any fix.
"""

import statistics
from analysis.recommender import build_recommendations
from scrapers.news_scraper import scrape_news
from scrapers.reddit_scraper import scrape_reddit


def main() -> None:
    mentions = []
    try:
        mentions += scrape_news()
    except Exception as e:
        print(f"news scraper failed: {e}")
    try:
        mentions += scrape_reddit()
    except Exception as e:
        print(f"reddit scraper failed: {e}")

    if not mentions:
        print("No mentions scraped — check API keys / rate limits before continuing.")
        return

    results = build_recommendations(mentions)
    if not results:
        print("build_recommendations returned 0 results.")
        return

    print(f"\n=== {len(results)} tickers analyzed ===\n")

    for tf in ("short_term", "long_term"):
        print(f"--- {tf} composite distribution ---")
        scores = [(r["ticker"], (r.get(tf) or {}).get("composite_score"))
                  for r in results]
        scores = [(t, s) for t, s in scores if s is not None]
        if not scores:
            print("  no scores available\n")
            continue
        vals = [s for _, s in scores]
        print(f"  n          = {len(vals)}")
        print(f"  mean       = {statistics.mean(vals):+.4f}")
        print(f"  stdev      = {statistics.stdev(vals) if len(vals) > 1 else 0:+.4f}")
        print(f"  min / max  = {min(vals):+.4f}  /  {max(vals):+.4f}")
        thr_buy = 0.25; thr_strong = 0.55
        print(f"  pct > +0.55 (STRONG BUY):  "
              f"{sum(1 for v in vals if v >= thr_strong)/len(vals)*100:.1f}%")
        print(f"  pct > +0.25 (BUY):         "
              f"{sum(1 for v in vals if v >= thr_buy)/len(vals)*100:.1f}%")
        print(f"  pct between (HOLD):        "
              f"{sum(1 for v in vals if -thr_buy < v < thr_buy)/len(vals)*100:.1f}%")
        print(f"  pct < -0.25 (SELL):        "
              f"{sum(1 for v in vals if v <= -thr_buy)/len(vals)*100:.1f}%")
        print(f"  pct < -0.55 (STRONG SELL): "
              f"{sum(1 for v in vals if v <= -thr_strong)/len(vals)*100:.1f}%")
        print(f"\n  per-ticker scores (sorted):")
        for t, s in sorted(scores, key=lambda x: -x[1]):
            sig = (next(r for r in results if r["ticker"] == t).get(tf) or {}).get("signal")
            print(f"    {t:6s}  {s:+.4f}  →  {sig}")
        print()


if __name__ == "__main__":
    main()
