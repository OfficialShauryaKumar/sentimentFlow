# SentimentFlow Backtest

Validates that daily sentiment scores from the SentimentFlow pipeline have predictive power for next-day equity returns. Implements the standard quant equity-signal evaluation stack with three ablations.

## What it computes

**Headline metrics** (saved to `results.csv`):

- **Information Coefficient (mean_IC):** daily Spearman rank correlation between sentiment score and 1-day forward return, averaged across the eval period. An IC of 0.02 is real signal; 0.05+ is strong; 0.10+ is suspicious (recheck for look-ahead leakage).
- **Information Ratio (IR):** mean IC divided by std of IC across days. Above 0.5 is meaningful.
- **Hit rate:** fraction of high-conviction signals (|sentiment| > 0.1) where direction matched. Random is 50%; 54–56% on a real signal is publishable.
- **Long-short Sharpe (gross / net):** annualized Sharpe ratio of an equal-weight portfolio that goes long the top sentiment quintile and short the bottom quintile, with and without 10bps round-trip transaction costs.
- **Total return / max drawdown:** standard portfolio statistics for the net long-short strategy.

**Ablations:**

- **Signal decay** (`decay.csv` and `ic_by_horizon.png`): IC at 1-day, 5-day, and 20-day forward horizons. Real sentiment signals usually peak at 1 day and decay.
- **News-volume slice** (`volume_slice.csv`, requires `article_count` column): IC for low-volume vs high-volume news days.
- **Quintile returns** (`quintile_returns.png`): mean forward return by sentiment quintile — should be roughly monotonic if the signal works.

**Charts:**

- `cumulative.png`: Long-short portfolio equity curve (gross and net) versus SPY.
- `ic_by_horizon.png`: Bar chart of IC at three holding periods.
- `quintile_returns.png`: Bar chart of mean forward return by sentiment quintile.

## Input format

A CSV with three required columns and one optional:

| column | type | description |
|---|---|---|
| `date` | ISO date | The date the sentiment is "as of" — articles must be published BEFORE market close on this date |
| `ticker` | string | Stock symbol, e.g. `AAPL` |
| `sentiment_score` | float in [-1, 1] | Aggregated sentiment from articles published before close on `date` |
| `article_count` | int (optional) | Number of articles aggregated, used for the news-volume slice |

## Usage

```bash
pip install pandas numpy yfinance scipy matplotlib
python backtest.py --data sentiment.csv --out results/
```

For the FinBERT baseline ablation (optional, not yet wired into `main()` — see "Extending"):

```bash
pip install transformers torch
```

## Pitfalls to actively avoid

1. **Look-ahead bias.** Article timestamps must be *publish* time, not *scrape* time, and you must use UTC-aware datetimes. The script enters positions at next-day's close, but if your sentiment score for date `t` includes articles published after 4pm ET on `t`, you've leaked future information. Verify before running.

2. **Survivorship bias.** Backtesting on today's S&P 500 implicitly selects winners. Either use historical index membership data, or disclose the limitation explicitly in your writeup.

3. **Test-set tuning.** Don't pick hyperparameters on the same period you report. Hold out the last 30% of dates for final evaluation; only touch them once.

4. **Transaction costs.** The script includes a 10bps round-trip default. For a more realistic high-turnover strategy, increase to 20–30bps.

## Resume bullets this generates

Once you run it, fill in the numbers from `results.csv`:

> Backtested over [N] months across [K] tickers; long-short portfolio achieved annualized Sharpe of [X.XX] (vs [Y.YY] for SPY) after 10bps round-trip costs.
>
> Achieved Information Coefficient of [0.0XX] on next-day returns (IR = [0.XX]), with directional hit rate of [XX]% on high-conviction signals.
>
> Signal decay analysis showed predictive power persisted through 5-day horizons, with peak IC at 1-day forward returns.

## Extending

**FinBERT baseline.** Run your full SentimentFlow pipeline a second time using FinBERT (`ProsusAI/finbert`) as the sentiment model instead of yours. Save the output as `sentiment_finbert.csv`, run the backtest on both, and compare ICs. If your model beats FinBERT zero-shot by even 0.005 IC, that's a publishable result.

**Sector slice.** Add a `sector` column to the input and slice IC by sector — sometimes sentiment works in tech but not utilities.

**Train/test split.** Add `--train-end YYYY-MM-DD` to split data into train (for any threshold tuning) and test (for final reported numbers).
