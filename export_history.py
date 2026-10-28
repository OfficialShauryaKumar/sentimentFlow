"""
export_history.py — Download the sentiment history kept in Postgres to a CSV
that backtest/backtest.py can read.

Needs DATABASE_URL (in .env or the environment). Usage:
    python export_history.py                      # writes data/sentiment_history.csv
    python export_history.py --out history.csv
    python backtest/backtest.py --data data/sentiment_history.csv
"""

import argparse
import csv
import os
import sys

from database import HISTORY_COLUMNS, USE_POSTGRES, history_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join("data", "sentiment_history.csv"))
    args = parser.parse_args()

    if not USE_POSTGRES:
        print("DATABASE_URL is not set, so there is no Postgres history to export.", file=sys.stderr)
        return 1

    rows = history_all()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    days = len({r["date"] for r in rows})
    print(f"Wrote {len(rows)} rows covering {days} days to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
