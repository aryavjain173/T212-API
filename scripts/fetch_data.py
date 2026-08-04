"""
Milestone 2b: pull the price history and check it's usable.

Run:  python -m scripts.fetch_data           (uses cache if fresh)
      python -m scripts.fetch_data --refresh (force a redownload)

Prints a quality report and a preview of the momentum signal so you can
eyeball whether the numbers are plausible before anything trades on them.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from marketdata import (  # noqa: E402
    TICKERS,
    load_prices,
    print_quality_report,
    quality_report,
    sector_of,
    trailing_return,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")


def section(name: str) -> None:
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="force redownload")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    section("FETCH")
    prices, volumes = load_prices(
        TICKERS, refresh=args.refresh, start=args.start
    )

    section("QUALITY")
    report = quality_report(prices, volumes)
    print_quality_report(report)

    if report["dead_tickers"]:
        print("\nDead tickers returned no data at all. Fix these in")
        print("marketdata/universe.py before going further.")

    if report["days_since_last_bar"] > 5:
        print("\nThe most recent bar is over 5 days old. Either the market")
        print("has been shut, or the cache is stale. Try --refresh.")

    section("MOMENTUM SIGNAL PREVIEW  (12-1: 252d lookback, 21d skip)")
    mom = trailing_return(prices, lookback=252, skip=21)
    latest = mom.iloc[-1].dropna().sort_values(ascending=False)

    if latest.empty:
        print("no signal yet, not enough history")
        return 1

    print(f"as of {prices.index[-1].date()}, {len(latest)} names ranked\n")

    def show(label: str, s: pd.Series) -> None:
        print(f"{label}")
        for tkr, val in s.items():
            print(f"  {tkr:<8} {val:>8.1%}   {sector_of(tkr)}")

    show("TOP 10", latest.head(10))
    print()
    show("BOTTOM 10", latest.tail(10))

    section("SECTOR TILT OF THE TOP DECILE")
    decile = max(1, len(latest) // 10)
    top = latest.head(decile)
    counts: dict[str, int] = {}
    for tkr in top.index:
        counts[sector_of(tkr)] = counts.get(sector_of(tkr), 0) + 1
    for sec, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:<14} {n}/{decile}  {'#' * n}")
    print("\nIf one sector dominates, the 'stock selection' strategy is")
    print("partly a sector bet. Worth knowing before you defend it.")

    section("SHORT-HORIZON VIEW  (5d return, for reversal later)")
    rev = trailing_return(prices, lookback=5, skip=0).iloc[-1].dropna()
    rev = rev.sort_values()
    print("biggest 5-day fallers (reversal would buy these):")
    for tkr, val in rev.head(5).items():
        print(f"  {tkr:<8} {val:>8.1%}")
    print("\nbiggest 5-day risers:")
    for tkr, val in rev.tail(5).items():
        print(f"  {tkr:<8} {val:>8.1%}")

    print("\nNote these two signals often disagree, which is the point.")
    print("Momentum and short-term reversal work at different horizons.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
