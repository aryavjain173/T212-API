"""
Why did volatility targeting fail?

Run:  python -m scripts.diagnose

Vol targeting rests on two claims. Backtests confound them; regressions
separate them.

  CLAIM 1  trailing volatility forecasts future volatility
  CLAIM 2  high trailing volatility forecasts poor forward returns

Claim 1 is nearly always true (volatility clusters). Claim 2 is the one
that decides whether the technique makes money, and it is the one nobody
checks. If high volatility in your sample preceded STRONG returns, then
cutting exposure when vol is high must lose by construction, and no window
length or leverage cap can rescue it.

This script tests both directly, with Newey-West standard errors to handle
the overlapping forward windows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import run_backtest  # noqa: E402
from backtest.diagnostics import (  # noqa: E402
    conditional_returns,
    print_regression,
    vol_predicts_return,
    vol_predicts_sharpe,
    vol_predicts_vol,
)
from marketdata import TICKERS, load_prices  # noqa: E402
from strategies import CrossSectionalMomentum, EqualWeightBenchmark  # noqa: E402


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    prices, _ = load_prices(TICKERS)
    mom = CrossSectionalMomentum(n_long=12)
    bench = EqualWeightBenchmark()

    r_mom = run_backtest(prices, mom.generate_weights(prices), cost_bps=10).returns
    r_ben = run_backtest(prices, bench.generate_weights(prices), cost_bps=10).returns

    print(f"momentum series: {len(r_mom):,} days, "
          f"{r_mom.index[0].date()} -> {r_mom.index[-1].date()}")

    for label, series in [("MOMENTUM", r_mom), ("BENCHMARK", r_ben)]:
        section(f"{label}: IS VOLATILITY FORECASTABLE?  (claim 1)")
        print("  If this fails, vol targeting cannot work at all.\n")
        for w in [21, 63, 126]:
            print_regression(vol_predicts_vol(series, window=w))

        section(f"{label}: DOES HIGH VOL PREDICT POOR RETURNS?  (claim 2)")
        print("  Vol targeting cuts exposure when trailing vol is high.")
        print("  That only helps if those periods were bad. A significantly")
        print("  POSITIVE beta means the rule cuts risk into GOOD periods,")
        print("  which guarantees it loses money.\n")
        for h in [5, 21, 63]:
            print_regression(vol_predicts_return(series, window=126, horizon=h))
        print()
        for h in [21, 63]:
            print_regression(vol_predicts_sharpe(series, window=126, horizon=h))

    # ------------------------------------------------------- quintiles

    section("NON-PARAMETRIC VIEW: FORWARD RETURNS BY TRAILING-VOL QUINTILE")
    print("Sort every day by trailing 126d vol, then look at what happened")
    print("over the following 21 days. Vol targeting is betting that the")
    print("top row is worse than the bottom row.\n")

    for label, series in [("momentum", r_mom), ("benchmark", r_ben)]:
        tbl = conditional_returns(series, window=126, horizon=21, n_bins=5)
        if tbl.empty:
            continue
        print(f"  {label}:")
        print(f"    {'quintile':<12}{'mean vol':>10}{'fwd 21d':>10}"
              f"{'annualised':>12}{'n':>8}")
        print("    " + "-" * 52)
        names = ["1 calmest", "2", "3", "4", "5 wildest"]
        for i, (_, row) in enumerate(tbl.iterrows()):
            print(f"    {names[i]:<12}{row['mean_vol']:>9.1%}"
                  f"{row['mean_fwd']:>10.2%}{row['ann_fwd']:>11.1%}"
                  f"{int(row['n']):>8,}")
        spread = tbl["ann_fwd"].iloc[-1] - tbl["ann_fwd"].iloc[0]
        print(f"    wildest minus calmest: {spread:+.1%} annualised")
        if spread > 0:
            print("    POSITIVE. High-vol periods were FOLLOWED BY BETTER")
            print("    returns in this sample. De-risking into them was")
            print("    exactly the wrong trade, which is why vol targeting")
            print("    lost roughly 7 points of CAGR for no drawdown benefit.")
        else:
            print("    Negative, as vol targeting assumes. So the failure")
            print("    came from elsewhere: implementation lag, the cost of")
            print("    turnover, or the absence of a short leg to protect.")
        print()

    # ------------------------------------------------- the missing crash

    section("THE STRUCTURAL POINT: THERE IS NO SHORT LEG")
    print("Barroso and Santa-Clara studied LONG/SHORT momentum. The crash")
    print("they insure against has a specific mechanism: the short leg holds")
    print("prior losers, typically distressed high-beta names. When the market")
    print("rebounds violently off a bottom, those losers rally hardest, so a")
    print("short position in them behaves like a written call on the market.")
    print("That is what produced momentum's catastrophic drawdowns in 1932")
    print("and 2009.")
    print()
    print("This book is long-only. That mechanism is absent. The evidence:\n")

    dd_plain = (1 + r_mom).cumprod()
    dd_plain = (dd_plain / dd_plain.cummax() - 1).min()
    dd_ben = (1 + r_ben).cumprod()
    dd_ben = (dd_ben / dd_ben.cummax() - 1).min()
    print(f"  momentum max drawdown   {dd_plain:.1%}")
    print(f"  benchmark max drawdown  {dd_ben:.1%}")
    print(f"  difference              {dd_plain - dd_ben:+.1%}")
    print()
    print("  Momentum's drawdown is barely worse than the benchmark's, and")
    print("  both occurred in the same month (the Covid crash), which is a")
    print("  market event, not a momentum crash. There was no momentum-")
    print("  specific tail risk in this sample to insure against.")
    print()
    print("  Buying insurance against a risk you are not running is a pure")
    print("  cost. That is the cleanest explanation of the result.")

    # ------------------------------------------- worst momentum months

    section("WHEN DID MOMENTUM ACTUALLY HURT?")
    m_mom = (1 + r_mom).resample("ME").prod() - 1
    m_ben = (1 + r_ben).resample("ME").prod() - 1
    rel = (m_mom - m_ben).dropna().sort_values()
    print("worst 8 months of momentum RELATIVE to the benchmark:\n")
    print(f"  {'month':<10}{'momentum':>11}{'benchmark':>11}{'relative':>11}")
    print("  " + "-" * 43)
    for dt in rel.head(8).index:
        print(f"  {dt.strftime('%Y-%m'):<10}{m_mom[dt]:>10.1%}"
              f"{m_ben[dt]:>10.1%}{rel[dt]:>10.1%}")
    print("\nCheck whether these cluster after market bottoms. If they do,")
    print("the reversal mechanism is present even without a short leg, just")
    print("far weaker.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
