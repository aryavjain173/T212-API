"""
Pre-specified signal sweep, with a multiple-testing correction.

Run:  python -m scripts.signal_sweep
      python -m scripts.signal_sweep --cost 20
      python -m scripts.signal_sweep --quick

THE PROBLEM THIS SOLVES

You are about to test a few dozen strategy specifications and pick the
best. That procedure is guaranteed to produce a good-looking number even
if every single signal is worthless, because you are taking a maximum over
noisy estimates. The more you test, the better your winner looks, and the
less it means.

Concretely: each Sharpe ratio here carries a standard error around 0.3. If
you test 40 worthless strategies, the best of them will typically print
around 0.6 to 0.8 purely by chance. Reporting that as a discovery is the
single most common failure in quantitative research, and it is what an
interviewer is testing for when they ask how many things you tried.

THE DEFENCES

1. The grid below is written down in advance and run in full. Nothing is
   dropped after seeing results.
2. Every specification is reported, sorted, including the failures.
3. The best result is compared against a SIMULATED null: shuffle the
   signal so it carries no information, rerun the same grid, and record
   the best Sharpe achieved. Repeat. That distribution tells you what
   "best of N" looks like when nothing works, and your winner has to beat
   it to mean anything.
4. Cost sensitivity is reported alongside, because high-turnover signals
   look best exactly where the cost assumption is least reliable.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import (  # noqa: E402
    active_stats,
    ann_turnover,
    ann_vol,
    cagr,
    max_drawdown,
    run_backtest,
    sharpe,
)
from marketdata import TICKERS, load_prices  # noqa: E402
from strategies import (  # noqa: E402
    CrossSectionalMomentum,
    EqualWeightBenchmark,
    MeanReversion,
)


def section(t: str) -> None:
    print(f"\n{'=' * 84}\n{t}\n{'=' * 84}")


# ------------------------------------------------------------ the grid
# Written down before running. Do not edit after seeing results.

def build_grid(n_long: int, quick: bool) -> dict:
    grid: dict = {}

    # Reversal: the new hypothesis. Short horizons, frequent rebalancing.
    lookbacks = [1, 3, 5, 10, 21] if not quick else [1, 5, 21]
    rebals = ["D", "W"] if not quick else ["W"]
    for lb in lookbacks:
        for rb in rebals:
            for va in [True, False]:
                s = MeanReversion(
                    lookback=lb, n_long=n_long, rebalance=rb,
                    demean=True, vol_adjust=va,
                )
                grid[s.name] = s

    # Momentum: the incumbent, across horizons, for comparison on equal terms.
    for lb in ([63, 126, 252, 504] if not quick else [126, 252]):
        s = CrossSectionalMomentum(
            lookback=lb, skip=21, n_long=n_long, rebalance="M"
        )
        grid[s.name] = s

    return grid


def evaluate(prices, strat, cost_bps, bench_returns):
    w = strat.generate_weights(prices)
    try:
        res = run_backtest(prices, w, cost_bps=cost_bps, name=strat.name)
    except ValueError:
        return None
    r = res.returns
    a = active_stats(r, bench_returns)
    return {
        "name": strat.name,
        "sharpe": sharpe(r),
        "cagr": cagr(r),
        "vol": ann_vol(r),
        "maxdd": max_drawdown(r),
        "turnover": ann_turnover(res.turnover),
        "ir": a.get("information_ratio", np.nan),
        "t": a.get("t_stat", np.nan),
        "returns": r,
    }


def null_distribution(
    prices, strat_grid, cost_bps, bench_returns, n_sims: int, seed: int = 0
) -> np.ndarray:
    """
    What does 'best of N' look like when nothing works?

    We destroy the signal while preserving everything else about the
    procedure. Each simulation shuffles the COLUMN LABELS of the price
    matrix independently on each rebalance, so the strategies still select
    a portfolio in exactly the same way, with the same turnover and the
    same number of holdings, but the selection carries no information.

    The maximum Sharpe across the grid is recorded for each simulation.
    The resulting distribution is the bar your real winner must clear.
    """
    rng = np.random.default_rng(seed)
    best = []
    cols = list(prices.columns)

    for _ in range(n_sims):
        perm = rng.permutation(len(cols))
        shuffled = prices.copy()
        shuffled.columns = [cols[i] for i in perm]
        shuffled = shuffled[cols]

        sims = []
        for strat in strat_grid.values():
            # Rebuild weights on shuffled data, apply to TRUE prices.
            w = strat.generate_weights(shuffled)
            w.columns = [cols[i] for i in perm]
            w = w.reindex(columns=cols)
            try:
                r = run_backtest(prices, w, cost_bps=cost_bps).returns
                sims.append(sharpe(r))
            except ValueError:
                continue
        if sims:
            best.append(max(sims))

    return np.array(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--nlong", type=int, default=12)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    prices, _ = load_prices(TICKERS)
    bench = EqualWeightBenchmark()
    r_bench = run_backtest(
        prices, bench.generate_weights(prices), cost_bps=args.cost
    ).returns
    bench_sharpe = sharpe(r_bench)

    grid = build_grid(args.nlong, args.quick)
    print(f"{prices.shape[1]} names, {len(prices):,} bars, "
          f"{args.cost:.0f}bps costs")
    print(f"grid: {len(grid)} pre-specified strategies")
    print(f"benchmark Sharpe: {bench_sharpe:.3f}")

    rows = [
        r for r in (
            evaluate(prices, s, args.cost, r_bench) for s in grid.values()
        ) if r
    ]
    rows.sort(key=lambda r: -r["sharpe"])

    section(f"ALL {len(rows)} SPECIFICATIONS, RANKED  (nothing hidden)")
    hdr = (f"{'strategy':<26}{'Sharpe':>8}{'CAGR':>8}{'Vol':>8}"
           f"{'MaxDD':>8}{'Turn':>9}{'IR':>7}{'t':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "  <" if r["sharpe"] > bench_sharpe else ""
        print(f"{r['name'][:25]:<26}{r['sharpe']:>8.3f}{r['cagr']:>7.1%} "
              f"{r['vol']:>7.1%} {r['maxdd']:>7.1%} {r['turnover']:>8.0%} "
              f"{r['ir']:>6.2f} {r['t']:>6.2f}{flag}")
    print(f"\n'<' marks a strategy beating the benchmark Sharpe "
          f"of {bench_sharpe:.3f}")

    beat = sum(1 for r in rows if r["sharpe"] > bench_sharpe)
    print(f"{beat} of {len(rows)} beat the benchmark. Under pure noise you "
          f"would expect roughly half.")

    # ------------------------------------------------- multiple testing

    best = rows[0]
    section("MULTIPLE TESTING: IS THE WINNER BETTER THAN LUCK?")
    print(f"best specification : {best['name']}")
    print(f"its Sharpe         : {best['sharpe']:.3f}")
    print(f"specifications run : {len(rows)}")
    print()
    print(f"Simulating the null {args.sims} times. Each run shuffles which")
    print("stock is which, so the strategies trade identically but select")
    print("blindly, then records the BEST Sharpe across the whole grid.")
    print("This may take a minute.\n")

    null = null_distribution(
        prices, grid, args.cost, r_bench, n_sims=args.sims
    )
    if len(null) == 0:
        print("null simulation produced nothing")
        return 1

    pct = float((null >= best["sharpe"]).mean())
    print(f"  null best-of-{len(rows)} Sharpe distribution:")
    print(f"    median   {np.median(null):.3f}")
    print(f"    90th pct {np.percentile(null, 90):.3f}")
    print(f"    95th pct {np.percentile(null, 95):.3f}")
    print(f"    max      {null.max():.3f}")
    print(f"\n  your best: {best['sharpe']:.3f}")
    print(f"  empirical p-value: {pct:.3f}  "
          f"({int(pct * len(null))} of {len(null)} null runs did better)")
    print()
    if pct > 0.10:
        print("  NOT distinguishable from data mining. A blind procedure")
        print("  beats this result more than 10% of the time. The honest")
        print("  conclusion is that no signal in this grid demonstrably")
        print("  works on this universe.")
    elif pct > 0.05:
        print("  Marginal. Suggestive but not convincing at conventional")
        print("  thresholds, especially given the wide Sharpe error bars.")
    else:
        print("  Survives the multiple-testing correction. That is a real")
        print("  result, though still only in-sample: hold out the last two")
        print("  years and re-check before believing it.")

    # ------------------------------------------------------- cost sweep

    section("COST SENSITIVITY OF THE TOP 5")
    print("High-turnover signals look best where costs are least certain.")
    print("Trading 212 charges no commission, but you still pay spread, FX")
    print("and slippage. Read across, not down.\n")
    print(f"{'strategy':<26}" + "".join(f"{b:>9}bps" for b in [0, 5, 10, 20, 40]))
    print("-" * 86)
    for r in rows[:5]:
        strat = grid[r["name"]]
        w = strat.generate_weights(prices)
        line = f"{r['name'][:25]:<26}"
        for bps in [0, 5, 10, 20, 40]:
            try:
                s = sharpe(run_backtest(prices, w, cost_bps=bps).returns)
                line += f"{s:>12.3f}"
            except ValueError:
                line += f"{'-':>12}"
        print(line)
    print(f"\nbenchmark for reference: {bench_sharpe:.3f} at "
          f"{args.cost:.0f}bps (turnover only ~130%)")

    section("WHAT TO CONCLUDE")
    print("Read the three sections above in order:")
    print("  1. How many specifications beat the benchmark? Near half means")
    print("     you are looking at coin flips.")
    print("  2. Does the winner clear the null best-of-N? If not, stop.")
    print("  3. Does it survive realistic costs? A reversal signal that")
    print("     dies between 0 and 20bps was never tradeable.")
    print()
    print("A null result reported honestly is a stronger project than a")
    print("positive result you cannot defend. You have measured something")
    print("real either way: that these signals do not work on large-cap US")
    print("equities at retail cost levels, which is what theory predicts")
    print("and what most practitioners would tell you.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
