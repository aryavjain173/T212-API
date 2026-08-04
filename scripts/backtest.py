"""
Milestone 3+4: run the backtest.

Run:  python -m scripts.backtest
      python -m scripts.backtest --cost 5 --nlong 10
      python -m scripts.backtest --plot

Reports momentum against an equal-weight benchmark of the same universe,
then sweeps the cost assumption, then sweeps the parameters. Read the
output in that order, because each section is designed to talk you out of
believing the one before it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backtest import (  # noqa: E402
    active_stats,
    cagr,
    compare,
    drawdown_series,
    print_significance,
    print_summary,
    run_backtest,
    sharpe,
    summarise,
    vol_matched,
)
from marketdata import TICKERS, load_prices, sector_of  # noqa: E402
from strategies import (  # noqa: E402
    CrossSectionalMomentum,
    EqualWeightBenchmark,
    ShortTermReversal,
)


def section(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=10.0, help="bps per turnover")
    ap.add_argument("--nlong", type=int, default=12)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--skip", type=int, default=21)
    ap.add_argument("--rebalance", default="M", choices=["D", "W", "M", "Q"])
    ap.add_argument("--start", default=None, help="e.g. 2018-01-01")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    prices, _ = load_prices(TICKERS)
    if args.start:
        prices = prices.loc[args.start:]

    print(f"{prices.shape[1]} names, {prices.index[0].date()} "
          f"-> {prices.index[-1].date()}, {len(prices):,} bars")

    mom = CrossSectionalMomentum(
        lookback=args.lookback,
        skip=args.skip,
        n_long=args.nlong,
        rebalance=args.rebalance,
    )
    bench = EqualWeightBenchmark(rebalance=args.rebalance)
    rev = ShortTermReversal(lookback=5, n_long=args.nlong, rebalance="W")

    results = {}
    for strat in (mom, bench, rev):
        w = strat.generate_weights(prices)
        results[strat.name] = run_backtest(
            prices, w, cost_bps=args.cost, name=strat.name
        )

    # ---------------------------------------------------------- headline

    section(f"MOMENTUM  ({args.cost:.0f}bps costs)")
    s_mom = summarise(results[mom.name])
    print_summary(s_mom)

    section(f"EQUAL-WEIGHT BENCHMARK  ({args.cost:.0f}bps costs)")
    s_bench = summarise(results[bench.name])
    print_summary(s_bench)

    section("SIDE BY SIDE")
    summaries = [summarise(r) for r in results.values()]
    compare(summaries)

    excess = s_mom["cagr"] - s_bench["cagr"]
    print(f"\nmomentum minus benchmark CAGR: {excess:+.2%}")

    # ------------------------------------------------------ significance

    section("IS THE DIFFERENCE REAL?")
    print("Raw CAGR comparisons are the easiest thing in finance to fool")
    print("yourself with. Two questions settle it: is the gap larger than")
    print("the noise, and did it come from skill or just from more risk?\n")

    r_mom = results[mom.name].returns
    r_bench = results[bench.name].returns

    act = active_stats(r_mom, r_bench)
    vm = vol_matched(r_mom, r_bench)
    print_significance(act, vm)

    print("\n  SHARPE UNCERTAINTY")
    print(f"    momentum   {s_mom['sharpe']:.2f}  "
          f"95% CI [{s_mom['sharpe_lo']:.2f}, {s_mom['sharpe_hi']:.2f}]")
    print(f"    benchmark  {s_bench['sharpe']:.2f}  "
          f"95% CI [{s_bench['sharpe_lo']:.2f}, {s_bench['sharpe_hi']:.2f}]")
    lo_hi_overlap = (
        s_mom["sharpe_lo"] <= s_bench["sharpe_hi"]
        and s_bench["sharpe_lo"] <= s_mom["sharpe_hi"]
    )
    if lo_hi_overlap:
        print("    The intervals overlap, so the two Sharpe ratios are not")
        print("    separable on this sample. Note the paired test above is")
        print("    the sharper tool: because the series are correlated, it")
        print("    can detect a difference these wide intervals cannot.")

    yrs_needed = None
    if act and act["information_ratio"] != 0:
        yrs_needed = (2.0 / act["information_ratio"]) ** 2
        print(f"\n    At an IR of {act['information_ratio']:.2f}, reaching")
        print(f"    t = 2.0 would take about {yrs_needed:.0f} years of data.")
        if yrs_needed > 30:
            print("    That is longer than the strategy has plausibly existed")
            print("    in its current form, which is the honest conclusion.")

    # -------------------------------------------------------- cost sweep

    section("COST SENSITIVITY")
    print("If the edge only survives at zero cost, there is no edge.\n")
    w_mom = mom.generate_weights(prices)
    print(f"{'bps':>6}{'CAGR':>10}{'Sharpe':>10}{'vs bench':>11}")
    print("-" * 37)
    for bps in [0, 2, 5, 10, 20, 35, 50]:
        r = run_backtest(prices, w_mom, cost_bps=bps)
        b = run_backtest(
            prices, bench.generate_weights(prices), cost_bps=bps
        )
        print(f"{bps:>6}{cagr(r.returns):>9.2%}{sharpe(r.returns):>10.2f}"
              f"{cagr(r.returns) - cagr(b.returns):>10.2%}")

    # ------------------------------------------------------ param sweep

    section("PARAMETER SENSITIVITY")
    print("A real effect degrades smoothly as you vary parameters.")
    print("A fitted one has a sharp peak at whatever you happened to pick.\n")

    print("holding N names (12-1, monthly):")
    print(f"{'N':>6}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}")
    print("-" * 36)
    for n in [3, 6, 9, 12, 18, 24, 30]:
        st = CrossSectionalMomentum(n_long=n, rebalance=args.rebalance)
        r = run_backtest(prices, st.generate_weights(prices), cost_bps=args.cost)
        s = summarise(r)
        print(f"{n:>6}{s['cagr']:>9.2%}{s['sharpe']:>10.2f}{s['max_dd']:>10.1%}")

    print("\nlookback window (top 12, monthly):")
    print(f"{'days':>6}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}")
    print("-" * 36)
    for lb in [63, 126, 189, 252, 378, 504]:
        st = CrossSectionalMomentum(
            lookback=lb, skip=args.skip, n_long=args.nlong,
            rebalance=args.rebalance,
        )
        r = run_backtest(prices, st.generate_weights(prices), cost_bps=args.cost)
        s = summarise(r)
        print(f"{lb:>6}{s['cagr']:>9.2%}{s['sharpe']:>10.2f}{s['max_dd']:>10.1%}")

    print("\nskip window (does excluding the last month matter?):")
    print(f"{'skip':>6}{'CAGR':>10}{'Sharpe':>10}")
    print("-" * 26)
    for sk in [0, 5, 10, 21, 42]:
        st = CrossSectionalMomentum(
            lookback=args.lookback, skip=sk, n_long=args.nlong,
            rebalance=args.rebalance,
        )
        r = run_backtest(prices, st.generate_weights(prices), cost_bps=args.cost)
        print(f"{sk:>6}{cagr(r.returns):>9.2%}{sharpe(r.returns):>10.2f}")

    # ------------------------------------------------------ worst years

    section("WORST CALENDAR YEARS")
    print("Averages hide the years that would have made you stop.\n")
    yr_mom = (1 + r_mom).resample("YE").prod() - 1
    yr_ben = (1 + r_bench).resample("YE").prod() - 1
    tbl = pd.DataFrame({"momentum": yr_mom, "benchmark": yr_ben}).dropna()
    tbl["excess"] = tbl["momentum"] - tbl["benchmark"]
    for dt, row in tbl.iterrows():
        flag = "  <-- underperformed" if row["excess"] < 0 else ""
        print(f"  {dt.year}  {row['momentum']:>8.1%}  "
              f"{row['benchmark']:>8.1%}  {row['excess']:>+8.1%}{flag}")

    # ------------------------------------------------- current portfolio

    section("CURRENT TARGET PORTFOLIO")
    w_now = w_mom.iloc[-1].dropna()
    w_now = w_now[w_now > 0].sort_values(ascending=False)
    print(f"as of {prices.index[-1].date()}, {len(w_now)} positions\n")
    for tkr, wt in w_now.items():
        print(f"  {tkr:<8} {wt:>7.2%}   {sector_of(tkr)}")

    counts: dict[str, float] = {}
    for tkr, wt in w_now.items():
        counts[sector_of(tkr)] = counts.get(sector_of(tkr), 0.0) + wt
    print("\nsector exposure:")
    for sec, wt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:<14} {wt:>6.1%}  {'#' * int(wt * 40)}")

    if args.plot:
        _plot(results)

    return 0


def _plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, res in results.items():
        ax1.plot(res.equity.index, res.equity.values, label=name, lw=1.2)
        ax2.fill_between(
            res.returns.index, drawdown_series(res.returns).values,
            0, alpha=0.3,
        )
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of 1.0 (log)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "logs" / "backtest.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nchart written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
