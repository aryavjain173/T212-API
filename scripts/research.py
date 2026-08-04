"""
Milestone 4b: does risk management rescue momentum?

Run:  python -m scripts.research
      python -m scripts.research --plot
      python -m scripts.research --target 0.17 --cost 10

THE QUESTION

The plain momentum backtest produced a clear negative: higher raw return,
but lower Sharpe than equal weighting, and once volatility-matched the
levered benchmark won outright. The active return was statistically
indistinguishable from zero.

Barroso and Santa-Clara (2015) argued the fix is not better selection but
better risk management: momentum's volatility is forecastable even when
its return is not, so scale exposure by inverse realised volatility. This
script tests that claim on your universe.

HOW TO READ THE OUTPUT

Two refinements are tested SEPARATELY and then together, so you can
attribute any improvement to a mechanism:

  IVMom  changes how the held names are weighted (composition)
  VT     changes how large the book is (exposure)

The default volatility target is set to the BENCHMARK's realised
volatility. That makes the comparison apples-to-apples by construction:
every strategy is then being asked to deliver return at the same risk
level, so raw CAGR becomes directly comparable without any levering
argument.

Be prepared for this to fail. Vol targeting reliably improves Sharpe in
long/short momentum, where crashes are severe. In a long-only book over a
decade-long bull market it may do very little, because the crash risk it
protects against barely materialised. That would be a real finding too.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backtest import (  # noqa: E402
    active_stats,
    ann_vol,
    cagr,
    drawdown_series,
    max_drawdown,
    run_backtest,
    sharpe,
    sharpe_ci,
    summarise,
    vol_matched,
)
from marketdata import TICKERS, load_prices, sector_of  # noqa: E402
from strategies import (  # noqa: E402
    CrossSectionalMomentum,
    EqualWeightBenchmark,
    InverseVolMomentum,
    VolTargeted,
)


def section(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--nlong", type=int, default=12)
    ap.add_argument("--target", type=float, default=None,
                    help="vol target; default = benchmark realised vol")
    ap.add_argument("--volwindow", type=int, default=126)
    ap.add_argument("--maxlev", type=float, default=1.0)
    ap.add_argument("--rf", type=float, default=0.0,
                    help="annual rate earned on the cash leg")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    prices, _ = load_prices(TICKERS)
    print(f"{prices.shape[1]} names, {prices.index[0].date()} -> "
          f"{prices.index[-1].date()}, {len(prices):,} bars")

    bench = EqualWeightBenchmark()
    r_bench = run_backtest(
        prices, bench.generate_weights(prices),
        cost_bps=args.cost, name="EqualWeight", rf=args.rf,
    )
    bench_vol = ann_vol(r_bench.returns)

    target = args.target if args.target is not None else bench_vol
    print(f"benchmark realised vol {bench_vol:.2%}  ->  vol target {target:.2%}")
    if args.maxlev <= 1.0:
        print("leverage capped at 1.0x: the book de-risks into cash but "
              "never borrows.")

    mom = CrossSectionalMomentum(n_long=args.nlong)
    ivm = InverseVolMomentum(n_long=args.nlong)

    variants = {
        "EqualWeight (bench)": bench,
        "Momentum": mom,
        "Momentum + invvol wts": ivm,
        "Momentum + vol target": VolTargeted(
            mom, target_vol=target, vol_window=args.volwindow,
            max_leverage=args.maxlev,
        ),
        "Momentum + both": VolTargeted(
            ivm, target_vol=target, vol_window=args.volwindow,
            max_leverage=args.maxlev,
        ),
    }

    results = {}
    for label, strat in variants.items():
        w = strat.generate_weights(prices)
        results[label] = run_backtest(
            prices, w, cost_bps=args.cost, name=label, rf=args.rf
        )

    # ------------------------------------------------------------- table

    section(f"RESULTS  ({args.cost:.0f}bps costs, target vol {target:.1%})")
    hdr = (f"{'':<24}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'95% CI':>16}"
           f"{'MaxDD':>8}{'Turn':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, res in results.items():
        s = summarise(res)
        lo, hi = sharpe_ci(res.returns)
        print(
            f"{label:<24}{s['cagr']:>7.1%} {s['ann_vol']:>7.1%} "
            f"{s['sharpe']:>7.2f} [{lo:>5.2f},{hi:>5.2f}] "
            f"{s['max_dd']:>7.1%} {s['ann_turnover']:>7.0%}"
        )

    # ------------------------------------------------- attribution

    section("ATTRIBUTION: WHICH CHANGE DID THE WORK?")
    base_sh = sharpe(results["Momentum"].returns)
    print(f"plain momentum Sharpe: {base_sh:.3f}\n")
    print(f"{'change':<26}{'Sharpe':>9}{'delta':>9}{'vol':>9}{'CAGR':>9}")
    print("-" * 62)
    for label in ["Momentum + invvol wts", "Momentum + vol target",
                  "Momentum + both"]:
        r = results[label].returns
        print(f"{label:<26}{sharpe(r):>9.3f}{sharpe(r) - base_sh:>+9.3f}"
              f"{ann_vol(r):>8.1%}{cagr(r):>9.1%}")

    iv_d = sharpe(results["Momentum + invvol wts"].returns) - base_sh
    vt_d = sharpe(results["Momentum + vol target"].returns) - base_sh
    both_d = sharpe(results["Momentum + both"].returns) - base_sh
    print(f"\nadditivity check: {iv_d:+.3f} + {vt_d:+.3f} = {iv_d + vt_d:+.3f} "
          f"vs combined {both_d:+.3f}")
    if abs((iv_d + vt_d) - both_d) > 0.1:
        print("The two effects are NOT additive, so they interact. That is")
        print("expected: inverse-vol weighting already lowers portfolio vol,")
        print("which leaves the vol targeter less work to do.")

    # ----------------------------------------------- vs benchmark tests

    section("EACH VARIANT vs THE EQUAL-WEIGHT BENCHMARK")
    rb = results["EqualWeight (bench)"].returns
    print(f"{'strategy':<24}{'active':>9}{'IR':>7}{'t':>7}{'p':>8}"
          f"{'volmatch edge':>15}")
    print("-" * 70)
    for label in list(variants)[1:]:
        r = results[label].returns
        a = active_stats(r, rb)
        vm = vol_matched(r, rb)
        if not a:
            continue
        print(
            f"{label:<24}{a['active_return']:>+8.2%}"
            f"{a['information_ratio']:>7.2f}{a['t_stat']:>7.2f}"
            f"{a['p_value']:>8.3f}{vm['edge_vs_levered']:>+14.2%}"
        )
    print("\n'volmatch edge' levers the benchmark to the strategy's own vol")
    print("and compares. Negative means the strategy lost to simply taking")
    print("the same risk more cheaply. That is the column that matters.")

    # --------------------------------------------------- exposure path

    section("VOL TARGETING: WHAT THE EXPOSURE ACTUALLY DID")
    vt = variants["Momentum + vol target"]
    exp = vt.exposure(prices).dropna()
    print(f"exposure range   {exp.min():.2f} - {exp.max():.2f}")
    print(f"mean exposure    {exp.mean():.2f}")
    print(f"days below 0.75  {(exp < 0.75).mean():.1%}")
    print(f"days at the cap  {(exp >= args.maxlev - 1e-9).mean():.1%}")

    print("\nmean exposure by calendar year:")
    for yr, v in exp.groupby(exp.index.year).mean().items():
        bar = "#" * int(v * 30)
        print(f"  {yr}  {v:.2f}  {bar}")
    print("\nLow readings mark periods the model judged risky. Check they")
    print("line up with events you recognise, and note the lag: the scalar")
    print("responds AFTER volatility rises, so the first hit is taken at")
    print("full size. Vol targeting softens crashes, it does not dodge them.")

    # ------------------------------------------------------ worst years

    section("CALENDAR YEARS")
    tbl = pd.DataFrame({
        lbl: (1 + res.returns).resample("YE").prod() - 1
        for lbl, res in results.items()
    }).dropna()
    cols = list(tbl.columns)
    print(f"{'year':<6}" + "".join(f"{c[:14]:>16}" for c in cols))
    print("-" * (6 + 16 * len(cols)))
    for dt, row in tbl.iterrows():
        print(f"{dt.year:<6}" + "".join(f"{row[c]:>15.1%} " for c in cols))

    section("DRAWDOWN COMPARISON")
    for label, res in results.items():
        dd = drawdown_series(res.returns)
        print(f"  {label:<24} worst {max_drawdown(res.returns):>7.1%}   "
              f"on {dd.idxmin().date()}")

    # --------------------------------------------------------- portfolio

    section("CURRENT TARGET PORTFOLIO  (momentum + both)")
    w = variants["Momentum + both"].generate_weights(prices).iloc[-1].dropna()
    w = w[w > 0].sort_values(ascending=False)
    invested = w.sum()
    print(f"as of {prices.index[-1].date()}\n")
    print(f"  invested  {invested:>6.1%}")
    print(f"  cash      {1 - invested:>6.1%}\n")
    for tkr, wt in w.items():
        print(f"  {tkr:<8} {wt:>7.2%}   {sector_of(tkr)}")

    if args.plot:
        _plot(results, vt.exposure(prices))

    return 0


def _plot(results: dict, exposure: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3, 1]},
    )
    for label, res in results.items():
        axes[0].plot(res.equity.index, res.equity.values, label=label, lw=1.2)
        axes[1].plot(
            res.returns.index, drawdown_series(res.returns).values, lw=0.8
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("growth of 1.0 (log)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(alpha=0.3)
    axes[2].fill_between(exposure.index, exposure.values, alpha=0.5)
    axes[2].set_ylabel("vol-target\nexposure")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "logs" / "research.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nchart written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
