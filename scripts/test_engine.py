"""
Engine tests. Run:  python -m scripts.test_engine

Every test here checks the engine against an answer computed independently,
usually by hand. A backtester that isn't tested this way is just a number
generator, and the numbers it generates will be wrong in the direction you
were hoping for.

These are the tests you should be able to describe if someone asks how you
know your backtest is right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import run_backtest, sharpe, max_drawdown, cagr, ann_turnover  # noqa: E402
from strategies import CrossSectionalMomentum, EqualWeightBenchmark  # noqa: E402

PASS, FAIL = 0, 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}" + (f"   {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {label}   {detail}")


def bdays(n: int):
    return pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n).values)


def section(t: str) -> None:
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


# ---------------------------------------------------------------- tests


def test_buy_and_hold():
    """One asset compounding at a known rate. The engine must reproduce it."""
    section("1. BUY AND HOLD REPRODUCES A KNOWN COMPOUND RETURN")
    n = 100
    idx = bdays(n)
    daily = 0.001
    px = pd.DataFrame({"A": 100 * (1 + daily) ** np.arange(n)}, index=idx)

    w = pd.DataFrame(1.0, index=idx, columns=["A"])
    res = run_backtest(px, w, cost_bps=0.0, lag=1)

    # Held from day 2 onward (lag 1, first target on day 0 -> held day 1,
    # earns return from day 2). Compare to the analytical figure.
    expected = (1 + daily) ** (n - 2) - 1
    actual = (1 + res.returns).prod() - 1
    check(
        "compound return matches analytical",
        abs(actual - expected) < 1e-9,
        f"got {actual:.8f}, expected {expected:.8f}",
    )
    check("no costs when cost_bps=0", res.costs.sum() == 0)


def test_equal_weight_matches_mean():
    """An equal-weight book must return the mean of its constituents."""
    section("2. EQUAL WEIGHT EQUALS THE CROSS-SECTIONAL MEAN RETURN")
    np.random.seed(7)
    n, k = 300, 5
    idx = bdays(n)
    rets = pd.DataFrame(
        np.random.normal(0.0005, 0.01, (n, k)),
        index=idx, columns=list("ABCDE"),
    )
    px = 100 * (1 + rets).cumprod()

    # Rebalance daily so the book is equal-weight every day, no drift.
    w = pd.DataFrame(1.0 / k, index=idx, columns=px.columns)
    res = run_backtest(px, w, cost_bps=0.0, lag=1)

    px_rets = px.pct_change(fill_method=None)
    aligned = px_rets.mean(axis=1).reindex(res.returns.index)

    # Skip the first row: that's the entry day, when the book is still
    # empty and correctly returns zero. Everything after must match to
    # floating-point precision.
    diff = (res.returns - aligned).abs().iloc[1:].max()
    check(
        "daily return equals mean of constituents",
        diff < 1e-12,
        f"max abs diff {diff:.2e} over {len(res.returns) - 1} days",
    )
    check(
        "entry day correctly returns zero (no position yet)",
        res.returns.iloc[0] == 0.0,
    )


def test_turnover_accounting():
    """A known switch between two assets must produce known turnover."""
    section("3. TURNOVER AND COSTS ARE ACCOUNTED CORRECTLY")
    n = 10
    idx = bdays(n)
    # Flat prices, so no drift. Any turnover is purely from trading.
    px = pd.DataFrame(100.0, index=idx, columns=["A", "B"])

    w = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    w.iloc[:5, 0] = 1.0   # hold A
    w.iloc[5:, 1] = 1.0   # switch entirely to B

    res = run_backtest(px, w, cost_bps=100.0, lag=1)  # 100bps = 1%

    # Two events: initial entry into A (0 -> 1, traded 1.0) and the switch
    # A -> B (sell 1, buy 1, traded 2.0). One-way turnover 0.5 and 1.0.
    total_one_way = res.turnover.sum()
    check(
        "one-way turnover is 1.5 (0.5 entry + 1.0 switch)",
        abs(total_one_way - 1.5) < 1e-12,
        f"got {total_one_way}",
    )

    # Cost = traded notional x rate. Traded = 1.0 + 2.0 = 3.0 at 1%.
    check(
        "total cost is 3.0%",
        abs(res.costs.sum() - 0.03) < 1e-12,
        f"got {res.costs.sum():.6f}",
    )

    # With flat prices, gross return must be exactly zero and net must be
    # exactly the cost.
    check("gross return is zero on flat prices", abs(res.gross_returns.sum()) < 1e-12)
    check(
        "net return is exactly minus the costs",
        abs(res.returns.sum() + 0.03) < 1e-12,
        f"got {res.returns.sum():.6f}",
    )


def test_drift_is_modelled():
    """Between rebalances the book must drift, not stay pinned to target."""
    section("4. PORTFOLIO DRIFT IS MODELLED")
    n = 30
    idx = bdays(n)
    px = pd.DataFrame(
        {
            "UP": 100 * 1.02 ** np.arange(n),   # rises 2% a day
            "FLAT": np.full(n, 100.0),
        },
        index=idx,
    )
    # Set a 50/50 target once, then never again.
    w = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    w.iloc[0] = [0.5, 0.5]
    w = w.ffill()
    w.iloc[1:] = np.nan   # only one instruction, on day 0

    res = run_backtest(px, w, cost_bps=0.0, lag=1)
    final = res.weights.iloc[-1]

    check(
        "winner's weight grew above 50%",
        final["UP"] > 0.6,
        f"UP ended at {final['UP']:.1%}",
    )
    check(
        "weights still sum to 1",
        abs(final.sum() - 1.0) < 1e-12,
        f"sum {final.sum():.10f}",
    )
    check(
        "no turnover after the single instruction",
        res.turnover.iloc[2:].sum() < 1e-12,
    )


def test_lag_prevents_lookahead():
    """
    Build a strategy that genuinely peeks at the future: it holds whichever
    name will go up most TOMORROW. With lag=0 that foresight pays off and
    the Sharpe should be absurd. With lag=1 the peek is shifted out of
    reach and the edge must vanish completely.
    """
    section("5. THE EXECUTION LAG ACTUALLY PREVENTS LOOKAHEAD")
    np.random.seed(11)
    n, k = 500, 6
    idx = bdays(n)
    rets = pd.DataFrame(
        np.random.normal(0, 0.02, (n, k)), index=idx, columns=list("ABCDEF")
    )
    px = 100 * (1 + rets).cumprod()

    # THE CHEAT: row i holds tomorrow's best performer.
    px_rets = px.pct_change(fill_method=None)
    tomorrow = px_rets.shift(-1)
    best = tomorrow.rank(axis=1, ascending=False, method="first") == 1
    w = best.astype(float)

    r0 = run_backtest(px, w, cost_bps=0.0, lag=0)
    r1 = run_backtest(px, w, cost_bps=0.0, lag=1)

    s0, s1 = sharpe(r0.returns), sharpe(r1.returns)
    check(
        "lag 0 produces an impossible Sharpe (cheating detected)",
        s0 > 5,
        f"lag0 Sharpe {s0:.2f}",
    )
    check(
        "lag 1 removes the edge entirely",
        s1 < 1.5,
        f"lag1 Sharpe {s1:.2f}",
    )


def test_metrics_against_hand_calcs():
    section("6. METRICS MATCH HAND CALCULATIONS")
    # A series with a known drawdown: +10%, -50%, +10%
    r = pd.Series([0.10, -0.50, 0.10], index=bdays(3))
    # equity: 1.1, 0.55, 0.605. peak 1.1, trough 0.55 -> dd = -50%
    check(
        "max drawdown is -50%",
        abs(max_drawdown(r) - (-0.50)) < 1e-12,
        f"got {max_drawdown(r):.4f}",
    )

    # Constant returns -> zero vol -> Sharpe defined as 0, not inf/nan
    flat = pd.Series([0.001] * 50, index=bdays(50))
    check("Sharpe of a constant series is 0, not nan", sharpe(flat) == 0.0)

    # CAGR of exactly doubling over exactly one trading year
    daily = 2 ** (1 / 252) - 1
    yr = pd.Series([daily] * 252, index=bdays(252))
    check(
        "CAGR of a doubling year is 100%",
        abs(cagr(yr) - 1.0) < 1e-9,
        f"got {cagr(yr):.6f}",
    )

    # Turnover annualisation: 0.5 per day over 252 days -> 126x
    t = pd.Series([0.5] * 252, index=bdays(252))
    check(
        "annualised turnover is 126x",
        abs(ann_turnover(t) - 126.0) < 1e-9,
        f"got {ann_turnover(t):.4f}",
    )


def test_strategy_shapes():
    section("7. STRATEGIES PRODUCE VALID WEIGHTS")
    np.random.seed(3)
    n, k = 900, 30
    idx = bdays(n)
    cols = [f"T{i:02d}" for i in range(k)]
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, (n, k)), axis=0)),
        index=idx, columns=cols,
    )

    strat = CrossSectionalMomentum(lookback=252, skip=21, n_long=6, rebalance="M")
    w = strat.generate_weights(px)

    live = w.dropna(how="all")
    sums = live.sum(axis=1)
    check(
        "long-only weights sum to 1",
        (sums.sub(1.0).abs() < 1e-9).all(),
        f"range {sums.min():.6f} - {sums.max():.6f}",
    )
    check("no negative weights in long-only", (live >= -1e-12).all().all())
    check(
        "holds exactly 6 names",
        ((live > 0).sum(axis=1) == 6).all(),
        f"counts {sorted((live > 0).sum(axis=1).unique())}",
    )
    check(
        "warmup rows are blank (no signal before 273 bars)",
        w.iloc[:272].isna().all().all(),
    )

    # Rebalance discipline: weights should only CHANGE on month ends.
    changes = (live.diff().abs().sum(axis=1) > 1e-9)
    change_months = live.index[changes].to_period("M")
    check(
        "weights change at most once per month",
        change_months.value_counts().max() <= 1,
        f"max changes in a month: {change_months.value_counts().max()}",
    )

    bench = EqualWeightBenchmark().generate_weights(px)
    bl = bench.dropna(how="all")
    check(
        "benchmark weights sum to 1",
        (bl.sum(axis=1).sub(1.0).abs() < 1e-9).all(),
    )


def test_cost_sensitivity_direction():
    section("8. HIGHER COSTS ALWAYS REDUCE RETURNS")
    np.random.seed(5)
    n, k = 900, 20
    idx = bdays(n)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, (n, k)), axis=0)),
        index=idx, columns=[f"T{i:02d}" for i in range(k)],
    )
    w = CrossSectionalMomentum(n_long=5).generate_weights(px)

    prev = None
    ok = True
    for bps in [0, 5, 10, 25, 50]:
        c = cagr(run_backtest(px, w, cost_bps=bps).returns)
        if prev is not None and c > prev + 1e-12:
            ok = False
        prev = c
    check("CAGR is monotonically decreasing in cost", ok)


def main() -> int:
    test_buy_and_hold()
    test_equal_weight_matches_mean()
    test_turnover_accounting()
    test_drift_is_modelled()
    test_lag_prevents_lookahead()
    test_metrics_against_hand_calcs()
    test_strategy_shapes()
    test_cost_sensitivity_direction()

    print(f"\n{'=' * 66}")
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 66)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
