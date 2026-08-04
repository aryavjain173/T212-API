"""
Tests for the risk-managed strategies.

Run:  python -m scripts.test_strategies

The centrepiece is the causality test. Performance-based lookahead checks
are weak: I tried one while building this and it was insensitive, because
a 126-day rolling volatility shifted by one day is 99.99% correlated with
itself, so the shift barely moves the numbers even though it is the
correct thing to do.

The causality test is definitive instead of suggestive. Compute weights on
the full price history, then recompute on the history truncated at some
earlier date T. If the strategy only uses past data, every weight up to T
must be BIT-IDENTICAL between the two runs. If any weight changes, the
strategy was using information from after T, and no amount of plausible
Sharpe ratio can rescue it.

This is the test to describe if someone asks how you know your backtest
isn't peeking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import ann_vol, run_backtest, sharpe  # noqa: E402
from strategies import (  # noqa: E402
    CrossSectionalMomentum,
    EqualWeightBenchmark,
    InverseVolMomentum,
    VolTargeted,
)

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}" + (f"   {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {label}   {detail}")


def section(t: str) -> None:
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


def make_prices(n=2500, k=30, seed=4, regime=True):
    idx = pd.DatetimeIndex(pd.bdate_range("2014-01-01", periods=n).values)
    if regime:
        sd = np.where(np.arange(n)[:, None] < n // 2, 0.008, 0.030)
    else:
        sd = 0.015
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 1, (n, k)) * sd
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)),
        index=idx, columns=[f"T{i:02d}" for i in range(k)],
    )


# ------------------------------------------------------------- the tests


def test_causality():
    """
    The definitive lookahead test. Truncate the data and check nothing in
    the past changes.
    """
    section("1. CAUSALITY: PAST WEIGHTS MUST NOT DEPEND ON FUTURE DATA")
    px = make_prices()
    cut = px.index[1800]

    base = CrossSectionalMomentum(n_long=8)
    strategies = {
        "CrossSectionalMomentum": base,
        "EqualWeightBenchmark": EqualWeightBenchmark(),
        "InverseVolMomentum": InverseVolMomentum(n_long=8),
        "VolTargeted": VolTargeted(base, target_vol=0.08, max_leverage=1.0),
    }

    # Exclude the cut date itself. In the truncated run it is the last
    # available bar, so a month-end rebalance correctly fires there; in
    # the full run it is an ordinary mid-month day. That difference is an
    # artefact of truncation, not lookahead, and live running always sees
    # the truncated case.
    for name, strat in strategies.items():
        full = strat.generate_weights(px).loc[:cut].iloc[:-1]
        trunc = strat.generate_weights(px.loc[:cut])
        aligned = trunc.reindex_like(full)
        both_nan = full.isna() & aligned.isna()
        equal = ((full - aligned).abs() < 1e-12) | both_nan
        n_diff = int((~equal).sum().sum())
        check(
            f"{name} is causal",
            n_diff == 0,
            f"{n_diff} weights changed when future data was removed",
        )


def test_causality_catches_a_cheat():
    """
    Prove the test above actually works, by deliberately breaking a
    strategy and confirming it gets caught. A test that never fails is
    not a test.
    """
    section("2. THE CAUSALITY TEST CATCHES A DELIBERATE CHEAT")
    px = make_prices()
    cut = px.index[1800]
    base = CrossSectionalMomentum(n_long=8)

    class Peeker(VolTargeted):
        def _scalar(self, port_rets):
            # Centred window: sees 63 days into the future.
            fut = port_rets.rolling(126, center=True).std() * np.sqrt(252)
            sc = (self.target_vol / fut.replace(0.0, np.nan)).clip(
                0, self.max_leverage
            )
            reb = self._rebalance_dates(port_rets.index, self.scalar_rebalance)
            return sc.where(reb).ffill()

    cheat = Peeker(base, target_vol=0.08, max_leverage=1.0)
    full = cheat.generate_weights(px).loc[:cut].iloc[:-1]
    trunc = cheat.generate_weights(px.loc[:cut]).reindex_like(full)
    both_nan = full.isna() & trunc.isna()
    equal = ((full - trunc).abs() < 1e-12) | both_nan
    n_diff = int((~equal).sum().sum())
    check(
        "a forward-looking scalar IS detected",
        n_diff > 0,
        f"{n_diff} weights changed, as they should",
    )


def test_vol_targeting_hits_its_target():
    section("3. VOL TARGETING ACTUALLY CONTROLS VOLATILITY")
    px = make_prices()
    base = CrossSectionalMomentum(n_long=8)
    unscaled = run_backtest(px, base.generate_weights(px), cost_bps=10)
    base_vol = ann_vol(unscaled.returns)

    prev = None
    monotone = True
    for tgt in [0.06, 0.08, 0.10]:
        vt = VolTargeted(base, target_vol=tgt, max_leverage=1.0)
        r = run_backtest(px, vt.generate_weights(px), cost_bps=10)
        v = ann_vol(r.returns)
        check(
            f"target {tgt:.0%} lands below unscaled ({base_vol:.1%})",
            v < base_vol,
            f"realised {v:.1%}",
        )
        if prev is not None and v < prev - 1e-9:
            monotone = False
        prev = v
    check("realised vol rises with the target", monotone)


def test_derisking_in_high_vol_regime():
    section("4. EXPOSURE FALLS WHEN VOLATILITY RISES")
    px = make_prices(regime=True)
    mid = px.index[len(px) // 2]
    base = CrossSectionalMomentum(n_long=8)
    vt = VolTargeted(base, target_vol=0.08, max_leverage=1.0)
    e = vt.exposure(px).dropna()

    calm = e.loc[:mid].mean()
    wild = e.loc[mid:].mean()
    check(
        "mean exposure is lower in the volatile regime",
        wild < calm,
        f"calm {calm:.2f} -> volatile {wild:.2f}",
    )


def test_no_leverage_cap_respected():
    section("5. THE LEVERAGE CAP IS HONOURED")
    px = make_prices(regime=False)
    base = CrossSectionalMomentum(n_long=8)

    for cap in [1.0, 1.5]:
        vt = VolTargeted(base, target_vol=0.60, max_leverage=cap)
        w = vt.generate_weights(px).dropna(how="all")
        mx = float(w.sum(axis=1).max())
        check(
            f"exposure never exceeds {cap:g}x",
            mx <= cap + 1e-9,
            f"max exposure {mx:.4f}",
        )


def test_cash_leg_is_not_renormalised():
    """
    The engine bug this whole milestone nearly hid: if drift renormalises
    a partially invested book back to 1.0, vol targeting silently does
    nothing at all.
    """
    section("6. A PARTIALLY INVESTED BOOK STAYS PARTIALLY INVESTED")
    n = 60
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n).values)
    px = pd.DataFrame({"A": 100 * 1.01 ** np.arange(n)}, index=idx)

    w = pd.DataFrame(0.5, index=idx, columns=["A"])   # half invested, half cash
    res = run_backtest(px, w, cost_bps=0.0, lag=1)

    # Half the book in an asset rising 1% a day must return ~0.5% a day,
    # not 1%. If the engine renormalises, we'd see the full 1%.
    mean_ret = res.returns.iloc[2:].mean()
    check(
        "half-invested book earns about half the return",
        abs(mean_ret - 0.005) < 5e-4,
        f"mean daily {mean_ret:.5f}, expected ~0.00500",
    )

    # And with a cash rate, the cash leg must contribute.
    res_rf = run_backtest(px, w, cost_bps=0.0, lag=1, rf=0.05)
    check(
        "cash leg earns the risk-free rate",
        res_rf.returns.iloc[2:].mean() > mean_ret,
        f"{res_rf.returns.iloc[2:].mean():.6f} > {mean_ret:.6f}",
    )


def test_inverse_vol_weighting():
    section("7. INVERSE-VOL WEIGHTING BEHAVES CORRECTLY")
    px = make_prices(regime=False)
    base = CrossSectionalMomentum(n_long=8)
    iv = InverseVolMomentum(n_long=8)

    wb = base.generate_weights(px).dropna(how="all")
    wi = iv.generate_weights(px).dropna(how="all")

    check("weights sum to 1", bool((wi.sum(axis=1).sub(1).abs() < 1e-9).all()))
    check(
        "holds exactly the same names as plain momentum",
        bool(((wi > 0) == (wb.reindex(wi.index) > 0)).all().all()),
    )
    check(
        "weights are genuinely unequal",
        float(wi.iloc[-1][wi.iloc[-1] > 0].std()) > 1e-6,
    )

    # The quieter name must get the larger weight.
    rets = px.pct_change(fill_method=None)
    vol = rets.rolling(63).std().shift(1)
    last = wi.iloc[-1]
    held = last[last > 0].index
    v = vol.loc[wi.index[-1], held]
    corr = float(pd.Series(last[held].values).corr(pd.Series((1 / v).values)))
    check(
        "weight correlates with inverse volatility",
        corr > 0.9,
        f"corr {corr:.3f}",
    )


def main() -> int:
    test_causality()
    test_causality_catches_a_cheat()
    test_vol_targeting_hits_its_target()
    test_derisking_in_high_vol_regime()
    test_no_leverage_cap_respected()
    test_cash_leg_is_not_renormalised()
    test_inverse_vol_weighting()

    print(f"\n{'=' * 68}")
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 68)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
