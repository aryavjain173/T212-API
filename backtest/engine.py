"""
The backtest engine.

This is the part of the project where the honesty lives, so it's worth
being explicit about what it does and does not model.

WHAT IT GETS RIGHT

Execution lag. Precise semantics, because vagueness here is where bugs
hide. Inside the loop, day i earns its return on the book held coming into
day i, and only then do we trade to that day's target. So:

  lag=0: signal formed at the close of day i is held from the close of
         day i, and first earns a return on day i+1. This is the standard
         "trade at the close you just observed" assumption. Achievable
         with market-on-close orders, mildly optimistic.

  lag=1: signal formed at the close of day i is held from the close of
         day i+1, first earning on day i+2. A full extra day of delay.

We default to lag=1 because it is the conservative choice and because it
matches how you will actually run this: the scheduler wakes up after the
close, computes a signal on yesterday's data, and trades into today's
close. Without a lag at all you would be assuming you saw the closing
price and traded at it simultaneously, which is the single most common way
amateur backtests manufacture fake returns.

Portfolio drift. Between rebalances, positions move with prices. A 12-name
equal-weight book is only equal-weight on the day you set it. Modelling
drift matters because it's the difference between the turnover you assume
and the turnover you actually pay for: if you assume you rebalance to
target every day you massively overstate costs, and if you ignore drift
entirely you understate them.

Transaction costs. Charged on realised turnover, in basis points of traded
value. See the note on calibration below.

Survivorship, partly. The engine handles names entering and leaving the
data. The UNIVERSE is still survivorship-biased, which is a separate
problem documented in marketdata/universe.py and not fixable with free
data.

WHAT IT DOES NOT MODEL

Market impact. We assume you get the close. At the size you'll trade,
this is close enough to true for large-cap US names.

Fractional-share rounding. T212 supports fractional shares so this is
minor, but a real book has integer constraints.

Borrow costs on shorts. If you enable n_short, the short leg is free here
and would not be in reality.

Slippage versus the close. The gap between the official close and what you
actually get. Milestone 7 measures this from your real fills rather than
assuming it, which is the entire point of that milestone.

COST CALIBRATION

Trading 212 charges no commission on stocks, which tempts people to set
costs to zero. That's wrong. You still pay:
  - the bid-ask spread, roughly 1-3bps on large-cap US names
  - an FX conversion fee on a GBP-denominated account buying USD stocks
  - slippage between decision and fill

The default of 10bps per unit of turnover is deliberately conservative.
Run the sensitivity sweep in scripts/backtest.py: if the strategy only
works at 0bps, it doesn't work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series          # net daily portfolio returns
    gross_returns: pd.Series    # before costs
    equity: pd.Series           # cumulative growth of 1.0
    weights: pd.DataFrame       # actual held weights, after drift
    turnover: pd.Series         # daily one-way turnover as fraction of book
    costs: pd.Series            # daily cost drag
    name: str = "strategy"
    meta: dict = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return len(self.returns)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 10.0,
    lag: int = 1,
    name: str = "strategy",
    rf: float = 0.0,
) -> BacktestResult:
    """
    Simulate holding `target_weights` against `prices`.

    Implemented as an explicit day loop rather than a vectorised
    expression. Vectorising is possible but obscures the drift and
    turnover accounting, and at 3,000 rows the speed difference is
    irrelevant. Clarity wins where correctness is the whole point.
    """
    if not prices.index.equals(target_weights.index):
        raise ValueError("prices and weights must share an index")

    # THE LAG. Targets formed on day t are held from day t+lag onward.
    targets = target_weights.shift(lag)

    rets = prices.pct_change(fill_method=None)
    cost_rate = cost_bps / 10_000.0
    rf_daily = rf / 252.0   # annual cash rate paid on the uninvested leg

    dates = prices.index
    cols = prices.columns
    n = len(dates)

    held = np.zeros(len(cols))          # weights we actually hold
    held_hist = np.zeros((n, len(cols)))
    gross = np.zeros(n)
    net = np.zeros(n)
    turn = np.zeros(n)
    cost_arr = np.zeros(n)

    target_arr = targets.to_numpy(dtype=float)
    ret_arr = rets.to_numpy(dtype=float)

    for i in range(n):
        r = ret_arr[i]
        r = np.where(np.isnan(r), 0.0, r)

        # 1. Earn today's return on yesterday's book. Any unallocated
        #    weight is cash, earning rf (0 by default).
        invested = held.sum()
        cash_w = 1.0 - invested
        day_gross = float(held @ r) + cash_w * rf_daily

        # 2. Let the book drift with prices.
        #    The denominator must include the cash leg, otherwise a
        #    partially invested portfolio gets silently renormalised back
        #    to fully invested and the vol-targeting does nothing.
        grown = held * (1.0 + r)
        total = grown.sum() + cash_w * (1.0 + rf_daily)
        drifted = grown / total if total != 0 else grown

        # 3. If we have a target for today, trade to it and pay for it.
        tgt = target_arr[i]
        if not np.all(np.isnan(tgt)):
            tgt = np.where(np.isnan(tgt), 0.0, tgt)
            traded = np.abs(tgt - drifted).sum()
            # One-way turnover: buying 0.5 and selling 0.5 is 0.5 traded,
            # not 1.0. Halving here is the standard convention.
            one_way = traded / 2.0
            day_cost = traded * cost_rate
            held = tgt
        else:
            one_way = 0.0
            day_cost = 0.0
            held = drifted

        gross[i] = day_gross
        cost_arr[i] = day_cost
        net[i] = day_gross - day_cost
        turn[i] = one_way
        held_hist[i] = held

    idx = dates
    net_s = pd.Series(net, index=idx, name=name)
    gross_s = pd.Series(gross, index=idx, name=name)

    # Trim the warmup: everything before the first non-zero position is
    # dead time and would flatter the drawdown statistics.
    first = np.argmax(np.abs(held_hist).sum(axis=1) > 0)
    if np.abs(held_hist).sum() == 0:
        raise ValueError("strategy never took a position")

    net_s = net_s.iloc[first:]
    gross_s = gross_s.iloc[first:]

    return BacktestResult(
        returns=net_s,
        gross_returns=gross_s,
        equity=(1.0 + net_s).cumprod(),
        weights=pd.DataFrame(held_hist, index=idx, columns=cols).iloc[first:],
        turnover=pd.Series(turn, index=idx).iloc[first:],
        costs=pd.Series(cost_arr, index=idx).iloc[first:],
        name=name,
        meta={"cost_bps": cost_bps, "lag": lag, "rf": rf},
    )


# --------------------------------------------------------------- checks


def lookahead_check(
    prices: pd.DataFrame,
    strategy,
    cost_bps: float = 0.0,
) -> dict:
    """
    A diagnostic, not a proof.

    Runs the same strategy at lag 0 and lag 1. Lag 0 means trading on the
    close you just observed, which is impossible. If lag 0 looks
    dramatically better, the strategy is leaning on same-bar information
    and the lag is doing real work. If lag 0 looks IDENTICAL, the signal
    may already contain future data, which is worse.

    Neither result is conclusive. It's a smell test that catches the
    obvious cases.
    """
    w = strategy.generate_weights(prices)
    r0 = run_backtest(prices, w, cost_bps=cost_bps, lag=0, name="lag0")
    r1 = run_backtest(prices, w, cost_bps=cost_bps, lag=1, name="lag1")

    from .metrics import sharpe

    s0, s1 = sharpe(r0.returns), sharpe(r1.returns)
    return {
        "sharpe_lag0": s0,
        "sharpe_lag1": s1,
        "delta": s0 - s1,
        "verdict": (
            "suspicious: lag makes no difference" if abs(s0 - s1) < 1e-6
            else "ok" if s0 >= s1
            else "unusual: lag improved things"
        ),
    }
