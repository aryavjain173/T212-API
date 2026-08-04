"""
Risk-managed momentum.

WHY THIS EXISTS

The plain momentum backtest produced a specific, measurable failure: it
beat the equal-weight benchmark on raw return but lost once you matched
volatility, and the active return was statistically indistinguishable from
zero. That is not a reason to abandon the signal. It is a reason to attack
the thing that broke it, which was risk, not selection.

The literature had already found this. Barroso and Santa-Clara (2015)
observed that momentum's volatility is highly forecastable from its own
recent realised volatility, far more so than its return is. Their
conclusion: if you cannot predict the return but you CAN predict the risk,
manage the risk. Scaling the portfolio's exposure by the inverse of its
own recent realised volatility roughly doubled momentum's Sharpe ratio in
their sample and largely removed the crashes.

Daniel and Moskowitz (2016) reached a compatible conclusion from a
different angle: momentum crashes cluster in high-volatility periods that
follow market declines, when the short leg of a long/short momentum book
behaves like a written call option on the market.

TWO SEPARATE IDEAS, DELIBERATELY KEPT APART

1. WITHIN-portfolio risk weighting. Instead of equal-weighting the twelve
   names you hold, weight them by inverse volatility so a wild name gets a
   smaller allocation than a placid one. This changes the composition of
   the book but leaves it fully invested.

2. PORTFOLIO-LEVEL volatility targeting. Scale the whole book up or down
   so its forecast volatility hits a fixed target, holding cash when the
   strategy is running hot. This changes the size of the book.

These are implemented separately, and the research script runs each alone
and both together, because the entire point is to know WHICH change did
the work. Bundling them and reporting one improved number tells you
nothing about mechanism, and mechanism is what an interviewer will probe.

THE LOOKAHEAD TRAP, SPELLED OUT

Volatility targeting is unusually easy to get wrong, because the natural
implementation quietly uses the future. The forecast for day t must be
built only from returns realised strictly BEFORE day t. In the code below
every volatility estimate is shifted by one bar before it is used, and
there is a dedicated test that catches the error if that shift is removed.
If you ever see vol-targeting double a Sharpe ratio, check the shift
before you celebrate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .momentum import CrossSectionalMomentum


class InverseVolMomentum(CrossSectionalMomentum):
    """
    Momentum, but the selected names are weighted by inverse volatility
    rather than equally.

    The rationale: an equal-weight book is not an equal-RISK book. If one
    holding runs at 60% annualised vol and another at 15%, the first
    contributes roughly four times the risk despite the identical dollar
    allocation. Since momentum systematically selects names that have
    recently moved a lot, and recent movers tend to keep being volatile,
    equal weighting concentrates risk in exactly the names most likely to
    reverse violently.

    This does NOT change which names you hold, only how much of each. Any
    performance difference is therefore attributable purely to weighting.
    """

    def __init__(self, vol_window: int = 63, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vol_window = vol_window
        self.name = self.name.replace("XSMom", "IVMom")

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        base = super().generate_weights(prices)
        selected = base > 0

        # Realised vol, SHIFTED so day t uses only data up to t-1.
        rets = prices.pct_change(fill_method=None)
        vol = rets.rolling(self.vol_window).std().shift(1)

        # Floor the vol to stop a near-zero estimate producing an enormous
        # weight. A stock that hasn't moved in three months is usually a
        # data problem, not a riskless asset.
        #
        # The floor is a CROSS-SECTIONAL quantile, computed across names
        # within each row. An earlier version used a full-sample quantile
        # and the causality test caught it: that version let the volatility
        # of 2024 influence weights held in 2016. The effect on returns was
        # small, but a backtest that uses future data is not a backtest.
        floor = vol.quantile(0.05, axis=1)
        vol = vol.clip(lower=floor, axis=0)

        inv = (1.0 / vol).where(selected)
        weights = inv.div(inv.sum(axis=1), axis=0)

        # Rows where we hold nothing, or vol is unavailable, stay blank.
        weights = weights.where(selected.any(axis=1), other=np.nan)
        return weights.astype(float)


class VolTargeted(Strategy):
    """
    A wrapper that scales ANY base strategy to a constant volatility
    target. This is the Barroso and Santa-Clara mechanism.

    How it works:
      1. Compute the base strategy's unscaled weights.
      2. Reconstruct the return series those weights would have produced.
      3. Estimate realised volatility over a trailing window.
      4. Scale exposure by target_vol / realised_vol, capped.

    Because the scalar multiplies every position, the book is no longer
    fully invested. Below 1.0 the remainder sits in cash; above 1.0 it is
    leverage. Set max_leverage=1.0 for a long-only, cash-holding version
    that requires no borrowing, which is what you can actually run in a
    Trading 212 account.

    WHAT THIS ASSUMES, AND WHERE IT COULD MISLEAD

    It assumes volatility is persistent, which is one of the most robust
    facts in empirical finance (volatility clusters), and it assumes the
    trailing window is a decent forecast, which is true on average and
    badly false at regime breaks. In a sudden crash the scalar responds
    with a lag, so you take the first hit at full size and only then
    de-risk. Vol targeting reduces crash severity; it does not prevent it.

    It also mechanically raises turnover, because the scalar moves daily
    even when the underlying selection doesn't. Watch the cost sweep.
    """

    def __init__(
        self,
        base: Strategy,
        target_vol: float = 0.15,
        vol_window: int = 126,
        max_leverage: float = 1.0,
        min_exposure: float = 0.0,
        scalar_rebalance: str = "M",
    ) -> None:
        self.base = base
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.min_exposure = min_exposure
        self.scalar_rebalance = scalar_rebalance
        lev = "no lev" if max_leverage <= 1.0 else f"max {max_leverage:g}x"
        self.name = f"VT({base.name}, {target_vol:.0%}, {vol_window}d, {lev})"

    # -------------------------------------------------------- internals

    def _unscaled_returns(
        self, prices: pd.DataFrame, weights: pd.DataFrame
    ) -> pd.Series:
        """
        Reconstruct what the base strategy would have returned.

        Uses weights.shift(1) against same-day returns, matching the
        engine's accounting. This is an approximation: it ignores
        intra-month drift and costs. That is fine here, because we only
        need a volatility estimate, and vol is far less sensitive to those
        details than return is.
        """
        rets = prices.pct_change(fill_method=None)
        return (weights.shift(1) * rets).sum(axis=1, min_count=1)

    def _scalar(self, port_rets: pd.Series) -> pd.Series:
        """
        Exposure multiplier. The .shift(1) is the lookahead guard: the
        estimate available for day t uses returns through t-1 only.
        """
        realised = port_rets.rolling(self.vol_window).std() * np.sqrt(252)
        realised = realised.shift(1)

        scalar = self.target_vol / realised.replace(0.0, np.nan)
        scalar = scalar.clip(lower=self.min_exposure, upper=self.max_leverage)

        # Recomputing the scalar every day is a turnover disaster for a
        # monthly strategy, so hold it fixed between scalar rebalances.
        if self.scalar_rebalance != "D":
            reb = self._rebalance_dates(port_rets.index, self.scalar_rebalance)
            scalar = scalar.where(reb).ffill()

        return scalar

    # ---------------------------------------------------------- weights

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        base_w = self.base.generate_weights(prices)
        port_rets = self._unscaled_returns(prices, base_w)
        scalar = self._scalar(port_rets)

        scaled = base_w.mul(scalar, axis=0)

        # Blank rows where we have no scalar yet (warmup) rather than
        # falling back to full exposure, which would be a silent default
        # to the riskiest setting.
        scaled = scaled.where(scalar.notna(), other=np.nan)
        return scaled.astype(float)

    def exposure(self, prices: pd.DataFrame) -> pd.Series:
        """The scalar itself, for plotting and diagnosis."""
        base_w = self.base.generate_weights(prices)
        return self._scalar(self._unscaled_returns(prices, base_w))
