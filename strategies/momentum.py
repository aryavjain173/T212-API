"""
Cross-sectional momentum.

The idea, in one line: rank the universe by trailing return, hold the
winners, and rebalance periodically.

Why it's the right first strategy for this project, and what you should be
able to say about it:

Momentum is the most heavily documented anomaly in the empirical asset
pricing literature, going back to Jegadeesh and Titman (1993) and confirmed
across markets, asset classes and decades since. That matters here not
because it guarantees returns, but because when an interviewer asks "why
would this work?" you have real answers to choose between:

  - Underreaction. Information diffuses gradually; prices adjust slowly.
  - Risk premium. Winners are exposed to some priced risk factor.
  - Behavioural. Disposition effect, herding, extrapolative expectations.

And real answers to "why might it not?":

  - Momentum crashes. It fails catastrophically at market turning points,
    where a portfolio of prior winners gets destroyed in a sharp reversal.
    2009 is the canonical example, when momentum had one of the worst
    drawdowns of any documented factor.
  - It has been public since 1993, so any easy version is arbitraged out.
  - It's turnover-heavy, so costs eat a meaningful share of gross return.

The 12-1 specification (twelve months of return, skipping the most recent
month) is standard precisely because short-horizon returns reverse rather
than persist. The skip is the difference between momentum and noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class CrossSectionalMomentum(Strategy):
    """
    Long the top `n_long` names by trailing return, equally weighted.

    Optionally short the bottom `n_short` for a market-neutral version.
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        n_long: int = 12,
        n_short: int = 0,
        rebalance: str = "M",
        min_history: int | None = None,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance = rebalance
        self.min_history = min_history or (lookback + skip)

        side = "long-only" if n_short == 0 else f"L{n_long}/S{n_short}"
        self.name = (
            f"XSMom({lookback}-{skip}, top{n_long}, {rebalance}, {side})"
        )

    # ------------------------------------------------------------- signal

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Trailing return over `lookback` bars, ending `skip` bars ago."""
        if self.skip:
            end = prices.shift(self.skip)
            begin = prices.shift(self.skip + self.lookback)
        else:
            end = prices
            begin = prices.shift(self.lookback)
        return end / begin - 1.0

    # ------------------------------------------------------------ weights

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.signal(prices)

        # Only rank names that have a live price AND a valid signal today.
        # A name with a signal but no price can't be traded.
        valid = sig.notna() & prices.notna()
        sig = sig.where(valid)

        # Rank descending: rank 1 is the strongest performer.
        ranks = sig.rank(axis=1, ascending=False, method="first")
        n_valid = valid.sum(axis=1)

        weights = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns
        )

        long_mask = ranks.le(self.n_long)
        n_actual_long = long_mask.sum(axis=1)
        # Guard against the warmup period where fewer names are available
        # than we want to hold.
        weights = weights.mask(
            long_mask, 1.0 / n_actual_long.replace(0, np.nan), axis=0
        )

        if self.n_short:
            short_mask = ranks.gt(n_valid - self.n_short) & ranks.notna()
            n_actual_short = short_mask.sum(axis=1)
            short_w = -1.0 / n_actual_short.replace(0, np.nan)
            weights = weights.mask(short_mask, short_w, axis=0)

        # Blank out rows before we have enough history to rank anything.
        insufficient = n_valid < max(self.n_long + self.n_short, 5)
        weights.loc[insufficient] = np.nan

        # Only act on rebalance dates; hold the target in between.
        reb = self._rebalance_dates(prices.index, self.rebalance)
        weights = self._hold_between_rebalances(weights, reb)

        return weights.astype(float)


class ShortTermReversal(Strategy):
    """
    The counterpart to momentum, at the opposite horizon.

    Buys the recent losers. Included so you can measure it against
    momentum rather than guess: the two are close to anticorrelated at
    the extremes, which is exactly why blending them naively cancels both
    out. Left out of the default run deliberately.

    Be aware this is far more turnover-heavy than momentum, so it is much
    more sensitive to the cost assumption. A reversal strategy that looks
    good at zero cost and bad at 10bps is telling you something real.
    """

    def __init__(
        self,
        lookback: int = 5,
        n_long: int = 12,
        rebalance: str = "W",
    ) -> None:
        self.lookback = lookback
        self.n_long = n_long
        self.rebalance = rebalance
        self.name = f"STRev({lookback}d, bottom{n_long}, {rebalance})"

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices / prices.shift(self.lookback) - 1.0

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.signal(prices)
        valid = sig.notna() & prices.notna()
        sig = sig.where(valid)

        # Ascending: rank 1 is the WORST performer, which is what we buy.
        ranks = sig.rank(axis=1, ascending=True, method="first")
        n_valid = valid.sum(axis=1)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        mask = ranks.le(self.n_long)
        n_actual = mask.sum(axis=1)
        weights = weights.mask(mask, 1.0 / n_actual.replace(0, np.nan), axis=0)
        weights.loc[n_valid < self.n_long] = np.nan

        reb = self._rebalance_dates(prices.index, self.rebalance)
        return self._hold_between_rebalances(weights, reb).astype(float)


class EqualWeightBenchmark(Strategy):
    """
    Hold the whole universe equally. The honest benchmark.

    Beating cash is not interesting. Beating the universe you selected
    from is the only comparison that tells you whether the SIGNAL added
    anything, as opposed to the universe simply having gone up.
    """

    name = "EqualWeight"

    def __init__(self, rebalance: str = "M") -> None:
        self.rebalance = rebalance

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        valid = prices.notna()
        n = valid.sum(axis=1)
        weights = valid.div(n.replace(0, np.nan), axis=0)
        weights.loc[n == 0] = np.nan
        reb = self._rebalance_dates(prices.index, self.rebalance)
        return self._hold_between_rebalances(weights, reb).astype(float)
