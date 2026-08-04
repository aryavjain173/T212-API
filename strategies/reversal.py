"""
Mean reversion.

THE HONEST FRAMING

You are testing this because momentum failed. That is a legitimate reason
to look, and also exactly the situation in which people fool themselves,
so the discipline matters more here than it did the first time.

The danger is specification search. If you try enough signals, one will
look good by chance. With twenty independent specifications, the best of
them will show a Sharpe roughly 0.5 higher than its true value purely from
sampling noise. Reporting only that one, which is what almost every
published backtest does, produces a result that will not survive contact
with new data.

The defences used here:
  1. The parameter grid is written down BEFORE looking at results.
  2. Every specification is reported, including the failures.
  3. The best result is compared against what the best-of-N would look
     like under the null of no skill (see scripts/signal_sweep.py).

WHAT THE LITERATURE SAYS

Short-horizon reversal is real and well documented (Lehmann 1990, Lo and
MacKinlay 1990). But the modern interpretation matters: it is largely
compensation for providing liquidity. A stock falls because someone needed
to sell in size; you buy, absorb their impatience, and get paid for it.

That framing predicts two things about your setup, and both are bad news:

  - The effect is strongest in small, illiquid, hard-to-trade names. In 60
    of the largest US stocks, liquidity provision is exactly the service
    that is least scarce, so the premium should be smallest.
  - It requires very high turnover to harvest, so transaction costs eat it.
    Your existing 5-day reversal run showed 4,138% annual turnover.

So the prior is genuinely unfavourable. Test it anyway, but the cost
sensitivity is the column to read first, not the headline Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class MeanReversion(Strategy):
    """
    Cross-sectional reversal: buy the recent losers, sell the recent
    winners.

    Parameters deliberately exposed so the sweep can vary them:

      lookback   horizon over which "recent" is measured, in bars
      n_long     how many names to hold
      rebalance  how often to reform the book
      demean     if True, measure each name's return RELATIVE to the
                 cross-sectional mean that day. This strips out the market
                 move, so you are betting on relative dislocation rather
                 than accidentally shorting the market on down days.
      vol_adjust if True, standardise each name's return by its own
                 volatility before ranking. Without this, the ranking is
                 dominated by whichever names are structurally volatile,
                 and you end up with a high-beta portfolio rather than a
                 reversal portfolio.
    """

    def __init__(
        self,
        lookback: int = 5,
        n_long: int = 12,
        rebalance: str = "W",
        demean: bool = True,
        vol_adjust: bool = True,
        vol_window: int = 63,
    ) -> None:
        self.lookback = lookback
        self.n_long = n_long
        self.rebalance = rebalance
        self.demean = demean
        self.vol_adjust = vol_adjust
        self.vol_window = vol_window
        tags = []
        if demean:
            tags.append("dm")
        if vol_adjust:
            tags.append("va")
        tag = ("+" + "".join(tags)) if tags else ""
        self.name = f"MR({lookback}d,n{n_long},{rebalance}{tag})"

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        raw = prices / prices.shift(self.lookback) - 1.0

        if self.vol_adjust:
            # Standardise by each name's own vol so the ranking measures
            # "how unusual is this move for THIS stock", not "which stock
            # is jumpiest". Shifted to stay causal.
            vol = (
                prices.pct_change(fill_method=None)
                .rolling(self.vol_window).std().shift(1)
            )
            floor = vol.quantile(0.05, axis=1)
            vol = vol.clip(lower=floor, axis=0)
            raw = raw.div(vol * np.sqrt(self.lookback))

        if self.demean:
            # Subtract the cross-sectional mean: bet on relative moves,
            # not on the market direction.
            raw = raw.sub(raw.mean(axis=1), axis=0)

        return raw

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.signal(prices)
        valid = sig.notna() & prices.notna()
        sig = sig.where(valid)

        # Ascending: rank 1 is the WORST recent performer, which we buy.
        ranks = sig.rank(axis=1, ascending=True, method="first")
        n_valid = valid.sum(axis=1)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        mask = ranks.le(self.n_long)
        n_actual = mask.sum(axis=1)
        weights = weights.mask(mask, 1.0 / n_actual.replace(0, np.nan), axis=0)
        weights.loc[n_valid < self.n_long] = np.nan

        reb = self._rebalance_dates(prices.index, self.rebalance)
        return self._hold_between_rebalances(weights, reb).astype(float)
