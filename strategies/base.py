"""
The strategy interface.

Every strategy is a function from a price history to a set of target
portfolio weights, and nothing else. It does not know what broker you use,
what your account is worth, or whether it's being backtested or run live.
That separation is what lets the same object be measured historically and
then traded, with no code path that only exists in one of the two.

The contract, which the backtester and the live loop both rely on:

  generate_weights(prices) -> DataFrame of target weights

  - indexed identically to `prices`
  - one column per ticker
  - row t holds the weights you want to HOLD given information available
    at the close of day t
  - rows sum to 1.0 for long-only, or 0.0 for long/short; NaN where the
    strategy has no opinion yet (during warmup)

The single most important rule: row t may only use data from row t or
earlier. The backtester lags execution by one bar on top of that, so a
signal formed on Monday's close is traded at Tuesday's close. If you ever
see a Sharpe ratio above about 3 on daily equity data, assume you've broken
this rule before you assume you've found something.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Base class. Subclass this and implement generate_weights."""

    name: str = "unnamed"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights to hold, given information up to and including
        each row's date."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.Series:
        """
        Boolean series marking rebalance days.

        freq: 'D' daily, 'W' weekly (last bar of week), 'M' monthly
              (last bar of month), 'Q' quarterly.

        We take the last available BAR in each period rather than a
        calendar date, because calendar month-ends land on weekends and
        holidays and you'd silently skip rebalances.
        """
        s = pd.Series(False, index=index)
        if freq == "D":
            s[:] = True
            return s

        periods = {"W": "W", "M": "M", "Q": "Q"}
        if freq not in periods:
            raise ValueError(f"unsupported rebalance frequency: {freq}")

        grouper = index.to_period(periods[freq])
        last_bars = pd.Series(index, index=index).groupby(grouper).max()
        s.loc[last_bars.values] = True
        return s

    @staticmethod
    def _hold_between_rebalances(
        weights: pd.DataFrame, rebalance: pd.Series
    ) -> pd.DataFrame:
        """
        Carry target weights forward between rebalance dates.

        Note this holds the TARGET weight constant, which implies drifting
        back to target continuously. Real portfolios drift with prices and
        only get corrected on rebalance days. The backtester models the
        drift properly; this just supplies the targets.
        """
        held = weights.where(rebalance, other=np.nan)
        return held.ffill()
