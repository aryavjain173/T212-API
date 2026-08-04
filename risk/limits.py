"""
The risk layer.

This sits between "what the strategy wants to hold" and "orders that reach
the broker". Nothing else in the project is allowed to call an order
method directly; the live loop goes through GuardedTrader.

WHY IT IS SPLIT IN TWO

RiskEngine is pure. It takes target weights, the current portfolio, cash,
and prices, and returns a decision. It never touches the network. That is
what makes scripts/test_limits.py possible: every limit can be tested
against a hand-constructed case with a known answer, with no broker and no
API key. A risk check you cannot test offline is a risk check you do not
actually know works.

GuardedTrader is the thin wrapper. It holds the client, calls the engine,
and places whatever the engine approved. It contains no judgement of its
own, deliberately, because logic that lives next to a network call is
logic that only ever gets exercised in production.

THE TWO CLASSES OF CHECK

Halts abort the entire run and place nothing: kill switch, stale data, FX
sanity, drawdown breaker. These are conditions under which you do not
believe your own inputs, and the correct response to not believing your
inputs is to do nothing at all.

Shaping checks clip the order list rather than rejecting it: position size,
cash, minimum notional, turnover, order count. These are conditions where
the trade is legitimate but too large.

A NOTE ON CLIPPING, WHICH YOU SHOULD BE ABLE TO DEFEND

Clipping means a partially rebalanced book. The book you end up holding is
then one that neither the strategy nor the backtest ever described, and its
performance is not the performance you measured. That is a real cost and it
is the reason every clip is recorded in the decision log with the size of
the shortfall. Milestone 7 can then measure how far live weights drifted
from intended weights, alongside the slippage measurement, and quantify
what the clipping actually cost. An unlogged clip is a silent divergence
between research and reality, which is the exact failure this project
exists to avoid.

TURNOVER CONVENTION

backtest/engine.py computes traded = sum(|target - drifted|) and charges
cost on that full sum, then reports turnover as traded / 2. This module
uses the same halved, one-way figure, so a cap set here is on the same
scale as the 418% pa reported in the research. The traded notional behind
it is twice that number.

CURRENCY

The account is GBP-denominated and the positions are USD-listed. T212
reports position currentPrice in the instrument currency but cash figures
in the account currency, so market values and cash are not directly
comparable. Rather than hardcode a rate, the engine derives the implied
rate from the account itself:

    implied_fx = (cash.invested + cash.ppl) / sum(position market values)

and halts if that lands outside a plausible band. Self-calibrating, and it
fails loudly rather than quietly mis-sizing every order. If the account is
empty there is nothing to imply a rate from, so the configured fallback is
used and flagged.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import pandas as pd

from marketdata.loader import fetch_gbpusd_rate

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from t212.client import Cash, Position, T212Client

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE_MAP = ROOT / "data" / "universe_map.json"
DEFAULT_STATE_FILE = ROOT / "data" / "risk_state.json"
DEFAULT_KILL_FILE = ROOT / "HALT"

def truncate(x: float, decimals: int) -> float:
    """
    Round toward zero at a fixed number of decimal places.

    Deliberately not round(). Rounding to nearest can round UP, so a book of
    twelve positions can collectively exceed its cash budget or its turnover
    cap by a few pence purely through rounding, and the risk checks then
    fire on an artefact rather than on a decision.

    Truncating gives an invariant worth having: an executed order is never
    larger than the intended one. You never buy more than the target, and
    you never sell more than you hold. The cost is a tiny permanent
    under-investment, which is the right direction for the error to run.
    """
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    scale = 10 ** decimals
    return math.trunc(x * scale) / scale


NY = "America/New_York"
REGULAR_CLOSE = (16, 0)     # 16:00 ET
EARLY_CLOSE = (13, 0)       # 13:00 ET on half days
REGULAR_OPEN = (9, 30)

# NYSE full closures. Verified against the published calendar; the day of
# week for every entry was checked, because an observed-holiday shift is
# exactly the kind of thing that is wrong by one day and never noticed.
#
# THIS TABLE EXPIRES. It runs to the end of 2027 and the engine warns once
# the current year passes HOLIDAY_TABLE_THROUGH, rather than silently
# treating an unlisted holiday as a trading day. The proper fix is
# pandas_market_calendars, which is a dependency this project does not
# currently carry and does not otherwise need.
HOLIDAY_TABLE_THROUGH = 2027

US_MARKET_HOLIDAYS: frozenset[str] = frozenset({
    # 2026
    "2026-01-01",  # New Year's Day, Thu
    "2026-01-19",  # MLK, Mon
    "2026-02-16",  # Presidents' Day, Mon
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day, Mon
    "2026-06-19",  # Juneteenth, Fri
    "2026-07-03",  # Independence Day observed (4th is a Saturday)
    "2026-09-07",  # Labor Day, Mon
    "2026-11-26",  # Thanksgiving, Thu
    "2026-12-25",  # Christmas, Fri
    # 2027
    "2027-01-01",  # New Year's Day, Fri
    "2027-01-18",  # MLK, Mon
    "2027-02-15",  # Presidents' Day, Mon
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day, Mon
    "2027-06-18",  # Juneteenth observed (19th is a Saturday)
    "2027-07-05",  # Independence Day observed (4th is a Sunday)
    "2027-09-06",  # Labor Day, Mon
    "2027-11-25",  # Thanksgiving, Thu
    "2027-12-24",  # Christmas observed (25th is a Saturday)
})

# Half days, closing at 13:00 ET. These matter more than they look: the
# trade window is defined relative to the close, so treating a 13:00 close
# as 16:00 means the window opens three hours after the market has shut and
# every order queues to the next session.
US_EARLY_CLOSES: frozenset[str] = frozenset({
    "2026-11-27",  # day after Thanksgiving
    "2026-12-24",  # Christmas Eve
    "2027-11-26",  # day after Thanksgiving
})


# --------------------------------------------------------------- config


@dataclass(frozen=True)
class RiskConfig:
    """
    Every limit in one place, so a change is a one-line diff and shows up
    in the decision log rather than being buried in a call site.

    Defaults are sized for the deployed specification in PROJECT_STATUS §8:
    XSMom(252-21, top 12, monthly), long-only. Twelve equal-weighted names
    is 8.33% each, so max_position_weight of 0.15 gives headroom for drift
    between monthly rebalances without permitting a concentrated book.
    """

    # Broker quantity precision. T212 rejects orders carrying more decimal
    # places than it accepts, with a quantity-precision-mismatch 400. The
    # exact allowance is not documented and is not in the instrument
    # metadata, so scripts/probe_precision.py determines it empirically
    # against demo. Measured as 4 on 2026-07-30. Rounding happens in the engine rather than at the call
    # site, so that reported notional and turnover describe the order that
    # will actually be sent rather than the one we wished for.
    quantity_decimals: int = 4

    # Position sizing
    max_position_weight: float = 0.15
    min_order_notional: float = 20.0        # account currency; skip dust

    # Turnover, one-way, as a fraction of book value, per run
    max_turnover_per_run: float = 0.60

    # Order count
    max_orders_per_run: int = 25

    # Cash
    cash_buffer: float = 0.02               # keep this fraction unspent
    count_sell_proceeds: bool = False       # sells may not settle instantly

    # Drawdown circuit breaker, peak-to-trough, both bases
    max_drawdown_account: float = 0.25
    max_drawdown_strategy: float = 0.25

    # Data freshness
    max_bar_age_days: int = 4               # calendar days; survives a long weekend

    # Trading window. Orders sent outside US market hours are not rejected,
    # they QUEUE and fill at the next open. That silently tightens the
    # execution lag by most of a session relative to the lag=1 convention
    # the backtest measured, on every fill, forever. So the script decides
    # whether now is a sensible moment to trade rather than trusting the
    # scheduler to have fired at the right time.
    require_market_hours: bool = True
    trade_window_minutes: int = 30          # minutes before the close

    # FX sanity band for the implied rate (GBP account, USD instruments)
    fx_band: tuple[float, float] = (0.50, 2.00)
    fx_fallback: float = 1.0

    # Paths
    universe_map_path: Path = DEFAULT_UNIVERSE_MAP
    state_path: Path = DEFAULT_STATE_FILE
    kill_switch_path: Path = DEFAULT_KILL_FILE

    # Master switch. Setting this False disables every shaping check and is
    # used by the test suite as a negative control: if the limits are doing
    # nothing, the disabled and enabled runs produce identical output, and
    # the tests are worthless.
    enabled: bool = True


# ---------------------------------------------------------------- types


@dataclass
class Order:
    """A single proposed trade. Quantity is signed; negative sells."""

    yf_symbol: str
    t212_ticker: str
    quantity: float
    price: float                # instrument currency
    fx: float                   # instrument ccy -> account ccy
    target_weight: float
    current_weight: float
    quantity_decimals: int = 4

    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"

    @property
    def notional(self) -> float:
        """Absolute traded value in ACCOUNT currency."""
        return abs(self.quantity) * self.price * self.fx

    @property
    def weight_delta(self) -> float:
        return self.target_weight - self.current_weight

    def scaled_to_notional(self, target_notional: float, equity: float) -> "Order":
        """Return a copy trimmed to a smaller absolute notional."""
        if self.notional <= 0:
            return self
        scale = min(1.0, target_notional / self.notional)
        new_qty = truncate(self.quantity * scale, self.quantity_decimals)
        if new_qty == 0.0:
            new_qty = 0.0
        realised = 0.0 if self.quantity == 0 else new_qty / self.quantity
        new_delta = self.weight_delta * realised
        return Order(
            yf_symbol=self.yf_symbol,
            t212_ticker=self.t212_ticker,
            quantity=new_qty,
            price=self.price,
            fx=self.fx,
            target_weight=self.current_weight + new_delta,
            current_weight=self.current_weight,
            quantity_decimals=self.quantity_decimals,
        )


@dataclass
class Note:
    """One recorded decision. Everything that shapes or blocks lands here."""

    kind: str                   # 'halt' | 'clip' | 'drop' | 'info'
    check: str
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.kind.upper():<4}] {self.check}: {self.message}"


@dataclass
class RiskDecision:
    approved: list[Order]
    notes: list[Note]
    halted: bool
    equity: float
    fx: float
    turnover: float             # one-way, fraction of equity
    state: dict
    placed: list = field(default_factory=list)
    execution_error: str | None = None

    @property
    def halts(self) -> list[Note]:
        return [n for n in self.notes if n.kind == "halt"]

    @property
    def executed_cleanly(self) -> bool:
        """True only if every approved order actually reached the broker."""
        return (
            not self.halted
            and self.execution_error is None
            and len(self.placed) == len(self.approved)
        )

    def summary(self) -> str:
        if self.halted:
            reasons = "; ".join(n.message for n in self.halts)
            return f"HALTED, no orders placed. {reasons}"
        buys = sum(1 for o in self.approved if o.quantity > 0)
        sells = len(self.approved) - buys
        return (
            f"{len(self.approved)} orders approved ({buys} buy, {sells} sell), "
            f"one-way turnover {self.turnover:.2%}, equity {self.equity:,.2f}"
        )


class RiskHalt(RuntimeError):
    """
    Raised by GuardedTrader when a halt gate fires and orders were requested.

    Carries the decision, because a halt is the single event most worth
    logging and a caller that only catches the exception must still be able
    to record what happened and why.
    """

    def __init__(self, decision: "RiskDecision"):
        super().__init__(decision.summary())
        self.decision = decision


# --------------------------------------------------------- market hours


def market_close(day) -> tuple[int, int] | None:
    """
    Closing time in ET for a given date, or None if the market is shut.

    Weekend and holiday closures return None. Half days return 13:00.
    """
    iso = day.isoformat()
    if day.weekday() >= 5 or iso in US_MARKET_HOLIDAYS:
        return None
    return EARLY_CLOSE if iso in US_EARLY_CLOSES else REGULAR_CLOSE


def trade_window(now_ny, window_minutes: int) -> tuple[bool, str]:
    """
    Is now inside the window in which an order will fill at today's close?

    Returns (ok, human-readable reason). The window runs from
    `window_minutes` before the close up to the close itself, and never
    starts before the open, so a pathologically long window on a half day
    cannot open before trading does.
    """
    from datetime import timedelta

    close = market_close(now_ny.date())
    if close is None:
        why = "weekend" if now_ny.weekday() >= 5 else "US market holiday"
        return False, f"{now_ny.date()} is a {why}"

    close_dt = now_ny.replace(hour=close[0], minute=close[1], second=0, microsecond=0)
    open_dt = now_ny.replace(
        hour=REGULAR_OPEN[0], minute=REGULAR_OPEN[1], second=0, microsecond=0
    )
    start = max(open_dt, close_dt - timedelta(minutes=window_minutes))

    half = " (half day)" if close == EARLY_CLOSE else ""
    if now_ny < start:
        mins = (start - now_ny).total_seconds() / 60
        return False, (
            f"{mins:.0f} min too early. Window is "
            f"{start:%H:%M}-{close_dt:%H:%M} ET{half}, now {now_ny:%H:%M} ET"
        )
    if now_ny >= close_dt:
        return False, (
            f"market has closed. Window was {start:%H:%M}-{close_dt:%H:%M} "
            f"ET{half}, now {now_ny:%H:%M} ET"
        )
    return True, f"{(close_dt - now_ny).total_seconds() / 60:.0f} min to the close"


# ----------------------------------------------------------- ticker map


class TickerMap:
    """
    Bidirectional yfinance <-> T212 symbol map, loaded from
    data/universe_map.json (written by scripts/build_universe.py).

    This file is the whitelist. A ticker absent from it is a ticker nobody
    has confirmed against the live instrument list, and per the OVERRIDES
    note in marketdata/universe.py the cost of guessing wrong is holding a
    2x leveraged tracker or the EUR line instead of the name you meant.
    """

    def __init__(self, mapping: dict[str, dict]):
        self._raw = mapping
        self.to_t212: dict[str, str] = {}
        self.to_yf: dict[str, str] = {}
        for yf_sym, rec in mapping.items():
            t212 = rec["t212_ticker"]
            if t212 in self.to_yf and self.to_yf[t212] != yf_sym:
                raise ValueError(
                    f"two yfinance symbols map to {t212}: "
                    f"{self.to_yf[t212]} and {yf_sym}"
                )
            self.to_t212[yf_sym] = t212
            self.to_yf[t212] = yf_sym

    @classmethod
    def load(cls, path: Path | str = DEFAULT_UNIVERSE_MAP) -> "TickerMap":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m scripts.build_universe"
            )
        return cls(json.loads(path.read_text()))

    def __len__(self) -> int:
        return len(self.to_t212)

    def __contains__(self, yf_symbol: object) -> bool:
        return yf_symbol in self.to_t212

    def currency(self, yf_symbol: str) -> str:
        return self._raw.get(yf_symbol, {}).get("currency", "USD")


# ----------------------------------------------------------- NAV state


def load_state(path: Path | str = DEFAULT_STATE_FILE) -> dict:
    path = Path(path)
    if not path.exists():
        return {
            "strategy_nav": 1.0,
            "peak_strategy_nav": 1.0,
            "peak_account_equity": None,
            "last_weights": {},
            "last_price_date": None,
            "last_run": None,
        }
    return json.loads(path.read_text())


def save_state(state: dict, path: Path | str = DEFAULT_STATE_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def advance_strategy_nav(state: dict, prices: pd.DataFrame) -> dict:
    """
    Roll the modelled strategy NAV forward using the book we believed we
    held and the price move since we last looked.

    This is deliberately independent of the account. Account equity moves
    when you deposit or withdraw; this does not, which is why the breaker
    watches both. A deposit should not reset a drawdown, and a strategy
    that is down 30% should not look fine because you topped the account up.

    Uninvested weight is treated as cash earning nothing, matching the
    engine's rf=0 default.
    """
    state = dict(state)
    last_w = state.get("last_weights") or {}
    last_date = state.get("last_price_date")

    if not last_w or last_date is None:
        return state

    last_date = pd.Timestamp(last_date)
    if last_date not in prices.index:
        log.warning("last price date %s not in price index; NAV not advanced", last_date)
        return state

    now_date = prices.index[-1]
    if now_date <= last_date:
        return state

    p0 = prices.loc[last_date]
    p1 = prices.loc[now_date]

    ret = 0.0
    for sym, w in last_w.items():
        if sym not in prices.columns:
            continue
        a, b = float(p0.get(sym, np.nan)), float(p1.get(sym, np.nan))
        if not np.isfinite(a) or not np.isfinite(b) or a == 0:
            continue
        ret += w * (b / a - 1.0)

    nav = float(state.get("strategy_nav", 1.0)) * (1.0 + ret)
    state["strategy_nav"] = nav
    state["peak_strategy_nav"] = max(float(state.get("peak_strategy_nav", nav)), nav)
    return state


# ---------------------------------------------------------- risk engine


class RiskEngine:
    """
    Pure. Takes a picture of the world, returns a decision. No network.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        ticker_map: TickerMap | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.map = ticker_map or TickerMap.load(self.config.universe_map_path)
        # Snapshot the whitelist now. marketdata.universe.drop() mutates
        # module state, and a mid-run mutation must not change what has
        # already been approved.
        self.whitelist: frozenset[str] = frozenset(self.map.to_t212)

    # ------------------------------------------------------------ helpers

    def implied_fx(
        self,
        positions: Sequence["Position"],
        cash: "Cash",
        fx_override: float | None = None,
    ) -> tuple[float, Note | None]:
        """
        Derive instrument-currency -> account-currency rate from the account.

        Current value of the book in account currency is invested + ppl.
        Sum of quantity * currentPrice is the same book in instrument
        currency. The ratio is the rate actually being applied by the
        broker, which beats any rate we could look up.
        """
        gross = sum(p.quantity * p.current_price for p in positions)
        if not positions or gross <= 0:
            if fx_override is None:
                return self.config.fx_fallback, Note(
                    "halt", "fx",
                    "no positions to imply FX from, and no fx_override supplied. "
                    "Refusing to guess; a wrong FX silently mis-sizes every order.",
                )
            lo, hi = self.config.fx_band
            if not (lo <= fx_override <= hi):
                return fx_override, Note(
                    "halt", "fx",
                    f"supplied fx_override {fx_override:.4f} outside band [{lo}, {hi}]. "
                    "Refusing to trade on a rate that looks wrong.",
                    {"implied_fx": fx_override, "source": "fx_override"},
                )
            return fx_override, Note(
                "info", "fx",
                f"no positions to imply FX from; using supplied fx_override {fx_override:.4f}",
            )
        acct_value = cash.invested + cash.ppl
        if acct_value <= 0:
            if fx_override is None:
                return self.config.fx_fallback, Note(
                    "halt", "fx",
                    f"account value non-positive ({acct_value:.2f}) while holding "
                    "positions, and no fx_override supplied.",
                )
            return fx_override, Note(
                "halt", "fx",
                f"account value non-positive ({acct_value:.2f}) while holding "
                "positions. This shouldn't happen on a long-only book and needs "
                "a human look regardless of the FX rate.",
                {"implied_fx": fx_override, "source": "fx_override",
                 "acct_value": acct_value},
            )
        fx = acct_value / gross
        lo, hi = self.config.fx_band
        if not (lo <= fx <= hi):
            return fx, Note(
                "halt", "fx",
                f"implied FX rate {fx:.4f} outside band [{lo}, {hi}]. "
                "Position and cash figures disagree; order sizing would be wrong.",
                {"implied_fx": fx, "invested": cash.invested, "ppl": cash.ppl,
                 "gross_instrument_ccy": gross},
            )
        return fx, None

    def _bar_age_days(self, prices: pd.DataFrame, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        last_bar = pd.Timestamp(prices.index[-1])
        if last_bar.tzinfo is None:
            last_bar = last_bar.tz_localize("UTC")
        return (now - last_bar.to_pydatetime()).total_seconds() / 86400.0

    @staticmethod
    def _priority(order: Order) -> tuple[int, float]:
        """
        Sort key for clipping. Sells first, because they reduce exposure
        and free cash, then buys by descending target weight so the names
        the strategy has most conviction in survive the clip.
        """
        is_buy = 1 if order.quantity > 0 else 0
        return (is_buy, -order.target_weight)

    # ------------------------------------------------------------ the check

    def evaluate(
        self,
        target_weights: pd.Series,
        positions: Sequence["Position"],
        cash: "Cash",
        prices: pd.DataFrame,
        state: dict | None = None,
        now: datetime | None = None,
        fx_override: float | None = None,
    ) -> RiskDecision:
        """
        target_weights: index yfinance symbol, values desired weight of book
        positions:      current T212 positions
        cash:           current T212 cash snapshot
        prices:         price history, columns yfinance symbols
        """
        cfg = self.config
        notes: list[Note] = []
        state = advance_strategy_nav(state or load_state(cfg.state_path), prices)

        equity = float(cash.total)

        # ---- HALT GATES. Any one of these places nothing at all. --------

        if cfg.kill_switch_path.exists():
            notes.append(Note(
                "halt", "kill_switch",
                f"kill switch present at {cfg.kill_switch_path}. Delete it to resume.",
            ))

        age = self._bar_age_days(prices, now)
        if age > cfg.max_bar_age_days:
            notes.append(Note(
                "halt", "stale_data",
                f"last price bar is {age:.1f} days old, limit {cfg.max_bar_age_days}. "
                "Refusing to trade on prices this old.",
                {"age_days": age, "last_bar": str(prices.index[-1])},
            ))

        if cfg.require_market_hours:
            from zoneinfo import ZoneInfo
            now_ny = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(NY))

            if now_ny.year > HOLIDAY_TABLE_THROUGH:
                notes.append(Note(
                    "halt", "market_hours",
                    f"the holiday table only runs to {HOLIDAY_TABLE_THROUGH} and it "
                    f"is now {now_ny.year}. Update US_MARKET_HOLIDAYS rather than "
                    "trading blind through an unlisted closure.",
                ))
            else:
                ok, why = trade_window(now_ny, cfg.trade_window_minutes)
                if not ok:
                    notes.append(Note(
                        "halt", "market_hours",
                        f"outside the trading window: {why}. Orders sent now would "
                        "queue and fill at the next open, which is not the lag=1 "
                        "close the backtest measured.",
                        {"now_ny": str(now_ny), "reason": why},
                    ))
                else:
                    notes.append(Note("info", "market_hours", why))

        fx, fx_note = self.implied_fx(positions, cash, fx_override=fx_override)
        if fx_note:
            notes.append(fx_note)

        # Account drawdown. Peak is carried in state, so it survives runs.
        peak_acct = state.get("peak_account_equity")
        peak_acct = equity if peak_acct is None else max(float(peak_acct), equity)
        state["peak_account_equity"] = peak_acct
        acct_dd = 0.0 if peak_acct <= 0 else 1.0 - equity / peak_acct
        if acct_dd > cfg.max_drawdown_account:
            notes.append(Note(
                "halt", "drawdown_account",
                f"account equity is {acct_dd:.1%} below its peak, limit "
                f"{cfg.max_drawdown_account:.1%}.",
                {"equity": equity, "peak": peak_acct, "drawdown": acct_dd},
            ))

        # Strategy drawdown, immune to deposits and withdrawals.
        nav = float(state.get("strategy_nav", 1.0))
        peak_nav = float(state.get("peak_strategy_nav", nav))
        strat_dd = 0.0 if peak_nav <= 0 else 1.0 - nav / peak_nav
        if strat_dd > cfg.max_drawdown_strategy:
            notes.append(Note(
                "halt", "drawdown_strategy",
                f"modelled strategy NAV is {strat_dd:.1%} below its peak, limit "
                f"{cfg.max_drawdown_strategy:.1%}.",
                {"nav": nav, "peak": peak_nav, "drawdown": strat_dd},
            ))

        if any(n.kind == "halt" for n in notes):
            state["last_run"] = str(now or datetime.now(timezone.utc))
            return RiskDecision([], notes, True, equity, fx, 0.0, state)

        # ---- SHAPING. Build the order list and trim it. -----------------

        if equity <= 0:
            notes.append(Note("halt", "equity", "account equity is zero or negative"))
            return RiskDecision([], notes, True, equity, fx, 0.0, state)

        last_px = prices.iloc[-1]

        # 1. Whitelist. Drop unknown names to cash. Deliberately NOT
        #    renormalised: renormalising would silently push the dropped
        #    weight into the surviving names and increase concentration
        #    without anyone asking for it.
        #
        #    Note this is NOT gated on cfg.enabled. The other checks are
        #    limits, and a limit is something you can choose to relax. This
        #    is a prerequisite: without an entry in the map there is no
        #    T212 symbol to put in the order body, so there is no trade to
        #    place whatever the config says.
        targets: dict[str, float] = {}
        for sym, w in target_weights.dropna().items():
            if sym in self.whitelist:
                targets[str(sym)] = float(w)
            else:
                notes.append(Note(
                    "drop", "whitelist",
                    f"{sym} is not in the confirmed universe map, dropping to cash "
                    f"(weight {float(w):.2%}). Book will be under-invested by this much.",
                    {"symbol": sym, "weight": float(w)},
                ))

        # 2. Max position size. Clip; residual stays in cash.
        if cfg.enabled:
            for sym, w in list(targets.items()):
                if w > cfg.max_position_weight:
                    notes.append(Note(
                        "clip", "max_position",
                        f"{sym} target {w:.2%} exceeds cap {cfg.max_position_weight:.2%}, "
                        f"clipped (shortfall {w - cfg.max_position_weight:.2%} to cash)",
                        {"symbol": sym, "requested": w, "allowed": cfg.max_position_weight},
                    ))
                    targets[sym] = cfg.max_position_weight

        # 3. Current weights from the live portfolio.
        current: dict[str, float] = {}
        for p in positions:
            sym = self.map.to_yf.get(p.ticker)
            if sym is None:
                notes.append(Note(
                    "info", "unknown_holding",
                    f"holding {p.ticker} is not in the universe map. Not traded by "
                    "this system; left alone.",
                    {"ticker": p.ticker, "quantity": p.quantity},
                ))
                continue
            current[sym] = (p.quantity * p.current_price * fx) / equity

        # 4. Build orders from weight deltas.
        orders: list[Order] = []
        for sym in sorted(set(targets) | set(current)):
            tgt = targets.get(sym, 0.0)
            cur = current.get(sym, 0.0)
            delta = tgt - cur
            if abs(delta) < 1e-12:
                continue
            px = float(last_px.get(sym, np.nan))
            if not np.isfinite(px) or px <= 0:
                notes.append(Note(
                    "drop", "no_price",
                    f"{sym} has no usable price in the latest bar, skipping",
                    {"symbol": sym},
                ))
                continue
            qty = (delta * equity) / (px * fx)

            # Round to the broker's precision BEFORE anything downstream
            # reads the notional, and recompute the weight the rounded
            # quantity actually represents. Rounding after the turnover and
            # cash checks would mean those checks validated a different
            # order to the one placed.
            rounded = truncate(qty, cfg.quantity_decimals)
            if rounded == 0.0:
                notes.append(Note(
                    "drop", "quantity_precision",
                    f"{sym} order of {qty:.6f} truncates to zero at "
                    f"{cfg.quantity_decimals} dp, skipping",
                    {"symbol": sym, "requested_qty": qty},
                ))
                continue
            tgt = cur + (rounded * px * fx) / equity
            orders.append(Order(
                yf_symbol=sym,
                t212_ticker=self.map.to_t212[sym],
                quantity=rounded,
                price=px,
                fx=fx,
                target_weight=tgt,
                current_weight=cur,
                quantity_decimals=cfg.quantity_decimals,
            ))

        # 5. Minimum notional. Dust orders cost spread and achieve nothing.
        if cfg.enabled:
            kept = []
            for o in orders:
                if o.notional < cfg.min_order_notional:
                    notes.append(Note(
                        "drop", "min_notional",
                        f"{o.yf_symbol} order of {o.notional:,.2f} is below the "
                        f"{cfg.min_order_notional:,.2f} minimum, skipping",
                        {"symbol": o.yf_symbol, "notional": o.notional},
                    ))
                else:
                    kept.append(o)
            orders = kept

        # 6. Cash sufficiency. Buys only; sells free cash rather than use it.
        if cfg.enabled:
            budget = float(cash.free) * (1.0 - cfg.cash_buffer)
            if cfg.count_sell_proceeds:
                budget += sum(o.notional for o in orders if o.quantity < 0)
            orders = self._clip_buys_to_budget(orders, budget, equity, notes)

        # 7. Turnover cap, one-way, matching backtest/engine.py convention.
        turnover = self._one_way_turnover(orders, equity)
        if cfg.enabled and turnover > cfg.max_turnover_per_run:
            orders, turnover = self._clip_to_turnover(
                orders, equity, cfg.max_turnover_per_run, notes
            )

        # 8. Order count.
        if cfg.enabled and len(orders) > cfg.max_orders_per_run:
            orders.sort(key=self._priority)
            dropped = orders[cfg.max_orders_per_run:]
            orders = orders[: cfg.max_orders_per_run]
            notes.append(Note(
                "clip", "max_orders",
                f"{len(dropped) + len(orders)} orders exceeds the "
                f"{cfg.max_orders_per_run} per-run cap; dropped "
                f"{', '.join(o.yf_symbol for o in dropped)} (lowest priority)",
                {"dropped": [o.yf_symbol for o in dropped]},
            ))
            turnover = self._one_way_turnover(orders, equity)

        orders.sort(key=self._priority)

        # Record intended book for the next NAV roll-forward.
        state["last_weights"] = {
            **{s: 0.0 for s in current},
            **{o.yf_symbol: o.current_weight + o.weight_delta for o in orders},
        }
        for sym, w in current.items():
            state["last_weights"].setdefault(sym, w)
        state["last_price_date"] = str(prices.index[-1])
        state["last_run"] = str(now or datetime.now(timezone.utc))

        return RiskDecision(orders, notes, False, equity, fx, turnover, state)

    # ----------------------------------------------------------- clipping

    @staticmethod
    def _one_way_turnover(orders: Iterable[Order], equity: float) -> float:
        if equity <= 0:
            return 0.0
        traded = sum(o.notional for o in orders)
        return (traded / equity) / 2.0

    def _clip_buys_to_budget(
        self, orders: list[Order], budget: float, equity: float, notes: list[Note]
    ) -> list[Order]:
        buys = [o for o in orders if o.quantity > 0]
        sells = [o for o in orders if o.quantity < 0]
        want = sum(o.notional for o in buys)
        # Tolerance only for floating-point noise. Quantities are truncated
        # toward zero, so rounding can never push the total ABOVE the target;
        # anything materially over budget is a real breach.
        if want <= budget * (1.0 + 1e-9):
            return orders

        notes.append(Note(
            "clip", "cash",
            f"buys total {want:,.2f} against an available {budget:,.2f}; "
            "trimming in priority order",
            {"requested": want, "budget": budget},
        ))

        buys.sort(key=self._priority)
        remaining = budget
        out: list[Order] = []
        for o in buys:
            if remaining <= 0:
                notes.append(Note(
                    "drop", "cash",
                    f"{o.yf_symbol} buy dropped, no cash left", {"symbol": o.yf_symbol},
                ))
                continue
            if o.notional <= remaining:
                out.append(o)
                remaining -= o.notional
            else:
                trimmed = o.scaled_to_notional(remaining, equity)
                notes.append(Note(
                    "clip", "cash",
                    f"{o.yf_symbol} buy trimmed from {o.notional:,.2f} to "
                    f"{trimmed.notional:,.2f}",
                    {"symbol": o.yf_symbol, "from": o.notional, "to": trimmed.notional},
                ))
                if trimmed.notional >= self.config.min_order_notional:
                    out.append(trimmed)
                remaining = 0.0
        return sells + out

    def _clip_to_turnover(
        self, orders: list[Order], equity: float, cap: float, notes: list[Note]
    ) -> tuple[list[Order], float]:
        """
        Trim the order list until one-way turnover is at or below the cap,
        walking in priority order and partially filling the marginal order
        so the result sits exactly on the cap rather than somewhere below it.
        """
        budget_notional = cap * equity * 2.0     # undo the one-way halving
        orders = sorted(orders, key=self._priority)
        out: list[Order] = []
        remaining = budget_notional
        before = self._one_way_turnover(orders, equity)

        for o in orders:
            if remaining <= 0:
                notes.append(Note(
                    "drop", "turnover",
                    f"{o.yf_symbol} dropped, turnover cap exhausted",
                    {"symbol": o.yf_symbol},
                ))
                continue
            if o.notional <= remaining:
                out.append(o)
                remaining -= o.notional
            else:
                trimmed = o.scaled_to_notional(remaining, equity)
                if trimmed.notional >= self.config.min_order_notional:
                    out.append(trimmed)
                notes.append(Note(
                    "clip", "turnover",
                    f"{o.yf_symbol} trimmed from {o.notional:,.2f} to "
                    f"{trimmed.notional:,.2f} to fit the cap",
                    {"symbol": o.yf_symbol},
                ))
                remaining = 0.0

        after = self._one_way_turnover(out, equity)
        notes.append(Note(
            "clip", "turnover",
            f"one-way turnover {before:.2%} exceeds the {cap:.2%} cap, "
            f"clipped to {after:.2%}. The book will be partially rebalanced and "
            "will not match the backtested weights.",
            {"before": before, "after": after, "cap": cap},
        ))
        return out, after


# -------------------------------------------------------- guarded trader


class GuardedTrader:
    """
    The wrapper. Holds the client, calls the engine, places what it approved.

    It contains no risk logic of its own. That is the point: everything
    that decides anything lives in RiskEngine where it can be tested
    without a broker.
    """

    def __init__(
        self,
        client: "T212Client",
        engine: RiskEngine | None = None,
        config: RiskConfig | None = None,
    ) -> None:
        self.client = client
        self.engine = engine or RiskEngine(config)

    def snapshot(self) -> tuple[list["Position"], "Cash"]:
        """Two calls, ~37s of rate limiting. Do this once per run."""
        return self.client.portfolio(), self.client.cash()

    def rebalance(
        self,
        target_weights: pd.Series,
        prices: pd.DataFrame,
        dry_run: bool = True,
    ) -> RiskDecision:
        positions, cash = self.snapshot()
        state = load_state(self.engine.config.state_path)
        fx_rate, fx_err = fetch_gbpusd_rate()
        if fx_err is not None:
            log.warning("fx fallback fetch failed: %s", fx_err)
        decision = self.engine.evaluate(
            target_weights, positions, cash, prices, state, fx_override=fx_rate
        )

        for note in decision.notes:
            (log.error if note.kind == "halt" else log.info)("%s", note)

        if decision.halted:
            save_state(decision.state, self.engine.config.state_path)
            self._log_run(decision, [])
            if not dry_run:
                raise RiskHalt(decision)
            return decision

        if dry_run:
            log.info("DRY RUN. %s", decision.summary())
            return decision

        for o in decision.approved:
            try:
                resp = self.client.market_order(o.t212_ticker, o.quantity)
                decision.placed.append(
                    {"ticker": o.t212_ticker, "quantity": o.quantity, "response": resp}
                )
            except Exception as exc:
                # Stop on the first failure. A partially placed list is
                # already a divergence; continuing compounds it blindly.
                log.error("order failed for %s, stopping: %s", o.t212_ticker, exc)
                decision.execution_error = f"{o.t212_ticker}: {exc}"
                decision.notes.append(Note(
                    "halt", "execution",
                    f"order for {o.t212_ticker} failed after "
                    f"{len(decision.placed)} of {len(decision.approved)} placed; "
                    f"the rest were abandoned: {exc}",
                    {"ticker": o.t212_ticker, "placed": len(decision.placed)},
                ))
                break

        save_state(decision.state, self.engine.config.state_path)
        self._log_run(decision, decision.placed)
        return decision

    @staticmethod
    def _log_run(decision: RiskDecision, placed: list[dict]) -> None:
        """Append the whole run to JSONL for milestone 7 reconciliation."""
        path = ROOT / "logs" / "rebalance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "halted": decision.halted,
            "equity": decision.equity,
            "fx": decision.fx,
            "turnover": decision.turnover,
            "approved": [asdict(o) for o in decision.approved],
            "placed": placed,
            "execution_error": decision.execution_error,
            "notes": [asdict(n) for n in decision.notes],
        }
        with path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
