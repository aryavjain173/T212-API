"""
Risk layer tests. Run:  python -m scripts.test_limits

Same discipline as test_engine.py: every limit is checked against an answer
worked out independently, and every test is one that could fail. A risk
check that has never been observed to block anything is not a risk check,
it is a comment.

Three things are tested that are easy to skip and worth having:

  - a NEGATIVE CONTROL. The same inputs are run with enabled=False, and the
    output must differ. If the limits were doing nothing, every other test
    here would still pass and mean nothing.

  - the CLIP LANDS ON THE CAP. Not merely below it. Clipping to something
    conservative and calling it correct hides an off-by-half in the
    one-way turnover convention, which is exactly the kind of error that
    would make the live cap disagree with the backtested one.

  - HALT PRECEDENCE. A halt must place nothing at all, even when the order
    list would otherwise have been perfectly legal.

No network, no API key, no broker. That is the whole reason RiskEngine is
separated from GuardedTrader.
"""

import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from risk.limits import (  # noqa: E402
    Note,
    Order,
    RiskConfig,
    RiskEngine,
    TickerMap,
    advance_strategy_nav,
)

PASS, FAIL = 0, 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}" + (f"   {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {label}   {detail}")


def section(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ------------------------------------------------------------- fixtures
#
# Stand-ins for t212.client.Position and Cash. Duck-typed deliberately, so
# these tests import nothing that needs `requests` or a credential.


@dataclass(frozen=True)
class FakePosition:
    ticker: str
    quantity: float
    current_price: float
    average_price: float = 0.0
    ppl: float = 0.0


@dataclass(frozen=True)
class FakeCash:
    free: float
    invested: float
    total: float
    blocked: float = 0.0
    ppl: float = 0.0


SYMBOLS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO"]


def fake_map() -> TickerMap:
    return TickerMap({
        s: {"t212_ticker": f"{s}_US_EQ", "name": s, "currency": "USD",
            "sector": "X", "match": "exact"}
        for s in SYMBOLS
    })


def fake_prices(last: datetime | None = None, price: float = 100.0) -> pd.DataFrame:
    """Flat prices at a round number, so every hand-calculation is exact."""
    last = last or datetime.now(timezone.utc)
    idx = pd.DatetimeIndex(pd.bdate_range(end=last.date(), periods=300))
    return pd.DataFrame(price, index=idx, columns=SYMBOLS)


def engine(**overrides) -> RiskEngine:
    """
    An engine with paths pointed at a temp dir, so tests never touch the
    real state file or notice a real HALT file.
    """
    tmp = Path(tempfile.mkdtemp())
    # Market hours default to OFF here. Every test above is about a limit,
    # not about the clock, and leaving the gate on would make all of them
    # pass or fail depending on what time the suite happened to be run.
    # Test 17 turns it on explicitly and is the only one that should.
    overrides.setdefault("require_market_hours", False)
    cfg = RiskConfig(
        state_path=tmp / "state.json",
        kill_switch_path=tmp / "HALT",
        **overrides,
    )
    return RiskEngine(cfg, fake_map())


def clean_state() -> dict:
    return {
        "strategy_nav": 1.0, "peak_strategy_nav": 1.0,
        "peak_account_equity": None, "last_weights": {},
        "last_price_date": None, "last_run": None,
    }


def kinds(notes: list[Note], check_name: str) -> list[str]:
    return [n.kind for n in notes if n.check == check_name]


# ---------------------------------------------------------------- tests


def test_flat_book_produces_expected_orders():
    """
    Baseline with no limit binding. Equity 10,000, two names at 25% each,
    price 100, FX 1. Each order must be 2,500 notional, 25 shares.
    If this is wrong, nothing below means anything.
    """
    section("1. BASELINE ORDER CONSTRUCTION")
    e = engine(max_position_weight=1.0, cash_buffer=0.0)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.25, "MSFT": 0.25})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)

    check("not halted", not d.halted, d.summary())
    check("two orders", len(d.approved) == 2, f"got {len(d.approved)}")
    check("each order is 25 shares",
          all(abs(o.quantity - 25.0) < 1e-9 for o in d.approved),
          str([round(o.quantity, 4) for o in d.approved]))
    check("each notional is 2,500",
          all(abs(o.notional - 2_500) < 1e-6 for o in d.approved))
    # One-way turnover: traded 5,000 on 10,000 of equity = 50%, halved = 25%.
    check("one-way turnover is 25%", abs(d.turnover - 0.25) < 1e-9,
          f"{d.turnover:.6f}")


def test_turnover_matches_engine_convention():
    """
    The cap must be on the same scale as backtest/engine.py's reported
    turnover, which is sum(|dw|) / 2. Selling one full 100% position and
    buying another is sum|dw| = 2.0, one-way 1.0, NOT 2.0.
    """
    section("2. TURNOVER CONVENTION MATCHES THE BACKTEST ENGINE")
    e = engine(max_turnover_per_run=10.0, max_position_weight=1.0,
               cash_buffer=0.0, count_sell_proceeds=True)
    cash = FakeCash(free=0, invested=10_000, total=10_000)
    pos = [FakePosition("AAPL_US_EQ", quantity=100.0, current_price=100.0)]
    tgt = pd.Series({"MSFT": 1.0})

    d = e.evaluate(tgt, pos, cash, fake_prices(), clean_state())
    check("full switch is 100% one-way turnover, not 200%",
          abs(d.turnover - 1.0) < 1e-9, f"{d.turnover:.6f}")


def test_max_position_clips_and_leaves_cash():
    """
    A 50% target against a 15% cap must produce a 15% position, and the
    missing 35% must sit in cash rather than being redistributed.
    """
    section("3. MAX POSITION SIZE CLIPS, RESIDUAL GOES TO CASH")
    e = engine(max_position_weight=0.15)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.50, "MSFT": 0.10})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    by_sym = {o.yf_symbol: o for o in d.approved}

    check("AAPL clipped to 15%", abs(by_sym["AAPL"].target_weight - 0.15) < 1e-12,
          f"{by_sym['AAPL'].target_weight:.4f}")
    check("MSFT untouched at 10%", abs(by_sym["MSFT"].target_weight - 0.10) < 1e-12)
    check("clip was logged", "clip" in kinds(d.notes, "max_position"))
    total = sum(o.target_weight for o in d.approved)
    check("book is 25% invested, not renormalised to 60%",
          abs(total - 0.25) < 1e-12, f"{total:.4f}")


def test_turnover_cap_clips_exactly_to_cap():
    """
    Request 60% one-way turnover against a 20% cap. The result must be
    exactly 20%, not merely under it, and sells must survive ahead of buys.
    """
    section("4. TURNOVER CAP CLIPS TO EXACTLY THE CAP, SELLS FIRST")
    e = engine(max_turnover_per_run=0.20, max_position_weight=1.0,
               min_order_notional=0.0)
    cash = FakeCash(free=10_000, invested=4_000, total=10_000)
    pos = [FakePosition("AAPL_US_EQ", quantity=40.0, current_price=100.0)]
    # Sell all of AAPL (0.40), buy MSFT 0.40, buy NVDA 0.40 -> sum|dw| 1.20,
    # one-way 0.60.
    tgt = pd.Series({"MSFT": 0.40, "NVDA": 0.40})

    d = e.evaluate(tgt, pos, cash, fake_prices(), clean_state())

    check("turnover lands exactly on the 20% cap",
          abs(d.turnover - 0.20) < 1e-9, f"{d.turnover:.6f}")
    check("the sell survived the clip",
          any(o.quantity < 0 for o in d.approved),
          str([(o.yf_symbol, round(o.quantity, 2)) for o in d.approved]))
    check("sells are ordered before buys",
          [o.side for o in d.approved] == sorted(
              [o.side for o in d.approved], key=lambda s: s == "BUY"))
    check("clip was logged", "clip" in kinds(d.notes, "turnover"))


def test_min_notional_drops_dust():
    """A 0.01% target on a 10,000 book is 1.00, below a 20.00 floor."""
    section("5. MINIMUM NOTIONAL DROPS DUST ORDERS")
    e = engine(min_order_notional=20.0)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.0001, "MSFT": 0.10})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    syms = {o.yf_symbol for o in d.approved}

    check("dust order dropped", "AAPL" not in syms)
    check("real order kept", "MSFT" in syms)
    check("drop was logged", "drop" in kinds(d.notes, "min_notional"))


def test_whitelist_blocks_unknown_ticker():
    """
    A name absent from universe_map.json must never be traded, and its
    weight must NOT be pushed into the surviving names.
    """
    section("6. WHITELIST BLOCKS UNCONFIRMED TICKERS")
    e = engine(max_position_weight=1.0)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.30, "GME": 0.30})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    syms = {o.yf_symbol for o in d.approved}

    check("unknown ticker not traded", "GME" not in syms)
    check("known ticker still traded", "AAPL" in syms)
    check("AAPL not inflated to absorb the dropped weight",
          abs(d.approved[0].target_weight - 0.30) < 1e-12)
    check("drop was logged", "drop" in kinds(d.notes, "whitelist"))


def test_cash_sufficiency():
    """
    Buys worth 8,000 against 3,000 free cash and a 2% buffer. Budget is
    2,940. Total approved buy notional must not exceed it.
    """
    section("7. CASH SUFFICIENCY TRIMS BUYS TO THE BUDGET")
    e = engine(cash_buffer=0.02, max_position_weight=1.0, max_turnover_per_run=10.0)
    cash = FakeCash(free=3_000, invested=7_000, total=10_000)
    tgt = pd.Series({"AAPL": 0.40, "MSFT": 0.40})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    spend = sum(o.notional for o in d.approved if o.quantity > 0)

    check("buys fit the 2,940 budget", spend <= 2_940 + 1e-6, f"{spend:,.2f}")
    check("budget is actually used, not abandoned", spend > 2_900, f"{spend:,.2f}")
    check("clip was logged", "clip" in kinds(d.notes, "cash"))


def test_max_orders_truncates_by_priority():
    section("8. MAX ORDERS PER RUN TRUNCATES, LOWEST CONVICTION FIRST")
    e = engine(max_orders_per_run=3, max_position_weight=1.0,
               max_turnover_per_run=10.0)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.25, "MSFT": 0.20, "NVDA": 0.15,
                     "JPM": 0.10, "XOM": 0.05})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    syms = [o.yf_symbol for o in d.approved]

    check("truncated to 3 orders", len(d.approved) == 3, str(syms))
    check("kept the three largest targets", set(syms) == {"AAPL", "MSFT", "NVDA"},
          str(syms))
    check("truncation was logged", "clip" in kinds(d.notes, "max_orders"))


def test_kill_switch_halts_everything():
    section("9. KILL SWITCH HALTS, PLACING NOTHING")
    e = engine()
    e.config.kill_switch_path.write_text("stopped by hand\n")
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.25})

    d = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)

    check("halted", d.halted)
    check("no orders at all", len(d.approved) == 0)
    check("halt was logged", "halt" in kinds(d.notes, "kill_switch"))

    e.config.kill_switch_path.unlink()
    d2 = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    check("removing the file resumes trading", not d2.halted and len(d2.approved) == 1)


def test_stale_data_halts():
    section("10. STALE PRICE DATA HALTS")
    e = engine(max_bar_age_days=4)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.25})

    old = datetime.now(timezone.utc) - timedelta(days=30)
    d = e.evaluate(tgt, [], cash, fake_prices(last=old), clean_state(), fx_override=1.0)
    check("30-day-old data halts", d.halted and len(d.approved) == 0)
    check("halt was logged", "halt" in kinds(d.notes, "stale_data"))

    d2 = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    check("today's data does not halt", not d2.halted)


def test_drawdown_breakers_fire_independently():
    """
    Both bases must be able to fire on their own. The account breaker
    responds to equity below its stored peak; the strategy breaker responds
    to modelled NAV below its peak and is immune to deposits.
    """
    section("11. DRAWDOWN BREAKERS, BOTH BASES, INDEPENDENTLY")
    cash_now = FakeCash(free=10_000, invested=0, total=7_000)
    tgt = pd.Series({"AAPL": 0.25})

    # Account basis only: equity 7,000 against a 10,000 peak = -30%.
    e = engine(max_drawdown_account=0.25, max_drawdown_strategy=0.99)
    s = clean_state() | {"peak_account_equity": 10_000}
    d = e.evaluate(tgt, [], cash_now, fake_prices(), s, fx_override=1.0)
    check("account breaker fires at -30% vs a 25% limit", d.halted)
    check("account halt logged", "halt" in kinds(d.notes, "drawdown_account"))

    # Strategy basis only: NAV 0.70 against a 1.00 peak, account at its peak.
    e2 = engine(max_drawdown_account=0.99, max_drawdown_strategy=0.25)
    s2 = clean_state() | {"strategy_nav": 0.70, "peak_strategy_nav": 1.00}
    healthy = FakeCash(free=10_000, invested=0, total=10_000)
    d2 = e2.evaluate(tgt, [], healthy, fake_prices(), s2, fx_override=1.0)
    check("strategy breaker fires while the account looks fine", d2.halted)
    check("strategy halt logged", "halt" in kinds(d2.notes, "drawdown_strategy"))

    # Neither: shallow drawdown on both.
    e3 = engine(max_drawdown_account=0.25, max_drawdown_strategy=0.25)
    s3 = clean_state() | {"peak_account_equity": 10_500, "strategy_nav": 0.95,
                          "peak_strategy_nav": 1.00}
    d3 = e3.evaluate(tgt, [], FakeCash(10_000, 0, 10_000), fake_prices(), s3, fx_override=1.0)
    check("shallow drawdown does not halt", not d3.halted, d3.summary())


def test_strategy_nav_ignores_deposits():
    """
    The reason for tracking a second basis at all. A deposit doubles
    account equity; modelled NAV must not move.
    """
    section("12. MODELLED NAV IS IMMUNE TO DEPOSITS")
    px = fake_prices()
    px.iloc[-1] = 110.0     # every name +10% on the last bar

    s = clean_state() | {
        "strategy_nav": 1.0,
        "peak_strategy_nav": 1.0,
        "last_weights": {"AAPL": 0.5},          # half invested, half cash
        "last_price_date": str(px.index[-2]),
    }
    out = advance_strategy_nav(s, px)

    check("half-invested book earns half the move",
          abs(out["strategy_nav"] - 1.05) < 1e-9, f"{out['strategy_nav']:.6f}")

    # Now the same roll-forward with a deposit reflected in the account.
    e = engine()
    before = e.evaluate(pd.Series({"AAPL": 0.25}), [],
                        FakeCash(10_000, 0, 10_000), px, dict(s), fx_override=1.0)
    after = e.evaluate(pd.Series({"AAPL": 0.25}), [],
                       FakeCash(20_000, 0, 20_000), px, dict(s), fx_override=1.0)
    check("NAV identical across a doubled account balance",
          abs(before.state["strategy_nav"] - after.state["strategy_nav"]) < 1e-12)


def test_implied_fx():
    """
    Positions worth 10,000 USD against a book worth 8,000 GBP implies 0.80.
    Order sizing depends on this, so an unnoticed error here mis-sizes
    every trade by the FX error.
    """
    section("13. IMPLIED FX IS DERIVED FROM THE ACCOUNT, AND BANDED")
    e = engine()
    pos = [FakePosition("AAPL_US_EQ", quantity=100.0, current_price=100.0)]
    cash = FakeCash(free=0, invested=7_500, total=8_000, ppl=500)

    fx, note = e.implied_fx(pos, cash)
    check("implied rate is 0.80", abs(fx - 0.80) < 1e-12, f"{fx:.6f}")
    check("in-band rate raises no note", note is None)

    absurd = FakeCash(free=0, invested=400_000, total=400_000, ppl=0)
    fx2, note2 = e.implied_fx(pos, absurd)
    check("out-of-band rate halts", note2 is not None and note2.kind == "halt",
          f"{fx2:.2f}")


def test_halt_beats_a_legal_order_list():
    """Precedence. A halt must place nothing even when nothing else objects."""
    section("14. HALT PRECEDENCE OVER AN OTHERWISE LEGAL RUN")
    e = engine()
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.10, "MSFT": 0.10})

    ok = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    check("run is legal without the halt", not ok.halted and len(ok.approved) == 2)

    e.config.kill_switch_path.write_text("x")
    halted = e.evaluate(tgt, [], cash, fake_prices(), clean_state(), fx_override=1.0)
    check("same run places nothing once halted",
          halted.halted and len(halted.approved) == 0)
    e.config.kill_switch_path.unlink()


def test_negative_control_limits_actually_bind():
    """
    The test that makes the rest of this file meaningful. Run an input that
    every shaping check should object to, with limits on and off. If the
    two agree, the limits are inert and every PASS above is vacuous.
    """
    section("15. NEGATIVE CONTROL: DISABLING THE LIMITS CHANGES THE OUTPUT")
    cash = FakeCash(free=500, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.60, "MSFT": 0.30, "GME": 0.05, "NVDA": 0.0001})
    px = fake_prices()

    on = engine(max_position_weight=0.15, max_turnover_per_run=0.20,
                min_order_notional=20.0, max_orders_per_run=2)
    off = engine(max_position_weight=0.15, max_turnover_per_run=0.20,
                 min_order_notional=20.0, max_orders_per_run=2, enabled=False)

    d_on = on.evaluate(tgt, [], cash, px, clean_state(), fx_override=1.0)
    d_off = off.evaluate(tgt, [], cash, px, clean_state(), fx_override=1.0)

    check("the whitelist is NOT relaxable, it holds in both runs",
          "GME" not in {o.yf_symbol for o in d_off.approved}
          and "GME" not in {o.yf_symbol for o in d_on.approved})
    check("unconstrained run keeps the full 60% position",
          any(abs(o.target_weight - 0.60) < 1e-12 for o in d_off.approved))
    check("unconstrained run exceeds the turnover cap",
          d_off.turnover > 0.20, f"{d_off.turnover:.4f}")
    check("constrained run is materially different",
          len(d_on.approved) != len(d_off.approved)
          or abs(d_on.turnover - d_off.turnover) > 1e-9,
          f"on={len(d_on.approved)} orders/{d_on.turnover:.4f}, "
          f"off={len(d_off.approved)}/{d_off.turnover:.4f}")
    check("constrained run respects every binding limit",
          d_on.turnover <= 0.20 + 1e-9
          and len(d_on.approved) <= 2
          and all(o.target_weight <= 0.15 + 1e-12 for o in d_on.approved)
          and all(o.notional >= 20.0 for o in d_on.approved)
          and "GME" not in {o.yf_symbol for o in d_on.approved})


def test_unknown_holding_is_left_alone():
    """
    A position the system did not put on, and cannot map, must not be
    liquidated on the assumption that a zero target means sell.
    """
    section("16. HOLDINGS OUTSIDE THE MAP ARE NOT LIQUIDATED")
    e = engine()
    cash = FakeCash(free=5_000, invested=5_000, total=10_000)
    pos = [FakePosition("TSLA_US_EQ", quantity=50.0, current_price=100.0)]
    tgt = pd.Series({"AAPL": 0.20})

    d = e.evaluate(tgt, pos, cash, fake_prices(), clean_state())
    check("no sell order for the unmapped holding",
          "TSLA_US_EQ" not in {o.t212_ticker for o in d.approved})
    check("it was flagged rather than silently ignored",
          "info" in kinds(d.notes, "unknown_holding"))


def test_market_hours_gate():
    """
    Orders sent outside US hours queue to the next open rather than being
    rejected, which quietly shortens the execution lag on every fill. The
    gate must catch weekends, holidays, half days and simply-too-early.
    """
    section("17. TRADING WINDOW GATE")
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")

    e = engine(require_market_hours=True, trade_window_minutes=30)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.10})

    def at(y, m, d, hh, mm):
        when = datetime(y, m, d, hh, mm, tzinfo=NY)
        return e.evaluate(tgt, [], cash, fake_prices(last=when), clean_state(), now=when, fx_override=1.0)

    check("15:45 on a normal session trades", not at(2026, 7, 30, 15, 45).halted)
    check("06:30 is too early", at(2026, 7, 30, 6, 30).halted)
    check("16:30 is after the close", at(2026, 7, 30, 16, 30).halted)
    check("Saturday halts", at(2026, 8, 1, 15, 45).halted)
    check("Labor Day halts", at(2026, 9, 7, 15, 45).halted)

    # The case a naive implementation gets wrong: on a 13:00 half day,
    # 15:45 is nearly three hours after the market shut.
    check("half day at 15:45 halts", at(2026, 11, 27, 15, 45).halted)
    check("half day at 12:45 trades", not at(2026, 11, 27, 12, 45).halted)

    # And the gate must be switchable, or the other 16 tests are hostage
    # to the wall clock.
    off = engine(require_market_hours=False)
    when = datetime(2026, 8, 1, 3, 0, tzinfo=NY)     # Saturday, 3am
    d = off.evaluate(tgt, [], cash, fake_prices(last=when), clean_state(), now=when, fx_override=1.0)
    check("gate off trades at 3am on a Saturday", not d.halted)


def test_holiday_table_expiry():
    """
    A stale holiday table is worse than none, because it looks authoritative
    while treating unlisted closures as ordinary sessions. It must fail
    rather than degrade.
    """
    section("18. THE HOLIDAY TABLE EXPIRES LOUDLY")
    from zoneinfo import ZoneInfo
    from risk.limits import HOLIDAY_TABLE_THROUGH
    NY = ZoneInfo("America/New_York")

    e = engine(require_market_hours=True)
    cash = FakeCash(free=10_000, invested=0, total=10_000)
    tgt = pd.Series({"AAPL": 0.10})

    # A Wednesday afternoon well past the table's coverage.
    when = datetime(HOLIDAY_TABLE_THROUGH + 1, 6, 16, 15, 45, tzinfo=NY)
    d = e.evaluate(tgt, [], cash, fake_prices(last=when), clean_state(), now=when, fx_override=1.0)

    check("halts once past the table's last year", d.halted)
    check("the reason names the table",
          any("holiday table" in n.message for n in d.halts),
          str([n.message[:50] for n in d.halts]))


def main() -> int:
    test_flat_book_produces_expected_orders()
    test_turnover_matches_engine_convention()
    test_max_position_clips_and_leaves_cash()
    test_turnover_cap_clips_exactly_to_cap()
    test_min_notional_drops_dust()
    test_whitelist_blocks_unknown_ticker()
    test_cash_sufficiency()
    test_max_orders_truncates_by_priority()
    test_kill_switch_halts_everything()
    test_stale_data_halts()
    test_drawdown_breakers_fire_independently()
    test_strategy_nav_ignores_deposits()
    test_implied_fx()
    test_halt_beats_a_legal_order_list()
    test_negative_control_limits_actually_bind()
    test_unknown_holding_is_left_alone()
    test_market_hours_gate()
    test_holiday_table_expiry()

    print(f"\n{'=' * 70}")
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
