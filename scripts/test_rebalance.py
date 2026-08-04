"""
Rebalance loop tests. Run:  python -m scripts.test_rebalance

The loop is the one component that can lose money, so the properties worth
testing are the ones whose failure is expensive rather than merely wrong:

  - a dry run places NOTHING, even when the order list is perfectly legal
  - --execute places exactly the approved list and nothing else
  - a halt places nothing even with --execute set
  - the specification fingerprint catches an edited parameter
  - the cash buffer scaling stops the cash limit firing on every run

No network and no credentials. The broker and the price loader are both
replaced with fakes, so this exercises the real decision path and none of
the real side effects.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from marketdata.universe import TICKERS  # noqa: E402
from risk.limits import RiskConfig  # noqa: E402
import scripts.rebalance as rb  # noqa: E402

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


# ------------------------------------------------------------------ fakes


@dataclass(frozen=True)
class FakeCash:
    free: float = 10_000.0
    invested: float = 0.0
    total: float = 10_000.0
    blocked: float = 0.0
    ppl: float = 0.0


@dataclass
class FakeClient:
    """Records every order rather than sending it."""
    environment: str = "demo"
    allow_live: bool = False
    placed: list = field(default_factory=list)

    def portfolio(self):
        return []

    def cash(self):
        return FakeCash()

    def market_order(self, ticker, quantity):
        self.placed.append((ticker, quantity))
        return {"id": len(self.placed), "ticker": ticker, "quantity": quantity}


def fake_prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.DatetimeIndex(pd.bdate_range(end=pd.Timestamp.now().date(), periods=700))
    data = 100 * np.exp(
        np.cumsum(rng.normal(0.0004, 0.015, (len(idx), len(TICKERS))), axis=0)
    )
    return pd.DataFrame(data, index=idx, columns=TICKERS)


def install_fakes(monkey_state: dict, tmp: Path, kill: bool = False) -> FakeClient:
    """Point the module's collaborators at fakes and a temp directory."""
    client = FakeClient()
    prices = fake_prices()
    tmp.mkdir(parents=True, exist_ok=True)

    monkey_state["client"] = client
    rb.build_client = lambda: client
    rb.loader.load_prices = lambda *a, **k: (prices, prices)

    kill_path = tmp / "HALT"
    if kill:
        kill_path.write_text("halted for test\n")

    real_cfg = RiskConfig
    rb.RiskConfig = lambda **kw: real_cfg(
        **{**kw,
           "state_path": tmp / "state.json",
           "kill_switch_path": kill_path,
           "require_market_hours": False}
    )
    rb.ROOT = tmp
    return client


# ------------------------------------------------------------------ tests


def test_fingerprint_detects_an_edited_parameter():
    section("1. SPECIFICATION FINGERPRINT")
    base = rb.spec_hash(rb.DEPLOYED_SPEC)

    changed = dict(rb.DEPLOYED_SPEC, n_long=13)
    check("changing n_long changes the hash", rb.spec_hash(changed) != base)

    cosmetic = dict(rb.DEPLOYED_SPEC, authority="somewhere else")
    check("prose fields do not change the hash", rb.spec_hash(cosmetic) == base)

    reordered = {k: rb.DEPLOYED_SPEC[k] for k in reversed(list(rb.DEPLOYED_SPEC))}
    check("key order does not change the hash", rb.spec_hash(reordered) == base)
    print(f"\n  current fingerprint: {base}")


def test_dry_run_places_nothing(tmp_path: Path):
    section("2. DRY RUN PLACES NOTHING")
    state: dict = {}
    client = install_fakes(state, tmp_path / "dry")
    rb.SPEC_FINGERPRINT = rb.spec_hash(rb.DEPLOYED_SPEC)

    code = rb.main([])
    check("exit code 0", code == 0, str(code))
    check("no orders reached the broker", client.placed == [],
          str(client.placed[:3]))


def test_execute_places_the_approved_list(tmp_path: Path):
    section("3. --execute PLACES EXACTLY THE APPROVED ORDERS")
    state: dict = {}
    client = install_fakes(state, tmp_path / "exec")
    rb.SPEC_FINGERPRINT = rb.spec_hash(rb.DEPLOYED_SPEC)

    code = rb.main(["--execute"])
    check("exit code 0", code == 0, str(code))
    check("12 orders placed, matching n_long", len(client.placed) == 12,
          f"got {len(client.placed)}")
    check("all buys from an empty book",
          all(q > 0 for _, q in client.placed))
    check("all tickers are T212 format",
          all(t.endswith("_EQ") for t, _ in client.placed),
          str([t for t, _ in client.placed][:3]))


def test_halt_places_nothing_even_with_execute(tmp_path: Path):
    section("4. A HALT BEATS --execute")
    state: dict = {}
    client = install_fakes(state, tmp_path / "halt", kill=True)
    rb.SPEC_FINGERPRINT = rb.spec_hash(rb.DEPLOYED_SPEC)

    try:
        code = rb.main(["--execute"])
    except Exception as exc:
        code = 1
        check("halt surfaced rather than passing silently", "HALT" in str(exc).upper()
              or "kill" in str(exc).lower(), str(exc)[:80])
    else:
        check("exit code 1 signals the halt", code == 1, str(code))

    check("nothing was placed", client.placed == [], str(client.placed[:3]))


def test_cash_buffer_scaling_prevents_a_permanent_clip(tmp_path: Path):
    section("5. CASH BUFFER SCALING STOPS THE CASH LIMIT FIRING EVERY RUN")
    from risk.limits import RiskEngine, TickerMap

    cfg = RiskConfig(state_path=tmp_path / "s.json",
                     kill_switch_path=tmp_path / "NOHALT",
                     require_market_hours=False)
    e = RiskEngine(cfg, TickerMap.load())
    prices = fake_prices()
    strat = rb.build_strategy()
    row = strat.generate_weights(prices).iloc[-1].dropna()

    unscaled = e.evaluate(row, [], FakeCash(), prices)
    scaled = e.evaluate(row * (1.0 - cfg.cash_buffer), [], FakeCash(), prices)

    unscaled_clip = any(n.check == "cash" for n in unscaled.notes)
    scaled_clip = any(n.check == "cash" for n in scaled.notes)

    check("a 100%-invested target does trip the cash limit", unscaled_clip)
    check("the scaled target does not", not scaled_clip)
    achieved = sum(o.target_weight for o in scaled.approved)
    check("scaled book is 98% invested, within rounding",
          abs(achieved - 0.98) < 5e-3, f"{achieved:.6f}")

    # Rounding is not free, and the residual should be visible rather than
    # assumed away. It is also the reason the check above is not exact.
    # The invariant truncation buys us: never larger than intended.
    check("no executed order exceeds its intended size",
          all(abs(o.target_weight - o.current_weight)
              <= abs(row[o.yf_symbol] * (1 - cfg.cash_buffer)) + 1e-12
              for o in scaled.approved))
    check("every quantity is at the broker's precision",
          all(o.quantity == round(o.quantity, cfg.quantity_decimals)
              for o in scaled.approved),
          str([o.quantity for o in scaled.approved][:3]))
    print(f"\n  rounding residual: {0.98 - achieved:+.6f} of book "
          f"({(0.98 - achieved) * 10_000:+,.2f} on 10,000)")


def test_failed_execution_is_not_reported_as_success(tmp_path: Path):
    """
    The bug that reached the broker. A job that reports success after
    placing nothing is worse than one that crashes, because nobody looks.
    """
    section("6. A FAILED ORDER IS REPORTED AS A FAILURE")

    class FailingClient(FakeClient):
        def market_order(self, ticker, quantity):
            if len(self.placed) >= 3:
                raise RuntimeError("[400] quantity-precision-mismatch")
            return super().market_order(ticker, quantity)

    state: dict = {}
    install_fakes(state, tmp_path / "fail")
    client = FailingClient()
    rb.build_client = lambda: client
    rb.SPEC_FINGERPRINT = rb.spec_hash(rb.DEPLOYED_SPEC)

    code = rb.main(["--execute"])

    check("exit code 3 flags a partial fill", code == 3, str(code))
    check("stopped at the first failure, did not plough on",
          len(client.placed) == 3, str(len(client.placed)))
    check("not reported as clean", not rb._last_decision.executed_cleanly)
    check("the failure was recorded on the decision",
          rb._last_decision.execution_error is not None,
          str(rb._last_decision.execution_error)[:60])


def main() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    original_fp = rb.SPEC_FINGERPRINT
    try:
        test_fingerprint_detects_an_edited_parameter()
        test_dry_run_places_nothing(tmp)
        test_execute_places_the_approved_list(tmp)
        test_halt_places_nothing_even_with_execute(tmp)
        test_cash_buffer_scaling_prevents_a_permanent_clip(tmp)
        test_failed_execution_is_not_reported_as_success(tmp)
    finally:
        rb.SPEC_FINGERPRINT = original_fp

    print(f"\n{'=' * 70}")
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
