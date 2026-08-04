"""
The live rebalance loop. Run:

    python -m scripts.rebalance              # dry run, places nothing
    python -m scripts.rebalance --execute    # actually places orders

WHAT THIS RUNS, AND WHY IT RUNS DAILY

The deployed specification is XSMom(252-21, top 12, monthly), long-only, per
PROJECT_STATUS §8. "Monthly" refers to the SIGNAL: the ranking is recomputed
on the last bar of each month and held in between.

It does not mean the book is only traded monthly. strategies/base.py
forward-fills the target between rebalance dates, so a target is present on
every bar, and backtest/engine.py trades to whatever target is present,
comparing it against the drifted book. The measured portfolio in §5
therefore corrects drift back to target every session and refreshes its
signal monthly. Roughly two thirds of the reported 418% annual turnover is
drift correction rather than signal change.

That is the portfolio that was measured, so that is the portfolio this
loop trades, and it must therefore run every trading session. Running it
monthly would trade a different and unmeasured strategy.

The consequence to keep in mind: only about twelve runs a year change what
you hold. The rest are maintenance. The script labels which is which, so
the runs that matter are visible in the log rather than buried among 250
identical ones.

THE SPECIFICATION IS FINGERPRINTED

§8 makes the point that running live is the only unbiased out-of-sample
evidence this project will ever produce, and that fixing the specification
and not changing it is what makes that evidence valid. A constant at the
top of a file is not a commitment, because constants get edited. So the
parameters are hashed, the hash is asserted at startup, and it is written
into every log record.

Changing a parameter therefore fails loudly and requires setting
T212_SPEC_OVERRIDE=true to proceed, which is recorded. The point is not to
make change impossible. It is to make change impossible to do accidentally,
and impossible to do quietly.

SAFETY

Dry run is the default and --execute is required to place anything. Every
order passes through risk/limits.py first; nothing here calls the client's
order methods directly. The client's own live-order guard sits underneath
that, so reaching a real-money order requires T212_ENVIRONMENT=live,
T212_ALLOW_LIVE=true, and --execute, three independent switches.

Per §8, stay on demo until milestone 7 has measured real slippage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from marketdata import loader  # noqa: E402
from marketdata.universe import TICKERS  # noqa: E402
from risk.limits import (  # noqa: E402
    GuardedTrader,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    RiskHalt,
)
from strategies.momentum import CrossSectionalMomentum  # noqa: E402
from t212.config import build_client  # noqa: E402

log = logging.getLogger("rebalance")

# Last decision from main(), for tests and for interactive inspection.
_last_decision: RiskDecision | None = None

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------- the frozen specification

DEPLOYED_SPEC: dict = {
    "strategy": "CrossSectionalMomentum",
    "lookback": 252,
    "skip": 21,
    "n_long": 12,
    "n_short": 0,
    "rebalance": "M",
    "long_only": True,
    "vol_targeting": False,
    "inverse_vol_weighting": False,
    "frozen_on": "2026-07-30",
    "authority": "PROJECT_STATUS.md section 8",
}

# sha256 of the canonical JSON of the above, excluding the two prose fields.
SPEC_FINGERPRINT = "8b28cf703fbdb9c1"


def spec_hash(spec: dict) -> str:
    payload = {k: v for k, v in spec.items() if k not in {"frozen_on", "authority"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def assert_spec_unchanged() -> tuple[str, bool]:
    """Returns (hash, overridden). Raises unless the hash matches or is overridden."""
    actual = spec_hash(DEPLOYED_SPEC)
    override = os.getenv("T212_SPEC_OVERRIDE", "false").strip().lower() == "true"

    if SPEC_FINGERPRINT == "PLACEHOLDER":
        print(
            "\nThe deployed specification has not been fingerprinted yet.\n"
            f"Set SPEC_FINGERPRINT in {Path(__file__).name} to:\n\n"
            f"    SPEC_FINGERPRINT = \"{actual}\"\n\n"
            "Do this once, now, before the first live run. After that any edit\n"
            "to the parameters above will fail this check.\n"
        )
        raise SystemExit(2)

    if actual != SPEC_FINGERPRINT:
        msg = (
            f"DEPLOYED SPECIFICATION HAS CHANGED.\n"
            f"  expected {SPEC_FINGERPRINT}\n"
            f"  actual   {actual}\n\n"
            "PROJECT_STATUS section 8: running live is the only unbiased "
            "out-of-sample evidence this project will generate, and changing the\n"
            "specification after the fact forfeits it. If the change is deliberate,\n"
            "update SPEC_FINGERPRINT and record what changed and why. To run once\n"
            "without updating it, set T212_SPEC_OVERRIDE=true. That fact is logged."
        )
        if not override:
            raise SystemExit(msg)
        log.warning("SPEC OVERRIDE IN EFFECT\n%s", msg)
    return actual, override


def build_strategy() -> CrossSectionalMomentum:
    return CrossSectionalMomentum(
        lookback=DEPLOYED_SPEC["lookback"],
        skip=DEPLOYED_SPEC["skip"],
        n_long=DEPLOYED_SPEC["n_long"],
        rebalance=DEPLOYED_SPEC["rebalance"],
    )


# ------------------------------------------------------------- presentation


def is_signal_change(strategy, prices: pd.DataFrame) -> bool:
    """
    True if the latest bar is a rebalance bar, meaning the ranking was
    recomputed and the target set may actually differ. Every other run is
    drift maintenance against an unchanged target.
    """
    reb = strategy._rebalance_dates(prices.index, strategy.rebalance)
    return bool(reb.iloc[-1])


def print_plan(decision: RiskDecision, signal_day: bool, dry_run: bool) -> None:
    bar = "=" * 78
    print(f"\n{bar}")
    print("SIGNAL REFRESH (rebalance bar)" if signal_day
          else "DRIFT MAINTENANCE (target unchanged since last rebalance)")
    print(bar)

    if decision.halted:
        print("\nHALTED. Nothing will be placed.\n")
        for n in decision.halts:
            print(f"  {n}")
        print()
        return

    print(f"\nequity {decision.equity:,.2f}   implied FX {decision.fx:.4f}   "
          f"one-way turnover {decision.turnover:.2%}")

    if not decision.approved:
        print("\nNo orders. The book already matches the target within the limits.\n")
    else:
        print(f"\n{'side':<5} {'ticker':<15} {'quantity':>12} {'notional':>12} "
              f"{'now':>8} {'target':>8}")
        print("-" * 78)
        for o in decision.approved:
            print(f"{o.side:<5} {o.t212_ticker:<15} {o.quantity:>12.4f} "
                  f"{o.notional:>12.2f} {o.current_weight:>7.2%} {o.target_weight:>8.2%}")

    shaping = [n for n in decision.notes if n.kind in {"clip", "drop"}]
    if shaping:
        print(f"\n{len(shaping)} risk adjustments:")
        for n in shaping:
            print(f"  {n}")

    dust = sum(
        n.detail.get("notional", 0.0) for n in decision.notes if n.check == "min_notional"
    )
    if dust:
        print(f"\nSkipped as dust: {dust:,.2f} of intended trading. This is the gap "
              f"between\nthe backtested book and the live one, and milestone 7 should "
              f"measure it.")

    if dry_run:
        print("\nDRY RUN, nothing placed. Add --execute to trade.\n")
    elif decision.execution_error:
        print(f"\nEXECUTION FAILED after {len(decision.placed)} of "
              f"{len(decision.approved)} orders.")
        print(f"  {decision.execution_error}")
        print("\nThe book is now PARTIALLY REBALANCED and matches neither the "
              "previous\nnor the intended weights. Resolve before the next run.\n")
    else:
        print(f"\n{len(decision.placed)} orders placed.\n")


def append_run_summary(record: dict) -> None:
    path = ROOT / "logs" / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one rebalance against Trading 212.")
    ap.add_argument("--execute", action="store_true",
                    help="actually place orders. Without this, nothing is placed.")
    ap.add_argument("--no-refresh", action="store_true",
                    help="use the cached prices instead of refetching")
    ap.add_argument("--ignore-market-hours", action="store_true",
                    help="trade outside the pre-close window. Orders will QUEUE "
                         "to the next open and fill away from the close the "
                         "backtest assumed. For manual testing only.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    )

    fingerprint, overridden = assert_spec_unchanged()
    log.info("specification %s%s", fingerprint, "  (OVERRIDDEN)" if overridden else "")

    # 1. Prices. Refresh by default; a stale cache is a halt in the risk
    #    layer, so silently reusing an old one just wastes a run.
    prices, _ = loader.load_prices(TICKERS, refresh=not args.no_refresh)
    log.info("prices: %d bars to %s, %d names",
             len(prices), prices.index[-1].date(), prices.shape[1])

    # 2. Target weights from the frozen specification.
    strategy = build_strategy()
    weights = strategy.generate_weights(prices)
    row = weights.iloc[-1].dropna()
    if row.empty:
        log.error("strategy produced no target for the latest bar (warmup?)")
        return 2
    signal_day = is_signal_change(strategy, prices)

    # 3. Scale to leave the cash buffer unspent. Without this the cash check
    #    clips one name on literally every run, and a limit that always
    #    fires is a limit nobody reads.
    cfg = RiskConfig(require_market_hours=not args.ignore_market_hours)
    if args.ignore_market_hours:
        log.warning(
            "market-hours gate DISABLED. Orders placed outside US hours queue "
            "to the next open and fill away from the close, so this run is not "
            "comparable to the backtest."
        )
    row = row * (1.0 - cfg.cash_buffer)
    log.info("target: %d names, %.2f%% invested, signal_day=%s",
             len(row), row.sum() * 100, signal_day)

    # 4. Risk layer, then the broker.
    client = build_client()
    trader = GuardedTrader(client, RiskEngine(cfg))
    log.info("environment=%s allow_live=%s execute=%s",
             client.environment, client.allow_live, args.execute)

    # A halt raises when executing, so that a caller cannot mistake it for a
    # quiet no-op. Catch it here rather than letting it escape, because the
    # run summary below is the record of WHY nothing traded, and a halt is
    # the run you most want that record for.
    try:
        decision = trader.rebalance(row, prices, dry_run=not args.execute)
    except RiskHalt as halt:
        decision = halt.decision

    global _last_decision
    _last_decision = decision
    print_plan(decision, signal_day, dry_run=not args.execute)

    append_run_summary({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spec": fingerprint,
        "spec_overridden": overridden,
        "environment": client.environment,
        "executed": args.execute and not decision.halted,
        "n_placed": len(decision.placed),
        "execution_error": decision.execution_error,
        "clean": decision.executed_cleanly if args.execute else None,
        "signal_day": signal_day,
        "last_bar": str(prices.index[-1].date()),
        "halted": decision.halted,
        "halt_reasons": [n.check for n in decision.halts],
        "n_orders": len(decision.approved),
        "turnover": decision.turnover,
        "equity": decision.equity,
    })

    if decision.halted:
        return 1
    if args.execute and not decision.executed_cleanly:
        return 3        # placed some but not all; needs a human
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
