"""
Milestone 1: prove the connection works.

Run:  python -m scripts.smoke_test

This performs read-only calls only. It places no orders. If every section
prints OK, your key, secret, scopes and rate limiting are all correct and
you have a working foundation to build on.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t212 import T212Error  # noqa: E402
from t212.config import build_client, Settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")


def main() -> int:
    settings = Settings.load()

    section("ENVIRONMENT")
    print(f"environment : {settings.environment}")
    print(f"auth style  : {'key + secret (Basic)' if settings.api_secret else 'legacy single token'}")
    print(f"live orders : {'ENABLED' if settings.allow_live else 'blocked'}")
    if settings.environment == "live":
        print("\n  WARNING: pointed at the live environment.\n")

    client = build_client()
    failures = []

    section("ACCOUNT INFO  (scope: account)")
    try:
        info = client.account_info()
        print(json.dumps(info, indent=2))
        print("OK")
    except T212Error as exc:
        print(f"FAILED: {exc}")
        failures.append("account_info")

    section("CASH  (scope: account)")
    print("this call waits ~30s for the account rate limit, be patient")
    try:
        cash = client.cash()
        print(f"free     : {cash.free:,.2f}")
        print(f"invested : {cash.invested:,.2f}")
        print(f"total    : {cash.total:,.2f}")
        print(f"open P&L : {cash.ppl:,.2f}")
        print("OK")
    except T212Error as exc:
        print(f"FAILED: {exc}")
        failures.append("cash")

    section("PORTFOLIO  (scope: portfolio)")
    try:
        positions = client.portfolio()
        if not positions:
            print("no open positions")
        for p in positions:
            print(
                f"  {p.ticker:<16} qty={p.quantity:<10.4f} "
                f"avg={p.average_price:<10.2f} last={p.current_price:<10.2f} "
                f"mv={p.market_value:<12.2f} ppl={p.ppl:,.2f}"
            )
        print("OK")
    except T212Error as exc:
        print(f"FAILED: {exc}")
        failures.append("portfolio")

    section("OPEN ORDERS  (scope: orders:read)")
    try:
        orders = client.open_orders()
        print(f"{len(orders)} open order(s)")
        for o in orders[:5]:
            print(f"  {o}")
        print("OK")
    except T212Error as exc:
        print(f"FAILED: {exc}")
        failures.append("open_orders")

    section("INSTRUMENT UNIVERSE  (scope: metadata)")
    try:
        instruments = client.instruments()
        print(f"{len(instruments):,} tradeable instruments")
        for inst in instruments[:3]:
            print(f"  {inst.get('ticker'):<18} {inst.get('name')}")
        cache = Path(__file__).resolve().parent.parent / "data" / "instruments.json"
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(instruments))
        print(f"cached to {cache}")
        print("OK")
    except T212Error as exc:
        print(f"FAILED: {exc}")
        failures.append("instruments")

    section("RESULT")
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        print("403 errors mean a missing scope. Regenerate the key with the")
        print("required permissions ticked.")
        return 1
    print("All checks passed. Milestone 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
