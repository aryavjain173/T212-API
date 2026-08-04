"""
Determine the quantity precision Trading 212 actually accepts.

    python -m scripts.probe_precision

WHY THIS EXISTS

The first live order was rejected with quantity-precision-mismatch. The
accepted precision is not documented, is not in the instrument metadata,
and the error text ("invalid quantity precision 4") does not say what the
allowance is, only that ours was wrong. So rather than guessing a number
and hardcoding it, this measures it.

That is the same move as everywhere else in this project: the 10bps cost
assumption gets tested against real fills in milestone 7, the lookahead
assumption gets tested by the causality check, and this assumption gets
tested here. An assumption you have not measured is a number you made up.

HOW IT WORKS

It walks decimal places from coarse to fine, placing the smallest order it
can for each, and records which succeed. It stops at the first rejection,
since precision limits are monotonic: if 3dp is refused, 4dp will be too.

It then immediately cancels anything that is still open, and reports what
was filled.

THIS PLACES REAL ORDERS ON WHATEVER ENVIRONMENT YOUR .env POINTS AT.

On demo that is harmless. It refuses to run against live at all, because
the entire purpose is to trigger rejections, and deliberately provoking
rejections with real money is not a thing to do. Check the environment
line it prints before confirming.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.limits import TickerMap  # noqa: E402
from t212.client import T212Error  # noqa: E402
from t212.config import build_client  # noqa: E402

log = logging.getLogger("probe")

# A cheap, liquid, high-volume name, so the test order is as small as
# possible in cash terms and fills predictably.
PROBE_SYMBOL = "CSCO"

# Coarse to fine. 0 decimals is a whole share, which must work.
DECIMALS = [0, 1, 2, 3, 4, 5, 6]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    client = build_client()
    print(f"\nenvironment: {client.environment}   allow_live: {client.allow_live}")

    if client.environment != "demo":
        print(
            "\nRefusing to run against anything but demo. This script works by "
            "provoking\nrejections, which is not something to do with real money. "
            "Set T212_ENVIRONMENT=demo.\n"
        )
        return 2

    ticker = TickerMap.load().to_t212[PROBE_SYMBOL]
    print(f"probe instrument: {ticker}")
    print(
        "\nThis will place small real orders on the demo account, one per "
        "precision\nlevel, and cancel anything left open. Type 'yes' to continue: ",
        end="",
    )
    if input().strip().lower() != "yes":
        print("aborted, nothing placed\n")
        return 1

    results: dict[int, str] = {}
    placed_ids: list = []

    for dp in DECIMALS:
        # Smallest representable quantity at this precision, plus a whole
        # share so the order has enough notional to be accepted on grounds
        # other than precision. This isolates the variable being measured.
        qty = round(1 + (10 ** -dp if dp > 0 else 0), dp)
        try:
            resp = client.market_order(ticker, qty)
            results[dp] = "accepted"
            if isinstance(resp, dict) and resp.get("id"):
                placed_ids.append(resp["id"])
            print(f"  {dp} dp   qty {qty:<12} ACCEPTED")
        except T212Error as exc:
            detail = str(exc)
            results[dp] = f"rejected: {detail[:90]}"
            print(f"  {dp} dp   qty {qty:<12} REJECTED  {detail[:70]}")
            if "precision" in detail.lower():
                print(f"\n  precision limit found: {dp - 1} decimal places\n")
                break

    # Tidy up. Market orders usually fill rather than rest, so most of these
    # will 404. That is fine and expected.
    for oid in placed_ids:
        try:
            client.cancel_order(oid)
        except T212Error:
            pass

    accepted = [dp for dp, r in results.items() if r == "accepted"]
    print("=" * 66)
    if accepted:
        best = max(accepted)
        print(f"Highest accepted precision: {best} decimal places.\n")
        print(f"Set this in risk/limits.py:\n\n    quantity_decimals: int = {best}\n")
        if best < 2:
            print(
                "Note: at this precision, position sizing on high-priced names is\n"
                "coarse. On a 5,000 book a 1-share step in a 1,200 stock is 24% of\n"
                "the book, so the position cap and min notional will be doing more\n"
                "work than intended. Worth checking the plan output carefully.\n"
            )
    else:
        print("Nothing was accepted. Something other than precision is wrong.\n")

    print("Whatever you set, the demo account now holds a few probe shares.")
    print("Flatten them before the next rebalance, or the first run will")
    print("generate sell orders you did not intend.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
