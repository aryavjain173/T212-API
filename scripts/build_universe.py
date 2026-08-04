"""
Milestone 2a: resolve every universe name to a real Trading 212 ticker.

Run:  python -m scripts.build_universe

Why this is its own step rather than an assumption buried in the code:

T212's ticker format is SYMBOL_COUNTRY_TYPE, and the naive mapping from a
yfinance symbol is right most of the time and wrong in ways that only show
up when you place an order. Share classes are the usual culprit (BRK-B),
and some names simply are not offered on the platform at all.

Discovering that at order time means a failed rebalance and a portfolio
that silently drifts from its target. Discovering it now means editing a
list. This script writes data/universe_map.json, and everything downstream
reads that rather than guessing.

Read-only. Places no orders.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketdata.universe import TICKERS, sector_of, to_t212_ticker  # noqa: E402
from t212 import T212Error  # noqa: E402
from t212.config import build_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENTS_CACHE = ROOT / "data" / "instruments.json"
OUTPUT = ROOT / "data" / "universe_map.json"


def load_instruments(refresh: bool = False) -> list[dict]:
    """Use the cached instrument list if we have it. It's a big payload."""
    if INSTRUMENTS_CACHE.exists() and not refresh:
        logging.info("using cached instrument list")
        return json.loads(INSTRUMENTS_CACHE.read_text())

    logging.info("fetching instrument universe from T212 (rate limited, ~30s)")
    client = build_client()
    instruments = client.instruments()
    INSTRUMENTS_CACHE.parent.mkdir(exist_ok=True)
    INSTRUMENTS_CACHE.write_text(json.dumps(instruments))
    return instruments


def main() -> int:
    try:
        instruments = load_instruments()
    except T212Error as exc:
        print(f"could not load instruments: {exc}")
        return 1

    print(f"{len(instruments):,} instruments available on the platform\n")

    # Index by exact ticker, and by short name for fallback searching.
    by_ticker = {i["ticker"]: i for i in instruments if i.get("ticker")}
    us_equities = {
        t: i for t, i in by_ticker.items() if t.endswith("_US_EQ")
    }

    resolved: dict[str, dict] = {}
    unresolved: list[str] = []

    for yf_sym in TICKERS:
        guess = to_t212_ticker(yf_sym)

        if guess in by_ticker:
            inst = by_ticker[guess]
            resolved[yf_sym] = {
                "t212_ticker": guess,
                "name": inst.get("name"),
                "currency": inst.get("currencyCode"),
                "sector": sector_of(yf_sym),
                "match": "exact",
            }
            continue

        # Fallback: look for a US equity whose ticker starts with the base
        # symbol. Catches share-class naming differences.
        base = yf_sym.replace("-", "").replace(".", "")
        candidates = [
            t for t in us_equities
            if t.split("_")[0] == base or t.split("_")[0].startswith(base)
        ]

        if len(candidates) == 1:
            inst = by_ticker[candidates[0]]
            resolved[yf_sym] = {
                "t212_ticker": candidates[0],
                "name": inst.get("name"),
                "currency": inst.get("currencyCode"),
                "sector": sector_of(yf_sym),
                "match": "fuzzy",
            }
        elif candidates:
            print(f"  AMBIGUOUS  {yf_sym:<8} guessed {guess}")
            for c in candidates[:5]:
                print(f"               candidate: {c}  ({by_ticker[c].get('name')})")
            unresolved.append(yf_sym)
        else:
            print(f"  NOT FOUND  {yf_sym:<8} guessed {guess}")
            unresolved.append(yf_sym)

    exact = sum(1 for v in resolved.values() if v["match"] == "exact")
    fuzzy = sum(1 for v in resolved.values() if v["match"] == "fuzzy")

    print(f"\nresolved   : {len(resolved)}/{len(TICKERS)}"
          f"  ({exact} exact, {fuzzy} fuzzy)")

    if fuzzy:
        print("\nfuzzy matches, check these are the right instrument:")
        for yf_sym, v in resolved.items():
            if v["match"] == "fuzzy":
                print(f"  {yf_sym:<8} -> {v['t212_ticker']:<16} {v['name']}")

    if unresolved:
        print(f"\nunresolved : {', '.join(unresolved)}")
        print("Either drop these from marketdata/universe.py or add an")
        print("explicit override once you've found the right ticker.")

    # Flag anything not priced in USD. Not a blocker, but it means the
    # position sizing has an FX leg you need to think about.
    non_usd = {k: v for k, v in resolved.items() if v.get("currency") != "USD"}
    if non_usd:
        print("\nnot quoted in USD:")
        for k, v in non_usd.items():
            print(f"  {k:<8} {v['currency']}")

    OUTPUT.write_text(json.dumps(resolved, indent=2))
    print(f"\nwrote {OUTPUT}")
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
