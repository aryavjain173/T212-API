"""
Search the cached Trading 212 instrument list.

Run:  python -m scripts.find_ticker meta
      python -m scripts.find_ticker "berkshire"
      python -m scripts.find_ticker booking raytheon

Use this whenever build_universe reports a name it couldn't resolve. It
searches both the ticker string and the company name, case-insensitively,
so you can find an instrument even when its ticker format is not what you
guessed.

Reads the local cache only. No API calls, no orders.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "instruments.json"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    if not CACHE.exists():
        print(f"No instrument cache at {CACHE}")
        print("Run: python -m scripts.smoke_test")
        return 1

    instruments = json.loads(CACHE.read_text())

    for term in sys.argv[1:]:
        q = term.lower()
        hits = [
            i for i in instruments
            if q in str(i.get("ticker", "")).lower()
            or q in str(i.get("name", "")).lower()
            or q in str(i.get("shortName", "")).lower()
        ]

        print(f"\n{'=' * 70}")
        print(f"'{term}'  -  {len(hits)} match(es)")
        print("=" * 70)

        if not hits:
            print("  nothing found")
            continue

        # US equities first, they're almost always what we want.
        hits.sort(key=lambda i: (
            not str(i.get("ticker", "")).endswith("_US_EQ"),
            str(i.get("ticker", "")),
        ))

        for i in hits[:25]:
            print(
                f"  {str(i.get('ticker')):<22} "
                f"{str(i.get('currencyCode')):<5} "
                f"{str(i.get('shortName') or ''):<10} "
                f"{i.get('name')}"
            )

        if len(hits) > 25:
            print(f"  ... and {len(hits) - 25} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
