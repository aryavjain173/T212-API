"""
The tradeable universe: 60 large-cap US equities.

Why these, and why sector labels:

Cross-sectional momentum ranks names against each other, so the universe
composition IS a design decision, not a detail. Sixty names gives roughly
six per decile, which is enough that a decile portfolio is diversified
rather than a bet on one company. Below about forty you are mostly
measuring idiosyncratic noise.

Sector labels are carried so the backtest can report how concentrated the
long book gets. Naive momentum has a well-known habit of loading up on
whichever sector has run hardest, which means your "stock selection"
strategy quietly becomes a sector bet. You want to be able to see that.

A caveat you should be able to articulate: this list is chosen with
hindsight from today's large caps, so it carries survivorship bias. Names
that were large caps in 2015 and later collapsed are absent, which flatters
any backtest. Fixing it properly needs point-in-time index constituents,
which are not freely available. Acknowledging the bias honestly is worth
more in an interview than pretending it isn't there.
"""

from __future__ import annotations

# yfinance symbol -> sector
UNIVERSE: dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "AMD": "Technology",
    "CSCO": "Technology",
    "ACN": "Technology",
    "QCOM": "Technology",
    "INTU": "Technology",
    # Communication services
    "GOOGL": "Communication",
    "META": "Communication",
    "NFLX": "Communication",
    "DIS": "Communication",
    "CMCSA": "Communication",
    "T": "Communication",
    "VZ": "Communication",
    # Consumer discretionary
    "AMZN": "Cons Disc",
    "TSLA": "Cons Disc",
    "HD": "Cons Disc",
    "MCD": "Cons Disc",
    "NKE": "Cons Disc",
    "LOW": "Cons Disc",
    "SBUX": "Cons Disc",
    "BKNG": "Cons Disc",
    # Consumer staples
    "WMT": "Cons Staples",
    "PG": "Cons Staples",
    "KO": "Cons Staples",
    "PEP": "Cons Staples",
    "COST": "Cons Staples",
    "PM": "Cons Staples",
    "MDLZ": "Cons Staples",
    # Financials
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "AXP": "Financials",
    # Healthcare
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "PFE": "Healthcare",
    "AMGN": "Healthcare",
    # Industrials
    "CAT": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "UNP": "Industrials",
    "RTX": "Industrials",
    # Energy / Utilities
    "XOM": "Energy",
    "CVX": "Energy",
    "NEE": "Utilities",
}

TICKERS: list[str] = sorted(UNIVERSE.keys())


def sector_of(ticker: str) -> str:
    return UNIVERSE.get(ticker, "Unknown")


def sectors() -> dict[str, list[str]]:
    """Sector -> list of tickers, for concentration reporting."""
    out: dict[str, list[str]] = {}
    for tkr, sec in UNIVERSE.items():
        out.setdefault(sec, []).append(tkr)
    return {k: sorted(v) for k, v in sorted(out.items())}


# Explicit yfinance -> T212 ticker overrides.
#
# The guessed SYMBOL_US_EQ format is right for most US listings and wrong
# for a handful. Rather than making the guessing logic ever more baroque,
# pin the exceptions here once you've confirmed them with:
#     python -m scripts.find_ticker <name>
#
# Leave a comment saying what you confirmed, so future-you doesn't have to
# re-derive it.
#
# All four below were confirmed against the live instrument list on
# 2026-07-29. Note the pattern: every one is a ticker frozen at the time
# of listing and never updated through a later rebrand or merger. This is
# why venues and vendors key off stable identifiers such as ISINs rather
# than tickers, and why "which instrument is this actually" is real work.
OVERRIDES: dict[str, str] = {
    "META":  "FB_US_EQ",     # pre-2021-rebrand Facebook ticker
    "BKNG":  "PCLN_US_EQ",   # pre-2018-rebrand Priceline ticker
    "BRK-B": "BRK_B_US_EQ",  # underscore separator; Class B, NOT BRK/A
    "RTX":   "UTX_US_EQ",    # legacy United Technologies ticker, USD line
}


def to_t212_ticker(yf_symbol: str) -> str:
    """
    Best-guess mapping from a yfinance symbol to the Trading 212 format.

    T212 uses SYMBOL_COUNTRY_TYPE, e.g. AAPL_US_EQ. yfinance uses a hyphen
    for share classes where T212 tends to use nothing or a different
    separator, so BRK-B is the kind of name that needs checking rather
    than trusting.

    This is a GUESS. Always validate the output against the live
    instruments list before trading it. scripts/build_universe.py does
    exactly that.
    """
    if yf_symbol in OVERRIDES:
        return OVERRIDES[yf_symbol]
    base = yf_symbol.replace("-", "")
    return f"{base}_US_EQ"


def drop(*symbols: str) -> None:
    """Remove names from the universe, e.g. ones T212 doesn't offer."""
    for s in symbols:
        UNIVERSE.pop(s, None)
    TICKERS[:] = sorted(UNIVERSE.keys())


if __name__ == "__main__":
    print(f"{len(TICKERS)} tickers")
    for sec, names in sectors().items():
        print(f"  {sec:<14} {len(names):>2}  {' '.join(names)}")
