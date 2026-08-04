"""
Market data layer: fetch daily bars, cache them, and check they're sane.

Two things this module exists to protect you from.

1. Silent corporate-action errors. A 4-for-1 split looks exactly like a 75%
   crash if you use raw closes. We use adjusted closes throughout, which
   handle splits and dividends, so a momentum signal measures return rather
   than share-count arithmetic. This single issue invalidates a large share
   of amateur backtests.

2. Lookahead bias creeping in through the data layer. Everything here is
   daily bars stamped at the close. Anything that turns bars into positions
   must lag the signal by at least one bar, and the backtester enforces
   that. Getting it wrong produces spectacular fake returns.

Caching is to parquet because it round-trips dtypes and a DatetimeIndex
without mangling them, unlike CSV, and because refetching 60 names on every
run is slow and rude to the data provider.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRICES_FILE = CACHE_DIR / "prices.parquet"
VOLUME_FILE = CACHE_DIR / "volume.parquet"

# If the cache is older than this many hours, refetch rather than trust it.
STALE_HOURS = 18


class DataError(RuntimeError):
    """Raised when data fails a quality check we refuse to trade through."""


# ------------------------------------------------------------------ fetch


def fetch_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily adjusted closes and volumes.

    Returns (prices, volumes), both wide DataFrames indexed by date with
    one column per ticker.
    """
    import yfinance as yf

    log.info("fetching %d tickers from %s", len(tickers), start)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # splits and dividends folded into OHLC
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        raise DataError("yfinance returned nothing. Check your connection.")

    # With multiple tickers yfinance returns a column MultiIndex of
    # (field, ticker). With a single ticker it returns flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        volumes = raw["Volume"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers
        volumes = raw[["Volume"]].copy()
        volumes.columns = tickers

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    volumes.index = pd.to_datetime(volumes.index).tz_localize(None)

    prices = prices.sort_index()
    volumes = volumes.sort_index().reindex(columns=prices.columns)

    prices, volumes = drop_partial_bar(prices, volumes)

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        log.warning("no data returned for: %s", ", ".join(missing))

    return prices, volumes


def drop_partial_bar(
    prices: pd.DataFrame, volumes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove today's bar if the US session hasn't closed yet.

    yfinance happily returns a bar for the current session, but it's a
    live intraday snapshot rather than a close. Left in, it silently
    corrupts every signal computed from it, and worse, the number changes
    every time you refetch, so a backtest stops being reproducible.

    US equities close at 16:00 America/New_York. We drop the final row if
    it's dated today and that time hasn't passed.
    """
    if prices.empty:
        return prices, volumes

    try:
        from zoneinfo import ZoneInfo
        now_ny = datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover - zoneinfo should always exist
        return prices, volumes

    last = prices.index[-1].date()
    if last == now_ny.date() and now_ny.hour < 16:
        log.warning(
            "dropping partial bar for %s (US session still open, %02d:%02d NY)",
            last, now_ny.hour, now_ny.minute,
        )
        prices = prices.iloc[:-1]
        volumes = volumes.iloc[:-1]

    return prices, volumes


def save_cache(prices: pd.DataFrame, volumes: pd.DataFrame) -> None:
    prices.to_parquet(PRICES_FILE)
    volumes.to_parquet(VOLUME_FILE)
    log.info("cached %d rows x %d cols to %s", *prices.shape, CACHE_DIR)


def cache_age_hours() -> float | None:
    if not PRICES_FILE.exists():
        return None
    mtime = datetime.fromtimestamp(PRICES_FILE.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 3600


def load_prices(
    tickers: list[str] | None = None,
    refresh: bool = False,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load from cache, refetching when the cache is missing, stale or forced.

    This is the function everything else should call.
    """
    age = cache_age_hours()
    need_fetch = refresh or age is None or age > STALE_HOURS

    if need_fetch:
        if tickers is None:
            raise DataError("cache miss and no tickers given to fetch")
        prices, volumes = fetch_prices(tickers, start=start)
        save_cache(prices, volumes)
    else:
        log.info("using cache (%.1fh old)", age)
        prices = pd.read_parquet(PRICES_FILE)
        volumes = pd.read_parquet(VOLUME_FILE)

    return prices, volumes


# ---------------------------------------------------------------- quality


def quality_report(prices: pd.DataFrame, volumes: pd.DataFrame) -> dict:
    """
    Check the data before it reaches a strategy. Returns a dict of findings.

    Nothing here is clever. It's the boring stuff that silently ruins
    backtests when nobody checks it.
    """
    report: dict = {}
    report["rows"] = len(prices)
    report["cols"] = prices.shape[1]
    report["start"] = str(prices.index.min().date())
    report["end"] = str(prices.index.max().date())

    # Columns that are entirely empty: a bad ticker symbol.
    dead = prices.columns[prices.isna().all()].tolist()
    report["dead_tickers"] = dead

    # Per-ticker missing-data rate, ignoring the leading NaNs that a later
    # listing date legitimately produces.
    gaps = {}
    for col in prices.columns:
        s = prices[col]
        first = s.first_valid_index()
        if first is None:
            continue
        live = s.loc[first:]
        rate = live.isna().mean()
        if rate > 0.01:
            gaps[col] = round(float(rate), 4)
    report["gappy_tickers"] = gaps

    # Implausible single-day moves. Adjusted data should not produce these.
    # If it does, suspect an unadjusted split or a bad print.
    rets = prices.pct_change(fill_method=None)
    extreme = {}
    for col in rets.columns:
        hits = rets[col].abs() > 0.40
        if hits.any():
            dates = rets.index[hits].strftime("%Y-%m-%d").tolist()
            extreme[col] = dates[:5]
    report["extreme_moves"] = extreme

    # Stale prices: identical closes several days running usually means a
    # halt, a delisting, or forward-filled junk.
    stale = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < 10:
            continue
        run = (s.diff() == 0).astype(int)
        longest = int(run.groupby((run != run.shift()).cumsum()).sum().max() or 0)
        if longest >= 5:
            stale[col] = longest
    report["stale_runs"] = stale

    # Recency: is the most recent bar actually recent?
    last = prices.index.max()
    report["days_since_last_bar"] = int((datetime.now() - last).days)

    return report


def print_quality_report(report: dict) -> None:
    print(f"rows            : {report['rows']:,}")
    print(f"tickers         : {report['cols']}")
    print(f"date range      : {report['start']} -> {report['end']}")
    print(f"last bar age    : {report['days_since_last_bar']} days")

    def block(label: str, val) -> None:
        if not val:
            print(f"{label:<16}: none")
        else:
            print(f"{label:<16}: {val}")

    block("dead tickers", report["dead_tickers"])
    block("gappy tickers", report["gappy_tickers"])
    block("extreme moves", report["extreme_moves"])
    block("stale runs", report["stale_runs"])


# ------------------------------------------------------------- transforms


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. fill_method=None so gaps stay NaN, not zero."""
    return prices.pct_change(fill_method=None)


def trailing_return(
    prices: pd.DataFrame, lookback: int, skip: int = 0
) -> pd.DataFrame:
    """
    Return over `lookback` bars, ending `skip` bars ago.

    The skip is what makes this momentum rather than a mess. Classic
    12-1 momentum measures the 12 months to one month ago, deliberately
    excluding the most recent month because short-horizon returns tend to
    reverse rather than persist. Including it dilutes the signal.

    lookback=252, skip=21  -> 12-1 momentum on daily bars
    lookback=5,   skip=0   -> last-week return, for short-term reversal
    """
    if skip:
        end = prices.shift(skip)
        begin = prices.shift(skip + lookback)
    else:
        end = prices
        begin = prices.shift(lookback)
    return end / begin - 1.0


def realised_vol(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Annualised trailing volatility, for risk scaling later."""
    return to_returns(prices).rolling(window).std() * np.sqrt(252)


def dollar_volume(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 21):
    """
    Average traded value. A liquidity filter you'll want before sizing
    anything, and a useful sanity check that a name is really tradeable.
    """
    return (prices * volumes).rolling(window).mean()


def fetch_gbpusd_rate() -> tuple[float, str | None]:
    """
    GBP-per-USD rate, for RiskEngine's cold-start FX fallback.

    yfinance's GBPUSD=X quotes USD per GBP (e.g. ~1.27), the opposite of
    what implied_fx needs, so this returns the reciprocal.

    Returns (rate, error). error is None on success; rate is 0.0 on
    failure, which the caller must treat as unusable rather than as a
    literal rate of zero.
    """
    try:
        import yfinance as yf
        data = yf.Ticker("GBPUSD=X").history(period="1d")
        if data.empty:
            return 0.0, "yfinance returned no data for GBPUSD=X"
        usd_per_gbp = float(data["Close"].iloc[-1])
        if usd_per_gbp <= 0:
            return 0.0, f"yfinance returned a non-positive rate: {usd_per_gbp}"
        return 1.0 / usd_per_gbp, None
    except Exception as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"
