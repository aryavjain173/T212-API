"""
Thin client for the Trading 212 Public API (beta).

Design notes:
  - Rate limits are per-endpoint-family and strict, so we enforce them
    client-side rather than relying on 429s to tell us we got it wrong.
  - Auth has two supported styles. Older keys are a single token sent
    raw in the Authorization header. Newer keys are a key+secret pair
    sent as HTTP Basic. We detect which based on whether a secret is set.
  - Any order-placing method refuses to run against the live environment
    unless allow_live=True was passed explicitly at construction. This is
    a deliberate speed bump, not real security.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterator

import requests

log = logging.getLogger(__name__)

DEMO_URL = "https://demo.trading212.com"
LIVE_URL = "https://live.trading212.com"

# Minimum seconds between calls, per endpoint family.
# Padded slightly above the documented limits to leave headroom.
RATE_LIMITS: dict[str, float] = {
    "account": 31.0,
    "portfolio": 6.0,
    "orders": 2.5,
    "history": 11.0,
    "metadata": 31.0,
    "pies": 6.0,
    "exports": 31.0,
}
DEFAULT_LIMIT = 2.0


class T212Error(RuntimeError):
    """Raised when the API returns an error we can't transparently retry."""

    def __init__(self, status: int, message: str, payload: Any = None):
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.payload = payload


class LiveTradingBlocked(RuntimeError):
    """Raised when an order is attempted on live without explicit opt-in."""


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: float
    average_price: float
    current_price: float
    ppl: float  # unrealised P&L in account currency

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @classmethod
    def from_api(cls, d: dict) -> "Position":
        return cls(
            ticker=d["ticker"],
            quantity=float(d.get("quantity", 0.0)),
            average_price=float(d.get("averagePrice", 0.0)),
            current_price=float(d.get("currentPrice", 0.0)),
            ppl=float(d.get("ppl", 0.0)),
        )


@dataclass(frozen=True)
class Cash:
    free: float
    invested: float
    total: float
    blocked: float
    ppl: float

    @classmethod
    def from_api(cls, d: dict) -> "Cash":
        return cls(
            free=float(d.get("free", 0.0)),
            invested=float(d.get("invested", 0.0)),
            total=float(d.get("total", 0.0)),
            blocked=float(d.get("blocked") or 0.0),
            ppl=float(d.get("ppl", 0.0)),
        )


class RateLimiter:
    """Blocking, per-key minimum-interval limiter. Thread safe."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = Lock()

    def wait(self, key: str) -> None:
        interval = RATE_LIMITS.get(key, DEFAULT_LIMIT)
        with self._lock:
            last = self._last.get(key)
            now = time.monotonic()
            if last is not None:
                sleep_for = interval - (now - last)
                if sleep_for > 0:
                    log.debug("rate limit: sleeping %.1fs for '%s'", sleep_for, key)
                    time.sleep(sleep_for)
            self._last[key] = time.monotonic()


class T212Client:
    def __init__(
        self,
        api_key: str,
        api_secret: str | None = None,
        environment: str = "demo",
        allow_live: bool = False,
        timeout: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        if environment not in {"demo", "live"}:
            raise ValueError("environment must be 'demo' or 'live'")
        if environment == "live" and not allow_live:
            log.warning(
                "Client pointed at LIVE but allow_live=False. "
                "Reads will work; order placement will raise."
            )

        self.environment = environment
        self.allow_live = allow_live
        self.base_url = DEMO_URL if environment == "demo" else LIVE_URL
        self.timeout = timeout
        self.max_retries = max_retries

        self._limiter = RateLimiter()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": self._build_auth(api_key, api_secret),
                "Content-Type": "application/json",
                "User-Agent": "t212-quant/0.1",
            }
        )

    @staticmethod
    def _build_auth(api_key: str, api_secret: str | None) -> str:
        if api_secret:
            token = base64.b64encode(
                f"{api_key}:{api_secret}".encode("utf-8")
            ).decode("ascii")
            return f"Basic {token}"
        # Legacy single-token style: the key goes in the header verbatim.
        return api_key

    # ---------------------------------------------------------------- core

    def _request(
        self,
        method: str,
        path: str,
        limit_key: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"

        for attempt in range(1, self.max_retries + 1):
            self._limiter.wait(limit_key)
            try:
                resp = self._session.request(
                    method, url, params=params, json=json_body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise T212Error(0, f"network failure: {exc}") from exc
                backoff = 2 ** attempt
                log.warning("network error (%s), retrying in %ss", exc, backoff)
                time.sleep(backoff)
                continue

            if resp.status_code == 429:
                backoff = 2 ** attempt * 5
                log.warning("429 rate limited on %s, backing off %ss", path, backoff)
                time.sleep(backoff)
                continue

            if resp.status_code == 401:
                raise T212Error(401, "bad API key or secret", resp.text)
            if resp.status_code == 403:
                raise T212Error(
                    403, f"missing scope for {path} - check key permissions", resp.text
                )
            if not resp.ok:
                raise T212Error(resp.status_code, resp.text[:500])

            if not resp.content:
                return None
            try:
                return resp.json()
            except ValueError:
                return resp.text

        raise T212Error(429, f"gave up after {self.max_retries} attempts on {path}")

    def _guard_live_orders(self) -> None:
        if self.environment == "live" and not self.allow_live:
            raise LiveTradingBlocked(
                "Refusing to place a real-money order. Construct the client with "
                "allow_live=True if that is genuinely what you intend."
            )

    # ------------------------------------------------------------- account

    def account_info(self) -> dict:
        return self._request("GET", "/api/v0/equity/account/info", "account")

    def cash(self) -> Cash:
        return Cash.from_api(self._request("GET", "/api/v0/equity/account/cash", "account"))

    # ----------------------------------------------------------- portfolio

    def portfolio(self) -> list[Position]:
        raw = self._request("GET", "/api/v0/equity/portfolio", "portfolio") or []
        return [Position.from_api(p) for p in raw]

    def position(self, ticker: str) -> Position | None:
        try:
            raw = self._request(
                "GET", f"/api/v0/equity/portfolio/{ticker}", "portfolio"
            )
        except T212Error as exc:
            if exc.status == 404:
                return None
            raise
        return Position.from_api(raw) if raw else None

    # ------------------------------------------------------------ metadata

    def instruments(self) -> list[dict]:
        """Full tradeable universe. Large payload, cache it locally."""
        return self._request("GET", "/api/v0/equity/metadata/instruments", "metadata")

    def exchanges(self) -> list[dict]:
        return self._request("GET", "/api/v0/equity/metadata/exchanges", "metadata")

    # -------------------------------------------------------------- orders

    def open_orders(self) -> list[dict]:
        return self._request("GET", "/api/v0/equity/orders", "orders") or []

    def get_order(self, order_id: int | str) -> dict:
        return self._request("GET", f"/api/v0/equity/orders/{order_id}", "orders")

    def cancel_order(self, order_id: int | str) -> Any:
        return self._request("DELETE", f"/api/v0/equity/orders/{order_id}", "orders")

    def market_order(self, ticker: str, quantity: float) -> dict:
        """Negative quantity sells. Fractional quantities are allowed."""
        self._guard_live_orders()
        log.info("MARKET %s qty=%s (%s)", ticker, quantity, self.environment)
        return self._request(
            "POST",
            "/api/v0/equity/orders/market",
            "orders",
            json_body={"ticker": ticker, "quantity": quantity},
        )

    def limit_order(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        time_validity: str = "DAY",
    ) -> dict:
        self._guard_live_orders()
        log.info(
            "LIMIT %s qty=%s @ %s (%s)", ticker, quantity, limit_price, self.environment
        )
        return self._request(
            "POST",
            "/api/v0/equity/orders/limit",
            "orders",
            json_body={
                "ticker": ticker,
                "quantity": quantity,
                "limitPrice": limit_price,
                "timeValidity": time_validity,
            },
        )

    def stop_order(
        self,
        ticker: str,
        quantity: float,
        stop_price: float,
        time_validity: str = "DAY",
    ) -> dict:
        self._guard_live_orders()
        return self._request(
            "POST",
            "/api/v0/equity/orders/stop",
            "orders",
            json_body={
                "ticker": ticker,
                "quantity": quantity,
                "stopPrice": stop_price,
                "timeValidity": time_validity,
            },
        )

    # ------------------------------------------------------------- history

    def order_history(self, limit: int = 50) -> Iterator[dict]:
        """Paginated order history. Yields items, follows the cursor."""
        cursor = None
        while True:
            params: dict[str, Any] = {"limit": limit}
            if cursor is not None:
                params["cursor"] = cursor
            page = self._request(
                "GET", "/api/v0/equity/history/orders", "history", params=params
            )
            items = page.get("items", []) if isinstance(page, dict) else []
            for item in items:
                yield item
            cursor = (page or {}).get("nextPagePath") or (page or {}).get("cursor")
            if not cursor or not items:
                return

    def transactions(self, limit: int = 50) -> dict:
        return self._request(
            "GET", "/api/v0/history/transactions", "history", params={"limit": limit}
        )
