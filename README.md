# t212-quant

A modular Python framework for systematic equity strategies, with a Trading 212
broker integration, a vectorised backtester and a hard risk-limit layer.

Built against the Trading 212 Public API (beta). Runs against the paper-trading
environment by default.

## Why it's structured this way

The Trading 212 API is an **execution and account** API, not a market data API.
There is no price feed and no historical OHLCV endpoint. So the system is split
into three layers that know nothing about each other:

```
data/      market data in  (yfinance, or any other source)
strategies/ signal generation: prices -> target weights
t212/      execution out: target weights -> orders
```

Swapping the data source or the broker touches one layer, not the whole codebase.

Rate limits also shape the design. The documented limits are roughly 1 request
per 30s for account info, 1 per 5s for portfolio, 1 per 1-2s for orders and 6
per minute for history. That rules out anything intraday. The system is built
as a **scheduled rebalancer**, not a tick-driven trader, and the client enforces
those intervals itself rather than waiting to be told off with a 429.

## Setup

```bash
cd t212-quant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then fill in your key
```

Generate an API key in the Trading 212 app under Settings > API (Beta). You will
be shown an API Key and an API Secret; the secret is displayed **once only**.
Tick these scopes:

| Scope | Needed for |
|---|---|
| `account` | cash balance, account info |
| `portfolio` | open positions |
| `orders:read` | viewing orders |
| `orders:execute` | placing and cancelling orders |
| `metadata` | the tradeable instrument universe |
| `history:orders` | fill history, needed for the P&L attribution later |

Restrict the key to your IP if you can. Put nothing in the repo: `.env` is
gitignored, keep it that way.

## Verify the connection

```bash
python -m scripts.smoke_test
```

Read-only, places no orders. It takes about a minute because it deliberately
respects the account rate limit. If every section prints OK, you have a working
foundation.

A 403 means a missing scope, not a bad key. Regenerate with the right
permissions ticked.

## Safety model

Three independent guards, because one is never enough:

1. `T212_ENVIRONMENT=demo` points the base URL at the paper environment.
2. `T212_ALLOW_LIVE` must be exactly `true` before any order method will run
   against live. Otherwise it raises `LiveTradingBlocked`.
3. The risk layer (milestone 5) applies position and turnover caps regardless
   of what the strategy asks for.

Reads against live are always permitted. It's order placement that's gated.

## Running it

```bash
python -m scripts.smoke_test       # 1. verify the broker connection
python -m scripts.build_universe   # 2a. resolve tickers to T212 format
python -m scripts.fetch_data       # 2b. pull prices, check quality

python -m scripts.find_ticker meta   # search the instrument list by name

python -m scripts.test_engine        # verify the backtester is correct
python -m scripts.test_strategies    # verify the strategies are causal
python -m scripts.backtest --plot    # 3+4. plain momentum vs benchmark
python -m scripts.research --plot    # 4b. does risk management help?
python -m scripts.diagnose           # 4c. WHY did it help or not?
python -m scripts.signal_sweep       # 4d. pre-specified grid + null test
```

If `build_universe` reports a name it can't resolve, use `find_ticker` to
find the real ticker, then pin it in `OVERRIDES` in `marketdata/universe.py`.
If T212 genuinely doesn't offer it, drop it from the universe instead.

## Roadmap

- [x] **1. Broker client** — auth, rate limiting, retries, live-order guard
- [x] **2. Data layer** — 60-name universe, adjusted daily bars cached to
      parquet, quality checks, validated yfinance -> T212 ticker mapping
- [x] **3. Strategy interface** — abstract base returning target weights;
      cross-sectional momentum, short-term reversal, equal-weight benchmark
- [x] **4. Backtester** — explicit execution lag, portfolio drift, cash leg,
      turnover and cost accounting; 24 tests against analytical answers
- [x] **4c. Diagnostics** — regressions testing the assumptions behind vol
      targeting, with Newey-West errors for overlapping windows
- [x] **4d. Signal sweep** — pre-specified grid of momentum and reversal
      specifications, with a simulated best-of-N null to correct for
      multiple testing
- [x] **4b. Risk-managed momentum** — inverse-vol weighting and Barroso &
      Santa-Clara volatility targeting, with a strict causality test suite
- [x] **5. Risk layer** — max position size, max daily turnover, stale-data
      halt, kill switch
- [x] **6. Rebalance loop** — scheduled run: compute targets, diff against the
      live portfolio, emit the minimal order set, log every decision
- [ ] **7. Reconciliation** — pull fill history, compare realised execution
      against the backtest's assumed fills, quantify the slippage gap

Step 7 is the one that makes this interesting to a trading desk. Backtest-versus-
live divergence is the actual problem in systematic trading, and most student
projects stop before they get there.

## Disclaimer

Educational project. Not investment advice. The API is in beta and Trading 212
provides it as-is with liability sitting with the user. Anything you run against
a live account is your own decision and your own risk.
