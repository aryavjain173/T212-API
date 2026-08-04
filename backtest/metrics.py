"""
Performance statistics.

Deliberately a small set. A page of thirty ratios is a way of hiding that
you don't know which three matter. These are the ones you'll be asked
about, and you should be able to say what each one misses.

Sharpe. Excess return per unit of volatility. Its blind spot is that it
treats upside and downside volatility identically and assumes returns are
roughly normal, which equity strategies are not. It flatters anything that
sells tail risk.

Max drawdown. Peak-to-trough loss. The number that actually determines
whether a strategy is survivable, because it's what a real allocator (or
your own nerve) reacts to. A 0.9 Sharpe with a 55% drawdown is unrunnable.

Turnover. Annualised fraction of the book traded. Determines cost
sensitivity and tells you how much of the gross return is at risk from
your cost assumption being wrong.

Hit rate. Fraction of positive days. Mostly diagnostic. High hit rate with
low returns means you're picking up pennies; the opposite means a few big
days carry everything, which is fragile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Annualised Sharpe. rf is an annual rate, converted to daily.

    Note this is a naive Sharpe on daily data. It ignores autocorrelation,
    which inflates the figure for strategies with persistent positions.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Sharpe but penalising only downside deviation."""
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_detail(returns: pd.Series) -> dict:
    """Where the worst drawdown happened and how long recovery took."""
    dd = drawdown_series(returns)
    trough = dd.idxmin()
    peak = dd.loc[:trough]
    peak = peak[peak == 0].index[-1] if (peak == 0).any() else dd.index[0]
    after = dd.loc[trough:]
    recovered = after[after >= -1e-9]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "depth": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "days_underwater": (
            int((recovery - peak).days) if recovery is not None else None
        ),
    }


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else 0.0


def hit_rate(returns: pd.Series) -> float:
    live = returns[returns != 0]
    if len(live) == 0:
        return 0.0
    return float((live > 0).mean())


def ann_turnover(turnover: pd.Series) -> float:
    if len(turnover) == 0:
        return 0.0
    return float(turnover.sum() / len(turnover) * TRADING_DAYS)


# ------------------------------------------------------- significance


def sharpe_se(returns: pd.Series) -> float:
    """
    Standard error of the annualised Sharpe ratio.

    Uses the asymptotic result from Lo (2002) for iid returns:

        SE(SR_period) = sqrt( (1 + SR_period^2 / 2) / T )

    then annualised by sqrt(252). T is the number of observations.

    The practical upshot, and the reason this belongs in every backtest:
    with ten years of daily data the standard error on a Sharpe ratio is
    roughly 0.3. So a strategy printing 1.0 and one printing 0.75 are not
    distinguishable, and most of the strategy comparisons people make are
    comparing noise.

    Caveat worth knowing: the iid assumption is wrong for real returns,
    which are autocorrelated and fat-tailed. Autocorrelation in particular
    makes the true standard error LARGER than this, so treat the number as
    a floor on your uncertainty rather than the whole of it.
    """
    n = len(returns)
    if n < 3:
        return float("nan")
    sr_period = sharpe(returns) / np.sqrt(TRADING_DAYS)
    return float(np.sqrt((1 + 0.5 * sr_period ** 2) / n) * np.sqrt(TRADING_DAYS))


def sharpe_ci(returns: pd.Series, conf: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the annualised Sharpe."""
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(conf, 1.960)
    s, se = sharpe(returns), sharpe_se(returns)
    return (s - z * se, s + z * se)


def _norm_sf(z: float) -> float:
    """One-sided normal tail probability, without pulling in scipy."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def active_stats(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """
    Paired test of a strategy against its benchmark.

    This is the right comparison, and it is much more powerful than
    comparing two Sharpe ratios side by side. Because the two return
    series are highly correlated (they hold overlapping names in the same
    market), the difference between them is far less noisy than either
    series alone. Testing the DIFFERENCE directly uses that.

    Reports the information ratio, which is active return divided by
    tracking error, and its t-statistic. As a rule of thumb the t-stat is
    IR multiplied by the square root of the number of years, so an IR of
    0.5 needs four years to reach t = 1.0 and sixteen to reach t = 2.0.
    Genuine skill takes a long time to prove.
    """
    idx = strategy.index.intersection(benchmark.index)
    a = (strategy.loc[idx] - benchmark.loc[idx]).dropna()
    if len(a) < 3:
        return {}

    years = len(a) / TRADING_DAYS
    active_ann = float(a.mean() * TRADING_DAYS)
    te = float(a.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ir = active_ann / te if te > 0 else 0.0
    t_stat = ir * np.sqrt(years)
    p_two = 2 * _norm_sf(abs(t_stat))

    return {
        "active_return": active_ann,
        "tracking_error": te,
        "information_ratio": ir,
        "t_stat": float(t_stat),
        "p_value": float(p_two),
        "years": years,
        "significant_5pct": bool(p_two < 0.05),
        "correlation": float(strategy.loc[idx].corr(benchmark.loc[idx])),
    }


def vol_matched(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """
    The comparison that usually settles the argument.

    A strategy that runs hotter than its benchmark will show a higher
    return even with no skill whatsoever, because it is simply taking more
    risk. To strip that out, scale the benchmark's returns by the ratio of
    volatilities so both sit at the same risk level, then compare.

    If the levered benchmark beats the strategy, the strategy's excess
    return was risk, not skill: you could have got the same outcome more
    cheaply by borrowing to hold the benchmark.

    Simplifying assumption: the leverage is free and continuously
    rebalanced. Real financing costs money, so this slightly flatters the
    levered benchmark. At the leverage ratios involved here the effect is
    small, but it is an assumption and you should name it.
    """
    idx = strategy.index.intersection(benchmark.index)
    s, b = strategy.loc[idx], benchmark.loc[idx]

    vs, vb = ann_vol(s), ann_vol(b)
    if vb == 0:
        return {}
    k = vs / vb

    levered = b * k
    return {
        "leverage": float(k),
        "strategy_cagr": cagr(s),
        "benchmark_cagr": cagr(b),
        "levered_benchmark_cagr": cagr(levered),
        "strategy_vol": vs,
        "levered_vol": ann_vol(levered),
        "edge_vs_levered": cagr(s) - cagr(levered),
        "levered_max_dd": max_drawdown(levered),
        "strategy_max_dd": max_drawdown(s),
    }


def print_significance(act: dict, vm: dict) -> None:
    if act:
        print("  PAIRED TEST vs BENCHMARK")
        print(f"    active return      {act['active_return']:>9.2%} pa")
        print(f"    tracking error     {act['tracking_error']:>9.2%}")
        print(f"    information ratio  {act['information_ratio']:>9.2f}")
        print(f"    t-statistic        {act['t_stat']:>9.2f}   "
              f"over {act['years']:.1f} years")
        print(f"    p-value            {act['p_value']:>9.3f}")
        print(f"    correlation        {act['correlation']:>9.2f}")
        verdict = (
            "distinguishable from zero at 5%"
            if act["significant_5pct"]
            else "NOT distinguishable from zero"
        )
        print(f"    verdict            {verdict}")

    if vm:
        print("\n  VOLATILITY-MATCHED COMPARISON")
        print(f"    strategy vol       {vm['strategy_vol']:>9.2%}")
        print(f"    leverage needed    {vm['leverage']:>9.2f}x  "
              f"to match with the benchmark")
        print(f"    strategy CAGR      {vm['strategy_cagr']:>9.2%}")
        print(f"    levered bench CAGR {vm['levered_benchmark_cagr']:>9.2%}")
        print(f"    edge               {vm['edge_vs_levered']:>+9.2%}")
        if vm["edge_vs_levered"] < 0:
            print("    The levered benchmark wins. The strategy's extra")
            print("    return was risk taking, not selection skill.")
        else:
            print("    The strategy beats the levered benchmark, so the")
            print("    excess is not purely a volatility effect.")
        print(f"    max DD strat/lev   {vm['strategy_max_dd']:>9.1%} / "
              f"{vm['levered_max_dd']:.1%}")


def summarise(result) -> dict:
    """Headline statistics for a BacktestResult."""
    r = result.returns
    g = result.gross_returns
    dd = drawdown_detail(r)
    return {
        "name": result.name,
        "start": r.index[0].date(),
        "end": r.index[-1].date(),
        "years": round(len(r) / TRADING_DAYS, 1),
        "total_return": total_return(r),
        "cagr": cagr(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sharpe_se": sharpe_se(r),
        "sharpe_lo": sharpe_ci(r)[0],
        "sharpe_hi": sharpe_ci(r)[1],
        "sortino": sortino(r),
        "max_dd": dd["depth"],
        "dd_peak": dd["peak"].date(),
        "dd_trough": dd["trough"].date(),
        "dd_recovery": dd["recovery"].date() if dd["recovery"] is not None else None,
        "calmar": calmar(r),
        "hit_rate": hit_rate(r),
        "ann_turnover": ann_turnover(result.turnover),
        "cost_drag_pa": float(result.costs.sum() / len(r) * TRADING_DAYS),
        "gross_cagr": cagr(g),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }


def print_summary(s: dict) -> None:
    print(f"  period          {s['start']} -> {s['end']}  ({s['years']}y)")
    print(f"  total return    {s['total_return']:>10.1%}")
    print(f"  CAGR            {s['cagr']:>10.2%}   (gross {s['gross_cagr']:.2%})")
    print(f"  ann vol         {s['ann_vol']:>10.2%}")
    print(f"  Sharpe          {s['sharpe']:>10.2f}   "
          f"95% CI [{s['sharpe_lo']:.2f}, {s['sharpe_hi']:.2f}]  "
          f"+/- {1.96 * s['sharpe_se']:.2f}")
    print(f"  Sortino         {s['sortino']:>10.2f}")
    print(f"  max drawdown    {s['max_dd']:>10.1%}   "
          f"{s['dd_peak']} -> {s['dd_trough']}")
    rec = s["dd_recovery"] or "not recovered"
    print(f"  recovered       {str(rec):>10}")
    print(f"  Calmar          {s['calmar']:>10.2f}")
    print(f"  hit rate        {s['hit_rate']:>10.1%}")
    print(f"  ann turnover    {s['ann_turnover']:>10.1%}")
    print(f"  cost drag pa    {s['cost_drag_pa']:>10.2%}")
    print(f"  best/worst day  {s['best_day']:>9.1%} / {s['worst_day']:.1%}")


def compare(summaries: list[dict]) -> None:
    """Side-by-side table. The comparison is the point, not the level."""
    hdr = f"{'':<28}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>9}{'Turn':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(
            f"{s['name'][:27]:<28}"
            f"{s['cagr']:>8.1%} "
            f"{s['ann_vol']:>8.1%} "
            f"{s['sharpe']:>8.2f} "
            f"{s['max_dd']:>8.1%} "
            f"{s['ann_turnover']:>8.0%}"
        )
