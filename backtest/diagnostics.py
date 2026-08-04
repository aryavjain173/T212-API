"""
Diagnostics: test the ASSUMPTIONS behind a technique, not just its output.

A backtest tells you a strategy didn't work. It does not tell you why, and
without why you cannot tell the difference between "wrong technique" and
"wrong parameters". These functions test the underlying claims directly,
using regressions rather than simulated portfolios.

The distinction matters because a regression on the raw data has far more
statistical power than a comparison of two backtested equity curves. A
backtest reduces ten years of information to one number; a regression uses
every observation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Univariate OLS with Newey-West standard errors.

    Plain OLS standard errors are badly wrong here because overlapping
    forward-return windows induce serial correlation: if you regress the
    next 21 days of return on something, consecutive observations share 20
    days of data. Ignoring that inflates t-statistics by roughly the square
    root of the overlap, which is how a lot of weak results get published
    as strong ones.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return {}

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    # Newey-West with a lag chosen by the usual rule of thumb.
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = (resid ** 2)[:, None, None] * (X[:, :, None] * X[:, None, :])
    S = S.sum(axis=0)
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1)
        e1, e2 = resid[lag:], resid[:-lag]
        X1, X2 = X[lag:], X[:-lag]
        G = ((e1 * e2)[:, None, None] * (X1[:, :, None] * X2[:, None, :])).sum(axis=0)
        S += w * (G + G.transpose(0, 1) if G.ndim == 2 else G)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))

    t = beta[1] / se[1] if se[1] > 0 else 0.0
    p = 2 * 0.5 * math.erfc(abs(t) / math.sqrt(2))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0

    return {
        "n": n, "alpha": float(beta[0]), "beta": float(beta[1]),
        "se": float(se[1]), "t": float(t), "p": float(p), "r2": float(r2),
        "nw_lags": L,
    }


def vol_predicts_vol(returns: pd.Series, window: int = 126) -> dict:
    """
    Claim 1 behind vol targeting: trailing volatility forecasts future
    volatility.

    This is the part that is almost always true. Volatility clusters, and
    it is one of the most robust regularities in finance. If this fails,
    the technique cannot work at all.
    """
    trail = returns.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    fwd = (
        returns.rolling(window).std().shift(-window) * np.sqrt(TRADING_DAYS)
    )
    r = _ols(trail.to_numpy(), fwd.to_numpy())
    r["label"] = f"trailing {window}d vol -> forward {window}d vol"
    return r


def vol_predicts_return(
    returns: pd.Series, window: int = 126, horizon: int = 21
) -> dict:
    """
    Claim 2 behind vol targeting, and the one that actually decides it:
    high trailing volatility forecasts POOR forward returns.

    Vol targeting cuts exposure when trailing vol is high. That only helps
    if the periods it cuts into were bad ones. If high volatility instead
    preceded strong returns in this sample, the rule must lose money by
    construction, and no choice of window or leverage cap can rescue it.

    A significantly POSITIVE beta here is the smoking gun.
    """
    trail = returns.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    fwd = (
        (1 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon)
        - 1
    )
    r = _ols(trail.to_numpy(), fwd.to_numpy())
    r["label"] = f"trailing {window}d vol -> next {horizon}d return"
    return r


def vol_predicts_sharpe(
    returns: pd.Series, window: int = 126, horizon: int = 63
) -> dict:
    """
    The cleanest version of the test: does high trailing volatility
    forecast a poor forward RISK-ADJUSTED return?

    This is what vol targeting implicitly assumes, since it trades off
    return against risk. Regressing on forward Sharpe tests the assumption
    in the units the technique actually cares about.
    """
    trail = returns.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    fwd_mean = returns.rolling(horizon).mean().shift(-horizon)
    fwd_sd = returns.rolling(horizon).std().shift(-horizon)
    fwd_sharpe = (fwd_mean / fwd_sd.replace(0, np.nan)) * np.sqrt(TRADING_DAYS)
    r = _ols(trail.to_numpy(), fwd_sharpe.to_numpy())
    r["label"] = f"trailing {window}d vol -> forward {horizon}d Sharpe"
    return r


def conditional_returns(
    returns: pd.Series, window: int = 126, horizon: int = 21, n_bins: int = 5
) -> pd.DataFrame:
    """
    Non-parametric version of the same question, which is often more
    persuasive than a regression coefficient.

    Sort every day into quintiles by trailing volatility, then report the
    average forward return in each bucket. If vol targeting is sound, the
    high-vol bucket should show clearly worse forward returns. If the
    buckets look flat, or the high-vol bucket is the BEST, the technique
    has nothing to work with.
    """
    trail = returns.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    fwd = (
        (1 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon)
        - 1
    )
    df = pd.DataFrame({"vol": trail, "fwd": fwd}).dropna()
    if len(df) < n_bins * 20:
        return pd.DataFrame()

    df["bin"] = pd.qcut(df["vol"], n_bins, labels=False, duplicates="drop")
    out = df.groupby("bin").agg(
        n=("fwd", "size"),
        mean_vol=("vol", "mean"),
        mean_fwd=("fwd", "mean"),
        median_fwd=("fwd", "median"),
        sd_fwd=("fwd", "std"),
    )
    out["ann_fwd"] = (1 + out["mean_fwd"]) ** (TRADING_DAYS / horizon) - 1
    return out


def print_regression(r: dict) -> None:
    if not r:
        print("    insufficient data")
        return
    stars = (
        "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05
        else "*" if r["p"] < 0.10 else ""
    )
    print(f"    {r['label']}")
    print(f"      beta {r['beta']:>+9.4f}   t {r['t']:>+6.2f}{stars:<4} "
          f"p {r['p']:.4f}   R2 {r['r2']:.4f}   n {r['n']:,} "
          f"(NW lags {r['nw_lags']})")
