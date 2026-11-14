"""Daily information coefficients and their inference.

Conventions follow Qlib's ``SigAnaRecord``: on each date, IC is the Pearson
correlation and Rank IC the Spearman correlation between the model score and
the raw label ``Ref($close, -2) / Ref($close, -1) - 1`` across the stocks in
the universe that day.  ICIR is the mean daily IC over its standard deviation
(not annualised).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def daily_ic(score: pd.Series, label: pd.Series, min_names: int = 20) -> pd.DataFrame:
    """Daily Pearson IC and Spearman Rank IC.

    ``score`` and ``label`` are indexed by ``(datetime, instrument)``.  Rows
    with a missing score or label are dropped; dates with fewer than
    ``min_names`` stocks are skipped.
    """
    df = pd.concat({"s": score, "y": label}, axis=1, join="inner").dropna()
    g = df.groupby(level="datetime")
    n = g.size()
    keep = n[n >= min_names].index
    df = df[df.index.get_level_values("datetime").isin(keep)]
    g = df.groupby(level="datetime")
    ic = g.apply(lambda d: _pearson(d["s"].to_numpy(), d["y"].to_numpy()))
    ranks = g.rank(method="average")
    rank_ic = ranks.groupby(level="datetime").apply(lambda d: _pearson(d["s"].to_numpy(), d["y"].to_numpy()))
    out = pd.DataFrame({"ic": ic, "rank_ic": rank_ic, "n": n.loc[keep]})
    out.index.name = "datetime"
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the mean of ``x`` with Bartlett weights.

    ``lags`` defaults to ``floor(4 (T/100)^(2/9))`` (Newey and West, 1994).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    t = len(x)
    if t < 2:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (t / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    s = e @ e / t
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * (e[k:] @ e[:-k]) / t
    return float(np.sqrt(max(s, 0.0) / t))


@dataclass(frozen=True)
class ICSummary:
    ic: float
    icir: float
    rank_ic: float
    rank_icir: float
    rank_ic_t: float
    days: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def summarize(ic: pd.DataFrame) -> ICSummary:
    """Qlib's four signal metrics plus a Newey–West t-statistic for Rank IC."""
    r = ic["rank_ic"].to_numpy()
    return ICSummary(
        ic=float(ic["ic"].mean()),
        icir=float(ic["ic"].mean() / ic["ic"].std()),
        rank_ic=float(np.nanmean(r)),
        rank_icir=float(ic["rank_ic"].mean() / ic["rank_ic"].std()),
        rank_ic_t=float(np.nanmean(r) / newey_west_se(r)),
        days=int(ic["ic"].notna().sum()),
    )


def window(df: pd.DataFrame | pd.Series, start: str, end: str):
    """Rows whose ``datetime`` level lies in ``[start, end)``."""
    d = df.index.get_level_values("datetime")
    return df[(d >= pd.Timestamp(start)) & (d < pd.Timestamp(end))]
