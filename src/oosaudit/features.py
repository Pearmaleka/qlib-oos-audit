"""Daily Rank IC of each raw Alpha158 feature against the label."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
_ROLLING = re.compile(r"^([A-Z]+?)(5|10|20|30|60)$")


def family(name: str) -> tuple[str, int | None]:
    """Operator family and rolling window of an Alpha158 column name.

    ``KMID2`` and ``OPEN0`` are single-day features (window ``None``);
    ``CORD10`` is family ``CORD`` with window 10.
    """
    if name.startswith("K"):
        return "KBAR", None
    if name.endswith("0") and name[:-1] in ("OPEN", "HIGH", "LOW", "VWAP"):
        return "PRICE", None
    m = _ROLLING.match(name)
    if m is None:
        return name, None
    return m.group(1), int(m.group(2))


def _rank_ic_one(x: np.ndarray, y: np.ndarray, day: np.ndarray, n_days: int) -> np.ndarray:
    ok = ~(np.isnan(x) | np.isnan(y))
    df = pd.DataFrame({"d": day[ok], "x": x[ok], "y": y[ok]})
    r = df.groupby("d")[["x", "y"]].rank(method="average")
    d = df["d"].to_numpy()
    rx, ry = r["x"].to_numpy(), r["y"].to_numpy()
    cnt = np.bincount(d, minlength=n_days).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.bincount(d, rx, n_days) / cnt
        my = np.bincount(d, ry, n_days) / cnt
        ex, ey = rx - mx[d], ry - my[d]
        cov = np.bincount(d, ex * ey, n_days)
        vx = np.bincount(d, ex * ex, n_days)
        vy = np.bincount(d, ey * ey, n_days)
        out = cov / np.sqrt(vx * vy)
    out[cnt < 20] = np.nan
    return out


def feature_rank_ic(raw: pd.DataFrame) -> pd.DataFrame:
    """Dates by features matrix of daily Spearman correlations with the label."""
    dates = raw.index.get_level_values("datetime")
    codes, uniq = pd.factorize(dates, sort=True)
    y = raw[("label", "LABEL0")].to_numpy(dtype=float)
    cols = [c for c in raw.columns if c[0] == "feature"]
    out = {}
    for c in cols:
        out[c[1]] = _rank_ic_one(raw[c].to_numpy(dtype=float), y, codes, len(uniq))
    res = pd.DataFrame(out, index=pd.DatetimeIndex(uniq, name="datetime"))
    return res
