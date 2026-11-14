"""Collect predictions into daily ICs and the audit's statistics.

Periods (half-open ``[start, end)``):

* ``published``: 2017-01-01 to 2020-08-01, the Qlib benchmark's test window;
* ``post``: 2020-08-01 to the end of the data, unseen by the benchmark;
* ``recent``: 2023-01-01 to the end, the refit protocol's test window.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import data, metrics
from .run import PRED_DIR

END_EXCL = "2026-11-14"
PERIODS = {
    "published": ("2017-01-01", "2020-08-01"),
    "post": ("2020-08-01", END_EXCL),
    "recent": ("2023-01-01", END_EXCL),
}
ENSEMBLE = -1  # seed label of the seed-averaged score


def label_series() -> pd.Series:
    return data.load_raw_store()[("label", "LABEL0")].rename("label")


def load_preds(protocol: str, model: str) -> dict[int, pd.Series]:
    out = {}
    for p in sorted((PRED_DIR / protocol / model).glob("seed*.parquet")):
        out[int(p.stem[4:])] = pd.read_parquet(p)["score"]
    return out


def seed_ensemble(preds: dict[int, pd.Series]) -> pd.Series:
    """Average of the per-date cross-sectionally z-scored seed scores."""
    z = []
    for s in preds.values():
        g = s.groupby(level="datetime")
        z.append((s - g.transform("mean")) / g.transform("std"))
    return pd.concat(z, axis=1).mean(axis=1).rename("score")


def collect_daily_ic(protocols=("pub", "refit", "roll"), cache: Path | None = None) -> pd.DataFrame:
    """Long table of daily IC / Rank IC for every (protocol, model, seed)."""
    label = label_series()
    rows = []
    for protocol in protocols:
        pdir = PRED_DIR / protocol
        if not pdir.exists():
            continue
        for mdir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            preds = load_preds(protocol, mdir.name)
            if not preds:
                continue
            items = dict(preds)
            # The seed ensemble; for a single seed it is that seed's score.
            items[ENSEMBLE] = seed_ensemble(preds) if len(preds) > 1 else next(iter(preds.values()))
            for seed, score in items.items():
                ic = metrics.daily_ic(score, label).reset_index()
                ic["protocol"], ic["model"], ic["seed"] = protocol, mdir.name, seed
                rows.append(ic)
    out = pd.concat(rows, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
    return out


def in_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    a, b = PERIODS[period]
    d = pd.to_datetime(df["datetime"])
    return df[(d >= pd.Timestamp(a)) & (d < pd.Timestamp(b))]


def summary_table(ic: pd.DataFrame, protocol: str, period: str) -> pd.DataFrame:
    """Per model: mean and std over seeds of the four Qlib signal metrics."""
    sub = in_period(ic[(ic.protocol == protocol) & (ic.seed != ENSEMBLE)], period)
    per_seed = (
        sub.groupby(["model", "seed"])
        .apply(lambda d: pd.Series(metrics.summarize(d).as_dict()), include_groups=False)
        .reset_index()
    )
    agg = per_seed.groupby("model").agg(
        ic=("ic", "mean"),
        ic_sd=("ic", "std"),
        icir=("icir", "mean"),
        rank_ic=("rank_ic", "mean"),
        rank_ic_sd=("rank_ic", "std"),
        rank_icir=("rank_icir", "mean"),
        seeds=("seed", "nunique"),
    )
    return agg.fillna({"ic_sd": 0.0, "rank_ic_sd": 0.0})


def period_difference(x_a: np.ndarray, x_b: np.ndarray) -> tuple[float, float]:
    """Difference of means of two non-overlapping daily series and its HAC SE."""
    d = np.nanmean(x_b) - np.nanmean(x_a)
    se = np.sqrt(metrics.newey_west_se(x_a) ** 2 + metrics.newey_west_se(x_b) ** 2)
    return float(d), float(se)


def paired_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Mean of ``x - y`` over common days, its HAC SE and two-sided p-value."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    m, se = float(d.mean()), metrics.newey_west_se(d)
    return m, se, float(2 * sps.norm.sf(abs(m / se)))


def wide(ic: pd.DataFrame, protocol: str, period: str, seed: int = ENSEMBLE, col: str = "rank_ic") -> pd.DataFrame:
    """Dates by models matrix of a daily metric, on dates every model covers."""
    sub = in_period(ic[(ic.protocol == protocol) & (ic.seed == seed)], period)
    return sub.pivot(index="datetime", columns="model", values=col).dropna()


def seed_variance_share(ic: pd.DataFrame, protocol: str, period: str, col: str = "rank_ic") -> dict[str, float]:
    """One-way random-effects split of run-level mean ``col`` into model and seed parts.

    Uses only models with at least two seeds.  Returns the between-model
    variance component, the within-model (seed) variance, and the share of
    run-to-run variance due to the model (the intraclass correlation).
    """
    sub = in_period(ic[(ic.protocol == protocol) & (ic.seed != ENSEMBLE)], period)
    runs = sub.groupby(["model", "seed"])[col].mean().reset_index()
    counts = runs.groupby("model").seed.nunique()
    runs = runs[runs.model.isin(counts[counts >= 2].index)]
    k = runs.model.nunique()
    n_i = runs.groupby("model").size()
    n = len(runs)
    grand = runs[col].mean()
    means = runs.groupby("model")[col].mean()
    ssb = float((n_i * (means - grand) ** 2).sum())
    ssw = float(((runs[col] - runs.model.map(means)) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (n_i**2).sum() / n) / (k - 1)
    var_model = max((msb - msw) / n0, 0.0)
    return {
        "models": int(k),
        "runs": int(n),
        "var_model": var_model,
        "var_seed": msw,
        "icc": var_model / (var_model + msw) if var_model + msw > 0 else np.nan,
    }


def rank_agreement(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Kendall's tau between two model scores on their common models."""
    common = a.index.intersection(b.index)
    tau, p = sps.kendalltau(a.loc[common], b.loc[common])
    return float(tau), float(p), int(len(common))


def pairwise_tests(w: pd.DataFrame) -> pd.DataFrame:
    """HAC-paired differences in daily Rank IC for every pair of models."""
    rows = []
    for a, b in combinations(w.columns, 2):
        m, se, p = paired_test(w[a].to_numpy(), w[b].to_numpy())
        rows.append({"a": a, "b": b, "diff": m, "se": se, "p": p})
    return pd.DataFrame(rows)


def model_confidence_set(w: pd.DataFrame, size: float = 0.10, block: int = 10, reps: int = 2000, seed: int = 0):
    """Hansen–Lunde–Nason MCS on the loss ``-Rank IC`` (stationary bootstrap)."""
    from arch.bootstrap import MCS

    mcs = MCS(-w, size=size, block_size=block, reps=reps, method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    return mcs.pvalues["Pvalue"].sort_values(ascending=False), list(mcs.included)


def spa_vs(w: pd.DataFrame, benchmark: str, block: int = 10, reps: int = 2000, seed: int = 0) -> dict[str, float]:
    """Hansen's SPA test: does any model beat ``benchmark`` on daily Rank IC?"""
    from arch.bootstrap import SPA

    others = [c for c in w.columns if c != benchmark]
    spa = SPA(-w[benchmark], -w[others], block_size=block, reps=reps, bootstrap="stationary", seed=seed)
    spa.compute()
    p = spa.pvalues
    return {"lower": float(p["lower"]), "consistent": float(p["consistent"]), "upper": float(p["upper"])}
