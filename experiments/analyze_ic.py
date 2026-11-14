"""Signal-level audit: reproduction, decay, staleness, seed noise, MCS/SPA.

Reads the saved predictions, writes ``results/daily_ic.parquet`` and
``results/ic_audit.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oosaudit import audit, metrics
from published import published_table

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

ORDER = [
    "linear",
    "lightgbm",
    "xgboost",
    "catboost",
    "doubleensemble",
    "mlp",
    "gru",
    "lstm",
    "alstm",
    "transformer",
    "localformer",
    "tcn",
]


def ordered(models) -> list[str]:
    return [m for m in ORDER if m in set(models)]


def ens_daily(ic: pd.DataFrame, protocol: str, model: str, period: str) -> pd.DataFrame:
    sub = ic[(ic.protocol == protocol) & (ic.model == model) & (ic.seed == audit.ENSEMBLE)]
    return audit.in_period(sub, period).set_index("datetime")


def reproduction(ic: pd.DataFrame, pub: pd.DataFrame) -> dict:
    ours = audit.summary_table(ic, "pub", "published")
    rows = {}
    for m in ordered(ours.index):
        r = {f"ours_{k}": float(ours.loc[m, k]) for k in ["ic", "ic_sd", "icir", "rank_ic", "rank_ic_sd", "rank_icir"]}
        r["seeds"] = int(ours.loc[m, "seeds"])
        if m in pub.index:
            r.update({f"pub_{k}": float(pub.loc[m, k]) for k in ["ic", "icir", "rank_ic", "rank_icir"]})
        rows[m] = r
    common = [m for m in rows if "pub_ic" in rows[m]]
    tau = {}
    for k in ["ic", "rank_ic", "icir", "rank_icir"]:
        a = pd.Series({m: rows[m][f"pub_{k}"] for m in common})
        b = pd.Series({m: rows[m][f"ours_{k}"] for m in common})
        t, p, n = audit.rank_agreement(a, b)
        tau[k] = {"tau": t, "p": p, "n": n}
    gap = np.array([rows[m]["ours_rank_ic"] - rows[m]["pub_rank_ic"] for m in common])
    return {"models": rows, "kendall_published_vs_ours": tau, "rank_ic_gap_mean_abs": float(np.abs(gap).mean())}


def decay(ic: pd.DataFrame) -> dict:
    a = audit.summary_table(ic, "pub", "published")
    b = audit.summary_table(ic, "pub", "post")
    rows = {}
    for m in ordered(a.index.intersection(b.index)):
        xa = ens_daily(ic, "pub", m, "published")["rank_ic"].to_numpy()
        xb = ens_daily(ic, "pub", m, "post")["rank_ic"].to_numpy()
        d, se = audit.period_difference(xa, xb)
        rows[m] = {
            "published": float(a.loc[m, "rank_ic"]),
            "published_sd": float(a.loc[m, "rank_ic_sd"]),
            "post": float(b.loc[m, "rank_ic"]),
            "post_sd": float(b.loc[m, "rank_ic_sd"]),
            "seeds": int(a.loc[m, "seeds"]),
            "post_t": float(np.nanmean(xb) / metrics.newey_west_se(xb)),
            "change_pct": float(100 * (b.loc[m, "rank_ic"] / a.loc[m, "rank_ic"] - 1)),
            "ens_published": float(np.nanmean(xa)),
            "ens_post": float(np.nanmean(xb)),
            "ens_diff": d,
            "ens_diff_se": se,
            "ic_published": float(a.loc[m, "ic"]),
            "ic_post": float(b.loc[m, "ic"]),
        }
    pa = np.mean([r["published"] for r in rows.values()])
    pb = np.mean([r["post"] for r in rows.values()])
    return {
        "models": rows,
        "mean_published": float(pa),
        "mean_post": float(pb),
        "mean_change_pct": float(100 * (pb / pa - 1)),
    }


def staleness(ic: pd.DataFrame) -> dict:
    """Rank IC on 2023-2026 of stale (pub), refit-once (refit) and annual (roll) models."""
    rows = {}
    for m in ORDER:
        r = {}
        for proto in ("pub", "refit", "roll"):
            sub = ic[(ic.protocol == proto) & (ic.model == m) & (ic.seed != audit.ENSEMBLE)]
            if sub.empty:
                continue
            per_seed = audit.in_period(sub, "recent").groupby("seed")["rank_ic"].mean()
            r[proto] = float(per_seed.mean())
            r[f"{proto}_sd"] = float(per_seed.std()) if len(per_seed) > 1 else 0.0
            x = ens_daily(ic, proto, m, "recent")["rank_ic"].to_numpy()
            r[f"{proto}_ens_t"] = float(np.nanmean(x) / metrics.newey_west_se(x))
        if "refit" in r:
            fresh_pub = ic[(ic.protocol == "pub") & (ic.model == m) & (ic.seed != audit.ENSEMBLE)]
            r["pub_in_published"] = float(
                audit.in_period(fresh_pub, "published").groupby("seed")["rank_ic"].mean().mean()
            )
            if "pub" in r:
                xa = ens_daily(ic, "pub", m, "recent")["rank_ic"]
                xb = ens_daily(ic, "refit", m, "recent")["rank_ic"]
                j = pd.concat([xa, xb], axis=1, join="inner").dropna().to_numpy()
                d, se, p = audit.paired_test(j[:, 1], j[:, 0])
                r["refit_minus_pub"] = {"diff": d, "se": se, "p": p}
            rows[m] = r
    return rows


def seed_noise(ic: pd.DataFrame) -> dict:
    out = {}
    for proto, period in [("pub", "published"), ("pub", "post"), ("refit", "recent")]:
        try:
            out[f"{proto}/{period}"] = audit.seed_variance_share(ic, proto, period)
        except (ZeroDivisionError, ValueError):
            continue
    return out


def tests(ic: pd.DataFrame) -> dict:
    out = {}
    for proto, period in [("pub", "published"), ("pub", "post"), ("refit", "recent")]:
        w = audit.wide(ic, proto, period)
        w = w[ordered(w.columns)]
        if w.shape[1] < 3:
            continue
        pvals, included = audit.model_confidence_set(w)
        pair = audit.pairwise_tests(w)
        k = len(pair)
        holm = np.sort(pair["p"].to_numpy()) <= 0.05 / (k - np.arange(k))
        n_holm = int(np.argmin(holm)) if not holm.all() else k
        res = {
            "days": int(len(w)),
            "mean_rank_ic": {m: float(w[m].mean()) for m in w.columns},
            "hac_se": {m: metrics.newey_west_se(w[m].to_numpy()) for m in w.columns},
            "mcs_pvalues": {m: float(v) for m, v in pvals.items()},
            "mcs_included": included,
            "pairs": int(k),
            "pairs_p05": int((pair["p"] < 0.05).sum()),
            "pairs_holm": n_holm,
            "pairwise": pair.to_dict(orient="records"),
        }
        for bench in ("lightgbm", "linear"):
            if bench in w.columns:
                res[f"spa_vs_{bench}"] = audit.spa_vs(w, bench)
        out[f"{proto}/{period}"] = res
    return out


def rank_stability(ic: pd.DataFrame, pub: pd.DataFrame) -> dict:
    s_pub = audit.summary_table(ic, "pub", "published")["rank_ic"]
    s_post = audit.summary_table(ic, "pub", "post")["rank_ic"]
    out = {}
    pairs = {
        "published_table_vs_ours_published": (pub["rank_ic"], s_pub),
        "ours_published_vs_ours_post": (s_pub, s_post),
        "published_table_vs_ours_post": (pub["rank_ic"], s_post),
    }
    try:
        s_rec = audit.summary_table(ic, "refit", "recent")["rank_ic"]
        pairs["ours_published_vs_refit_recent"] = (s_pub, s_rec)
        pairs["published_table_vs_refit_recent"] = (pub["rank_ic"], s_rec)
    except (KeyError, ValueError):
        pass
    for name, (a, b) in pairs.items():
        t, p, n = audit.rank_agreement(a, b)
        out[name] = {"tau": t, "p": p, "n": n}
    return out


def yearly(ic: pd.DataFrame) -> dict:
    """Mean Rank IC per calendar year, seed ensemble, per protocol and model."""
    sub = ic[ic.seed == audit.ENSEMBLE].copy()
    sub["year"] = pd.to_datetime(sub["datetime"]).dt.year
    g = sub.groupby(["protocol", "model", "year"])["rank_ic"].mean()
    out: dict = {}
    for (proto, m, y), v in g.items():
        out.setdefault(proto, {}).setdefault(m, {})[int(y)] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute daily ICs")
    args = ap.parse_args()
    cache = RES / "daily_ic.parquet"
    if args.refresh or not cache.exists():
        ic = audit.collect_daily_ic(cache=cache)
    else:
        ic = pd.read_parquet(cache)
    pub = published_table()
    out = {
        "reproduction": reproduction(ic, pub),
        "decay": decay(ic),
        "staleness": staleness(ic),
        "seed_noise": seed_noise(ic),
        "tests": tests(ic),
        "rank_stability": rank_stability(ic, pub),
        "yearly": yearly(ic),
        "runs": {
            p: {
                m: int(ic[(ic.protocol == p) & (ic.model == m) & (ic.seed != audit.ENSEMBLE)].seed.nunique())
                for m in ordered(ic[ic.protocol == p].model.unique())
            }
            for p in sorted(ic.protocol.unique())
        },
    }
    (RES / "ic_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("decay", "rank_stability", "seed_noise", "runs")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
