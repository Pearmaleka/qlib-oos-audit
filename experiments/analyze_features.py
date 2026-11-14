"""Feature-level decay: daily Rank IC of each raw Alpha158 feature.

Writes ``results/feature_rank_ic.parquet`` and ``results/feature_audit.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oosaudit import data, features, metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PERIODS = {
    "train": ("2008-01-01", "2017-01-01"),
    "published": ("2017-01-01", "2020-08-01"),
    "post": ("2020-08-01", "2026-11-14"),
}


def period_means(fic: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, (a, b) in PERIODS.items():
        sub = fic[(fic.index >= a) & (fic.index < b)]
        out[name] = sub.mean()
        out[f"{name}_t"] = sub.apply(lambda s: s.mean() / metrics.newey_west_se(s.to_numpy()))
    return pd.DataFrame(out)


def main() -> None:
    cache = RES / "feature_rank_ic.parquet"
    if cache.exists():
        fic = pd.read_parquet(cache)
    else:
        fic = features.feature_rank_ic(data.load_raw_store())
        RES.mkdir(exist_ok=True)
        fic.to_parquet(cache)
    pm = period_means(fic)
    fam = pd.DataFrame([features.family(c) for c in pm.index], index=pm.index, columns=["family", "window"])
    pm = pm.join(fam)

    # Signed predictive strength measured in the direction learned before 2017.
    sign = np.sign(pm["train"])
    for p in ("train", "published", "post"):
        pm[f"{p}_aligned"] = pm[p] * sign
    strong = pm[pm["train_t"].abs() > 3]
    slope_pub = float(np.polyfit(strong["published_aligned"], strong["post_aligned"], 1)[0])
    by_window = strong.groupby(strong["window"].fillna(1))[
        ["train_aligned", "published_aligned", "post_aligned"]
    ].mean()
    by_family = strong.groupby("family")[["train_aligned", "published_aligned", "post_aligned"]].mean()
    flips = int(((strong["post_aligned"] < 0) & (strong["published_aligned"] > 0)).sum())
    out = {
        "n_features": int(len(pm)),
        "n_strong": int(len(strong)),
        "mean_abs": {p: float(pm[p].abs().mean()) for p in ("train", "published", "post")},
        "strong_mean_aligned": {p: float(strong[f"{p}_aligned"].mean()) for p in ("train", "published", "post")},
        "strong_median_ratio_post_over_published": float(
            np.median(strong["post_aligned"] / strong["published_aligned"])
        ),
        "slope_post_on_published": slope_pub,
        "sign_flips_post": flips,
        "by_window": {str(int(k)): v for k, v in by_window.to_dict(orient="index").items()},
        "by_family": by_family.to_dict(orient="index"),
        "top_published": strong.sort_values("published_aligned", ascending=False)
        .head(10)[["published", "post", "family", "window"]]
        .to_dict(orient="index"),
    }
    (RES / "feature_audit.json").write_text(json.dumps(out, indent=1, default=float))
    pm.to_csv(RES / "feature_period_means.csv")
    print(json.dumps({k: out[k] for k in list(out)[:8]}, indent=1))
    print(by_window.round(4))


if __name__ == "__main__":
    main()
