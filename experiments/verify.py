"""Independent re-computation of the headline numbers.

Uses pandas' own correlation routines on the saved predictions, not the
``oosaudit.metrics`` code, and checks the values in ``results/*.json``.
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRED = RES / "preds"
WINDOWS = {
    "published": ("2017-01-01", "2020-08-01"),
    "post": ("2020-08-01", "2026-11-14"),
    "recent": ("2023-01-01", "2026-11-14"),
}


def rank_ic_mean(score: pd.Series, label: pd.Series, a: str, b: str) -> float:
    df = pd.concat({"s": score, "y": label}, axis=1, join="inner").dropna()
    d = df.index.get_level_values("datetime")
    df = df[(d >= a) & (d < b)]
    per_day = df.groupby(level="datetime").apply(lambda g: g["s"].corr(g["y"], method="spearman"))
    return float(per_day.mean())


def main() -> int:
    from oosaudit import data  # only for the cached label store

    label = data.load_raw_store()[("label", "LABEL0")]
    ic = json.loads((RES / "ic_audit.json").read_text())
    bad = 0
    checks = 0
    for m, row in ic["decay"]["models"].items():
        seeds = sorted((PRED / "pub" / m).glob("seed*.parquet"))
        for w, key in (("published", "published"), ("post", "post")):
            vals = [rank_ic_mean(pd.read_parquet(p)["score"], label, *WINDOWS[w]) for p in seeds]
            got, want = float(np.mean(vals)), row[key]
            checks += 1
            if abs(got - want) > 1e-9:
                bad += 1
                print(f"MISMATCH decay {m} {w}: {got} vs {want}")
    for m, row in ic["staleness"].items():
        for proto in ("refit", "roll"):
            if proto not in row:
                continue
            seeds = sorted((PRED / proto / m).glob("seed*.parquet"))
            vals = [rank_ic_mean(pd.read_parquet(p)["score"], label, *WINDOWS["recent"]) for p in seeds]
            got, want = float(np.mean(vals)), row[proto]
            checks += 1
            if abs(got - want) > 1e-9:
                bad += 1
                print(f"MISMATCH recent {proto} {m}: {got} vs {want}")
    pub_mean = np.mean([r["published"] for r in ic["decay"]["models"].values()])
    post_mean = np.mean([r["post"] for r in ic["decay"]["models"].values()])
    checks += 2
    if abs(pub_mean - ic["decay"]["mean_published"]) > 1e-12 or abs(post_mean - ic["decay"]["mean_post"]) > 1e-12:
        bad += 1
        print("MISMATCH decay means")
    print(f"{checks} checks, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
