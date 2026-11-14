"""Portfolio-level audit of the TopK-Dropout backtests.

Reads ``results/backtests``; writes ``results/backtest_audit.json`` and a
compact ``results/excess_returns.parquet`` (daily excess return per run).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oosaudit import backtest, sharpe
from oosaudit.audit import PERIODS
from published import published_table

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BT = RES / "backtests"


def load_reports() -> dict[tuple[str, str, str], pd.DataFrame]:
    out = {}
    for p in sorted(BT.glob("*/*.parquet")):
        model, tag = p.stem.rsplit("_", 1)
        out[(p.parent.name, model, tag)] = pd.read_parquet(p)
    return out


def main() -> None:
    reps = load_reports()
    pub = published_table()
    rows = []
    for (proto, model, tag), rep in reps.items():
        for period, (a, b) in PERIODS.items():
            if proto == "refit" and period != "recent":
                continue
            r = rep[(rep.index >= a) & (rep.index < b)]
            if len(r) < 100:
                continue
            perf = backtest.perf(r).as_dict()
            x = r["excess"].to_numpy()
            perf.update(
                {
                    "protocol": proto,
                    "model": model,
                    "tag": tag,
                    "period": period,
                    "psr0": sharpe.psr(x, 0.0),
                    "daily_sharpe": sharpe.sharpe(x),
                    "bench_annual": float(r["bench"].mean() * backtest.ANN),
                    "cost_annual": float(r["cost"].mean() * backtest.ANN),
                }
            )
            rows.append(perf)
    df = pd.DataFrame(rows)

    # Published window: per-seed mean and std, next to the published table.
    seeds = df[(df.protocol == "pub") & (df.period == "published") & (df.tag != "ens")]
    repro = seeds.groupby("model")[["annual_excess", "information_ratio", "max_drawdown"]].agg(["mean", "std"])
    reproduction = {}
    for m in repro.index:
        r = {
            f"ours_{k}": float(repro.loc[m, (k, "mean")])
            for k in ["annual_excess", "information_ratio", "max_drawdown"]
        }
        r.update({f"ours_{k}_sd": float(repro.loc[m, (k, "std")]) for k in ["annual_excess", "information_ratio"]})
        if m in pub.index:
            r.update({f"pub_{k}": float(pub.loc[m, k]) for k in ["annual_excess", "information_ratio", "max_drawdown"]})
        reproduction[m] = r
    single = df[(df.protocol == "pub") & (df.period == "published") & (df.tag == "ens")]
    for m in single.model:
        if m not in reproduction:
            row = single[single.model == m].iloc[0]
            reproduction[m] = {
                "ours_annual_excess": float(row.annual_excess),
                "ours_information_ratio": float(row.information_ratio),
                "ours_max_drawdown": float(row.max_drawdown),
            }
            if m in pub.index:
                reproduction[m].update(
                    {f"pub_{k}": float(pub.loc[m, k]) for k in ["annual_excess", "information_ratio", "max_drawdown"]}
                )

    # Null strategy: TopK-Dropout on random scores, mean and std over seeds.
    null = {}
    nul = df[(df.protocol == "null") & (df.tag != "ens")]
    for period, g in nul.groupby("period"):
        null[period] = {
            "annual_excess": float(g.annual_excess.mean()),
            "annual_excess_sd": float(g.annual_excess.std()),
            "information_ratio": float(g.information_ratio.mean()),
            "information_ratio_sd": float(g.information_ratio.std()),
            "seeds": int(len(g)),
        }

    ens = df[(df.tag == "ens") & (df.protocol != "null")]
    table = {}
    for (proto, model), g in ens.groupby(["protocol", "model"]):
        table.setdefault(proto, {})[model] = {
            row.period: {
                k: float(getattr(row, k))
                for k in [
                    "annual_excess",
                    "information_ratio",
                    "max_drawdown",
                    "turnover",
                    "psr0",
                    "bench_annual",
                    "cost_annual",
                ]
            }
            for row in g.itertuples()
        }

    # Deflated Sharpe of the best post-publication strategy, counting every
    # (model, seed) run of the published protocol as a trial.
    post = df[(df.protocol == "pub") & (df.period == "post") & (df.tag != "ens")]
    dsr = {}
    if len(post) >= 2:
        best = post.sort_values("daily_sharpe", ascending=False).iloc[0]
        rep = reps[("pub", best.model, best.tag)]
        a, b = PERIODS["post"]
        x = rep[(rep.index >= a) & (rep.index < b)]["excess"].to_numpy()
        d, sr0 = sharpe.dsr(x, post["daily_sharpe"].to_numpy())
        dsr = {
            "best": f"{best.model}/{best.tag}",
            "trials": int(len(post)),
            "information_ratio": float(best.information_ratio),
            "psr0": float(best.psr0),
            "dsr": d,
            "sr0_annualised": float(sr0 * np.sqrt(backtest.ANN)),
        }

    excess = pd.DataFrame({f"{p}/{m}/{t}": rep["excess"] for (p, m, t), rep in reps.items()}).astype("float32")
    excess.to_parquet(RES / "excess_returns.parquet")
    out = {"reproduction": reproduction, "ensembles": table, "null": null, "deflated_sharpe_post": dsr}
    (RES / "backtest_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["reproduction"], indent=1)[:3000])
    print(json.dumps(dsr, indent=1))


if __name__ == "__main__":
    main()
