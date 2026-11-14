"""TopK-Dropout backtests of every model's seed-ensemble score (and, for the
published window, of each seed), with Qlib's benchmark settings.

Writes daily reports to ``results/backtests/<protocol>/<model>[_seed<k>].parquet``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from oosaudit import audit, backtest, data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "backtests"

WINDOWS = {
    "pub": ("2017-01-01", data.END),
    "refit": ("2023-01-01", data.END),
    "roll": ("2017-01-01", data.END),
    "null": ("2017-01-01", data.END),
}


def tags(protocol: str, model: str, per_seed: bool) -> list[str]:
    seeds = sorted(int(p.stem[4:]) for p in (audit.PRED_DIR / protocol / model).glob("seed*.parquet"))
    out = ["ens"]
    if per_seed and len(seeds) > 1:
        out += [f"seed{s}" for s in seeds]
    return out


def run_job(protocol: str, model: str, tag: str) -> None:
    """One backtest; run in its own process because Qlib's caches grow across runs."""
    data.init_qlib(1)
    preds = audit.load_preds(protocol, model)
    if tag == "ens":
        score = audit.seed_ensemble(preds) if len(preds) > 1 else next(iter(preds.values()))
    else:
        score = preds[int(tag[4:])]
    start, end = WINDOWS[protocol]
    path = OUT / protocol / f"{model}_{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    backtest.run(score, start, end).to_parquet(path)
    print(f"{protocol} {model} {tag} {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="pub,refit,roll,null")
    ap.add_argument("--models", default="")
    ap.add_argument("--per-seed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--job", help="protocol:model:tag, run one backtest in this process")
    args = ap.parse_args()
    if args.job:
        run_job(*args.job.split(":"))
        return
    for protocol in args.protocols.split(","):
        pdir = audit.PRED_DIR / protocol
        if not pdir.exists():
            continue
        models = args.models.split(",") if args.models else sorted(p.name for p in pdir.iterdir() if p.is_dir())
        for m in models:
            for tag in tags(protocol, m, args.per_seed and protocol in ("pub", "null")):
                path = OUT / protocol / f"{m}_{tag}.parquet"
                if path.exists() and not args.force:
                    continue
                subprocess.run([sys.executable, __file__, "--job", f"{protocol}:{m}:{tag}"], check=False)


if __name__ == "__main__":
    main()
