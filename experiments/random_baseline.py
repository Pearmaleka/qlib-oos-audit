"""Null strategy: TopK-Dropout on random scores.

Scores are i.i.d. standard normal for every (date, stock) in the CSI 300
test index, so any excess return of the backtest comes from the strategy's
mechanics (equal weights, slow turnover, costs), not from prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oosaudit.audit import load_preds
from oosaudit.run import PRED_DIR

SEEDS = range(5)


def main() -> None:
    index = load_preds("pub", "linear")[0].index
    out = PRED_DIR / "null" / "random"
    out.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        pd.Series(rng.standard_normal(len(index)).astype("float32"), index=index, name="score").to_frame().to_parquet(
            out / f"seed{s}.parquet"
        )


if __name__ == "__main__":
    main()
