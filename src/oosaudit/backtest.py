"""Qlib's own TopK-Dropout backtest, with the benchmark's settings.

The strategy holds the 50 highest-scored stocks and replaces at most 5 per
day; trades execute at the close with 5 bp to buy, 15 bp to sell and a
minimum fee of 5 CNY; a stock that moved more than 9.5% that day cannot be
traded.  ``excess`` is the daily portfolio return minus the CSI 300 return
minus costs, as in Qlib's ``PortAnaRecord``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data

STRATEGY = {"topk": 50, "n_drop": 5}
EXCHANGE = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}
ACCOUNT = 100_000_000
# Qlib annualises daily figures with 238 trading days (``risk_analysis``).
ANN = 238


def run(score: pd.Series, start: str, end: str) -> pd.DataFrame:
    """Daily report with ``return``, ``bench``, ``cost``, ``turnover`` and ``excess``."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    from qlib.backtest import backtest, executor
    from qlib.contrib.strategy import TopkDropoutStrategy

    score = score.sort_index()
    # Load quotes only for the stocks that can be held (the full market does not fit in 8 GB).
    codes = sorted(score.index.get_level_values("instrument").unique())
    strategy = TopkDropoutStrategy(signal=score.to_frame("score"), **STRATEGY)
    exe = executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
    portfolio, _ = backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=exe,
        account=ACCOUNT,
        benchmark=data.BENCHMARK,
        exchange_kwargs={"freq": "day", "codes": codes, **EXCHANGE},
    )
    report, _ = portfolio["1day"]
    report = report[["return", "bench", "cost", "turnover"]].copy()
    report["excess"] = report["return"] - report["bench"] - report["cost"]
    return report


@dataclass(frozen=True)
class Perf:
    annual_excess: float
    information_ratio: float
    max_drawdown: float
    turnover: float
    days: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def perf(report: pd.DataFrame, start: str | None = None, end: str | None = None) -> Perf:
    """Qlib's ``risk_analysis`` conventions: additive annualisation, ``ANN`` days."""
    r = report
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index < pd.Timestamp(end)]
    x = r["excess"].to_numpy()
    cum = np.cumsum(x)
    return Perf(
        annual_excess=float(x.mean() * ANN),
        information_ratio=float(x.mean() / x.std(ddof=1) * np.sqrt(ANN)),
        max_drawdown=float((cum - np.maximum.accumulate(cum)).min()),
        turnover=float(r["turnover"].mean()),
        days=int(len(x)),
    )
