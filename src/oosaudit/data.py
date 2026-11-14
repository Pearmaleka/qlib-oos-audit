"""Build and load the raw Alpha158 feature store.

Features and the label are computed once with Qlib's own ``Alpha158`` handler
(no processors), over the whole sample, and cached as a pickle.  Every model
run then rebuilds its handler from this cache with the processors of its
official configuration, so normalisation is always fitted on that run's own
training window.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("OOSAUDIT_QLIB_DIR", Path.home() / "data" / "qlib_bin"))
CACHE_DIR = Path(os.environ.get("OOSAUDIT_CACHE", Path.home() / "data" / "cache"))

START = "2008-01-01"
END = "2026-11-13"
MARKET = "csi300"
BENCHMARK = "SH000300"


def init_qlib(kernels: int = 1) -> None:
    """Initialise Qlib on the local A-share data without on-disk caches."""
    import qlib
    from qlib.constant import REG_CN

    qlib.init(
        provider_uri=str(DATA_DIR),
        region=REG_CN,
        kernels=kernels,
        expression_cache=None,
        dataset_cache=None,
    )


def raw_store_path(market: str = MARKET) -> Path:
    return CACHE_DIR / f"alpha158_{market}_{START}_{END}.pkl"


def build_raw_store(market: str = MARKET, kernels: int = 2) -> Path:
    """Compute Alpha158 features and the default label for ``market``."""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    init_qlib(kernels)
    handler = Alpha158(
        instruments=market,
        start_time=START,
        end_time=END,
        fit_start_time=START,
        fit_end_time="2014-12-31",
        infer_processors=[],
        learn_processors=[],
    )
    df = handler.fetch(data_key=DataHandlerLP.DK_R).astype("float32")
    return save_raw_store(df, market)


def save_raw_store(df: pd.DataFrame, market: str = MARKET) -> Path:
    """Store with Qlib's two-level ``(feature|label, name)`` columns."""
    if df.columns.nlevels == 1:
        group = ["label" if c.startswith("LABEL") else "feature" for c in df.columns]
        df.columns = pd.MultiIndex.from_arrays([group, list(df.columns)])
    path = raw_store_path(market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return path


def load_raw_store(market: str = MARKET) -> pd.DataFrame:
    return pd.read_pickle(raw_store_path(market))


if __name__ == "__main__":
    import time

    t0 = time.time()
    p = build_raw_store()
    df = load_raw_store()
    print(p, df.shape, f"{time.time() - t0:.0f}s")
