"""Memory-lean drop-in replacements for Qlib processors.

``CSZFillna`` has the same semantics as ``qlib.data.dataset.processor.CSZFillna``
(fill each missing feature with that day's cross-sectional mean) but uses a
grouped ``transform`` instead of a per-day ``apply``; the latter needs more
memory than the 8 GB machine these experiments ran on.  ``tests/`` checks the
two give identical output.
"""

from __future__ import annotations

import pandas as pd
from qlib.data.dataset.processor import Processor, get_group_columns


class CSZFillna(Processor):
    """Cross-sectional mean fill, vectorised."""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        block = df[cols]
        means = block.groupby(level="datetime").transform("mean")
        df[cols] = block.fillna(means)
        return df
