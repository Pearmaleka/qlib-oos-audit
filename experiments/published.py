"""Parse Qlib's published Alpha158 / CSI300 benchmark table.

The table is vendored in ``configs/qlib/BENCHMARKS_README.md`` (Qlib commit
54355232, MIT licence).  Values are mean ± std over 20 seeds on the test
window 2017-01-01 to 2020-08-01.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "configs" / "qlib" / "BENCHMARKS_README.md"

# Our model keys -> the row label in the published table (Alpha158 section).
ROWS = {
    "tcn": ("TCN", "Alpha158"),
    "transformer": ("Transformer", "Alpha158"),
    "gru": ("GRU", "Alpha158(with selected 20 features)"),
    "lstm": ("LSTM", "Alpha158(with selected 20 features)"),
    "localformer": ("Localformer", "Alpha158"),
    "alstm": ("ALSTM", "Alpha158(with selected 20 features)"),
    "linear": ("Linear", "Alpha158"),
    "catboost": ("CatBoost", "Alpha158"),
    "xgboost": ("XGBoost", "Alpha158"),
    "mlp": ("MLP", "Alpha158"),
    "lightgbm": ("LightGBM", "Alpha158"),
    "doubleensemble": ("DoubleEnsemble", "Alpha158"),
}
COLS = ["ic", "icir", "rank_ic", "rank_icir", "annual_excess", "information_ratio", "max_drawdown"]


def _mean(cell: str) -> float:
    return float(cell.split("±")[0])


def published_table() -> pd.DataFrame:
    text = README.read_text()
    section = text.split("### Alpha158 dataset", 1)[1].split("### Alpha360 dataset", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 9 or "±" not in cells[2]:
            continue
        name = re.sub(r"\s*\(.*$", "", cells[0]).strip()
        rows[(name, cells[1])] = [_mean(c) for c in cells[2:]]
    out = {k: rows[v] for k, v in ROWS.items() if v in rows}
    return pd.DataFrame.from_dict(out, orient="index", columns=COLS)


if __name__ == "__main__":
    print(published_table())
