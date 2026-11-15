import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qlib")

from oosaudit import backtest, features, run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from published import published_table  # noqa: E402


def test_published_table_parses_all_models():
    t = published_table()
    assert set(t.index) == set(run.CONFIGS)
    assert t.loc["lightgbm", "ic"] == pytest.approx(0.0448)
    assert t.loc["doubleensemble", "rank_ic"] == pytest.approx(0.0502)
    assert t.loc["linear", "information_ratio"] == pytest.approx(0.9209)


def test_configs_parse_like_qrun():
    # "2e-4" is a float in YAML 1.2 (ruamel), a string in PyYAML.
    assert isinstance(run.model_config("gru", 0, 2)["kwargs"]["lr"], float)


@pytest.mark.parametrize("model", sorted(run.CONFIGS))
def test_model_config_seeds_and_threads(model):
    cfg = run.model_config(model, 7, 2)
    kw = cfg["kwargs"]
    for key in ("num_threads", "nthread", "thread_count", "n_jobs"):
        if key in kw:
            assert kw[key] == 2
    if model in run.DETERMINISTIC:
        assert "seed" not in kw
    elif cfg["class"] == "CatBoostModel":
        assert kw["random_seed"] == 7 and kw["bootstrap_type"] == "Bernoulli"
    else:
        assert kw["seed"] == 7


def test_handler_fits_processors_on_the_training_window():
    h = run.handler_config("gru", "refit")["kwargs"]
    norm = [p for p in h["infer_processors"] if p["class"] == "RobustZScoreNorm"][0]["kwargs"]
    assert (norm["fit_start_time"], norm["fit_end_time"]) == run.PROTOCOLS["refit"]["train"]
    assert h["start_time"] == run.PROTOCOLS["refit"]["train"][0]


def test_lean_processor_is_swapped_in():
    procs = run.handler_config("mlp", "pub")["kwargs"]["infer_processors"]
    fill = [p for p in procs if p["class"] == "CSZFillna"][0]
    assert fill["module_path"] == "oosaudit.processors"


def test_roll_segments():
    assert run.roll_segments(2017) == {
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2017-12-31"),
    }
    last = run.roll_segments(run.ROLL_YEARS[-1])
    assert last["test"][1] == run.data.END


def test_lean_cszfillna_matches_qlib():
    from qlib.data.dataset.processor import CSZFillna as Reference

    from oosaudit.processors import CSZFillna

    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=30), [f"S{i}" for i in range(40)]], names=["datetime", "instrument"]
    )
    cols = pd.MultiIndex.from_arrays([["feature"] * 5 + ["label"], list("abcde") + ["L"]])
    df = pd.DataFrame(rng.normal(size=(len(idx), 6)), index=idx, columns=cols)
    df[df > 1.2] = np.nan
    df.loc[(pd.Timestamp("2020-01-03"), slice(None)), ("feature", "a")] = np.nan
    a = Reference("feature")(df.copy())
    b = CSZFillna("feature")(df.copy())
    assert (a.isna() == b.isna()).all().all()
    np.testing.assert_allclose(a.fillna(0).to_numpy(), b.fillna(0).to_numpy(), atol=1e-12)


def test_family_names():
    assert features.family("KMID2") == ("KBAR", None)
    assert features.family("VWAP0") == ("PRICE", None)
    assert features.family("CORD10") == ("CORD", 10)
    assert features.family("MA5") == ("MA", 5)
    assert features.family("VSUMD60") == ("VSUMD", 60)
    assert features.family("IMIN30") == ("IMIN", 30)


def test_feature_rank_ic_matches_pandas_spearman():
    rng = np.random.default_rng(1)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=25), [f"S{i}" for i in range(50)]], names=["datetime", "instrument"]
    )
    y = rng.standard_normal(len(idx))
    x = np.round(0.5 * y + rng.standard_normal(len(idx)), 1)  # ties
    x[::11] = np.nan
    raw = pd.DataFrame({("feature", "F1"): x, ("label", "LABEL0"): y}, index=idx)
    got = features.feature_rank_ic(raw)["F1"]
    df = pd.DataFrame({"x": x, "y": y}, index=idx).dropna()
    want = df.groupby(level="datetime").apply(lambda d: d["x"].corr(d["y"], method="spearman"))
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), atol=1e-12)


def test_perf_conventions():
    days = pd.bdate_range("2021-01-04", periods=5)
    rep = pd.DataFrame(
        {"excess": [0.01, -0.02, 0.005, -0.01, 0.02], "turnover": [0.1] * 5},
        index=days,
    )
    p = backtest.perf(rep)
    x = rep["excess"].to_numpy()
    assert p.annual_excess == pytest.approx(x.mean() * 238)
    assert p.information_ratio == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(238))
    assert p.max_drawdown == pytest.approx(-0.025)
    assert p.days == 5


def _procs(model: str, key: str) -> list:
    return run.handler_config(model, "pub")["kwargs"][key]


def test_label_variants_change_only_the_label():
    for variant, (base, change) in run.VARIANTS.items():
        if "label" not in change:
            continue
        assert run.model_config(variant, 3, 2) == run.model_config(base, 3, 2)
        assert _procs(variant, "infer_processors") == _procs(base, "infer_processors")
        label = [
            p
            for p in _procs(variant, "learn_processors")
            if isinstance(p, dict) and p.get("kwargs", {}).get("fields_group") == "label"
        ]
        assert [p["class"] for p in label] == [change["label"]]
        rest = [p for p in _procs(variant, "learn_processors") if p not in label]
        base_rest = [
            p
            for p in _procs(base, "learn_processors")
            if not (isinstance(p, dict) and p.get("kwargs", {}).get("fields_group") == "label")
        ]
        assert rest == base_rest


def test_feature_variant_keeps_the_recurrent_columns():
    first = _procs("lightgbm_f20", "infer_processors")[0]
    assert first["class"] == "FilterCol"
    gru_cols = [p for p in _procs("gru", "infer_processors") if p["class"] == "FilterCol"][0]["kwargs"]["col_list"]
    assert first["kwargs"]["col_list"] == list(gru_cols)
