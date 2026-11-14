import numpy as np
import pandas as pd
import pytest

from oosaudit import audit, metrics, sharpe


def _panel(days=40, names=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=days), [f"S{i:03d}" for i in range(names)]],
        names=["datetime", "instrument"],
    )
    y = rng.standard_normal(len(idx))
    s = 0.3 * y + rng.standard_normal(len(idx))
    return pd.Series(s, idx), pd.Series(y, idx)


def test_daily_ic_matches_pandas():
    s, y = _panel()
    y.iloc[::7] = np.nan
    got = metrics.daily_ic(s, y)
    df = pd.concat({"s": s, "y": y}, axis=1).dropna()
    g = df.groupby(level="datetime")
    want_ic = g.apply(lambda d: d["s"].corr(d["y"]))
    want_rank = g.apply(lambda d: d["s"].corr(d["y"], method="spearman"))
    np.testing.assert_allclose(got["ic"], want_ic, atol=1e-12)
    np.testing.assert_allclose(got["rank_ic"], want_rank, atol=1e-12)


def test_newey_west_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    e = rng.standard_normal(600)
    x = np.convolve(e, [1.0, 0.6, 0.3], mode="same") + 0.1
    for lags in (0, 3, 9):
        fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
        assert metrics.newey_west_se(x, lags) == pytest.approx(float(fit.bse[0]), rel=1e-10)


def test_newey_west_without_lags_is_iid_se():
    x = np.random.default_rng(2).standard_normal(250)
    assert metrics.newey_west_se(x, 0) == pytest.approx(x.std(ddof=0) / np.sqrt(len(x)))


def test_period_difference_and_paired_test():
    rng = np.random.default_rng(3)
    a = 0.05 + 0.1 * rng.standard_normal(800)
    b = 0.01 + 0.1 * rng.standard_normal(800)
    d, se = audit.period_difference(a, b)
    assert d == pytest.approx(b.mean() - a.mean())
    assert se > 0
    m, se2, p = audit.paired_test(a, b)
    assert m == pytest.approx((a - b).mean())
    assert p < 1e-6


def _runs(var_model, var_seed, models=8, seeds=5, days=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(models):
        mu = rng.normal(0.03, np.sqrt(var_model))
        for s in range(seeds):
            level = mu + rng.normal(0, np.sqrt(var_seed))
            for d in pd.bdate_range("2017-01-02", periods=days):
                rows.append({"protocol": "pub", "model": f"m{m}", "seed": s, "datetime": d, "rank_ic": level})
    return pd.DataFrame(rows)


def test_seed_variance_share_recovers_components():
    ic = _runs(var_model=4e-6, var_seed=1e-6, models=40, seeds=6)
    out = audit.seed_variance_share(ic, "pub", "published")
    assert out["models"] == 40 and out["runs"] == 240
    assert out["var_seed"] == pytest.approx(1e-6, rel=0.25)
    assert out["var_model"] == pytest.approx(4e-6, rel=0.5)
    assert 0.6 < out["icc"] < 0.95


def test_seed_variance_share_is_zero_when_models_are_identical():
    ic = _runs(var_model=0.0, var_seed=1e-6, models=10, seeds=8, seed=4)
    assert audit.seed_variance_share(ic, "pub", "published")["icc"] < 0.2


def test_seed_ensemble_is_mean_of_zscores():
    s1, _ = _panel(seed=5)
    s2, _ = _panel(seed=6)
    ens = audit.seed_ensemble({0: s1, 1: s2})

    def z(s):
        g = s.groupby(level="datetime")
        return (s - g.transform("mean")) / g.transform("std")

    np.testing.assert_allclose(ens, (z(s1) + z(s2)) / 2)


def test_psr_and_dsr():
    rng = np.random.default_rng(7)
    x = 0.001 + 0.01 * rng.standard_normal(2000)
    assert sharpe.psr(x, 0.0) > 0.99
    assert sharpe.psr(-x, 0.0) < 0.01
    assert sharpe.expected_max_sharpe(100, 0.01) > sharpe.expected_max_sharpe(10, 0.01) > 0
    d, sr0 = sharpe.dsr(x, rng.normal(0, 0.05, size=50))
    assert sr0 > 0 and d < sharpe.psr(x, 0.0)


def test_mcs_keeps_the_best_model():
    pytest.importorskip("arch")
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2020-01-01", periods=600)
    w = pd.DataFrame(
        {
            "good": 0.06 + 0.1 * rng.standard_normal(600),
            "bad": -0.02 + 0.1 * rng.standard_normal(600),
            "mid": 0.055 + 0.1 * rng.standard_normal(600),
        },
        index=days,
    )
    _, included = audit.model_confidence_set(w, reps=500)
    assert "good" in included and "bad" not in included
