"""Probabilistic and deflated Sharpe ratios (Bailey and López de Prado, 2012, 2014).

All Sharpe ratios here are per period (daily), not annualised.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1))


def psr(x: np.ndarray, sr_star: float = 0.0) -> float:
    """P(true Sharpe > ``sr_star``) given the sample's skewness and kurtosis."""
    x = np.asarray(x, float)
    t = len(x)
    sr = sharpe(x)
    g3 = float(sps.skew(x))
    g4 = float(sps.kurtosis(x, fisher=False))
    den = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr)
    return float(sps.norm.cdf((sr - sr_star) * np.sqrt(t - 1) / den))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios with true value zero."""
    if n_trials < 2:
        return 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(x: np.ndarray, trial_sharpes: np.ndarray) -> tuple[float, float]:
    """Deflated Sharpe ratio of ``x`` after ``len(trial_sharpes)`` trials.

    Returns ``(dsr, sr0)``: the probability that the true Sharpe exceeds the
    expected maximum under the null, and that benchmark ``sr0``.
    """
    trial_sharpes = np.asarray(trial_sharpes, float)
    sr0 = expected_max_sharpe(len(trial_sharpes), float(trial_sharpes.var(ddof=1)))
    return psr(x, sr0), sr0
