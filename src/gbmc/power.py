"""Minimum detectable effect under the real decision rule (plan Step 2.1).

The data-generating model is gbm-4d's (``gbm4d.mde.power_at``): each simulated
patient's C0 PR-AUC is resampled from the observed LUMIERE C0 scores, a shared
patient difficulty is split from independent noise by the assumed correlation
``rho``, and both arms are clipped to [0, 1] symmetrically.

What differs from gbm-4d is the decision rule. This project's verdict uses a
**BCa** patient-bootstrap interval (Step 2.3), so a simulated study "detects"
the effect when the BCa lower bound is above zero, exactly as the verdict rule
would read it. gbm-4d's simulation used the percentile interval and a two-sided
test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gbm4d.mde import paired_difference_sd
from scipy import stats


@dataclass(frozen=True)
class PowerPoint:
    delta: float
    power: float
    n: int
    rho: float


def bca_lower_above_zero(differences: np.ndarray, n_boot: int, alpha: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Per simulated study (rows): is the BCa lower bound of the mean above zero?"""
    n_sim, n = differences.shape
    est = differences.mean(axis=1)
    boot = np.empty((n_sim, n_boot))
    for i in range(n_sim):
        boot[i] = differences[i][rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    prop = (boot < est[:, None]).mean(axis=1)
    prop = np.clip(prop, 1.0 / (n_boot + 1), 1.0 - 1.0 / (n_boot + 1))
    z0 = stats.norm.ppf(prop)
    jack = (differences.sum(axis=1, keepdims=True) - differences) / (n - 1)
    dev = jack.mean(axis=1, keepdims=True) - jack
    den = 6.0 * ((dev ** 2).sum(axis=1) ** 1.5)
    a = np.where(den == 0, 0.0, (dev ** 3).sum(axis=1) / np.where(den == 0, 1.0, den))
    z = z0 + stats.norm.ppf(alpha / 2)
    q = stats.norm.cdf(z0 + z / (1 - a * z))
    boot.sort(axis=1)
    idx = np.clip(np.floor(q * (n_boot - 1)).astype(int), 0, n_boot - 1)
    lower = boot[np.arange(n_sim), idx]
    return lower > 0.0


def power_at(c0_scores: np.ndarray, delta: float, n: int, rho: float, seed: int,
             n_sim: int = 1000, n_boot: int = 1000, alpha: float = 0.05) -> float:
    rng = np.random.default_rng(seed)
    c0_scores = np.asarray(c0_scores, dtype=float)
    c0_scores = c0_scores[np.isfinite(c0_scores)]
    sd = float(c0_scores.std(ddof=1))
    centre = float(c0_scores.mean())
    shared = centre + (rng.choice(c0_scores, size=(n_sim, n)) - centre) * np.sqrt(rho)
    within = paired_difference_sd(sd, rho) / np.sqrt(2.0)
    c0 = np.clip(shared + rng.normal(0.0, within, size=(n_sim, n)), 0.0, 1.0)
    ch = np.clip(shared + delta + rng.normal(0.0, within, size=(n_sim, n)), 0.0, 1.0)
    return float(bca_lower_above_zero(ch - c0, n_boot, alpha, rng).mean())


def minimum_detectable_effect(c0_scores: np.ndarray, n: int, rho: float, seed: int,
                              grid, target_power: float = 0.80, **kwargs
                              ) -> tuple[float | None, list[PowerPoint]]:
    """Smallest delta on ``grid`` with power >= target; None if none reaches it."""
    curve, found = [], None
    for delta in grid:
        p = power_at(c0_scores, float(delta), n, rho, seed, **kwargs)
        curve.append(PowerPoint(float(delta), p, n, rho))
        if found is None and p >= target_power:
            found = float(delta)
    return found, curve
