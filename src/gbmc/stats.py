"""Patient-level bootstrap with BCa intervals (plan Step 1.5, as in gbm-wm).

The BCa interval is adapted from gbm-wm ``src/gbmwm/spatial_null.py:_bca_interval``
(Efron 1987). gbm-wm is not a dependency, so the function is copied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def bca_interval(boot: np.ndarray, estimate: float, jackknife: np.ndarray,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected and accelerated percentile interval."""
    boot = boot[np.isfinite(boot)]
    if boot.size < 10:
        return float("nan"), float("nan")
    prop = float(np.mean(boot < estimate))
    prop = min(max(prop, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
    z0 = stats.norm.ppf(prop)
    jm = jackknife.mean()
    num = ((jm - jackknife) ** 3).sum()
    den = 6.0 * (((jm - jackknife) ** 2).sum() ** 1.5)
    a = 0.0 if den == 0 else num / den
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = z0 + stats.norm.ppf(q)
        adj = stats.norm.cdf(z0 + z / (1 - a * z))
        out.append(float(np.quantile(boot, min(max(adj, 0.0), 1.0))))
    return out[0], out[1]


@dataclass(frozen=True)
class MeanCI:
    n: int
    mean: float
    low: float
    high: float
    boot: np.ndarray | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "ci_low": self.low, "ci_high": self.high}


def mean_with_bca(values: np.ndarray, n_resamples: int, seed: int,
                  alpha: float = 0.05) -> MeanCI:
    """Mean of per-patient values with a patient-resampling BCa CI.

    Patients with an undefined value (NaN) are dropped before resampling, and
    the returned ``n`` says how many remained.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return MeanCI(n, float(values.mean()) if n else float("nan"),
                      float("nan"), float("nan"))
    boot = bootstrap_means(values, n_resamples, seed)
    jackknife = (values.sum() - values) / (n - 1)
    estimate = float(values.mean())
    low, high = bca_interval(boot, estimate, jackknife, alpha)
    return MeanCI(n, estimate, low, high, boot)


def bootstrap_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Means of ``n_resamples`` patient resamples of finite ``values``."""
    rng = np.random.default_rng(seed)
    return values[rng.integers(0, values.size, size=(n_resamples, values.size))].mean(axis=1)
