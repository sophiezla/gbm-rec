from __future__ import annotations

import numpy as np

from gbmc import power

SCORES = np.random.default_rng(0).gamma(1.2, 0.06, size=53).clip(0, 1)


def test_false_positive_rate_is_near_alpha_over_two():
    # one-sided reading of a 95% interval: about 2.5% under the null
    p = power.power_at(SCORES, 0.0, 30, 0.5, seed=1, n_sim=400, n_boot=400)
    assert p < 0.07


def test_power_rises_with_effect_and_n():
    small = power.power_at(SCORES, 0.02, 20, 0.5, seed=1, n_sim=200, n_boot=300)
    large = power.power_at(SCORES, 0.10, 20, 0.5, seed=1, n_sim=200, n_boot=300)
    more_n = power.power_at(SCORES, 0.02, 60, 0.5, seed=1, n_sim=200, n_boot=300)
    assert large > small
    assert more_n > small


def test_mde_is_deterministic_and_on_grid():
    grid = [0.02, 0.05, 0.1, 0.2]
    a = power.minimum_detectable_effect(SCORES, 25, 0.5, 3, grid, n_sim=150, n_boot=200)
    b = power.minimum_detectable_effect(SCORES, 25, 0.5, 3, grid, n_sim=150, n_boot=200)
    assert a == b
    assert a[0] in grid
