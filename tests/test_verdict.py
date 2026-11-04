from __future__ import annotations

import numpy as np
import pytest

from gbmc import verdict


@pytest.mark.parametrize("low, high, mde, expected", [
    (0.001, 0.02, 0.05, "Challenger champion"),
    (-0.01, 0.02, 0.05, "Bounded null"),
    (-0.01, 0.07, 0.05, "Inconclusive"),
    (-0.03, -0.001, 0.05, "Challenger worse"),
])
def test_classify(low, high, mde, expected):
    assert verdict.classify(low, high, mde) == expected


def test_champion_is_c0_unless_challenger_wins():
    assert verdict.champion("Bounded null", "C1") == "C0"
    assert verdict.champion("Challenger champion", "C1") == "C1"


def test_bh_matches_hand_computation():
    out = verdict.benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}, q=0.05)
    assert out["a"]["p_adjusted"] == pytest.approx(0.04)
    # c (rank 2, raw 0.06) inherits b's smaller 0.04 * 4 / 3 by the step-up minimum
    assert out["c"]["p_adjusted"] == pytest.approx(0.04 * 4 / 3)
    assert out["a"]["rejected"] and not out["d"]["rejected"]


def test_bootstrap_p():
    assert verdict.bootstrap_p(np.full(100, 0.1)) == 0.0
    assert verdict.bootstrap_p(np.linspace(-1, 1, 101)) == pytest.approx(1.0)


def test_sentences_follow_the_plan():
    s = verdict.sentence("Bounded null", "C1", -0.004, (-0.012, 0.006), 0.42, 0.44, 7)
    assert s.startswith("On an independent cohort, the best challenger from a")
    assert "seven-model" in s and "0.0060" in s and "42.0%" in s and "44.0%" in s
    s = verdict.sentence("Challenger champion", "C2", 0.02, (0.005, 0.03), 0.5, 0.4, 8)
    assert "C2 adds +0.0200 PR-AUC" in s
    s = verdict.sentence("Bounded null", "C2", 0.0, (-0.01, 0.01), 0.5, 0.4, 8,
                         coverage_ci=(0.02, 0.15), coverage_differs=True)
    assert s.startswith("The verdict follows the pre-registered primary")
