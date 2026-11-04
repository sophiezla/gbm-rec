from __future__ import annotations

import numpy as np
import pytest

from gbmc import candidates, metrics, stats


def test_coverage_shares_ties_at_the_cut():
    scores = np.array([3.0, 2.0, 2.0, 2.0, 2.0, 1.0])
    truth = np.array([0, 1, 0, 0, 0, 1], bool)
    # top 3: voxel 0 plus 2 of the 4 tied voxels -> expected half of one positive
    assert metrics.coverage_at_voxels(scores, truth, 3) == pytest.approx(0.25)
    assert metrics.coverage_at_voxels(scores, truth, 6) == 1.0
    assert metrics.coverage_at_voxels(scores, truth, 0) == 0.0


def test_undefined_patient_is_nan_not_zero():
    truth = np.zeros(10, bool)
    out = metrics.patient_scores(np.arange(10.0), truth, {"margin15": 3})
    assert all(np.isnan(v) for v in out.values())


def test_lift_of_random_ranking_is_about_one():
    rng = np.random.default_rng(0)
    truth = rng.random(20000) < 0.05
    ap = metrics.pr_auc(rng.random(20000), truth)
    assert metrics.lift_over_chance(ap, truth) == pytest.approx(1.0, abs=0.15)


def test_folds_are_disjoint_complete_and_deterministic():
    patients = [f"Patient-{i:03d}" for i in range(63)]
    a = candidates.patient_folds(patients, 7, seed=11)
    assert candidates.patient_folds(list(reversed(patients)), 7, seed=11) == a
    assert [len(f) for f in a] == [9] * 7
    assert sorted(p for f in a for p in f) == patients
    assert candidates.patient_folds(patients, 7, seed=12) != a


@pytest.mark.parametrize("means, expected", [
    ({"C4": 0.05, "C1": 0.08, "C3": 0.081, "C2": 0.07, "C6": 0.084}, "C1"),   # C6 within 0.005
    ({"C4": 0.05, "C1": 0.08, "C3": 0.07, "C2": 0.07, "C6": 0.0851}, "C6"),   # C6 clears margin
    ({"C4": 0.05, "C1": 0.06, "C3": 0.07, "C2": 0.09, "C6": 0.07}, "C2"),
])
def test_selection_rule(means, expected):
    assert candidates.select_challenger(means, 0.005) == expected


def test_selection_ignores_distance_baselines():
    means = {"C0": 0.5, "C0T": 0.5, "C4": 0.01, "C1": 0.02}
    assert candidates.select_challenger(means, 0.005) == "C1"


def test_rank_average_ties_and_scale():
    r = candidates.percentile_ranks(np.array([1.0, 1.0, 3.0]))
    np.testing.assert_allclose(r, [0.25, 0.25, 1.0])
    c6 = candidates.rank_average([np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])])
    np.testing.assert_allclose(c6, [0.5, 0.5, 0.5])


@pytest.mark.parametrize("name", ["C1", "C3", "C2"])
def test_learned_candidates_fit_and_predict(name):
    rng = np.random.default_rng(1)
    cand = candidates.LEARNED[name]
    matrices, truths = [], []
    for n in (400, 200):
        x = rng.normal(size=(n, 6)).astype(np.float32)
        matrices.append(x)
        truths.append(x[:, 0] < -1.0)
    model = candidates.fit(cand, matrices, truths, seed=3)
    p = candidates.predict(cand, model, matrices[0])
    assert p.shape == (400,)
    assert metrics.pr_auc(p, truths[0]) > 0.5


def test_bca_is_deterministic_and_brackets_the_mean():
    values = np.random.default_rng(2).normal(0.01, 0.02, size=50)
    a = stats.mean_with_bca(np.append(values, np.nan), 1000, seed=5)
    assert a == stats.mean_with_bca(np.append(values, np.nan), 1000, seed=5)
    assert a.n == 50
    assert a.low < a.mean < a.high
