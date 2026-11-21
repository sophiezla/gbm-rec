"""The confirmatory scoring path on a synthetic patient: every candidate, every metric."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from gbmc import candidates, confirm, rhuh_features


def _patient():
    shape = (48, 48, 48)
    z, y, x = np.indices(shape)
    r2 = lambda c: (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2  # noqa: E731
    brain = r2((24, 24, 24)) < 21 ** 2
    target = r2((24, 24, 16)) < 4 ** 2
    flair = (r2((24, 24, 16)) < 8 ** 2) & ~target
    recurrence = (r2((24, 24, 22)) < 3 ** 2) | (r2((24, 12, 34)) < 2 ** 2)   # one near, one far
    ref = sitk.GetImageFromArray(np.zeros(shape, np.float32))
    return confirm.PatientInputs("P", ref, brain, target, flair,
                                 np.zeros(shape, np.float32), target | flair, recurrence & brain)


def _models():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6)).astype(np.float32)
    y = x[:, 0] < -1
    return {n: candidates.fit(c, [x], [y], seed=1) for n, c in candidates.LEARNED.items()}


def test_score_patient_every_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(rhuh_features, "falx", lambda ref, brain, t1: np.zeros(brain.shape, bool))
    inp = _patient()
    n = int((inp.brain & ~inp.target).sum())
    c5_score = np.random.default_rng(2).random(n)
    with pytest.raises(RuntimeError, match="C5 scores missing"):
        confirm.score_patient(inp, _models())
    with pytest.raises(RuntimeError, match="evaluation voxels"):
        confirm.score_patient(inp, _models(), c5_score[:-1])
    row, scores, qc = confirm.score_patient(inp, _models(), c5_score)
    assert set(scores) == {"C0", "C0T", "C4", "C1", "C3", "C2", "C5", "C6", "E1"}
    # E1 (H2, D19) is exactly the rank average of C0 and C2
    assert np.allclose(scores["E1"], candidates.rank_average([scores["C0"], scores["C2"]]))
    assert row["evaluable"] and row["n_positive"] > 0
    assert 0 < row["share_beyond_20mm"] < 1          # the far lesion escapes the margin
    for name in scores:
        m = row[name]
        assert 0 < m["pr_auc"] <= 1 and 0 <= m["coverage_margin15"] <= 1
    # Distance ranks the near lesion first, so it covers it at the margin's volume
    assert row["C0"]["coverage_margin15"] > 0
    written = confirm.write_maps(tmp_path, "P", inp.reference, inp.brain & ~inp.target,
                                 {"C0": scores["C0"]}, volumes_ml=(1,))
    assert (tmp_path / "P" / written["C0_map"]).is_file()
    top = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "P" / written["C0_top1ml"])))
    assert int(top.sum()) == 1000


def test_score_patient_without_c5_scores_h2_only(monkeypatch):
    monkeypatch.setattr(rhuh_features, "falx", lambda ref, brain, t1: np.zeros(brain.shape, bool))
    row, scores, _ = confirm.score_patient(_patient(), _models(), include_c5=False)
    assert "C5" not in scores and "C6" not in scores and "E1" in scores
    assert 0 < row["E1"]["pr_auc"] <= 1


def test_h2_summary_classifies_and_fills_the_sentence():
    rng = np.random.default_rng(1)
    rows = {f"P{i}": {"evaluable": True,
                      "C0": {"pr_auc": 0.05 + 0.01 * rng.random(), "coverage_margin15": 0.7},
                      "E1": {"pr_auc": 0.30, "coverage_margin15": 0.9}} for i in range(12)}
    rows["P_undefined"] = {"evaluable": False}
    out = confirm.h2_summary(rows, "MU-Glioma-Post", 0.04, master=7, n_boot=200)
    assert out["n_evaluable"] == 12 and out["outcome"] == "Final model wins"
    assert out["sentence"].startswith("On MU-Glioma-Post, the final model adds +0.2")


def test_h2_sentences_cover_all_outcomes():
    from gbmc import verdict

    s = {o: verdict.h2_sentence(o, "RHUH-GBM", -0.01, (-0.02, 0.001)) for o in verdict.OUTCOMES}
    assert "adds less than 0.0010" in s["Bounded null"]
    assert "too small" in s["Inconclusive"]
    assert "distance beats the final model by 0.0100" in s["Challenger worse"]
    assert set(verdict.H2_OUTCOMES.values()) == {"Final model wins", "Bounded null",
                                                "Inconclusive", "Final model worse"}
