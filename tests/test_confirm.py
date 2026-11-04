"""The confirmatory scoring path on a synthetic patient: every candidate, every metric."""

from __future__ import annotations

import numpy as np
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
    row, scores, qc = confirm.score_patient(inp, _models())
    assert set(scores) == {"C0", "C0T", "C4", "C1", "C3", "C2", "C6"}
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
