from __future__ import annotations

import numpy as np
import pytest

from gbmc import consort, rhuh, screens


def _scan(labels=(1, 2, 3)):
    """Enhancing bright on T1ce, necrosis dark, peritumoral FLAIR-bright."""
    seg = np.zeros((10, 10, 30), np.uint8)
    t1 = np.zeros(seg.shape, np.float32)
    fl = np.zeros(seg.shape, np.float32)
    necro, peri, enh = labels
    seg[:, :, 0:10], t1[:, :, 0:10], fl[:, :, 0:10] = necro, -2.0, 0.5
    seg[:, :, 10:20], t1[:, :, 10:20], fl[:, :, 10:20] = peri, 0.2, 2.5
    seg[:, :, 20:30], t1[:, :, 20:30], fl[:, :, 20:30] = enh, 3.0, 1.0
    return seg, t1, fl, np.ones(seg.shape, bool)


def test_mapping_matches_published():
    ev = [rhuh.derive_mapping(f"P{i}", *_scan()) for i in range(5)]
    decision = rhuh.check_mapping(ev, min_agreement=0.8)
    assert decision.mapping == rhuh.EXPECTED_LABELS and not decision.corrected


def test_swapped_mapping_is_corrected():
    ev = [rhuh.derive_mapping(f"P{i}", *_scan(labels=(1, 3, 2))) for i in range(5)]
    decision = rhuh.check_mapping(ev, min_agreement=0.8)
    assert decision.corrected
    assert decision.mapping == {1: "necrosis", 3: "peritumoral", 2: "enhancing"}


def test_ambiguous_mapping_is_refused():
    ev = [rhuh.derive_mapping("A", *_scan(labels=(1, 3, 2))),
          rhuh.derive_mapping("B", *_scan(labels=(2, 1, 3))),
          rhuh.derive_mapping("C", *_scan(labels=(3, 2, 1)))]
    with pytest.raises(ValueError, match="not decidable"):
        rhuh.check_mapping(ev, min_agreement=0.8)


def test_scan_missing_a_label_does_not_vote():
    seg, t1, fl, brain = _scan()
    seg[seg == 1] = 0
    assert rhuh.derive_mapping("P", seg, t1, fl, brain).derived == {}


def test_target_rule_both_branches():
    seg = np.zeros((5, 5, 5), np.uint8)
    seg[0, 0, 0] = 3
    seg[1, 1, 1] = 4
    brain = np.ones(seg.shape, bool)
    brain[4] = False
    t, how = rhuh.treatment_target(seg, brain, cavity_label=4, preop_core_in_postop=None)
    assert t.sum() == 2 and "labelled" in how
    core = np.zeros(seg.shape, bool)
    core[3:5, 3, 3] = True
    t, how = rhuh.treatment_target(seg, brain, cavity_label=None, preop_core_in_postop=core)
    assert t.sum() == 2      # residual + core voxel inside the brain; the one outside is clipped
    with pytest.raises(ValueError):
        rhuh.treatment_target(seg, brain, cavity_label=None, preop_core_in_postop=None)


def test_layout_refuses_before_download():
    with pytest.raises(ValueError):
        rhuh.layout()


def test_inventory_reasons():
    ok = rhuh.inventory_row("P", {"preop": set(), "postop": {"T1ce", "FLAIR", "seg"},
                                  "recurrence": {"T1ce", "FLAIR", "seg"}}, 30.0, False)
    assert ok.usable
    late = rhuh.inventory_row("P", {"postop": {"T1ce", "FLAIR", "seg"},
                                    "recurrence": {"T1ce", "seg"}}, 80.0, None)
    assert not late.usable and len(late.reasons) == 2


def test_consort_counts_add_up():
    enrolled = [f"P{i}" for i in range(6)]
    outcomes = [consort.Outcome("P0", "failed transport", "metric never evaluated"),
                consort.Outcome("P1", "no enhancing recurrence outside the target"),
                *[consort.Outcome(p, None) for p in enrolled[2:]]]
    table = consort.flow(enrolled, outcomes)
    assert table["n_evaluable"] == 4 and table["rows"][-1]["remaining"] == 4
    assert "**4**" in consort.markdown(table)
    with pytest.raises(ValueError):
        consort.flow(enrolled + ["P9"], outcomes)


def test_consensus_names_the_outlier():
    base = np.zeros((6, 6, 6, 3))
    fields = {"T1c": base + [5.0, 0, 0], "T2": base, "FLAIR": base + [0.1, 0, 0],
              "shipped_rigid": base}
    out = screens.consensus(fields, np.ones((6, 6, 6), bool))
    assert out["dissenter"] == "T1c" and out["our_deviation_mm"] > 4


def test_synthetic_field_properties():
    z, y, x = np.indices((40, 40, 40))
    brain = (z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2 < 18 ** 2
    f = screens.synthetic_field(brain.shape, (1, 1, 1), brain, seed=1)
    assert np.all(f[~brain] == 0)
    assert np.linalg.norm(f, axis=-1).max() <= 8.0 + 1e-9
    err = screens.recovery_error(f, f, brain, 1.0)
    assert err["error_median_mm"] == 0 and err["recovered_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("rt, after, before, dissenter, expected", [
    (0.9, 1.0, 1.0, None, "SCREENED"),
    (2.5, 1.0, 1.0, None, "EXCLUDE"),
    (0.9, 4.0, 3.9, None, "REVIEW"),
    (0.9, 4.0, 3.9, "T1c", "EXCLUDE"),
    (0.9, 1.0, 1.0, "T1c", "REVIEW"),
])
def test_screen_verdict(rt, after, before, dissenter, expected):
    r = screens.ScreenResult("P", rt, after, before, dissenter)
    assert screens.verdict(r, 2.0, 3.0, 1.0) == expected
