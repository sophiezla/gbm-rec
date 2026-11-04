from __future__ import annotations

import pandas as pd
import pytest

from gbmc import progressions


def _tables():
    ratings = pd.DataFrame({
        "Patient": ["P"] * 6,
        "Timepoint": ["week-000-2", "week-010", "week-020", "week-030", "week-044", "week-060"],
        "Rating": ["Post-Op", "PD", "SD", "PD", "PD", "PD"],
    })
    completeness = pd.DataFrame({
        "Patient": ["P"] * 6,
        "Timepoint": ratings["Timepoint"],
        "DeepBraTumIA": [True, True, True, False, True, True],
    }).set_index(["Patient", "Timepoint"])
    return ratings, completeness


def test_later_progressions_skip_unsegmented_and_cap_at_two():
    ratings, completeness = _tables()
    out = progressions.later_progressions(ratings, completeness, "P", "week-010")
    assert out == ["week-044", "week-060"]      # week-030 has no segmentation; SD skipped


def test_later_progressions_orders_by_week_not_string():
    ratings, completeness = _tables()
    out = progressions.later_progressions(ratings, completeness, "P", "week-000-2", n_later=5)
    assert out == ["week-010", "week-044", "week-060"]


def test_frozen_script_requires_pinned_checkout(monkeypatch):
    monkeypatch.setattr(progressions, "checkout_commit", lambda: "0" * 40)
    with pytest.raises(RuntimeError, match="pinned"):
        progressions.frozen_gbm4d_script("04_label_transport", {})


def _vendored(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "gbm4d").mkdir(parents=True)
    (tmp_path / "scripts" / "04_label_transport.py").write_text("X = 1\n")
    (tmp_path / "src" / "gbm4d" / "transport.py").write_text("Y = 2\n")
    (tmp_path / "config.yaml").write_text("a: 1\n")
    progressions.write_vendored_manifest(tmp_path)
    return tmp_path


def test_vendored_copy_verifies_and_detects_tampering(tmp_path, monkeypatch):
    root = _vendored(tmp_path)
    monkeypatch.setenv("GBMC_GBM4D_ROOT", str(root))
    assert progressions.verify_frozen_gbm4d() == "vendored"
    (root / "src" / "gbm4d" / "transport.py").write_text("Y = 3\n")
    with pytest.raises(RuntimeError, match="modified"):
        progressions.verify_frozen_gbm4d()


def test_neither_git_nor_vendored_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("GBMC_GBM4D_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="neither"):
        progressions.verify_frozen_gbm4d()
