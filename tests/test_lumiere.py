from __future__ import annotations

import json

import pytest

from gbmc import lumiere


def _lock(tmp_path, cohort, train, test):
    d = tmp_path / "results" / "cohort_lock"
    d.mkdir(parents=True)
    (d / "cohort_lock.json").write_text(
        json.dumps({"cohort": cohort, "train": train, "test": test}), encoding="utf-8")


def test_cohort_reads_the_list_and_checks_it(tmp_path, monkeypatch):
    ids = [f"Patient-{i:03d}" for i in range(63)]
    _lock(tmp_path, list(reversed(ids)), ids[:50], ids[50:])
    monkeypatch.setattr(lumiere, "COHORT_LOCK", tmp_path / "results/cohort_lock/cohort_lock.json")
    assert lumiere.cohort() == ids


def test_cohort_refuses_inconsistent_lock(tmp_path, monkeypatch):
    ids = [f"Patient-{i:03d}" for i in range(63)]
    _lock(tmp_path, ids, ids[:50], ids[51:])
    monkeypatch.setattr(lumiere, "COHORT_LOCK", tmp_path / "results/cohort_lock/cohort_lock.json")
    with pytest.raises(ValueError):
        lumiere.cohort()


def test_committed_cohort_lock():
    patients = lumiere.cohort()
    assert len(patients) == 63
    assert "Patient-040" not in patients and "Patient-087" not in patients
