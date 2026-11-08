"""The C5 import (gbmc.tournament.c5_held_out) refuses anything that does not match the run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gbmc.provenance import file_digest
from gbmc.tournament import c5_held_out

PATIENTS = ["Patient-001", "Patient-002"]
FOLDS = [["Patient-001"], ["Patient-002"]]


def _write(tmp_path, folds=FOLDS, dirty=False, tamper=False):
    hashes = {}
    for i, patient in enumerate(PATIENTS):
        path = tmp_path / f"{patient}.npz"
        np.savez(path, score=np.full(5, float(i)))
        hashes[patient] = file_digest(path)
    if tamper:
        np.savez(tmp_path / "Patient-002.npz", score=np.zeros(5))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"folds": folds, "dirty": dirty, "sha256": hashes}), encoding="utf-8")


def test_accepts_matching_maps(tmp_path):
    _write(tmp_path)
    scores = c5_held_out(PATIENTS, FOLDS, tmp_path)
    assert scores["Patient-002"].tolist() == [1.0] * 5


def test_refuses_without_maps(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        c5_held_out(PATIENTS, FOLDS, tmp_path)


def test_refuses_other_folds(tmp_path):
    _write(tmp_path, folds=[PATIENTS])
    with pytest.raises(SystemExit, match="different folds"):
        c5_held_out(PATIENTS, FOLDS, tmp_path)


def test_refuses_uncommitted_code(tmp_path):
    _write(tmp_path, dirty=True)
    with pytest.raises(SystemExit, match="uncommitted"):
        c5_held_out(PATIENTS, FOLDS, tmp_path)


def test_refuses_a_changed_file(tmp_path):
    _write(tmp_path, tamper=True)
    with pytest.raises(SystemExit, match="hash"):
        c5_held_out(PATIENTS, FOLDS, tmp_path)
