from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from gbmc import freeze


def _model():
    x = np.random.default_rng(0).normal(size=(50, 2))
    return LogisticRegression().fit(x, x[:, 0] > 0)


def test_save_load_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "models" / "C1.joblib"
    digest = freeze.save_model(_model(), path)
    assert freeze.load_model(path, digest).coef_.shape == (1, 2)
    freeze.verify_hashes({"models/C1.joblib": digest}, tmp_path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(freeze.HashMismatchError):
        freeze.load_model(path, digest)
    with pytest.raises(freeze.HashMismatchError):
        freeze.verify_hashes({"models/C1.joblib": digest}, tmp_path)


def test_missing_model_fails_verification(tmp_path):
    with pytest.raises(freeze.HashMismatchError, match="missing"):
        freeze.verify_hashes({"models/C2.joblib": "0" * 64}, tmp_path)


def test_run_once_refuses_second_run_and_logs_resumes(tmp_path):
    freeze.open_once(tmp_path, ["a", "b"])
    state = freeze.open_once(tmp_path, ["a", "b"])        # interrupted run resumes
    assert len(state["resumed_attempts"]) == 1
    freeze.mark_done(tmp_path, {"n": 2})
    with pytest.raises(freeze.AlreadyRunError):
        freeze.open_once(tmp_path, ["a", "b"])
    assert json.loads((tmp_path / freeze.DONE).read_text())["n"] == 2


def test_mark_done_requires_marker(tmp_path):
    with pytest.raises(RuntimeError):
        freeze.mark_done(tmp_path, {})
