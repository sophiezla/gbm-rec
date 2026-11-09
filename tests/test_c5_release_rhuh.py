"""C5 release import (03_) and RHUH-GBM map import (30_/31_) refuse anything unchecked."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gbmc import c5_import, confirm
from gbmc.imaging import zscore_array
from gbmc.provenance import file_digest

SETTINGS = {"patch": [96, 96, 96], "batch": 2}
PATIENTS = ["Patient-001", "Patient-002"]


# ---------------------------------------------------------------- release model

def _release(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    release = tmp_path / "c5_release"
    release.mkdir()
    (release / "model.pt").write_bytes(b"weights")
    manifest = {"commit": "abc", "dirty": False, "smoke": False, "settings": SETTINGS,
                "patients": PATIENTS, "model_sha256": file_digest(release / "model.pt"),
                "torch": "2.15.0", "device": "cuda", "gpu": "RTX"} | overrides
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    oof = tmp_path / "oof_manifest.json"
    oof.write_text(json.dumps({"commit": "def"}), encoding="utf-8")
    return release, oof


def _import(tmp_path, release, oof, same=True, patients=PATIENTS):
    return c5_import.import_release(release, tmp_path / "out", oof, SETTINGS, patients,
                                    code_check=lambda a, b: same)


def test_release_import_copies_and_registers(tmp_path):
    release, oof = _release(tmp_path)
    rel, digest, prov = _import(tmp_path, release, oof)
    assert rel == "models/C5.pt"
    assert file_digest(tmp_path / "out" / rel) == digest
    assert prov["commit"] == "abc" and prov["tournament_maps_commit"] == "def"


@pytest.mark.parametrize("overrides, message", [
    ({"dirty": True}, "uncommitted"),
    ({"smoke": True}, "smoke"),
    ({"settings": {"patch": [32, 32, 32], "batch": 2}}, "other settings"),
    ({"patients": PATIENTS[:1]}, "locked cohort"),
    ({"model_sha256": "0" * 64}, "hash"),
])
def test_release_import_refuses(tmp_path, overrides, message):
    release, oof = _release(tmp_path, **overrides)
    with pytest.raises(SystemExit, match=message):
        _import(tmp_path, release, oof)
    assert not (tmp_path / "out" / "models" / "C5.pt").exists()


def test_release_import_refuses_changed_network_code(tmp_path):
    release, oof = _release(tmp_path)
    with pytest.raises(SystemExit, match="differs"):
        _import(tmp_path, release, oof, same=False)


def test_same_code_compares_only_the_network_file():
    # HEAD against itself is trivially the same; a missing commit is refused, not guessed.
    assert c5_import.same_code("HEAD", "HEAD")
    with pytest.raises(SystemExit, match="not in this clone"):
        c5_import.same_code("HEAD", "0" * 40)


# ---------------------------------------------------------------- RHUH-GBM maps

MODEL_SHA = "a" * 64


def _rhuh(tmp_path: Path, variant="tuned", model_sha=MODEL_SHA, tamper=False,
          other_inputs=False, count=5, dirty=False):
    inputs, maps = tmp_path / "inputs", tmp_path / "maps"
    (inputs / variant).mkdir(parents=True)
    (maps / variant).mkdir(parents=True)
    in_hashes, map_hashes = {}, {}
    for i, p in enumerate(PATIENTS):
        (inputs / variant / f"{p}.npz").write_bytes(f"input {i}".encode())
        in_hashes[p] = file_digest(inputs / variant / f"{p}.npz")
        np.savez(maps / variant / f"{p}.npz", score=np.full(count, float(i)))
        map_hashes[p] = file_digest(maps / variant / f"{p}.npz")
    if tamper:
        np.savez(maps / variant / "Patient-002.npz", score=np.zeros(count))
    (inputs / variant / "manifest.json").write_text(json.dumps(
        {"dirty": False, "sha256": in_hashes, "n_evaluate": dict.fromkeys(PATIENTS, 5)}))
    used = {p: "0" * 64 for p in PATIENTS} if other_inputs else in_hashes
    (maps / variant / "manifest.json").write_text(json.dumps(
        {"dirty": dirty, "smoke": False, "model_sha256": model_sha, "inputs_sha256": used,
         "sha256": map_hashes}))
    return inputs, maps


def test_rhuh_scores_accepts_matching_maps(tmp_path):
    inputs, maps = _rhuh(tmp_path)
    scores = c5_import.rhuh_scores("tuned", PATIENTS, MODEL_SHA, inputs, maps)
    assert scores["Patient-002"].tolist() == [1.0] * 5


@pytest.mark.parametrize("kwargs, message", [
    ({"model_sha": "b" * 64}, "registered release model"),
    ({"other_inputs": True}, "other inputs"),
    ({"tamper": True}, "hash"),
    ({"count": 4}, "voxel count"),
    ({"dirty": True}, "uncommitted"),
])
def test_rhuh_scores_refuses(tmp_path, kwargs, message):
    inputs, maps = _rhuh(tmp_path, **kwargs)
    with pytest.raises(SystemExit, match=message):
        c5_import.rhuh_scores("tuned", PATIENTS, MODEL_SHA, inputs, maps)


def test_rhuh_scores_refuses_missing_and_unknown(tmp_path):
    inputs, maps = _rhuh(tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        c5_import.rhuh_scores("tuned", [*PATIENTS, "Patient-003"], MODEL_SHA, inputs, maps)
    with pytest.raises(ValueError, match="variant"):
        c5_import.rhuh_scores("other", PATIENTS, MODEL_SHA, inputs, maps)


def test_c5_variants_cover_every_arm_and_the_primary():
    arms = {"a": {"patients": ["P1", "P2"], "c5": "tuned"},
            "b": {"patients": ["P3"], "c5": "original"},
            "c": {"patients": ["P2", "P4"], "c5": "tuned"}}
    assert confirm.c5_variants(arms, ["P5"]) == {"tuned": ["P1", "P2", "P4", "P5"],
                                                 "original": ["P3"]}


def test_zscore_array_is_scale_free():
    rng = np.random.default_rng(0)
    values = rng.normal(500, 80, size=(6, 6, 6)).astype(np.float32)
    mask = values > 450
    z = zscore_array(values, mask)
    assert z[~mask].tolist() == [0.0] * int((~mask).sum())
    assert abs(float(z[mask].mean())) < 1e-5 and abs(float(z[mask].std()) - 1) < 1e-4
    assert np.allclose(zscore_array(values * 3 + 7, mask), z, atol=1e-5)


# ---------------------------------------------------------------- GPU-side scorer

def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scorer_loads_only_the_registered_model(tmp_path):
    torch = pytest.importorskip("torch")
    from gbmc import c5

    tiny = c5.Settings(patch=(16, 16, 16), levels=3, base_channels=4)
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    torch.save(c5.build_unet(tiny).state_dict(), path)
    scorer = _script("14_c5_score_rhuh.py")
    model = scorer.load_model(path, file_digest(path), tiny, "cpu")
    shape = (20, 18, 22)
    evaluate = np.zeros(shape, bool)
    evaluate[2:-2, 2:-2, 2:-2] = True
    zeros = np.zeros(shape, np.float32)
    packed = c5.pack("P", zeros, zeros, ~evaluate, zeros.astype(bool), zeros, evaluate,
                     np.zeros(int(evaluate.sum()), bool), 2)
    np.savez(tmp_path / "P.npz", **packed)
    score = c5.predict(model, c5.load_case(tmp_path / "P.npz"), tiny, "cpu")
    assert score.shape == (int(evaluate.sum()),)
    with pytest.raises(SystemExit, match="registered"):
        scorer.load_model(path, "0" * 64, tiny, "cpu")
