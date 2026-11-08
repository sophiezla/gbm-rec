#!/usr/bin/env python
"""Pack C5's inputs for the GPU computer (D6 items 8-9). Scores nothing; no PyTorch needed.

For each of the 63 patients, writes ``data/interim/c5_inputs/<patient>.npz`` (not in git):
z-scored T1c and FLAIR, target mask, FLAIR mask (oedema) and the f1 map, cropped to the
brain, plus the evaluation mask and labels taken from the feature cache, so C5 trains and
scores on exactly the voxels every other candidate uses. ``manifest.json`` records each
file's SHA-256 and the commit; ``13_c5_train.py`` refuses files that do not match.

    python scripts/12_c5_pack.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import c5, config, features, lumiere
from gbmc.provenance import file_digest, git_state, require_clean_tree
from gbmc.tournament import CACHE

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "interim" / "c5_inputs"
F1 = features.COLUMNS.index("f1")


def pack_patient(patient: str) -> dict:
    case = lumiere.load_baseline(patient)
    target = case.target
    with np.load(CACHE / f"{patient}.npz") as cached:
        case.brain = features.brain_from_cache(cached["evaluate_packed"],
                                               cached["evaluate_shape"], case.brain, target)
        truth = cached["truth"].astype(bool)
        cached_f1 = cached["matrix"][:, F1]
    evaluate = case.brain & ~target
    base = lumiere.derived_root() / "data" / "interim" / "preprocessed" / patient / case.baseline
    flair = sitk.GetArrayFromImage(sitk.ReadImage(str(base / "FLAIR.nii.gz"), sitk.sitkFloat32))
    f1 = features.signed_distance_mm(target, case.reference)
    if not np.allclose(np.clip(f1, -300, 300)[evaluate], cached_f1, atol=1e-4):
        raise SystemExit(f"{patient}: recomputed f1 does not match the feature cache")
    return c5.pack(patient, case.ct1, flair, target, case.flair_abnormal, f1, evaluate, truth,
                   c5.lr_axis_from_direction(case.reference.GetDirection()))


def main() -> int:
    require_clean_tree()
    config.pin_itk_threads()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for patient in lumiere.cohort():
        path = OUT / f"{patient}.npz"
        np.savez_compressed(path, **pack_patient(patient))
        hashes[patient] = file_digest(path)
        print(patient, flush=True)
    state = git_state()
    (OUT / "manifest.json").write_text(json.dumps(
        {"commit": state["commit"], "dirty": state["dirty"], "sha256": hashes,
         "channels": list(c5.CHANNELS), "f1_scale_mm": c5.F1_SCALE_MM}, indent=2) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
