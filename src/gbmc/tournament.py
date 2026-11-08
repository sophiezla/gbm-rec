"""Shared access to the per-patient feature cache written by ``scripts/01_``.

Used by the tournament (``02_``) and the release refit (``03_``) so both train
on exactly the same rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gbmc import config
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data" / "interim" / "features"


def load(patient: str, cache: Path = CACHE) -> dict:
    with np.load(cache / f"{patient}.npz") as z:
        return {"matrix": z["matrix"], "truth": z["truth"].astype(bool),
                "matched": dict(zip(z["matched_labels"].tolist(),
                                    z["matched_voxels"].tolist(), strict=True))}


def subsample_negatives(truth: np.ndarray, ratio: int, seed: int) -> np.ndarray:
    """Row indices: every positive, plus ``ratio`` random negatives per positive."""
    positives = np.flatnonzero(truth)
    if positives.size == 0:
        raise ValueError("no positive voxels: this patient cannot train a fold")
    negatives = np.flatnonzero(~truth)
    count = min(negatives.size, ratio * positives.size)
    chosen = np.random.default_rng(seed).choice(negatives, size=count, replace=False)
    return np.concatenate([positives, chosen])


def missing(patients: list[str], cache: Path = CACHE) -> list[str]:
    return [p for p in patients if not (cache / f"{p}.npz").is_file()]


def training_rows(patient: str, master: int, cache: Path = CACHE
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Every positive plus seeded negatives, fixed per patient across folds (gbm-4d arm2).

    A patient with no positives still contributes random negatives.
    """
    data = load(patient, cache)
    truth = data["truth"]
    seed = stable_seed("tournament_negatives", patient, master_seed=master)
    if truth.any():
        rows = subsample_negatives(truth, int(config.get("tournament.negative_ratio")), seed)
    else:
        take = min(truth.size, int(config.get("tournament.empty_patient_negatives")))
        rows = np.random.default_rng(seed).choice(truth.size, size=take, replace=False)
    return data["matrix"][rows], truth[rows]


#: C5 held-out score maps, trained on the GPU computer and imported (D6 (g)). Not in git.
C5_MAPS = REPO_ROOT / "data" / "interim" / "c5_oof"


def c5_held_out(patients: list[str], folds: list[list[str]], maps: Path = C5_MAPS
                ) -> dict[str, np.ndarray]:
    """C5's imported held-out scores, refusing anything that does not match this run.

    ``maps/manifest.json`` (written on the GPU computer) must list the same folds, a clean
    tree, and a SHA-256 for every ``<patient>.npz``; each file holds ``score``, one value
    per evaluation voxel in the feature cache's order.
    """
    import json

    from gbmc.provenance import file_digest

    manifest_path = maps / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"C5 is a fixed candidate (D6 (g)) but {manifest_path} is missing: "
                         "train C5 on the GPU computer and import its maps first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("folds") != folds:
        raise SystemExit("C5 maps were made with different folds; refusing")
    if manifest.get("dirty", True):
        raise SystemExit("C5 maps were made from uncommitted code; refusing")
    scores = {}
    for patient in patients:
        path = maps / f"{patient}.npz"
        if not path.is_file():
            raise SystemExit(f"C5 map missing for {patient}")
        if file_digest(path) != manifest.get("sha256", {}).get(patient):
            raise SystemExit(f"C5 map for {patient} does not match its recorded hash")
        with np.load(path) as z:
            scores[patient] = z["score"].astype(np.float64)
    return scores
