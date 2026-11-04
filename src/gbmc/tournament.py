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
