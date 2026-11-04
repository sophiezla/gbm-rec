"""Per-patient scores (plan Step 1.5): PR-AUC, lift over chance, matched-volume coverage.

Everything here is computed for ONE patient over the evaluation region (brain
minus target) and aggregated across patients afterwards. Voxels are never
pooled across patients.
"""

from __future__ import annotations

import numpy as np
from gbm4d.arm0 import patient_pr_auc


def pr_auc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Average precision; NaN when the patient has no positive voxel (undefined, never 0).

    gbm-4d's implementation, reused so the endpoint is computed identically.
    """
    return patient_pr_auc(np.asarray(scores), np.asarray(truth, dtype=bool))


def lift_over_chance(ap: float, truth: np.ndarray) -> float:
    """PR-AUC divided by prevalence, the PR-AUC of a random ranking."""
    prevalence = float(np.mean(truth))
    return float(ap / prevalence) if prevalence > 0 and np.isfinite(ap) else float("nan")


def coverage_at_voxels(scores: np.ndarray, truth: np.ndarray, n_treated: int) -> float:
    """Share of recurrence inside the ``n_treated`` highest-scoring voxels.

    The denominator is recurrence in the evaluation region (outside the target).

    Ties at the cut are shared, not broken by voxel order: if the cut falls
    inside a group of equal scores, each voxel of that group counts as treated
    with the same probability, which is the expected coverage under a random
    tie-break. A distance map has many exact ties on a 1 mm lattice, and
    breaking them by array index would bias coverage in a direction nobody
    chose.
    """
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    n_treated = int(min(max(n_treated, 0), scores.size))
    if n_treated == 0:
        return 0.0
    if n_treated == scores.size:
        return 1.0
    threshold = np.partition(scores, scores.size - n_treated)[scores.size - n_treated]
    above = scores > threshold
    at = scores == threshold
    remaining = n_treated - int(above.sum())
    share = remaining / int(at.sum())
    captured = int((truth & above).sum()) + share * int((truth & at).sum())
    return float(captured / positives)


def coverage_curve(scores: np.ndarray, truth: np.ndarray, voxels_per_ml: float,
                   volumes_ml=None) -> dict[str, float]:
    """Coverage at each treated volume (ml) on gbm-4d's grid (``gbm4d.arm0.COVERAGE_VOLUMES_ML``).

    Keys are the volume as a string (JSON-safe). Volumes larger than the
    evaluation region are skipped.
    """
    from gbm4d.arm0 import COVERAGE_VOLUMES_ML

    out = {}
    for ml in volumes_ml or COVERAGE_VOLUMES_ML:
        n = int(round(ml * voxels_per_ml))
        if 0 < n <= len(scores):
            out[str(ml)] = coverage_at_voxels(scores, truth, n)
    return out


def voxels_for_volume(volume_ml: float, spacing_zyx) -> int:
    return int(round(volume_ml * 1000.0 / float(np.prod(spacing_zyx))))


def patient_scores(scores: np.ndarray, truth: np.ndarray, matched_voxels: dict[str, int]
                   ) -> dict[str, float]:
    """PR-AUC, lift and coverage at each matched volume for one patient and one candidate.

    ``matched_voxels`` maps a label (e.g. ``"margin15"``) to a treated-voxel count
    that is the same for every candidate of this patient.
    """
    ap = pr_auc(scores, truth)
    out = {"pr_auc": ap, "lift": lift_over_chance(ap, truth)}
    for label, n in matched_voxels.items():
        out[f"coverage_{label}"] = coverage_at_voxels(scores, truth, n)
    return out
