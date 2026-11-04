"""Voxel features f1-f5 and the trimmed guideline distance behind C0T (plan Step 1.3).

Every feature comes from the baseline scan only; nothing from the progression
scan enters a predictor. The same function builds training and scoring
features, so the two cannot drift apart (gbm-4d's rule).

========  ==========================================================  =============
Column    Definition                                                  Plan
========  ==========================================================  =============
f1        signed Euclidean distance (mm) to the target surface         f1
f2        1 inside the baseline oedema label (DeepBraTumIA 3), else 0  f2
f3        signed Euclidean distance (mm) to the f2 surface             f3
f4        f1 x f2                                                      f4
f5        geodesic distance (mm) from the target inside the brain,     f5
          with the falx and the ventricles closed
c0t       geodesic distance with the falx closed and the ventricles    C0T
          entered at most ``ventricle_depth_mm``
========  ==========================================================  =============

Signed distances are negative inside the structure. Inside the evaluation
region (brain minus target) f1 is therefore always >= 0.

Distances that never arrive (a hemisphere closed off by the falx; an absent
FLAIR region) are capped at :data:`MAX_PATH_MM`, as gbm-4d did, so no sentinel
of 1e6 reaches a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

#: Cap on geodesic path length (mm).
MAX_PATH_MM = 300.0

COLUMNS = ("f1", "f2", "f3", "f4", "f5", "c0t")


def _image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(array.astype(np.uint8))
    out.CopyInformation(reference)
    return out


def signed_distance_mm(mask: np.ndarray, reference: sitk.Image) -> np.ndarray | None:
    """Signed Euclidean distance to the surface of ``mask``; None if it is empty."""
    if not mask.any():
        return None
    distance = sitk.SignedMaurerDistanceMap(
        _image(mask, reference), insideIsPositive=False, squaredDistance=False,
        useImageSpacing=True)
    return sitk.GetArrayFromImage(distance).astype(np.float32)


def geodesic_distance_mm(target: np.ndarray, allowed: np.ndarray,
                         reference: sitk.Image) -> np.ndarray:
    """Distance from ``target`` travelling only through ``allowed``, capped.

    Uses fast marching (:func:`gbmc.anatomy.geodesic_distance`).
    """
    from gbmc import anatomy

    distance = anatomy.geodesic_distance(_image(target, reference), _image(allowed, reference))
    return np.minimum(sitk.GetArrayFromImage(distance), MAX_PATH_MM).astype(np.float32)


def ventricle_allowance(tissue_distance: np.ndarray, ventricles: np.ndarray,
                        spacing_zyx, depth_mm: float) -> np.ndarray:
    """Extend a tissue-only distance map at most ``depth_mm`` into the ventricles.

    A ventricle voxel inherits the distance of its nearest non-ventricle voxel
    plus its own Euclidean depth below the ventricle wall. Deeper than
    ``depth_mm`` it stays unreachable. With ``depth_mm = 0`` the ventricles are
    closed entirely.
    """
    out = tissue_distance.copy()
    if not ventricles.any():
        return out
    depth, nearest = ndimage.distance_transform_edt(
        ventricles, sampling=spacing_zyx, return_indices=True)
    inherited = tissue_distance[tuple(nearest)] + depth
    reach = ventricles & (depth <= depth_mm) & (inherited < MAX_PATH_MM)
    out[reach] = inherited[reach].astype(np.float32)
    out[ventricles & ~reach] = MAX_PATH_MM
    return out


@dataclass
class FeatureVolumes:
    """Whole-volume feature maps for one patient."""

    volumes: dict[str, np.ndarray]
    evaluate: np.ndarray   # brain minus target

    def matrix(self, columns=COLUMNS) -> np.ndarray:
        """(n_evaluation_voxels, n_columns) float32, in C order of the volume."""
        return np.column_stack([self.volumes[c][self.evaluate] for c in columns]
                               ).astype(np.float32)


def build(reference: sitk.Image, brain: np.ndarray, target: np.ndarray,
          flair_abnormal: np.ndarray, falx: np.ndarray, ventricles: np.ndarray,
          ventricle_depth_mm: float) -> FeatureVolumes:
    """f1-f5 and the C0T distance for one patient.

    ``falx`` and ``ventricles`` are boolean barrier masks on the same grid.
    """
    if not target.any():
        raise ValueError("empty treatment target: nothing to measure distance from")
    sx, sy, sz = reference.GetSpacing()
    spacing_zyx = (sz, sy, sx)
    target_in_brain = target & brain
    ventricles = ventricles & brain & ~target_in_brain
    falx = falx & brain & ~target_in_brain

    f1 = np.clip(signed_distance_mm(target, reference), -MAX_PATH_MM, MAX_PATH_MM)
    f2 = flair_abnormal.astype(np.float32)
    f3 = signed_distance_mm(flair_abnormal, reference)
    # An absent FLAIR region is not distance zero; every voxel is maximally far
    # from a structure that does not exist.
    f3 = (np.full(brain.shape, MAX_PATH_MM, dtype=np.float32) if f3 is None
          else np.clip(f3, -MAX_PATH_MM, MAX_PATH_MM))
    f4 = f1 * f2

    tissue = brain & ~falx & ~ventricles
    through_tissue = geodesic_distance_mm(target, tissue, reference)
    f5 = through_tissue
    c0t = ventricle_allowance(through_tissue, ventricles, spacing_zyx, ventricle_depth_mm)
    # The falx itself is never entered: it stays at the cap in both maps.

    return FeatureVolumes(
        volumes={"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "c0t": c0t},
        evaluate=brain & ~target)
