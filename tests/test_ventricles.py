"""Ventricle rule V4 (gbmc.anatomy.segment_ventricles) on a synthetic head."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from gbmc import anatomy

N = 120
CENTRE = np.array([60.0, 60.0, 60.0])


def _grid():
    z, y, x = np.indices((N, N, N), dtype=float)
    return z - CENTRE[0], y - CENTRE[1], x - CENTRE[2]


def _image(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _phantom(dark_ball_mm: float = 0.0):
    z, y, x = _grid()
    brain = np.sqrt(z**2 + y**2 + x**2) <= 55
    rng = np.random.default_rng(0)
    t1c = (0.3 + 0.1 * rng.standard_normal((N, N, N))).astype(np.float32)
    left = (z / 8) ** 2 + (y / 18) ** 2 + ((x + 7) / 5) ** 2 <= 1
    right = (z / 8) ** 2 + (y / 18) ** 2 + ((x - 7) / 5) ** 2 <= 1
    ventricles = left | right
    t1c[ventricles] = -2.5
    # A sulcus: a one-voxel plate of CSF reaching in from the surface, apart from the ventricles.
    sulcus = brain & (np.abs(x - 30) < 0.5)
    t1c[sulcus] = -2.5
    if dark_ball_mm:
        t1c[np.sqrt(z**2 + y**2 + x**2) <= dark_ball_mm] = -2.5
    t1c[~brain] = 0.0
    return _image(t1c), _image(brain.astype(np.uint8)), ventricles, sulcus


def test_finds_ventricles_and_ignores_a_thin_sulcus():
    image, brain, truth, sulcus = _phantom()
    mask_img, qc = anatomy.segment_ventricles(image, brain)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    assert qc.ok
    dice = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
    assert dice > 0.85
    assert (mask & sulcus).sum() == 0


def test_rejects_an_implausibly_large_mask():
    image, brain, _, _ = _phantom(dark_ball_mm=36.0)
    _, qc = anatomy.segment_ventricles(image, brain)
    assert not qc.ok
    assert "large" in qc.note


def test_a_lesion_excludes_nearby_voxels():
    image, brain, truth, _ = _phantom()
    lesion = np.zeros((N, N, N), np.uint8)
    lesion[55:65, 55:65, 70:80] = 1     # beside the right ventricle
    mask_img, _ = anatomy.segment_ventricles(image, brain, _image(lesion))
    mask = sitk.GetArrayFromImage(mask_img) > 0
    near = sitk.GetArrayFromImage(sitk.BinaryDilate(_image(lesion), [5, 5, 5])) > 0
    assert not (mask & near).any()


def test_original_rule_still_available_for_comparison():
    image, brain, _, _ = _phantom()
    _, qc = anatomy.segment_ventricles_otsu(image, brain)
    assert qc.volume_mm3 >= 0
