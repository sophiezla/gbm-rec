"""The anatomy, imaging, transport and PDE modules on small synthetic volumes."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from gbmc import anatomy, imaging, pde, transport


def _head(size: int = 64) -> np.ndarray:
    noise = gaussian_filter(np.random.default_rng(0).normal(size=(size,) * 3), 3)
    grid = np.mgrid[:size, :size, :size] - size / 2
    ball = (grid**2).sum(axis=0) < (0.4 * size) ** 2
    return ((noise - noise.min()) / (noise.max() - noise.min()) * ball).astype(np.float32)


def test_rigid_transform_aligns_when_passed_to_resample_as_returned():
    # Regression (deviation V3): the inverse moves the image further away.
    head = _head()
    fixed = sitk.GetImageFromArray(head)
    moving = sitk.Resample(fixed, fixed, sitk.TranslationTransform(3, (5.0, -3.0, 2.0)),
                           sitk.sitkLinear, 0.0)

    def error(tx: sitk.Transform) -> float:
        out = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0))
        return float(np.sqrt(((out - head) ** 2)[head > 0].mean()))

    rigid, _ = imaging.register_rigid(fixed, moving, seed=1)
    unaligned = error(sitk.Transform(3, sitk.sitkIdentity))
    assert error(rigid) < 0.05 < unaligned < error(rigid.GetInverse())


def test_zscore_uses_only_the_mask():
    image = sitk.GetImageFromArray(np.array([[[1.0, 3.0, 100.0]]], dtype=np.float32))
    mask = sitk.GetImageFromArray(np.array([[[1, 1, 0]]], dtype=np.uint8))
    assert sitk.GetArrayFromImage(imaging.zscore(image, mask)).ravel().tolist() == [-1.0, 1.0, 0.0]


def test_geodesic_distance_goes_around_a_wall():
    allowed = np.ones((1, 21, 21), np.uint8)
    allowed[0, :18, 10] = 0
    target = np.zeros_like(allowed)
    target[0, 2, 2] = 1
    d = sitk.GetArrayFromImage(anatomy.geodesic_distance(sitk.GetImageFromArray(target),
                                                         sitk.GetImageFromArray(allowed)))
    assert d[0, 2, 2] == 0
    assert d[0, 2, 18] > 16 + 5


def test_falx_is_the_midline_outside_white_matter():
    brain = np.ones((4, 4, 21), bool)
    reference = sitk.GetImageFromArray(brain.astype(np.uint8))
    t1 = -np.ones(brain.shape)
    t1[:2] = 1.0
    to_midline = sitk.TranslationTransform(3, (-10.0, 0.0, 0.0))
    falx = anatomy.falx_barrier(reference, to_midline, brain, t1)
    assert np.flatnonzero(falx[3, 0]).tolist() == [8, 9, 10, 11, 12]
    assert not falx[:2].any()


def test_surface_distance_of_identical_masks_is_zero():
    mask = sitk.GetImageFromArray((_head(32) > 0).astype(np.uint8))
    out = anatomy.surface_distance_mm(mask, mask)
    assert out["dice"] == 1.0 and out["max_mm"] == 0.0


def test_warp_label_keeps_label_values():
    label = np.zeros((16, 16, 16), np.uint8)
    label[4:8, 4:8, 4:8], label[8:12, 8:12, 8:12] = 1, 3
    image = sitk.GetImageFromArray(label)
    warped = transport.warp_label(image, image, sitk.TranslationTransform(3, (0.5, 0.5, 0.5)))
    assert set(np.unique(sitk.GetArrayFromImage(warped))) <= {0, 1, 3}


def test_round_trip_of_inverse_pair_is_zero():
    image = sitk.GetImageFromArray(_head(32))
    forward = sitk.TranslationTransform(3, (3.0, 0.0, 0.0))
    backward = sitk.TranslationTransform(3, (-3.0, 0.0, 0.0))
    qc = transport.round_trip_error(forward, backward, image, sitk.Cast(image > 0, sitk.sitkUInt8),
                                    threshold_mm=1.0)
    assert qc.median_mm == pytest.approx(0.0, abs=1e-6) and qc.verdict == "pass"


def test_pde_conserves_mass_without_growth_and_respects_the_brain():
    brain = np.zeros((9, 9, 9), bool)
    brain[1:8, 1:8, 1:8] = True
    c = np.zeros(brain.shape)
    c[4, 4, 4] = 0.5
    d = np.where(brain, 0.5, 0.0)
    out = pde.advance(c, d, np.zeros(brain.shape), brain, pde.stable_dt(1.0, 0.5), 50)
    assert out.sum() == pytest.approx(c.sum())
    assert out[~brain].max() == 0.0


def test_pde_front_moves_at_the_fisher_speed():
    length = 400
    brain = np.ones(length, bool)
    c = np.zeros(length)
    c[:10] = 1.0
    d_value, rho_value = 0.5, 0.05
    dt = pde.stable_dt(1.0, d_value, ndim=1)

    def front(field: np.ndarray) -> int:
        return int(np.argmax(field < 0.5))

    start = pde.advance(c, np.full(length, d_value), np.full(length, rho_value), brain, dt,
                        int(300 / dt))
    end = pde.advance(start, np.full(length, d_value), np.full(length, rho_value), brain, dt,
                      int(300 / dt))
    speed = (front(end) - front(start)) / 300
    assert speed == pytest.approx(2 * np.sqrt(d_value * rho_value), rel=0.1)
