from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from gbmc import features

SHAPE = (40, 40, 40)


def _reference() -> sitk.Image:
    img = sitk.GetImageFromArray(np.zeros(SHAPE, dtype=np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _ball(centre, radius):
    z, y, x = np.indices(SHAPE)
    return ((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) <= radius ** 2


def _case(falx=None, ventricles=None, flair=None):
    brain = _ball((20, 20, 20), 18)
    target = _ball((20, 20, 12), 3)
    flair = _ball((20, 20, 12), 6) & ~target if flair is None else flair
    falx = np.zeros(SHAPE, bool) if falx is None else falx
    ventricles = np.zeros(SHAPE, bool) if ventricles is None else ventricles
    return brain, target, features.build(_reference(), brain, target, flair, falx,
                                         ventricles, 5.0)


def test_f1_is_signed_and_nonnegative_in_evaluation_region():
    brain, target, vols = _case()
    f1 = vols.volumes["f1"]
    assert (f1[target] <= 0).all()
    assert (f1[vols.evaluate] >= 0).all()
    assert not (vols.evaluate & target).any()


def test_f4_is_interaction_and_absent_flair_is_capped():
    _, _, vols = _case(flair=np.zeros(SHAPE, bool))
    assert (vols.volumes["f3"] == features.MAX_PATH_MM).all()
    assert (vols.volumes["f4"] == 0).all()
    _, _, vols = _case()
    np.testing.assert_allclose(vols.volumes["f4"], vols.volumes["f1"] * vols.volumes["f2"])


def test_falx_closes_the_other_hemisphere():
    falx = np.zeros(SHAPE, bool)
    falx[:, :, 20] = True
    brain, _, vols = _case(falx=falx)
    far_side = brain & (np.indices(SHAPE)[2] > 21)
    assert (vols.volumes["f5"][far_side] == features.MAX_PATH_MM).all()
    assert (vols.volumes["c0t"][far_side] == features.MAX_PATH_MM).all()
    # Euclidean f1 ignores barriers.
    assert (vols.volumes["f1"][far_side] < features.MAX_PATH_MM).all()


def test_ventricle_allowance_depth():
    tissue = np.zeros((1, 1, 20), np.float32)
    vent = np.zeros((1, 1, 20), bool)
    vent[0, 0, 5:20] = True
    tissue[vent] = features.MAX_PATH_MM
    out = features.ventricle_allowance(tissue, vent, (1.0, 1.0, 1.0), depth_mm=5.0)
    # voxel 5 is 1 mm below the wall at voxel 4 (distance 0 there)
    assert out[0, 0, 5] == 1.0
    assert out[0, 0, 9] == 5.0
    assert out[0, 0, 10] == features.MAX_PATH_MM
    closed = features.ventricle_allowance(tissue, vent, (1.0, 1.0, 1.0), depth_mm=0.0)
    assert (closed[vent] == features.MAX_PATH_MM).all()


def test_matrix_columns():
    _, _, vols = _case()
    m = vols.matrix()
    assert m.shape == (int(vols.evaluate.sum()), len(features.COLUMNS))
    assert m.dtype == np.float32
