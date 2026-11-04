from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from gbmc import export


def _reference(spacing=(1.0, 1.0, 1.0), origin=(10.0, -20.0, 5.0)):
    img = sitk.GetImageFromArray(np.zeros((30, 30, 30), np.float32))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def test_index_to_physical_matches_itk():
    ref = _reference(spacing=(0.9, 1.1, 1.3))
    m = export.index_to_physical(ref)
    zyx = np.array([4.0, 7.0, 11.0, 1.0])
    expected = ref.TransformContinuousIndexToPhysicalPoint((11.0, 7.0, 4.0))
    np.testing.assert_allclose((m @ zyx)[:3], expected)


def test_surface_respects_budget_and_frame():
    z, y, x = np.indices((30, 30, 30))
    ball = (z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 8 ** 2
    ref = _reference()
    fine = export.surface(ball, ref, max_faces=100_000)
    coarse = export.surface(ball, ref, max_faces=1_000)
    assert len(coarse.faces) <= 1_000 < len(fine.faces)
    centre = ref.TransformContinuousIndexToPhysicalPoint((15.0, 15.0, 15.0))
    np.testing.assert_allclose(fine.vertices.mean(axis=0), centre, atol=0.5)
    assert export.surface(np.zeros((30, 30, 30), bool), ref, 1000) is None


def test_glb_and_slice_volume(tmp_path):
    z, y, x = np.indices((30, 30, 30))
    mesh = export.surface((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 36, _reference(), 5000)
    export.write_glb(mesh, tmp_path / "a.glb")
    assert (tmp_path / "a.glb").read_bytes()[:4] == b"glTF"
    export.slice_volume(_reference(), tmp_path / "s.nii.gz")
    out = sitk.ReadImage(str(tmp_path / "s.nii.gz"))
    assert out.GetSpacing() == (2.0, 2.0, 2.0) and out.GetPixelID() == sitk.sitkUInt8
