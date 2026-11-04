"""Tools for moving labels between scans and measuring how consistent the move is.

The round trip and the Jacobian measure consistency, not accuracy, so they can exclude a
registration but cannot certify one. No number from here is ever a TRE (plan section 13).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class RoundTrip:
    median_mm: float
    p95_mm: float
    max_mm: float
    n_voxels: int
    region: str
    threshold_mm: float | None = None

    @property
    def verdict(self) -> str:
        """``undetermined`` when no threshold is set or the region is under 100 voxels."""
        if self.threshold_mm is None or self.n_voxels < 100:
            return "undetermined"
        return "pass" if self.median_mm <= self.threshold_mm else "fail"

    def as_dict(self) -> dict:
        return asdict(self) | {"verdict": self.verdict}


def coarsen(image: sitk.Image, spacing_mm: float,
            interpolator: int = sitk.sitkLinear) -> sitk.Image:
    """Resample onto an isotropic grid of ``spacing_mm``."""
    if spacing_mm <= 0:
        return image
    size = [max(1, int(round(n * s / spacing_mm)))
            for n, s in zip(image.GetSize(), image.GetSpacing(), strict=True)]
    grid = sitk.Image(size, image.GetPixelID())
    grid.SetSpacing([spacing_mm] * 3)
    grid.SetOrigin(image.GetOrigin())
    grid.SetDirection(image.GetDirection())
    return sitk.Resample(image, grid, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0.0,
                         image.GetPixelID())


def metric_mask(brain: sitk.Image, cavity: sitk.Image | None, dilate_mm: float = 2.0) -> sitk.Image:
    """Brain minus a dilated cavity: the rim has no correspondence between scans."""
    inside = sitk.Cast(brain > 0, sitk.sitkUInt8)
    if cavity is None:
        return inside
    radius = max(1, int(round(dilate_mm / min(brain.GetSpacing()))))
    grown = sitk.BinaryDilate(sitk.Cast(cavity > 0, sitk.sitkUInt8), [radius] * 3)
    return sitk.Cast(sitk.And(inside, sitk.Not(sitk.Cast(grown > 0, sitk.sitkUInt8))),
                     sitk.sitkUInt8)


def warp_label(label: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """Nearest-neighbour only: linear interpolation would invent label values."""
    return sitk.Resample(label, reference, transform, sitk.sitkNearestNeighbor, 0,
                         label.GetPixelID())


def displacement_field(transform: sitk.Transform, reference: sitk.Image) -> sitk.Image:
    to_field = sitk.TransformToDisplacementFieldFilter()
    to_field.SetReferenceImage(reference)
    to_field.SetOutputPixelType(sitk.sitkVectorFloat64)
    return to_field.Execute(transform)


def jacobian_qc(transform: sitk.Transform, reference: sitk.Image, region: sitk.Image,
                sample_spacing_mm: float = 2.0) -> dict:
    """|J| inside ``region``: near 1 is volume-preserving, <= 0 means space folded."""
    grid = coarsen(reference, sample_spacing_mm)
    inside = sitk.GetArrayFromImage(
        coarsen(region, sample_spacing_mm, sitk.sitkNearestNeighbor)).astype(bool)
    if not inside.any():
        return {"n_voxels": 0, "verdict": "undetermined"}
    det = sitk.DisplacementFieldJacobianDeterminant(displacement_field(transform, grid))
    values = sitk.GetArrayFromImage(det)[inside]
    return {"median": float(np.median(values)), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "fraction_negative": float((values <= 0).mean()), "n_voxels": int(inside.sum())}


def round_trip_error(forward: sitk.Transform, backward: sitk.Transform, reference: sitk.Image,
                     region: sitk.Image, region_name: str = "peritumoral",
                     threshold_mm: float | None = None,
                     sample_spacing_mm: float = 2.0) -> RoundTrip:
    """Displacement left after applying forward then backward, inside ``region``."""
    grid = coarsen(reference, sample_spacing_mm)
    inside = sitk.GetArrayFromImage(
        coarsen(region, sample_spacing_mm, sitk.sitkNearestNeighbor)).astype(bool)
    if not inside.any():
        return RoundTrip(float("nan"), float("nan"), float("nan"), 0, region_name, threshold_mm)
    field = displacement_field(sitk.CompositeTransform([backward, forward]), grid)
    size = np.linalg.norm(sitk.GetArrayFromImage(field)[inside], axis=-1)
    return RoundTrip(float(np.median(size)), float(np.percentile(size, 95)), float(size.max()),
                     int(inside.sum()), region_name, threshold_mm)
