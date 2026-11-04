"""Anatomy derived from each scan: falx, ventricles, surface distance, geodesic distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

FALX_HALF_THICKNESS_MM = 2.0
#: T1 z-score above this counts as white matter, which keeps the corpus callosum open.
WHITE_MATTER_Z = 0.0
UNREACHABLE_MM = 1.0e6

VENTRICLE_EROSION_MM = 8.0    # removes sulcal CSF, leaves the deep ventricles
LESION_MARGIN_MM = 5.0        # necrosis and cavity are as dark as CSF on T1c
CENTRALITY_MM = 55.0          # further than this from the deep-brain centre is not ventricle
MIN_COMPONENT_MM3 = 1500.0


@dataclass
class VentricleQC:
    volume_mm3: float
    n_components: int
    threshold: float
    ok: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {"ventricle_volume_mm3": round(self.volume_mm3, 1),
                "n_components": self.n_components, "csf_threshold": round(self.threshold, 4),
                "segmentation_ok": self.ok, "note": self.note}


def falx_barrier(reference: sitk.Image, to_atlas: sitk.Transform, brain: np.ndarray,
                 t1_z: np.ndarray, half_thickness_mm: float = FALX_HALF_THICKNESS_MM,
                 white_matter_z: float = WHITE_MATTER_Z) -> np.ndarray:
    """Brain voxels within ``half_thickness_mm`` of the atlas midline and not white matter."""
    voxels = np.argwhere(brain)
    barrier = np.zeros(brain.shape, dtype=bool)
    if voxels.size == 0:
        return barrier
    axes = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    scale = np.asarray(reference.GetSpacing(), dtype=float)
    points = (voxels[:, ::-1].astype(float) @ (axes * scale).T
              + np.asarray(reference.GetOrigin(), dtype=float))
    atlas_x = _atlas_x(to_atlas, points)
    near = np.abs(atlas_x) <= float(half_thickness_mm)
    barrier[tuple(voxels[near].T)] = True
    return barrier & (t1_z < float(white_matter_z))


def _atlas_x(transform: sitk.Transform, points: np.ndarray) -> np.ndarray:
    """Atlas x for many points, as one matrix product when the transform is affine."""
    zero = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    row = np.array([transform.TransformPoint(tuple(unit))[0] - zero[0] for unit in np.eye(3)])
    checks = np.array([[13.0, -7.0, 41.0], [-60.0, 22.0, -18.0], [5.5, 90.0, 3.25]])
    linear = all(np.isclose(row @ p + zero[0], transform.TransformPoint(tuple(p))[0], atol=1e-4)
                 for p in checks)
    if linear:
        return (points @ row + zero[0]).astype(np.float32)
    return np.array([transform.TransformPoint(tuple(p))[0] for p in points], dtype=np.float32)


def barrier_report(barrier: np.ndarray, brain: np.ndarray, spacing_mm) -> dict:
    """An empty barrier, or one swallowing much of the brain, is a failed derivation."""
    count = int(barrier.sum())
    return {"barrier_voxels": count,
            "barrier_volume_ml": count * float(np.prod(spacing_mm)) / 1000.0,
            "fraction_of_brain": float(count / max(int(brain.sum()), 1)),
            "empty": count == 0}


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's split between the CSF and parenchyma intensity modes."""
    counts, edges = np.histogram(values, bins=bins)
    if counts.sum() == 0:
        return float("nan")
    centres = (edges[:-1] + edges[1:]) / 2.0
    share = np.cumsum(counts) / counts.sum()
    level = np.cumsum(counts * centres) / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (level[-1] * share - level) ** 2 / (share * (1.0 - share))
    spread[~np.isfinite(spread)] = -1.0
    return float(centres[int(np.argmax(spread))])


def segment_ventricles(image: sitk.Image, brain: sitk.Image, lesion: sitk.Image | None = None
                       ) -> tuple[sitk.Image, VentricleQC]:
    """Lateral ventricles from T1c: dark, deep, central, large, and away from the lesion."""
    def radius(mm: float) -> list[int]:
        return [max(1, int(round(mm / s))) for s in image.GetSpacing()]

    def as_mask(array: np.ndarray) -> sitk.Image:
        mask = sitk.GetImageFromArray(array.astype(np.uint8))
        mask.CopyInformation(image)
        return mask

    def failed(threshold: float, note: str) -> tuple[sitk.Image, VentricleQC]:
        empty = np.zeros(sitk.GetArrayFromImage(image).shape, np.uint8)
        return as_mask(empty), VentricleQC(0.0, 0, threshold, False, note)

    deep = sitk.BinaryErode(sitk.Cast(brain > 0, sitk.sitkUInt8), radius(VENTRICLE_EROSION_MM))
    if lesion is not None and sitk.GetArrayFromImage(lesion > 0).any():
        grown = sitk.BinaryDilate(sitk.Cast(lesion > 0, sitk.sitkUInt8), radius(LESION_MARGIN_MM))
        deep = sitk.And(deep, sitk.Cast(grown == 0, sitk.sitkUInt8))
    inside = sitk.GetArrayFromImage(deep).astype(bool)
    if not inside.any():
        return failed(float("nan"), "deep brain mask empty")

    intensity = sitk.GetArrayFromImage(image).astype(np.float64)
    threshold = otsu_threshold(intensity[inside])
    dark = as_mask(inside & (intensity <= threshold))
    opened = sitk.BinaryMorphologicalOpening(dark, radius(1.0))
    voxel_mm3 = float(np.prod(image.GetSpacing()))
    min_voxels = int(round(MIN_COMPONENT_MM3 / voxel_mm3))
    parts = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(opened),
                                                         minimumObjectSize=min_voxels))
    if parts.max() == 0:
        return failed(threshold, "no component survived the size filter")

    # Components are numbered largest first. Keep any at least a quarter of the largest
    # (the other ventricle, split by the septum) whose centre is central enough.
    sizes = np.bincount(parts.ravel())[1:]
    centre = np.argwhere(inside).mean(axis=0)
    scale = np.array(image.GetSpacing()[::-1])
    kept = [int(label) for label in 1 + np.flatnonzero(sizes >= 0.25 * sizes[0])
            if np.linalg.norm((np.argwhere(parts == label).mean(axis=0) - centre) * scale)
            <= CENTRALITY_MM]
    if not kept:
        return failed(threshold, "no central component; segmentation not trusted")
    mask = sitk.BinaryMorphologicalClosing(as_mask(np.isin(parts, kept)), radius(2.0))
    volume = float(sitk.GetArrayFromImage(mask).sum()) * voxel_mm3
    ok = volume >= MIN_COMPONENT_MM3
    return mask, VentricleQC(volume, len(kept), threshold, ok,
                             "" if ok else "ventricle volume implausibly small")


def surface_distance_mm(reference: sitk.Image, test: sitk.Image) -> dict:
    """Symmetric surface distance and Dice for two binary masks on the same grid."""
    for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(reference, getter)(), getattr(test, getter)(), atol=1e-4):
            raise ValueError(f"masks do not share a grid ({getter[3:].lower()} differs)")
    a, b = sitk.Cast(reference > 0, sitk.sitkUInt8), sitk.Cast(test > 0, sitk.sitkUInt8)
    if not (sitk.GetArrayFromImage(a).any() and sitk.GetArrayFromImage(b).any()):
        return {"mean_mm": None, "p95_mm": None, "max_mm": None, "dice": None}

    def one_way(source: sitk.Image, target: sitk.Image) -> np.ndarray:
        to_target = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False,
                                                          useImageSpacing=True))
        edge = sitk.GetArrayFromImage(sitk.LabelContour(source, fullyConnected=True)) > 0
        return sitk.GetArrayFromImage(to_target)[edge].astype(float)

    both = np.concatenate([one_way(a, b), one_way(b, a)])
    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(a, b)
    return {"mean_mm": float(both.mean()), "p95_mm": float(np.percentile(both, 95)),
            "max_mm": float(both.max()), "dice": float(overlap.GetDiceCoefficient())}


def geodesic_distance(target: sitk.Image, allowed: sitk.Image) -> sitk.Image:
    """Fast-marching distance (mm) from every target voxel, moving only inside ``allowed``."""
    seeds = np.argwhere(sitk.GetArrayFromImage(target) > 0)
    if seeds.size == 0:
        raise ValueError("empty treatment target: nothing to expand from")
    speed = sitk.Cast(sitk.Or(sitk.Cast(allowed > 0, sitk.sitkUInt8),
                              sitk.Cast(target > 0, sitk.sitkUInt8)), sitk.sitkFloat32)
    march = sitk.FastMarchingImageFilter()
    march.SetNormalizationFactor(1.0)
    march.SetTrialPoints([tuple(int(v) for v in seed[::-1]) for seed in seeds])
    march.SetStoppingValue(UNREACHABLE_MM)
    values = sitk.GetArrayFromImage(sitk.Cast(march.Execute(speed), sitk.sitkFloat32))
    values[~np.isfinite(values) | (values >= UNREACHABLE_MM)] = UNREACHABLE_MM
    out = sitk.GetImageFromArray(values)
    out.CopyInformation(target)
    return out
