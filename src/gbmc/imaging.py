"""MRI preprocessing: threads, rigid alignment, N4 and the per-timepoint chain.

The chain's order matters: N4 before registration, so shading does not drive the
metric, and z-score last, so it is computed on the final grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc.rng import stable_seed

REFERENCE = "CT1"
MOVING = ("T1", "T2", "FLAIR")
MAX_CENTROID_OFFSET_MM = 100.0


@dataclass
class RigidQC:
    sequence: str
    final_metric: float
    stop_condition: str
    translation_mm: float
    rotation_deg: float
    n_iterations: int


@dataclass
class TimepointQC:
    patient: str
    timepoint: str
    native_spacing: dict[str, list[float]] = field(default_factory=dict)
    max_native_slice_mm: float = 0.0
    intersequence_offset_mm: dict[str, float] = field(default_factory=dict)
    registrations: list[dict] = field(default_factory=list)
    brain_voxels: int = 0
    ok: bool = True
    note: str = ""


def set_itk_threads(n_threads: int) -> int:
    """Pin ITK's thread count and return the old one. Call in every worker process."""
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(n_threads))
    return int(previous)


def brain_mask(image: sitk.Image) -> sitk.Image:
    """Non-zero voxels. LUMIERE is already stripped; Otsu would cut into necrosis and CSF."""
    return sitk.Cast(image > 0, sitk.sitkUInt8)


def register_rigid(fixed: sitk.Image, moving: sitk.Image, seed: int,
                   sampling: float = 0.05) -> tuple[sitk.Transform, RigidQC]:
    """Rigid alignment by Mattes mutual information, with an explicit sampling seed."""
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(sampling, seed=int(seed))
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(
        sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                          sitk.CenteredTransformInitializerFilter.GEOMETRY),
        inPlace=False)
    transform = method.Execute(fixed, moving)
    wrapped = (transform.GetNthTransform(0) if isinstance(transform, sitk.CompositeTransform)
               else sitk.CompositeTransform(transform).GetNthTransform(0))
    euler = sitk.Euler3DTransform(wrapped)
    angles = (euler.GetAngleX(), euler.GetAngleY(), euler.GetAngleZ())
    return transform, RigidQC(
        sequence="", final_metric=float(method.GetMetricValue()),
        stop_condition=method.GetOptimizerStopConditionDescription(),
        translation_mm=round(float(np.linalg.norm(euler.GetTranslation())), 3),
        rotation_deg=round(float(np.degrees(np.linalg.norm(angles))), 3),
        n_iterations=int(method.GetOptimizerIteration()))


def n4_correct(image: sitk.Image, mask: sitk.Image, shrink: int = 2, iterations: int = 50,
               levels: int = 4) -> sitk.Image:
    """N4 fitted on a shrunk grid, with the bias field divided out at full resolution."""
    factors = [shrink] * image.GetDimension()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([iterations] * levels)
    corrector.Execute(sitk.Shrink(image, factors), sitk.Shrink(mask, factors))
    bias = sitk.Exp(corrector.GetLogBiasFieldAsImage(image))
    return sitk.Mask(image / bias, mask, outsideValue=0.0)


def resample_isotropic(image: sitk.Image, spacing_mm: float = 1.0) -> sitk.Image:
    size = [int(round(n * s / spacing_mm))
            for n, s in zip(image.GetSize(), image.GetSpacing(), strict=True)]
    grid = sitk.Image(size, image.GetPixelID())
    grid.SetSpacing([spacing_mm] * 3)
    grid.SetOrigin(image.GetOrigin())
    grid.SetDirection(image.GetDirection())
    return sitk.Resample(image, grid, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0,
                         image.GetPixelID())


def zscore(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Z-score over in-mask voxels only; background would make it depend on field of view."""
    values = sitk.GetArrayFromImage(image)
    inside = sitk.GetArrayFromImage(mask).astype(bool)
    voxels = values[inside]
    if voxels.size == 0 or float(voxels.std()) == 0.0:
        raise ValueError("cannot z-score: empty mask or zero variance inside it")
    scored = np.zeros_like(values, dtype=np.float32)
    scored[inside] = (voxels - voxels.mean()) / voxels.std()
    out = sitk.GetImageFromArray(scored)
    out.CopyInformation(image)
    return out


def _centroid_mm(image: sitk.Image) -> np.ndarray | None:
    voxels = np.argwhere(sitk.GetArrayFromImage(image) > 0)
    if voxels.size == 0:
        return None
    xyz = voxels.mean(axis=0)[::-1].tolist()
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(xyz))


def preprocess_timepoint(native_dir: Path, patient: str, timepoint: str, master_seed: int,
                         spacing_mm: float = 1.0) -> tuple[dict[str, sitk.Image], TimepointQC]:
    """N4, rigid to T1c, isotropic grid, one brain mask, z-score. Returns images by sequence."""
    qc = TimepointQC(patient=patient, timepoint=timepoint)
    raw: dict[str, sitk.Image] = {}
    for sequence in (REFERENCE, *MOVING):
        path = native_dir / f"{sequence}.nii.gz"
        if not path.is_file():
            qc.ok, qc.note = False, f"missing {sequence}.nii.gz"
            return {}, qc
        raw[sequence] = sitk.ReadImage(str(path), sitk.sitkFloat32)
        qc.native_spacing[sequence] = [round(float(s), 4) for s in raw[sequence].GetSpacing()]
    qc.max_native_slice_mm = max(max(s) for s in qc.native_spacing.values())

    centre = _centroid_mm(raw[REFERENCE])
    centres = {s: _centroid_mm(raw[s]) for s in MOVING}
    if centre is None or any(c is None for c in centres.values()):
        qc.ok, qc.note = False, "a sequence contains no non-zero voxels"
        return {}, qc
    offsets = {s: float(np.linalg.norm(c - centre)) for s, c in centres.items()}
    qc.intersequence_offset_mm = {s: round(v, 2) for s, v in offsets.items()}
    if max(offsets.values()) > MAX_CENTROID_OFFSET_MM:
        qc.ok, qc.note = False, "sequences are too far apart: corrupt header origin"
        return {}, qc

    corrected = {s: n4_correct(img, brain_mask(img)) for s, img in raw.items()}
    grid = resample_isotropic(corrected[REFERENCE], spacing_mm)
    images = {REFERENCE: grid}
    for sequence in MOVING:
        seed = stable_seed(patient, timepoint, sequence, "rigid", master_seed=master_seed)
        transform, rigid = register_rigid(corrected[REFERENCE], corrected[sequence], seed)
        rigid.sequence = sequence
        qc.registrations.append(asdict(rigid))
        images[sequence] = sitk.Resample(corrected[sequence], grid, transform, sitk.sitkLinear,
                                         0.0, corrected[sequence].GetPixelID())
    mask = brain_mask(grid)
    qc.brain_voxels = int(sitk.GetArrayFromImage(mask).sum())
    if qc.brain_voxels == 0:
        qc.ok, qc.note = False, "empty brain mask after resampling"
        return images, qc
    return {**{s: zscore(i, mask) for s, i in images.items()}, "brain_mask": mask}, qc
