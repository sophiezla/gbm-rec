"""Label transport with gbm-wm's tuned SyN recipe (plan Step 3.3, primary recipe).

``register_syn`` is copied from gbm-wm ``src/gbmwm/registration.py`` (gbm-wm is
not a dependency). Same algorithm as gbm-4d's frozen recipe: a Gaussian-smoothed
dense displacement field driven by ANTS neighbourhood cross-correlation. The
optimiser settings in :data:`TUNED_SYN` are gbm-wm's D11 lock. On synthetic
deformations they recovered to a 1.39 mm residual, against 2.13 mm for gbm-4d's
recipe (plan Step 3.3).

:func:`transport_pair` is the transport of one baseline/follow-up pair: rigid
pre-alignment, SyN forward and backward with the cavity masked out of the
metric, round-trip error in the peritumoral region, Jacobian QC, and a
nearest-neighbour warp of the follow-up segmentation into baseline space. It
follows gbm-wm ``12_mugp_transport.py:transport_one`` step for step, reusing
gbm-4d's frozen helpers for everything except the SyN call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

#: gbm-wm D11, locked 2026-09-19 on synthetic ground truth.
TUNED_SYN = dict(estimate="never", learning_rate=0.25, iterations=100, update_variance=3.0,
                 total_variance=0.1, convergence_window=100)


def coarsen(image: sitk.Image, spacing_mm: float, interpolator=sitk.sitkLinear) -> sitk.Image:
    size = [int(round(s * sp / spacing_mm))
            for s, sp in zip(image.GetSize(), image.GetSpacing(), strict=True)]
    return sitk.Resample(image, size, sitk.Transform(3, sitk.sitkIdentity), interpolator,
                         image.GetOrigin(), [spacing_mm] * 3, image.GetDirection(), 0.0,
                         image.GetPixelID())


def register_syn(fixed: sitk.Image, moving: sitk.Image, seed: int,
                 fixed_mask: sitk.Image | None = None, moving_mask: sitk.Image | None = None,
                 iterations: int = 120, learning_rate: float = 1.0, estimate: str = "each",
                 update_variance: float = 1.0, total_variance: float = 2.0, cc_radius: int = 4,
                 registration_spacing_mm: float = 2.0, convergence_window: int = 30,
                 convergence_minimum: float = 1e-9, shrink_factors=(2, 1),
                 smoothing_sigmas=(1, 0)) -> tuple[sitk.Transform, dict]:
    nn = sitk.sitkNearestNeighbor
    if fixed_mask is not None:
        fixed_mask = coarsen(fixed_mask, registration_spacing_mm, nn)
    if moving_mask is not None:
        moving_mask = coarsen(moving_mask, registration_spacing_mm, nn)
    fixed = coarsen(fixed, registration_spacing_mm)
    moving = coarsen(moving, registration_spacing_mm)

    field = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
    field.CopyInformation(fixed)
    initial = sitk.DisplacementFieldTransform(field)
    initial.SetSmoothingGaussianOnUpdate(varianceForUpdateField=update_variance,
                                         varianceForTotalField=total_variance)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsANTSNeighborhoodCorrelation(cc_radius)
    reg.SetMetricSamplingStrategy(reg.NONE)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8))
    if moving_mask is not None:
        reg.SetMetricMovingMask(sitk.Cast(moving_mask > 0, sitk.sitkUInt8))
    reg.SetInterpolator(sitk.sitkLinear)
    mode = {"each": reg.EachIteration, "once": reg.Once, "never": reg.Never}[estimate]
    reg.SetOptimizerAsGradientDescent(learningRate=learning_rate, numberOfIterations=iterations,
                                      estimateLearningRate=mode,
                                      convergenceMinimumValue=convergence_minimum,
                                      convergenceWindowSize=convergence_window)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=True)
    reg.SetShrinkFactorsPerLevel(list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.Execute(fixed, moving)
    metric = float(reg.GetMetricValue())
    return initial, {
        "final_metric": metric,
        "metric_valid": bool(np.isfinite(metric)) and abs(metric) < 1e300,
        "stop_condition": reg.GetOptimizerStopConditionDescription(),
        "n_iterations": int(reg.GetOptimizerIteration()),
        "settings": {"iterations": iterations, "learning_rate": learning_rate,
                     "estimate": estimate, "update_variance": update_variance,
                     "total_variance": total_variance, "cc_radius": cc_radius,
                     "registration_spacing_mm": registration_spacing_mm,
                     "convergence_window": convergence_window,
                     "shrink_factors": list(shrink_factors),
                     "smoothing_sigmas": list(smoothing_sigmas)},
        "seed": int(seed)}


@dataclass
class PairInputs:
    """One baseline/follow-up pair, already preprocessed onto their own grids."""

    patient: str
    fixed: Path              # baseline T1c (z-scored)
    fixed_brain: Path
    moving: Path             # follow-up T1c
    moving_brain: Path
    cavity: np.ndarray | None        # baseline cavity on the fixed grid; kept out of the metric
    peritumoral: np.ndarray | None   # baseline oedema mask on the fixed grid, for the round trip
    moving_seg: Path | None          # follow-up segmentation to warp (None: registration only)


def transport_pair(inputs: PairInputs, out_dir: Path, seeds: dict[str, int],
                   roundtrip_threshold_mm: float | None, recipe: dict | None = None) -> dict:
    """Register follow-up -> baseline and warp the follow-up labels. Returns a QC record.

    ``seeds`` holds ``prealign``, ``forward`` and ``backward``. ``recipe=None``
    means :data:`TUNED_SYN`; ``{}`` uses the original recipe's defaults (the sensitivity arm).
    """
    from gbmc import imaging, transport

    recipe = TUNED_SYN if recipe is None else recipe
    rec: dict = {"patient": inputs.patient, "ok": False, "note": "", "recipe": recipe}
    fixed = sitk.ReadImage(str(inputs.fixed), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(inputs.moving), sitk.sitkFloat32)
    fixed_brain = sitk.ReadImage(str(inputs.fixed_brain), sitk.sitkUInt8)
    moving_brain = sitk.ReadImage(str(inputs.moving_brain), sitk.sitkUInt8)

    def as_image(mask: np.ndarray | None) -> sitk.Image | None:
        if mask is None:
            return None
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        img.CopyInformation(fixed)
        return img

    fixed_metric = transport.metric_mask(fixed_brain, as_image(inputs.cavity))
    moving_metric = transport.metric_mask(moving_brain, None)
    rigid, rigid_qc = imaging.register_rigid(fixed, moving, seeds["prealign"])
    # The registration transform maps fixed points into moving space, which is what
    # Resample wants (deviation V3: gbm-wm and gbm-4d used the inverse here).
    moving_aligned = sitk.Resample(moving, fixed, rigid, sitk.sitkLinear, 0.0,
                                   moving.GetPixelID())
    moving_metric_aligned = sitk.Resample(moving_metric, fixed, rigid,
                                          sitk.sitkNearestNeighbor, 0, moving_metric.GetPixelID())
    deform_fwd, fwd_info = register_syn(fixed, moving_aligned, seeds["forward"], fixed_metric,
                                        moving_metric_aligned, **recipe)
    deform_bwd, bwd_info = register_syn(moving_aligned, fixed, seeds["backward"],
                                        moving_metric_aligned, fixed_metric, **recipe)
    rec["prealign"] = asdict(rigid_qc)
    rec["forward"], rec["backward"] = fwd_info, bwd_info
    out_dir.mkdir(parents=True, exist_ok=True)
    invalid = [n for n, i in (("forward", fwd_info), ("backward", bwd_info))
               if not i.get("metric_valid", True)]
    if invalid:
        # gbm-4d D11: a metric that was never evaluated returns an identity transform
        # that passes every QC below. Fail loudly and purge stale outputs.
        rec["note"] = f"registration metric never evaluated ({', '.join(invalid)})"
        for stale in ("recurrence_labels.nii.gz", "forward_transform.h5", "backward_transform.h5"):
            (out_dir / stale).unlink(missing_ok=True)
        (out_dir / "qc.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec

    forward = sitk.CompositeTransform([rigid, deform_fwd])
    backward = sitk.CompositeTransform([deform_bwd, rigid.GetInverse()])
    forward.FlattenTransform()
    backward.FlattenTransform()
    peritumoral = fixed_metric
    if inputs.peritumoral is not None and int(inputs.peritumoral.sum()) > 100:
        peritumoral = sitk.And(sitk.Cast(fixed_brain > 0, sitk.sitkUInt8),
                               as_image(inputs.peritumoral))
    rt = transport.round_trip_error(deform_fwd, deform_bwd, fixed, peritumoral,
                                    threshold_mm=roundtrip_threshold_mm)
    sitk.WriteTransform(forward, str(out_dir / "forward_transform.h5"))
    sitk.WriteTransform(backward, str(out_dir / "backward_transform.h5"))
    label_written = False
    if inputs.moving_seg is not None:
        seg = sitk.ReadImage(str(inputs.moving_seg), sitk.sitkUInt8)
        warped = transport.warp_label(seg, fixed, forward)
        sitk.WriteImage(warped, str(out_dir / "recurrence_labels.nii.gz"), useCompression=True)
        label_written = True
    rec.update(ok=True, roundtrip=rt.as_dict(),
               jacobian=transport.jacobian_qc(deform_fwd, fixed, peritumoral),
               label_written=label_written)
    (out_dir / "qc.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec
