#!/usr/bin/env python
"""Where do the voxels at the f5 cap sit? Anatomy only; scores nothing.

Splits every evaluation voxel at the 300 mm cap into falx, ventricle, or neither, and
describes the "neither" group (tissue the geodesic path never reached). Recurrence labels
are never read, so the answer cannot depend on an outcome.

    python scripts/11_f5_cap_survey.py --patients Patient-002   # prints only
    python scripts/11_f5_cap_survey.py                          # all 63, writes results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from gbmc import anatomy, config, features, lumiere
from gbmc.provenance import require_clean_tree, run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "interim" / "features"
OUT = REPO_ROOT / "results" / "f5_cap_survey"
F5 = features.COLUMNS.index("f5")


def _signed_atlas_x(case, mask: np.ndarray) -> np.ndarray:
    """Atlas x (mm) of each voxel in ``mask``; the sign gives the hemisphere."""
    voxels = np.argwhere(mask)
    axes = np.asarray(case.reference.GetDirection(), dtype=float).reshape(3, 3)
    scale = np.asarray(case.reference.GetSpacing(), dtype=float)
    points = (voxels[:, ::-1].astype(float) @ (axes * scale).T
              + np.asarray(case.reference.GetOrigin(), dtype=float))
    return anatomy._atlas_x(case.to_atlas, points)


def survey(patient: str) -> dict:
    case = lumiere.load_baseline(patient)
    target = case.target
    with np.load(CACHE / f"{patient}.npz") as cached:   # never touches "truth"
        case.brain = features.brain_from_cache(cached["evaluate_packed"],
                                               cached["evaluate_shape"], case.brain, target)
        f5_values = cached["matrix"][:, F5]
    evaluate = case.brain & ~target
    f5 = np.zeros(evaluate.shape, dtype=np.float32)
    f5[evaluate] = f5_values

    falx = anatomy.falx_barrier(case.reference, case.to_atlas, case.brain, case.t1)
    brain_img = sitk.GetImageFromArray(case.brain.astype(np.uint8))
    brain_img.CopyInformation(case.reference)
    lesion = sitk.GetImageFromArray((case.seg > 0).astype(np.uint8))
    lesion.CopyInformation(case.reference)
    vent_img, vent_qc = anatomy.segment_ventricles(case.reference, brain_img, lesion)
    vent = sitk.GetArrayFromImage(vent_img) > 0
    if not vent_qc.ok:
        vent = np.zeros_like(vent)

    capped = evaluate & (f5 >= features.MAX_PATH_MM)
    in_falx = capped & falx
    in_vent = capped & vent & ~falx
    unreached = capped & ~falx & ~vent
    labels, n_parts = ndimage.label(unreached)
    sizes = ndimage.sum_labels(unreached, labels, range(1, n_parts + 1)) if n_parts else []

    voxel_ml = float(np.prod(case.spacing_zyx)) / 1000.0
    target_x = float(np.median(_signed_atlas_x(case, target)))
    opposite = 0
    if unreached.any():
        opposite = int((np.sign(_signed_atlas_x(case, unreached)) == -np.sign(target_x)).sum())
    reached = f5[evaluate & (f5 < features.MAX_PATH_MM)]
    n = int(evaluate.sum())
    return {
        "patient": patient,
        "evaluation_voxels": n,
        "evaluation_ml": round(n * voxel_ml, 1),
        "capped_share": float(capped.sum() / n),
        "falx_share": float(in_falx.sum() / n),
        "ventricle_share": float(in_vent.sum() / n),
        "unreached_share": float(unreached.sum() / n),
        "unreached_ml": round(float(unreached.sum()) * voxel_ml, 3),
        "unreached_parts": int(n_parts),
        "largest_unreached_ml": round(float(max(sizes, default=0)) * voxel_ml, 3),
        "unreached_opposite_hemisphere_share": float(opposite / max(int(unreached.sum()), 1)),
        "target_median_atlas_x_mm": round(target_x, 1),
        "longest_reached_path_mm": round(float(reached.max()), 1),
        "ventricle_ml": round(vent_qc.volume_mm3 / 1000.0, 1),
        "ventricle_share_of_brain": float(vent.sum() / max(int(case.brain.sum()), 1)),
        "ventricle_parts": vent_qc.n_components,
        "csf_threshold": round(vent_qc.threshold, 4),
        "ventricle_segmentation_ok": bool(vent_qc.ok),
    }


def summarise(rows: list[dict]) -> dict:
    def spread(key: str) -> dict:
        values = np.array([row[key] for row in rows], dtype=float)
        return {"median": float(np.median(values)), "min": float(values.min()),
                "max": float(values.max())}

    keys = ("capped_share", "falx_share", "ventricle_share", "unreached_share", "unreached_ml",
            "largest_unreached_ml", "longest_reached_path_mm", "ventricle_ml")
    return {"patients": len(rows)} | {key: spread(key) for key in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None, help="print only; writes nothing")
    args = ap.parse_args()
    config.pin_itk_threads()

    if args.patients:
        for patient in args.patients:
            print(json.dumps(survey(patient), indent=2))
        return 0

    require_clean_tree()
    OUT.mkdir(parents=True, exist_ok=True)
    with run_manifest(OUT):
        rows = []
        for patient in lumiere.cohort():
            rows.append(survey(patient))
            print(patient, f"{rows[-1]['unreached_ml']} ml unreached", flush=True)
        (OUT / "survey.json").write_text(
            json.dumps({"summary": summarise(rows), "patients": rows}, indent=2) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
