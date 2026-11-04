#!/usr/bin/env python
"""Features f1-f5 and the C0T distance for the 63 LUMIERE patients (plan Step 1.3).

Reads gbm-4d's frozen preprocessed baselines and transported labels; writes one
cache file per patient to ``data/interim/features/`` (not in git):

* ``<patient>.npz``: ``matrix`` (n_eval x 6, float32, columns gbmc.features.COLUMNS),
  ``truth`` (recurrence per evaluation voxel), the packed evaluation mask, and
  the treated-voxel counts that define matched volume.
* ``<patient>.json``: barrier and region QC for the same patient.

No candidate is scored here. Resumable: a patient with a cache is skipped.

    python scripts/01_lumiere_features.py
    python scripts/01_lumiere_features.py --patients Patient-001 --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, features, lumiere, metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "interim" / "features"


def extract(patient: str, out_dir: Path) -> dict:
    from gbmc import anatomy

    case = lumiere.load_baseline(patient)
    target = case.target
    if not target.any():
        return {"patient": patient, "note": "empty treatment target"}

    falx = anatomy.falx_barrier(case.reference, case.to_atlas, case.brain, case.t1)

    brain_img = sitk.GetImageFromArray(case.brain.astype(np.uint8))
    brain_img.CopyInformation(case.reference)
    lesion = sitk.GetImageFromArray((case.seg > 0).astype(np.uint8))
    lesion.CopyInformation(case.reference)
    vent_img, vent_qc = anatomy.segment_ventricles(case.reference, brain_img, lesion)
    vent = sitk.GetArrayFromImage(vent_img) > 0
    if not vent_qc.ok:
        vent = np.zeros_like(vent)

    depth = float(config.get("tournament.ventricle_depth_mm"))
    vols = features.build(case.reference, case.brain, target, case.flair_abnormal,
                          falx, vent, depth)
    evaluate = vols.evaluate
    truth = lumiere.load_recurrence(patient)[evaluate]

    c0t = vols.volumes["c0t"][evaluate]
    matched = {f"margin{int(r)}": int((c0t <= float(r)).sum())
               for r in config.get("tournament.margin_mm")}
    matched |= {f"volume{int(v)}ml": metrics.voxels_for_volume(float(v), case.spacing_zyx)
                for v in config.get("tournament.fixed_volume_ml")}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{patient}.npz",
             matrix=vols.matrix(), truth=truth,
             evaluate_packed=np.packbits(evaluate.ravel()),
             evaluate_shape=np.array(evaluate.shape),
             matched_labels=np.array(list(matched)),
             matched_voxels=np.array(list(matched.values())))

    voxel_ml = float(np.prod(case.spacing_zyx)) / 1000.0
    qc = {
        "patient": patient,
        "baseline": case.baseline,
        "n_eval_voxels": int(evaluate.sum()),
        "n_positive": int(truth.sum()),
        "evaluable": bool(truth.any()),
        "target_ml": round(float(target.sum()) * voxel_ml, 2),
        "flair_abnormal_ml": round(float(case.flair_abnormal.sum()) * voxel_ml, 2),
        "falx": anatomy.barrier_report(falx, case.brain, case.spacing_zyx),
        "ventricles": vent_qc.as_dict() | {"used": bool(vent.any())},
        "matched_voxels": matched,
    }
    (out_dir / f"{patient}.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    return qc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()

    config.pin_itk_threads()
    patients = args.patients or lumiere.cohort()
    for patient in patients:
        if (args.out / f"{patient}.npz").is_file() and not args.overwrite:
            print(f"{patient}: cached, skipped")
            continue
        start = time.time()
        qc = extract(patient, args.out)
        summary = qc.get("note") or (f"eval {qc['n_eval_voxels']}, positives {qc['n_positive']}, "
                                     f"ventricles used {qc['ventricles']['used']}")
        print(f"{patient}: {summary} ({time.time() - start:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
