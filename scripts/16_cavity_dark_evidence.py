#!/usr/bin/env python
"""Evidence for the cavity darkness threshold (D17 item 2): T1c intensities on LUMIERE.

On 12 LUMIERE baselines drawn with the master seed, records the z-scored T1c (z in the brain
mask, from preprocessing) inside the V4 ventricle masks (pure fluid), inside the baseline
necrotic/non-enhancing label, and in tissue outside both. Reads baseline images and baseline
labels only; no outcome label.

Writes ``results/cavity_dark_evidence/evidence.json`` (clean tree required).

    python scripts/16_cavity_dark_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import anatomy, config, lumiere
from gbmc.provenance import require_clean_tree, run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "cavity_dark_evidence"
N_PATIENTS = 12


def main() -> int:
    require_clean_tree()
    config.pin_itk_threads()
    master = int(config.get("seeds.master"))
    rng = np.random.default_rng(master)
    patients = sorted(rng.choice(lumiere.cohort(), N_PATIENTS, replace=False).tolist())
    rows = []
    for p in patients:
        case = lumiere.load_baseline(p)
        z, lesion = case.ct1, case.seg > 0
        brain = sitk.GetImageFromArray(case.brain.astype(np.uint8))
        brain.CopyInformation(case.reference)
        les = sitk.GetImageFromArray(lesion.astype(np.uint8))
        les.CopyInformation(case.reference)
        vent_img, _ = anatomy.segment_ventricles(case.reference, brain, les)
        vent = sitk.GetArrayFromImage(vent_img) > 0
        cavity = case.seg == lumiere.NECROTIC_NONENHANCING
        tissue = case.brain & ~lesion & ~vent
        rows.append({"patient": p,
                     "ventricle_z_median": round(float(np.median(z[vent])), 2),
                     "cavity_label_z_median": (round(float(np.median(z[cavity])), 2)
                                               if cavity.any() else None),
                     "tissue_z_median": round(float(np.median(z[tissue])), 2),
                     "tissue_share_below_minus1": round(float((z[tissue] < -1.0).mean()), 3),
                     "ventricle_share_below_minus1": round(float((z[vent] < -1.0).mean()), 3)})
        print(rows[-1], flush=True)
    summary = {k: round(float(np.median([r[k] for r in rows])), 2)
               for k in ("ventricle_z_median", "tissue_z_median")}
    summary["midpoint"] = round((summary["ventricle_z_median"] + summary["tissue_z_median"]) / 2, 2)
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(master_seed=master, n_patients=N_PATIENTS)
        (OUT / "evidence.json").write_text(json.dumps(
            {"labels_read": "baseline only; no outcome", "summary": summary, "rows": rows},
            indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
