#!/usr/bin/env python
"""MU-Glioma-Post label transport, recurrence -> baseline (D19; registration section 8).

Same recipe as RHUH-GBM's ``21_`` (gbm-wm's tuned SyN, ``gbmc.registration``): driven by
T1ce, resection cavity masked out of the metric, round-trip error measured in the SNFH
region against ``registration.roundtrip_threshold_mm`` (3.54 mm). The baseline is the first
post-treatment MRI and the target is enhancing tissue + resection cavity on it
(``gbmc.mu.target``). Patients come from ``results/mu_eligibility/eligibility.json``.

Refuses until the MU file layout is recorded in ``config.yaml`` (after the registration
is public and the download is done), and on any label value the dataset does not document.
Per patient -> ``data/interim/mu/label_field_tuned/<patient>/`` (resumable). No model score.

    python scripts/26_mu_transport.py --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, mu
from gbmc.registration import PairInputs, transport_pair
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK = REPO_ROOT / "data" / "interim" / "mu" / "label_field_tuned"


def one(patient: str, master: int) -> dict:
    config.pin_itk_threads()
    out = WORK / patient
    if (out / "qc.json").is_file():
        return json.loads((out / "qc.json").read_text(encoding="utf-8")) | {"skipped": True}
    base, rec = mu.files(patient, "postop"), mu.files(patient, "recurrence")
    if not ({"T1ce", "seg"} <= set(base) and {"T1ce", "seg"} <= set(rec)):
        return {"patient": patient, "ok": False, "note": "missing T1ce or segmentation"}
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(base["seg"])))
    rec_seg = sitk.GetArrayFromImage(sitk.ReadImage(str(rec["seg"])))
    unknown = (set(np.unique(seg)) | set(np.unique(rec_seg))) - {0, *mu.EXPECTED_LABELS}
    if unknown:
        raise SystemExit(f"{patient}: label values {sorted(unknown)} are not documented")
    reference = sitk.ReadImage(str(base["T1ce"]))
    brain = sitk.GetArrayFromImage(reference) != 0          # images are skull-stripped
    target = mu.target(seg, brain)

    out.mkdir(parents=True, exist_ok=True)
    masks = {}
    for name, arr in (("brain", brain), ("target", target)):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(reference)
        masks[name] = out / f"{name}_mask.nii.gz"
        sitk.WriteImage(img, str(masks[name]), useCompression=True)
    rimg = sitk.ReadImage(str(rec["T1ce"]))
    rb = sitk.GetImageFromArray((sitk.GetArrayFromImage(rimg) != 0).astype(np.uint8))
    rb.CopyInformation(rimg)
    rec_brain = out / "recurrence_brain_mask.nii.gz"
    sitk.WriteImage(rb, str(rec_brain), useCompression=True)

    inputs = PairInputs(
        patient=patient, fixed=base["T1ce"], fixed_brain=masks["brain"],
        moving=rec["T1ce"], moving_brain=rec_brain,
        cavity=np.isin(seg, mu.label_values(("cavity",))),
        peritumoral=np.isin(seg, mu.label_values((mu.FLAIR_LABEL,))),
        moving_seg=rec["seg"])
    seeds = {k: stable_seed(patient, "mu-recipe", k, master_seed=master)
             for k in ("prealign", "forward", "backward")}
    qc = transport_pair(inputs, out, seeds,
                        float(config.get("registration.roundtrip_threshold_mm")), None)
    qc["target_rule"] = "enhancing + cavity on the baseline (gbmc.mu.target)"
    (out / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    mu.layout()
    master = int(config.get("registration.master_seed"))
    todo = json.loads(mu.ELIGIBILITY.read_text(encoding="utf-8"))["eligible"]
    print(f"MU transport: {len(todo)} eligible patients", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, p, master): p for p in todo}
        for fut in as_completed(futs):
            qc = fut.result()
            rt = (qc.get("roundtrip") or {}).get("median_mm", float("nan"))
            print(f"{futs[fut]}: {'ok' if qc.get('ok') else 'FAIL ' + qc.get('note', '')}; "
                  f"round-trip {rt:.2f} mm", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
