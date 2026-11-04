#!/usr/bin/env python
"""RHUH-GBM label transport, recurrence -> early post-op (plan Step 3.3).

Primary: gbm-wm's tuned SyN (``gbmc.registration.TUNED_SYN``). Sensitivity:
gbm-4d's original recipe (``--recipe gbm4d``), which made the LUMIERE labels.
Driven by T1ce, cavity masked from the metric, round-trip error in the
peritumoral region against the threshold fixed in Phase 2.

The cavity used for the metric mask, and the target, follow the Step 2.2 rule
as recorded after ``20_``'s cavity inspection (``config.yaml:rhuh.cavity_rule``:
"labelled:<value>", or "unlabelled", when the pre-op core is carried rigidly).

Per patient -> ``data/interim/rhuh/label_field_<recipe>/<patient>/`` (resumable).
No model score is computed.

    python scripts/21_rhuh_transport.py --workers 2
    python scripts/21_rhuh_transport.py --recipe gbm4d --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, rhuh
from gbmc.registration import PairInputs, transport_pair
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK = REPO_ROOT / "data" / "interim" / "rhuh"


def _mapping() -> dict[int, str]:
    inv = json.loads(
        (REPO_ROOT / "results" / "rhuh_inventory" / "inventory.json").read_text(encoding="utf-8")
    )
    return {int(k): v for k, v in inv["mapping"]["decision"]["mapping"].items()}


def one(patient: str, recipe_name: str, master: int, mapping: dict[int, str]) -> dict:
    from gbm4d.preprocess import use_deterministic_threads

    use_deterministic_threads(int(config.get("preprocessing.itk_threads")))
    out = WORK / f"label_field_{recipe_name}" / patient
    if (out / "qc.json").is_file():
        return json.loads((out / "qc.json").read_text(encoding="utf-8")) | {"skipped": True}
    post, rec = rhuh.files(patient, "postop"), rhuh.files(patient, "recurrence")
    postop = sitk.GetArrayFromImage(sitk.ReadImage(str(post["seg"])))
    brain_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(post["T1ce"]))) != 0
    cavity_label = rhuh.cavity_rule()
    if cavity_label is not None:
        cavity = postop == cavity_label
        target_core = None
    else:
        target_core = rhuh.carried_preop_core(
            patient, mapping, stable_seed(patient, "rhuh-core", master_seed=master)
        )
        cavity = target_core
    target, how = rhuh.treatment_target(postop, brain_arr, cavity_label, target_core, mapping)

    out.mkdir(parents=True, exist_ok=True)
    masks = {}
    for name, arr in (("brain", brain_arr), ("target", target)):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(sitk.ReadImage(str(post["T1ce"])))
        path = out / f"{name}_mask.nii.gz"
        sitk.WriteImage(img, str(path), useCompression=True)
        masks[name] = path
    rec_brain = out / "recurrence_brain_mask.nii.gz"
    rimg = sitk.ReadImage(str(rec["T1ce"]))
    rb = sitk.GetImageFromArray((sitk.GetArrayFromImage(rimg) != 0).astype(np.uint8))
    rb.CopyInformation(rimg)
    sitk.WriteImage(rb, str(rec_brain), useCompression=True)

    peri = next(v for v, n in mapping.items() if n == "peritumoral")
    inputs = PairInputs(
        patient=patient,
        fixed=post["T1ce"],
        fixed_brain=masks["brain"],
        moving=rec["T1ce"],
        moving_brain=rec_brain,
        cavity=cavity,
        peritumoral=postop == peri,
        moving_seg=rec["seg"],
    )
    seeds = {
        k: stable_seed(patient, "rhuh-recipe", k, master_seed=master)
        for k in ("prealign", "forward", "backward")
    }
    recipe = None if recipe_name == "tuned" else {}
    qc = transport_pair(
        inputs, out, seeds, float(config.get("registration.roundtrip_threshold_mm")), recipe
    )
    qc["target_rule"] = how
    (out / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--recipe", choices=("tuned", "gbm4d"), default="tuned")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    rhuh.layout()
    master = int(config.get("registration.master_seed"))
    mapping = _mapping()
    inv = json.loads(
        (REPO_ROOT / "results" / "rhuh_inventory" / "inventory.json").read_text(encoding="utf-8")
    )
    todo = [r["patient"] for r in inv["rows"] if r["usable"]]
    print(f"transport ({args.recipe}): {len(todo)} usable patients", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, p, args.recipe, master, mapping): p for p in todo}
        for fut in as_completed(futs):
            qc = fut.result()
            rt = (qc.get("roundtrip") or {}).get("median_mm", float("nan"))
            print(
                f"{futs[fut]}: {'ok' if qc.get('ok') else 'FAIL ' + qc.get('note', '')}; "
                f"round-trip {rt:.2f} mm",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
