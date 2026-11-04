#!/usr/bin/env python
"""Preprocess and transport the 2nd and 3rd progressions into baseline space (plan Step 1.8).

Uses the original SyN recipe, which made the first-progression labels (deviation V3).
Output goes under ``data/interim/`` (not in git):

* ``data/interim/preprocessed_later/<patient>/<timepoint>/``
* ``data/interim/label_field_later/<timepoint>/<patient>/`` (labels, transforms, qc.json)

The baselines are read from ``$GBMC_DERIVED_ROOT``. No candidate is scored here. Resumable;
a finished (patient, timepoint) is skipped.

    python scripts/05_later_progressions.py --list
    python scripts/05_later_progressions.py --workers 3
    python scripts/05_later_progressions.py --patients Patient-002
"""


from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import SimpleITK as sitk

from gbmc import config, imaging
from gbmc.lumiere import (
    NATIVE_SEG,
    NECROTIC_NONENHANCING,
    OEDEMA,
    cohort,
    derived_root,
    lumiere_root,
)
from gbmc.progressions import schedule
from gbmc.registration import PairInputs, transport_pair
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC = REPO_ROOT / "data" / "interim" / "preprocessed_later"
LABELS = REPO_ROOT / "data" / "interim" / "label_field_later"


def run_timepoint(patient: str, baseline: str, timepoint: str) -> dict:
    """Preprocess one later timepoint and transport it. Runs in a worker process."""
    config.pin_itk_threads()
    start = time.time()
    out = LABELS / timepoint / patient
    if (out / "qc.json").is_file():
        done = json.loads((out / "qc.json").read_text(encoding="utf-8"))
        return done | {"timepoint": timepoint, "skipped": True}
    master = int(config.get("later_progression.transport_master_seed"))
    native = lumiere_root() / "Imaging" / patient / timepoint
    moving = PREPROC / patient / timepoint
    if not (moving / "qc.json").is_file():
        images, pre_qc = imaging.preprocess_timepoint(native, patient, timepoint, master)
        if not pre_qc.ok:
            return {"patient": patient, "timepoint": timepoint, "ok": False,
                    "note": f"preprocessing failed: {pre_qc.note}", "seconds": time.time() - start}
        moving.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            sitk.WriteImage(image, str(moving / f"{name}.nii.gz"), useCompression=True)
        (moving / "qc.json").write_text(json.dumps(asdict(pre_qc), indent=2), encoding="utf-8")

    base = derived_root() / "data" / "interim" / "preprocessed" / patient / baseline
    reference = sitk.ReadImage(str(base / "CT1.nii.gz"), sitk.sitkFloat32)
    seg = sitk.GetArrayFromImage(sitk.Resample(
        sitk.ReadImage(str(lumiere_root() / "Imaging" / patient / baseline / NATIVE_SEG),
                       sitk.sitkUInt8),
        reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0,
        sitk.sitkUInt8))
    inputs = PairInputs(
        patient=patient, fixed=base / "CT1.nii.gz", fixed_brain=base / "brain_mask.nii.gz",
        moving=moving / "CT1.nii.gz", moving_brain=moving / "brain_mask.nii.gz",
        cavity=seg == NECROTIC_NONENHANCING, peritumoral=seg == OEDEMA,
        moving_seg=native / NATIVE_SEG)
    seeds = {step: stable_seed(patient, timepoint, step, master_seed=master)
             for step in ("prealign", "forward", "backward")}
    threshold = float(config.get("later_progression.roundtrip_threshold_mm"))
    qc = transport_pair(inputs, out, seeds, threshold, recipe={})
    return qc | {"timepoint": timepoint, "seconds": time.time() - start}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel timepoints; each needs about 3 GB of RAM")
    ap.add_argument("--list", action="store_true", help="print the schedule and exit")
    args = ap.parse_args()

    plan = schedule(args.patients or cohort())
    if args.list:
        for row in plan:
            print(f"{row['patient']}: first {row['first']}; later {row['later']}")
        print(f"{sum(len(r['later']) for r in plan)} later timepoints in "
              f"{sum(bool(r['later']) for r in plan)} patients")
        return 0

    LABELS.mkdir(parents=True, exist_ok=True)
    (LABELS / "schedule.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    jobs = [(row["patient"], row["baseline"], tp) for row in plan for tp in row["later"]]
    print(f"{len(jobs)} later timepoints, {args.workers} worker(s)", flush=True)

    def report(qc: dict) -> None:
        rt = (qc.get("roundtrip") or {}).get("median_mm", float("nan"))
        state = ("skipped (done)" if qc.get("skipped") else
                 "ok" if qc.get("ok") else f"FAIL {qc.get('note')}")
        print(f"{qc['patient']} {qc['timepoint']}: {state}; round-trip median {rt:.2f} mm "
              f"({qc.get('seconds', 0):.0f} s)", flush=True)

    if args.workers == 1:
        for job in jobs:
            report(run_timepoint(*job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(run_timepoint, *job) for job in jobs]):
                report(fut.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())
