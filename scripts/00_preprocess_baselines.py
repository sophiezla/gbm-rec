#!/usr/bin/env python
"""Preprocess each locked patient's baseline from raw LUMIERE (deviation V3).

Writes ``$GBMC_DERIVED_ROOT/data/interim/preprocessed/<patient>/<baseline>/``: the z-scored
sequences, the brain mask and qc.json. Resumable; a finished patient is skipped.

    python scripts/00_preprocess_baselines.py --workers 4
    python scripts/00_preprocess_baselines.py --patients Patient-001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import SimpleITK as sitk

from gbmc import config, imaging, lumiere


def one(patient: str, master: int) -> str:
    config.pin_itk_threads()
    start = time.time()
    baseline = lumiere.baseline_name(patient)
    out = lumiere.derived_root() / "data" / "interim" / "preprocessed" / patient / baseline
    if (out / "qc.json").is_file():
        return f"{patient}: skipped (done)"
    native = lumiere.lumiere_root() / "Imaging" / patient / baseline
    images, qc = imaging.preprocess_timepoint(native, patient, baseline, master)
    if not qc.ok:
        return f"{patient}: FAILED {qc.note}"
    out.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        sitk.WriteImage(image, str(out / f"{name}.nii.gz"), useCompression=True)
    (out / "qc.json").write_text(json.dumps(asdict(qc), indent=2) + "\n", encoding="utf-8")
    return f"{patient}: ok ({time.time() - start:.0f} s)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    master = int(config.get("later_progression.transport_master_seed"))
    patients = args.patients or lumiere.cohort()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in as_completed([pool.submit(one, p, master) for p in patients]):
            print(job.result(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
