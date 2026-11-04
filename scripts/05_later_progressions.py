#!/usr/bin/env python
"""Preprocess and transport the 2nd and 3rd progressions into baseline space (plan Step 1.8).

Runs gbm-4d's own frozen code: ``03_preprocess.process_one`` for each later
timepoint and ``04_label_transport.transport_one`` from it to the baseline,
with the same recipe, thread pinning and seed derivation. Only the output roots
change, to this repository's ``data/interim/`` (not in git):

* ``data/interim/preprocessed_later/<patient>/<timepoint>/``
* ``data/interim/label_field_later/<timepoint>/<patient>/`` (labels, transforms, qc.json)

The baseline preprocessing is read from gbm-4d, never recomputed. No candidate
is scored here. Resumable; a finished (patient, timepoint) is skipped.

The seed is gbm-4d's master seed (``config.yaml:later_progression.transport_master_seed``),
so the recipe is the one that made the first-progression labels (D9, proposed).

Long run: 83 timepoints at about 18 min each on one thread. Use --workers
(about 3 GB of RAM each) or the second machine (scripts/08_pack_bundle.py).

    python scripts/05_later_progressions.py --list
    python scripts/05_later_progressions.py --workers 3
    python scripts/05_later_progressions.py --patients Patient-002
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from gbmc import config
from gbmc.lumiere import cohort, gbm4d_root
from gbmc.progressions import frozen_gbm4d_script, schedule

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC = REPO_ROOT / "data" / "interim" / "preprocessed_later"
LABELS = REPO_ROOT / "data" / "interim" / "label_field_later"


def _stage_baseline(patient: str, baseline: str) -> None:
    """Make the frozen transport see baseline + later timepoint under one root.

    ``transport_one`` reads both timepoints from ``PREPROC_ROOT``; the baseline
    is copied (not recomputed) from gbm-4d so both come from one folder.
    """
    src = gbm4d_root() / "data" / "interim" / "preprocessed" / patient / baseline
    dst = PREPROC / patient / baseline
    if not (dst / "qc.json").is_file():
        shutil.copytree(src, dst, dirs_exist_ok=True)


_MODULES: dict = {}


def run_timepoint(patient: str, baseline: str, timepoint: str) -> dict:
    """Preprocess one later timepoint and transport it. Runs in a worker process.

    Each process loads (and verifies) the frozen gbm-4d scripts once.
    """
    if not _MODULES:
        _MODULES["pre"] = frozen_gbm4d_script("03_preprocess", {"OUT_ROOT": PREPROC})
        _MODULES["tr"] = frozen_gbm4d_script("04_label_transport", {"PREPROC_ROOT": PREPROC})
        from gbm4d import config as g4config  # gbm-4d's own config.yaml, via the loader
        _MODULES["roundtrip_mm"] = float(
            g4config.get("cohort.registration_qc.roundtrip_error_threshold_mm"))
    seed = int(config.get("later_progression.transport_master_seed"))
    threads = int(config.get("preprocessing.itk_threads"))
    start = time.time()
    _stage_baseline(patient, baseline)
    rec = _MODULES["pre"].process_one(patient, timepoint, 1.0, seed, False, threads)
    if not rec.get("ok", False):
        return {"patient": patient, "timepoint": timepoint, "ok": False,
                "note": f"preprocessing failed: {rec.get('note')}", "seconds": time.time() - start}
    tr = _MODULES["tr"]
    tr.OUT_ROOT = LABELS / timepoint
    qc = tr.transport_one(patient, baseline, timepoint, seed, threads,
                          _MODULES["roundtrip_mm"], False)
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
