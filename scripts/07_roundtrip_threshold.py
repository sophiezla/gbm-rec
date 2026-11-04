#!/usr/bin/env python
"""Round-trip distribution of the tuned recipe on LUMIERE (plan Step 3.3, fixed in Phase 2).

The RHUH-GBM exclusion threshold is set against the round-trip error that the
recipe which will actually run (gbm-wm's tuned SyN, ``gbmc.registration``)
produces on the 63 LUMIERE baseline -> first-progression pairs. gbm-wm lost 21
of 72 patients when a threshold written for one recipe met another (plan §2);
this measures the recipe's own distribution first.

Registration only: the progression segmentation is never read and nothing is
warped, so no LUMIERE label enters (the plan: "a check that uses no LUMIERE
labels"). The baseline segmentation supplies the cavity mask for the metric
and the oedema mask for the round-trip region, exactly as in transport.

Per patient -> ``data/interim/roundtrip_tuned/<patient>/qc.json`` (resumable).
``--summary`` writes ``results/registration_threshold/roundtrip.json`` (clean
tree required). The rule that turns this distribution into the threshold is a
pre-registration decision (``config.yaml:registration.roundtrip_threshold_rule``).

    python scripts/07_roundtrip_threshold.py --workers 2
    python scripts/07_roundtrip_threshold.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gbmc import config, lumiere
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.registration import PairInputs, transport_pair
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK = REPO_ROOT / "data" / "interim" / "roundtrip_tuned"
OUT = REPO_ROOT / "results" / "registration_threshold"


def one(patient: str, master: int, overwrite: bool = False) -> dict:
    import SimpleITK as sitk
    config.pin_itk_threads()
    out = WORK / patient
    if (out / "qc.json").is_file() and not overwrite:
        return json.loads((out / "qc.json").read_text(encoding="utf-8")) | {"skipped": True}
    qc = lumiere.transport_qc(patient)
    pre = lumiere.derived_root() / "data" / "interim" / "preprocessed" / patient
    base, prog = pre / qc["baseline"], pre / qc["progression"]
    seg_img = sitk.ReadImage(str(lumiere.lumiere_root() / "Imaging" / patient / qc["baseline"]
                                 / lumiere.NATIVE_SEG), sitk.sitkUInt8)
    reference = sitk.ReadImage(str(base / "CT1.nii.gz"), sitk.sitkFloat32)
    seg = sitk.GetArrayFromImage(sitk.Resample(
        seg_img, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0,
        sitk.sitkUInt8))
    inputs = PairInputs(
        patient=patient, fixed=base / "CT1.nii.gz", fixed_brain=base / "brain_mask.nii.gz",
        moving=prog / "CT1.nii.gz", moving_brain=prog / "brain_mask.nii.gz",
        cavity=seg == lumiere.NECROTIC_NONENHANCING, peritumoral=seg == lumiere.OEDEMA,
        moving_seg=None)
    seeds = {k: stable_seed(patient, "rhuh-recipe", k, master_seed=master)
             for k in ("prealign", "forward", "backward")}
    try:
        return transport_pair(inputs, out, seeds, roundtrip_threshold_mm=None)
    except Exception as exc:  # noqa: BLE001
        return {"patient": patient, "ok": False, "note": f"{type(exc).__name__}: {exc}"}


def summary() -> int:
    require_clean_tree()
    rows = []
    for patient in lumiere.cohort():
        path = WORK / patient / "qc.json"
        if path.is_file():
            q = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"patient": patient, "ok": q.get("ok"), "note": q.get("note", ""),
                         "roundtrip_median_mm": (q.get("roundtrip") or {}).get("median_mm"),
                         "roundtrip_max_mm": (q.get("roundtrip") or {}).get("max_mm"),
                         "jacobian_fraction_negative":
                             (q.get("jacobian") or {}).get("fraction_negative")})
    med = np.array([r["roundtrip_median_mm"] for r in rows if r["roundtrip_median_mm"] is not None])
    if med.size == 0:
        raise SystemExit("no finished patients")
    stats = {"n": int(med.size), "median": float(np.median(med)),
             "p90": float(np.percentile(med, 90)), "p95": float(np.percentile(med, 95)),
             "max": float(med.max())}
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(**stats, master_seed=int(config.get("registration.master_seed")))
        (OUT / "roundtrip.json").write_text(json.dumps(
            {"recipe": "gbm-wm D11 tuned SyN", "cohort": "LUMIERE baseline->first progression",
             "labels_read": False, "distribution": stats, "rows": rows}, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summary()
    master = int(config.get("registration.master_seed"))
    patients = args.patients or lumiere.cohort()
    start = time.time()

    def report(r: dict) -> None:
        rt = (r.get("roundtrip") or {}).get("median_mm", float("nan"))
        state = "skip" if r.get("skipped") else ("ok" if r.get("ok") else f"FAIL {r.get('note')}")
        print(f"{r['patient']}: {state}; round-trip median {rt:.2f} mm "
              f"({time.time() - start:.0f} s)", flush=True)

    if args.workers == 1:
        for p in patients:
            report(one(p, master))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(one, p, master) for p in patients]):
                report(fut.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())
