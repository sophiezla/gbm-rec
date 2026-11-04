#!/usr/bin/env python
"""RHUH-GBM registration screens (plan Step 3.4): exclude, never certify.

  --surface    brain-surface distance after transport vs the identity null
               (gbm-4d ``ventricles.surface_distance_mm``), every patient
  --arbiters   multi-arbiter consensus on every patient: the recipe re-driven by
               T2 and FLAIR, plus the shipped SRI24 alignment (identity here)
  --simulate   simulated deformation recovery on ``screens.simulate_n`` seeded patients
  --verdict    EXCLUDE / REVIEW / SCREENED per patient (``gbmc.screens.verdict``)

Thresholds come from ``config.yaml:screens`` and ``registration``, fixed in
Phase 2. Writes ``results/rhuh_screens/``; per-patient caches in
``data/interim/rhuh/screens/``.

    python scripts/22_rhuh_screens.py --surface --arbiters --simulate --workers 2
    python scripts/22_rhuh_screens.py --verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, rhuh, screens
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.registration import TUNED_SYN, register_syn
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
LF = REPO_ROOT / "data" / "interim" / "rhuh" / "label_field_tuned"
CACHE = REPO_ROOT / "data" / "interim" / "rhuh" / "screens"
OUT = REPO_ROOT / "results" / "rhuh_screens"
SPACING = 2.0


def transported() -> list[str]:
    return sorted(
        p.name
        for p in LF.iterdir()
        if (p / "qc.json").is_file()
        and json.loads((p / "qc.json").read_text(encoding="utf-8")).get("ok")
    )


def surface(patient: str) -> dict:
    from gbm4d import transport, ventricles

    fixed = sitk.ReadImage(str(LF / patient / "brain_mask.nii.gz"), sitk.sitkUInt8)
    moving = sitk.ReadImage(str(LF / patient / "recurrence_brain_mask.nii.gz"), sitk.sitkUInt8)
    forward = sitk.ReadTransform(str(LF / patient / "forward_transform.h5"))
    after = ventricles.surface_distance_mm(fixed, transport.warp_label(moving, fixed, forward))
    before = ventricles.surface_distance_mm(
        fixed, transport.warp_label(moving, fixed, sitk.Transform(3, sitk.sitkIdentity))
    )
    return {
        "surface_after_mm": after["mean_mm"],
        "surface_before_mm": before["mean_mm"],
        "dice_after": after["dice"],
        "dice_before": before["dice"],
    }


def arbiters(patient: str, master: int) -> dict:
    from gbm4d import preprocess, transport

    post, rec = rhuh.files(patient, "postop"), rhuh.files(patient, "recurrence")
    fixed = sitk.ReadImage(str(post["T1ce"]), sitk.sitkFloat32)
    reference = transport.coarsen(fixed, SPACING)
    brain = sitk.ReadImage(str(LF / patient / "brain_mask.nii.gz"), sitk.sitkUInt8)
    mask = sitk.GetArrayFromImage(transport.coarsen(brain, SPACING, sitk.sitkNearestNeighbor)) > 0
    fields = {
        "T1c": sitk.GetArrayFromImage(
            transport.displacement_field(
                sitk.ReadTransform(str(LF / patient / "forward_transform.h5")), reference
            )
        )
    }
    for seq in ("T2", "FLAIR"):
        if seq not in post or seq not in rec:
            continue
        f = sitk.ReadImage(str(post[seq]), sitk.sitkFloat32)
        m = sitk.ReadImage(str(rec[seq]), sitk.sitkFloat32)
        rigid, _ = preprocess.register_rigid(
            f, m, stable_seed(patient, "arb", seq, "rigid", master_seed=master)
        )
        aligned = sitk.Resample(m, f, rigid.GetInverse(), sitk.sitkLinear, 0.0, m.GetPixelID())
        deform, info = register_syn(
            f,
            aligned,
            stable_seed(patient, "arb", seq, master_seed=master),
            brain,
            None,
            **TUNED_SYN,
        )
        if not info["metric_valid"]:
            continue
        tx = sitk.CompositeTransform([rigid, deform])
        tx.FlattenTransform()
        fields[seq] = sitk.GetArrayFromImage(transport.displacement_field(tx, reference))
    fields["shipped_rigid"] = np.zeros_like(fields["T1c"])  # both scans already in SRI24
    return screens.consensus(fields, mask)


def simulate(patient: str, master: int) -> dict:
    from gbm4d import preprocess, transport

    post = rhuh.files(patient, "postop")
    fixed = sitk.ReadImage(str(post["T1ce"]), sitk.sitkFloat32)
    brain = sitk.GetArrayFromImage(fixed) != 0
    sp = fixed.GetSpacing()[::-1]
    truth = screens.synthetic_field(
        brain.shape, sp, brain, stable_seed(patient, "simfield", master_seed=master)
    )
    df = sitk.GetImageFromArray(truth[..., ::-1].astype(np.float64), isVector=True)
    df.CopyInformation(fixed)
    inverse = sitk.InvertDisplacementField(
        df,
        maximumNumberOfIterations=50,
        meanErrorToleranceThreshold=0.001,
        maxErrorToleranceThreshold=0.05,
        enforceBoundaryCondition=True,
    )
    moving = sitk.Resample(
        fixed,
        fixed,
        sitk.DisplacementFieldTransform(inverse),
        sitk.sitkLinear,
        0.0,
        fixed.GetPixelID(),
    )
    rigid, _ = preprocess.register_rigid(
        fixed, moving, stable_seed(patient, "sim-rigid", master_seed=master)
    )
    aligned = sitk.Resample(
        moving, fixed, rigid.GetInverse(), sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    deform, _ = register_syn(
        fixed, aligned, stable_seed(patient, "sim-syn", master_seed=master), None, None, **TUNED_SYN
    )
    tx = sitk.CompositeTransform([rigid, deform])
    tx.FlattenTransform()
    reference = transport.coarsen(fixed, SPACING)
    rec = sitk.GetArrayFromImage(transport.displacement_field(tx, reference))
    tru = sitk.GetArrayFromImage(
        transport.displacement_field(sitk.DisplacementFieldTransform(sitk.Image(df)), reference)
    )
    mask = (
        sitk.GetArrayFromImage(
            transport.coarsen(
                sitk.Cast(fixed != 0, sitk.sitkUInt8), SPACING, sitk.sitkNearestNeighbor
            )
        )
        > 0
    )
    return screens.recovery_error(rec, tru, mask, SPACING)


def _cached(kind: str, patient: str, fn, *a) -> dict:
    path = CACHE / kind / f"{patient}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    from gbm4d.preprocess import use_deterministic_threads

    use_deterministic_threads(int(config.get("preprocessing.itk_threads")))
    out = {"patient": patient} | fn(patient, *a)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def verdict() -> int:
    require_clean_tree()
    cfg = config.get("screens")
    rows = []
    for p in transported():
        qc = json.loads((LF / p / "qc.json").read_text(encoding="utf-8"))
        s = json.loads((CACHE / "surface" / f"{p}.json").read_text(encoding="utf-8"))
        a_path = CACHE / "arbiters" / f"{p}.json"
        a = json.loads(a_path.read_text(encoding="utf-8")) if a_path.is_file() else {}
        r = screens.ScreenResult(
            p,
            qc["roundtrip"]["median_mm"],
            s["surface_after_mm"],
            s["surface_before_mm"],
            a.get("dissenter"),
            a.get("our_deviation_mm"),
        )
        status = screens.verdict(
            r,
            float(config.get("registration.roundtrip_threshold_mm")),
            float(cfg["surface_review_mm"]),
            float(cfg["surface_worse_margin_mm"]),
        )
        rows.append(asdict(r) | {"status": status})
    sims = [json.loads(f.read_text(encoding="utf-8")) for f in (CACHE / "simulate").glob("*.json")]
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(n=len(rows), n_exclude=sum(r["status"] == "EXCLUDE" for r in rows))
        (OUT / "verdict.json").write_text(
            json.dumps({"gate": "SCREENED-NOT-LICENSED", "rows": rows, "simulate": sims}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print({s: sum(r["status"] == s for r in rows) for s in ("EXCLUDE", "REVIEW", "SCREENED")})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for flag in ("--surface", "--arbiters", "--simulate", "--verdict"):
        ap.add_argument(flag, action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    rhuh.layout()
    if args.verdict:
        return verdict()
    master = int(config.get("registration.master_seed"))
    patients = transported()
    jobs = []
    if args.surface:
        jobs += [("surface", p, surface) for p in patients]
    if args.arbiters:
        jobs += [("arbiters", p, arbiters, master) for p in patients]
    if args.simulate:
        n = int(config.get("screens.simulate_n"))
        pick = np.random.default_rng(stable_seed("rhuh_simulate", master_seed=master)).choice(
            patients, size=min(n, len(patients)), replace=False
        )
        jobs += [("simulate", p, simulate, master) for p in sorted(pick)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for out in pool.map(_run, jobs):
            print(out, flush=True)
    return 0


def _run(job) -> str:
    kind, patient, fn, *extra = job
    _cached(kind, patient, fn, *extra)
    return f"{kind} {patient}: done"


if __name__ == "__main__":
    sys.exit(main())
