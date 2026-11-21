#!/usr/bin/env python
"""THE MU-GLIOMA-POST TEST OF H2 (D19; registration section 8). It runs once.

H2b: the frozen final model E1 (rank average of C0 and C2) vs distance C0, delta mean PR-AUC
with a patient-level BCa 95% CI, classified with this cohort's achieved MDE at rho 0.7 and
reported with the pre-registered H2 sentence (``gbmc.confirm.h2_summary``). Scoring uses the
same code as RHUH-GBM's ``30_`` (``gbmc.confirm.score_patient``) without C5 or C6.

Order, and every refusal:
1. Refuses a dirty tree, a missing registration record (``config.yaml:osf``), the MU layout
   undecided, an unlocked cohort (``results/mu_lock/consort.json``) or missing models.
2. Every model's SHA-256 must equal ``results/release/model_hashes.json``.
3. Writes the irreversible marker ``results/mu_confirm/OPENED.json`` before the first MU
   recurrence label is read; once ``DONE.json`` exists it refuses to run again.
4. Scores every evaluable patient, then writes ``confirm.json`` (H2b) and the per-patient
   numbers. Only H2b is a registered test here; the other candidates' MU numbers are
   descriptive.

    python scripts/32_mu_confirm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, confirm, freeze, mu
from gbmc.imaging import zscore_array
from gbmc.provenance import require_clean_tree, run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "mu_confirm"
RELEASE = REPO_ROOT / "results" / "release"
LF = REPO_ROOT / "data" / "interim" / "mu" / "label_field_tuned"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def patient_inputs(patient: str) -> confirm.PatientInputs:
    """One MU patient on its baseline grid (target and FLAIR region as pre-registered)."""
    base = mu.files(patient, "postop")
    reference = sitk.ReadImage(str(base["T1ce"]), sitk.sitkFloat32)
    arr = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(str(p)))  # noqa: E731
    seg = arr(base["seg"])
    brain = arr(LF / patient / "brain_mask.nii.gz") > 0
    target = arr(LF / patient / "target_mask.nii.gz") > 0
    rec = arr(LF / patient / "recurrence_labels.nii.gz")
    t1 = arr(base["T1"]).astype(np.float32)
    return confirm.PatientInputs(
        patient=patient, reference=reference, brain=brain, target=target,
        flair_abnormal=np.isin(seg, mu.label_values((mu.FLAIR_LABEL,))) & brain,
        t1_z=zscore_array(t1, brain), lesion=seg > 0,
        recurrence=np.isin(rec, mu.label_values((mu.RECURRENCE_LABEL,))) & brain)


def main() -> int:
    require_clean_tree()
    osf = {"url": config.get("osf.url"), "timestamp": config.get("osf.timestamp")}
    mu.layout()
    master = int(config.get("seeds.master"))
    n_boot = int(config.get("seeds.n_bootstrap"))
    lock = _json(REPO_ROOT / "results/mu_lock/consort.json")
    register = _json(RELEASE / "model_hashes.json")
    freeze.verify_hashes(register["models"], RELEASE)
    models = confirm.load_models(RELEASE, register["models"])
    freeze.open_once(OUT, lock["evaluable"],
                     note="Written before any MU-Glioma-Post recurrence label was read here.")

    per_patient, qc = {}, {}
    for p in lock["evaluable"]:
        inp = patient_inputs(p)
        row, scores, feature_qc = confirm.score_patient(inp, models, include_c5=False)
        row["maps"] = confirm.write_maps(OUT / "maps", p, inp.reference,
                                         inp.brain & ~inp.target, scores)
        per_patient[p], qc[p] = row, feature_qc
        print(f"{p}: scored", flush=True)

    mde_rho = str(config.get("verdict.mde_rho"))
    mde = lock["achieved_mde"].get(mde_rho)
    h2 = confirm.h2_summary(per_patient, "MU-Glioma-Post", mde, master, n_boot)
    result = {"osf": osf, "test": "H2b", "mde_rho": mde_rho, "H2": h2,
              "language": "MU-Glioma-Post numbers are validation results for H2 only; "
                          "coverage denominator: recurrence outside the target"}
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml", RELEASE / "model_hashes.json"]) as m:
        m.record(master_seed=master, n_bootstrap=n_boot, outcome=h2["outcome"])
        (OUT / "confirm.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (OUT / "per_patient.json").write_text(json.dumps(per_patient, indent=2) + "\n",
                                              encoding="utf-8")
        (OUT / "feature_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    freeze.mark_done(OUT, {"outcome": h2["outcome"]})
    print(f"\nH2b: {h2['outcome']}.\n{h2['sentence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
