#!/usr/bin/env python
"""Lock the MU-Glioma-Post cohort (D19): flow table, evaluable n, achieved MDE. No model score.

Mirrors RHUH-GBM's ``23_``. Each eligible patient gets its first failing step in this order:
missing sequence or segmentation, failed transport (``26_``), screen failure (``27_``),
empty target, no enhancing recurrence outside the target (undefined, never scored 0).
Patients excluded by the clinical rules (``25_``) are listed with their reason. Checking
recurrence outside the target counts transported labels; it computes no model score.

The achieved MDE comes from the same simulation as the pre-registered power table, seeded
from C0's LUMIERE scores, at this cohort's evaluable n.

Writes ``results/mu_lock/`` (``consort.json``, ``consort.md``). Refuses a dirty tree.

    python scripts/28_mu_lock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, consort, mu, power
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
LF = REPO_ROOT / "data" / "interim" / "mu" / "label_field_tuned"
OUT = REPO_ROOT / "results" / "mu_lock"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def outcomes(eligible: list[str]) -> list[consort.Outcome]:
    screen = {r["patient"]: r for r in _json(REPO_ROOT / "results/mu_screens/verdict.json")["rows"]}
    enhancing = mu.label_values((mu.RECURRENCE_LABEL,))
    out = []
    for p in eligible:
        qc_path = LF / p / "qc.json"
        qc = _json(qc_path) if qc_path.is_file() else {}
        if qc.get("note") == "missing T1ce or segmentation":
            out.append(consort.Outcome(p, consort.STEPS[0], qc["note"]))
            continue
        if not qc.get("ok"):
            out.append(consort.Outcome(p, consort.STEPS[1], qc.get("note", "not transported")))
            continue
        if screen.get(p, {}).get("status") == "EXCLUDE":
            out.append(consort.Outcome(p, consort.STEPS[2], ", ".join(screen[p]["flags"])))
            continue
        target = sitk.GetArrayFromImage(sitk.ReadImage(str(LF / p / "target_mask.nii.gz"))) > 0
        if not target.any():
            out.append(consort.Outcome(p, consort.STEPS[3], ""))
            continue
        brain = sitk.GetArrayFromImage(sitk.ReadImage(str(LF / p / "brain_mask.nii.gz"))) > 0
        rec = sitk.GetArrayFromImage(sitk.ReadImage(str(LF / p / "recurrence_labels.nii.gz")))
        if not (np.isin(rec, enhancing) & brain & ~target).any():
            out.append(consort.Outcome(p, consort.STEPS[4], "undefined, not zero"))
            continue
        out.append(consort.Outcome(p, None))
    return out


def main() -> int:
    require_clean_tree()
    mu.layout()
    master = int(config.get("seeds.master"))
    elig = _json(mu.ELIGIBILITY)
    table = consort.flow(elig["eligible"], outcomes(elig["eligible"]))
    table["clinical_exclusions"] = elig["funnel"]
    per_patient = _json(REPO_ROOT / "results/tournament/per_patient.json")
    c0 = np.array([r["C0"]["pr_auc"] for r in per_patient.values() if r["evaluable"]])
    cfg = config.get("power")
    achieved = {}
    for rho in cfg["rho"]:
        mde, _ = power.minimum_detectable_effect(
            c0, table["n_evaluable"], float(rho),
            stable_seed("power", "mu", table["n_evaluable"], rho, master_seed=master),
            cfg["delta_grid"], target_power=float(cfg["target_power"]),
            n_sim=int(cfg["n_sim"]), n_boot=int(config.get("seeds.n_bootstrap")),
            alpha=float(cfg["alpha"]))
        achieved[str(rho)] = mde
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml", mu.ELIGIBILITY]) as m:
        m.record(n_evaluable=table["n_evaluable"], achieved_mde=achieved)
        (OUT / "consort.json").write_text(
            json.dumps(table | {"achieved_mde": achieved}, indent=2) + "\n", encoding="utf-8")
        (OUT / "consort.md").write_text(consort.markdown(table), encoding="utf-8")
    print(consort.markdown(table))
    print(f"achieved MDE by rho: {achieved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
