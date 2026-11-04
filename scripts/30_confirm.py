#!/usr/bin/env python
"""THE CONFIRMATORY RUN (plan Steps 4.2-4.6). It runs once.

Order, and every refusal:
1. Refuses a dirty tree, a missing OSF record (``config.yaml:osf``), an
   unlocked cohort (``results/rhuh_lock/consort.json``) or missing frozen models.
2. Hash check: every model's SHA-256 must equal ``results/release/model_hashes.json``
   (the values posted to OSF) before anything is loaded.
3. Writes the irreversible marker ``results/confirm/OPENED.json`` BEFORE the
   first RHUH-GBM label is read. Once ``DONE.json`` exists it refuses to run;
   there is no flag to bypass it. An interrupted run may resume, and each
   resumed attempt is logged in OPENED.json.
4. Scores every candidate on every evaluable patient (``gbmc.confirm``).
5. Primary: challenger vs C0, Δ mean PR-AUC, patient-level BCa 95% CI.
   Verdict by the Step 2.3 rule; outcome by Step 4.4 against the achieved MDE;
   the matching Step 2.4 sentence filled.
6. Secondaries under BH-FDR (q from ``config.yaml:verdict.fdr_q``): every other
   candidate vs C0; challenger vs C0T; challenger vs C0 in the escape region;
   lift over chance; coverage at matched 20 mm and at 50 ml. The hero secondary
   (coverage at the 15 mm margin's volume) and the complexity trend are
   reported outside the family, as the plan lists them.
7. Step 4.6 export: maps and top-ranked masks to ``results/confirm/maps/``
   (images are not in git), per-patient numbers in JSON.

    python scripts/30_confirm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import candidates, config, confirm, freeze, rhuh, verdict
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "confirm"
RELEASE = REPO_ROOT / "results" / "release"
LF = REPO_ROOT / "data" / "interim" / "rhuh" / "label_field_tuned"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def patient_inputs(patient: str, mapping: dict[int, str], label_dir: Path = LF,
                   positive: tuple[str, ...] = ("enhancing",),
                   target: np.ndarray | None = None) -> confirm.PatientInputs:
    """Load one RHUH-GBM patient. ``positive`` names the label(s) counted as recurrence."""
    post = rhuh.files(patient, "postop")
    reference = sitk.ReadImage(str(post["T1ce"]), sitk.sitkFloat32)
    arr = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(str(p)))  # noqa: E731
    seg = arr(post["seg"])
    brain = arr(label_dir / patient / "brain_mask.nii.gz") > 0
    if target is None:
        target = arr(label_dir / patient / "target_mask.nii.gz") > 0
    rec = arr(label_dir / patient / "recurrence_labels.nii.gz")
    peri = next(v for v, n in mapping.items() if n == "peritumoral")
    pos_values = [v for v, n in mapping.items() if n in positive]
    t1_seq = config.get("rhuh.falx_t1_sequence")
    return confirm.PatientInputs(
        patient=patient, reference=reference, brain=brain, target=target,
        flair_abnormal=(seg == peri) & brain, t1_z=arr(post[t1_seq]).astype(np.float32),
        lesion=seg > 0, recurrence=np.isin(rec, pos_values) & brain)


def summarise(per_patient: dict, challenger: str, master: int, n_boot: int) -> dict:
    """Means, Δ vs C0 (and challenger vs C0T) with BCa CIs and bootstrap p-values."""
    names = [c for c in candidates.ORDER if c in next(iter(per_patient.values()))]
    keys = [k for k in per_patient[next(iter(per_patient))]["C0"]]
    table = {}

    def delta(a: str, b: str, key: str) -> dict:
        va = np.array([r[a].get(key, np.nan) for r in per_patient.values()], float)
        vb = np.array([r[b].get(key, np.nan) for r in per_patient.values()], float)
        ci = mean_with_bca(va - vb, n_boot, stable_seed("confirm", a, b, key, master_seed=master))
        return ci.as_dict() | {"p": verdict.bootstrap_p(ci.boot) if ci.boot is not None else None}

    for name in names:
        table[name] = {k: {"mean": float(np.nanmean([r[name].get(k, np.nan)
                                                      for r in per_patient.values()]))}
                       for k in keys}
        if name != "C0":
            for k in keys:
                table[name][k]["delta_vs_C0"] = delta(name, "C0", k)
    table["_challenger_vs_C0T"] = {"pr_auc": delta(challenger, "C0T", "pr_auc")}
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    require_clean_tree()
    osf = {"url": config.get("osf.url"), "timestamp": config.get("osf.timestamp")}
    rhuh.layout()
    master = int(config.get("seeds.master"))
    n_boot = int(config.get("seeds.n_bootstrap"))
    lock = _json(REPO_ROOT / "results/rhuh_lock/consort.json")
    register = _json(RELEASE / "model_hashes.json")
    challenger = register["challenger"]
    mapping = {int(k): v for k, v in
               _json(REPO_ROOT / "results/rhuh_inventory/inventory.json")
               ["mapping"]["decision"]["mapping"].items()}

    freeze.verify_hashes(register["models"], RELEASE)                     # Step 4.2
    models = confirm.load_models(RELEASE, register["models"])
    freeze.open_once(OUT, lock["evaluable"],                              # Step 4.3
                     note="Written before any RHUH-GBM label was read by this script.")

    per_patient, qc = {}, {}
    for p in lock["evaluable"]:
        inp = patient_inputs(p, mapping)
        row, scores, feature_qc = confirm.score_patient(inp, models)
        row["maps"] = confirm.write_maps(OUT / "maps", p, inp.reference,
                                         inp.brain & ~inp.target, scores)
        per_patient[p], qc[p] = row, feature_qc
        print(f"{p}: scored", flush=True)

    table = summarise(per_patient, challenger, master, n_boot)
    primary = table[challenger]["pr_auc"]["delta_vs_C0"]
    mde_rho = str(config.get("verdict.mde_rho"))
    mde = lock["achieved_mde"].get(mde_rho)
    outcome = verdict.classify(primary["ci_low"], primary["ci_high"],
                               float("inf") if mde is None else float(mde))
    hero = table[challenger]["coverage_margin15"]["delta_vs_C0"]
    coverage_differs = hero["ci_low"] > 0 or hero["ci_high"] < 0
    text = verdict.sentence(
        outcome, challenger, primary["mean"], (primary["ci_low"], primary["ci_high"]),
        table[challenger]["coverage_margin15"]["mean"], table["C0"]["coverage_margin15"]["mean"],
        len([c for c in candidates.ORDER if c in table]),
        (hero["ci_low"], hero["ci_high"]), coverage_differs)

    family = {f"{c}_vs_C0_pr_auc": table[c]["pr_auc"]["delta_vs_C0"]["p"]
              for c in table if c not in ("C0", challenger, "_challenger_vs_C0T")}
    family |= {"challenger_vs_C0T": table["_challenger_vs_C0T"]["pr_auc"]["p"],
               "challenger_escape_pr_auc": table[challenger]["pr_auc_escape"]["delta_vs_C0"]["p"],
               "challenger_lift": table[challenger]["lift"]["delta_vs_C0"]["p"],
               "challenger_coverage_margin20": table[challenger]["coverage_margin20"]
               ["delta_vs_C0"]["p"],
               "challenger_coverage_volume50ml": table[challenger]["coverage_volume50ml"]
               ["delta_vs_C0"]["p"]}
    secondaries = verdict.benjamini_hochberg(family, float(config.get("verdict.fdr_q")))

    result = {"osf": osf, "challenger": challenger, "n_evaluable": len(per_patient),
              "primary": {"metric": "delta mean PR-AUC, challenger vs C0"} | primary,
              "achieved_mde": mde, "mde_rho": mde_rho, "outcome": outcome,
              "champion": verdict.champion(outcome, challenger), "sentence": text,
              "hero_coverage_margin15": hero, "secondaries_bh": secondaries,
              "candidates": table,
              "language": "RHUH-GBM numbers are validation results; coverage denominator: "
                          "recurrence outside the target"}
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml", RELEASE / "model_hashes.json"]) as m:
        m.record(master_seed=master, n_bootstrap=n_boot, challenger=challenger, outcome=outcome)
        (OUT / "confirm.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (OUT / "per_patient.json").write_text(json.dumps(per_patient, indent=2) + "\n",
                                              encoding="utf-8")
        (OUT / "feature_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        (OUT / "sentence.txt").write_text(text + "\n", encoding="utf-8")
    freeze.mark_done(OUT, {"outcome": outcome, "champion": result["champion"]})
    print(f"\n{outcome}. Champion: {result['champion']}.\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
