#!/usr/bin/env python
"""Pre-registered sensitivity analyses of the primary (plan Steps 2.3, 4.5). Runs once, after 30_.

Each arm re-scores the frozen challenger and C0 with ``gbmc.confirm`` (the
same code as the primary) and reports the primary Δ PR-AUC with its BCa CI.
The primary itself is never recomputed or replaced.

  unscreened        every transported patient with recurrence outside the
                    target, registration screens not applied
  gbm4d_recipe      labels transported with gbm-4d's original recipe
                    (``21_ --recipe gbm4d``), same screens
  alt_cavity        the other branch of the Step 2.2 cavity rule
  enh_necrosis      recurrence label = enhancing + necrosis
  no_secondary      the patients scanned partly at a secondary facility removed

Not built: the exploratory C1 + baseline ADC arm (first in the plan's cut order).

Refuses unless ``results/confirm/DONE.json`` exists; has its own run-once
marker in ``results/confirm_sensitivity/``.

    python scripts/31_sensitivity.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from gbmc import config, confirm, freeze, rhuh
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "confirm_sensitivity"
RELEASE = REPO_ROOT / "results" / "release"
RHUH = REPO_ROOT / "data" / "interim" / "rhuh"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_module():
    spec = importlib.util.spec_from_file_location("_confirm30",
                                                  REPO_ROOT / "scripts" / "30_confirm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transported(label_dir: Path) -> list[str]:
    return sorted(p.name for p in label_dir.iterdir()
                  if (p / "qc.json").is_file() and _json(p / "qc.json").get("ok"))


def arm_delta(rows: dict, challenger: str, master: int, n_boot: int, arm: str) -> dict:
    values = np.array([r[challenger]["pr_auc"] - r["C0"]["pr_auc"] for r in rows.values()
                       if r["evaluable"]])
    ci = mean_with_bca(values, n_boot, stable_seed("sensitivity", arm, master_seed=master))
    return ci.as_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    require_clean_tree()
    if not (REPO_ROOT / "results" / "confirm" / freeze.DONE).is_file():
        raise SystemExit("the primary (30_) has not completed")
    master = int(config.get("seeds.master"))
    n_boot = int(config.get("seeds.n_bootstrap"))
    primary = _primary_module()
    register = _json(RELEASE / "model_hashes.json")
    challenger = register["challenger"]
    freeze.verify_hashes(register["models"], RELEASE)
    models = confirm.load_models(RELEASE, register["models"])
    inv = _json(REPO_ROOT / "results/rhuh_inventory/inventory.json")
    mapping = {int(k): v for k, v in inv["mapping"]["decision"]["mapping"].items()}
    lock = _json(REPO_ROOT / "results/rhuh_lock/consort.json")
    screened_out = {r["patient"] for r in _json(REPO_ROOT / "results/rhuh_screens/verdict.json")
                    ["rows"] if r["status"] == "EXCLUDE"}
    secondary = {r["patient"] for r in inv["rows"] if r.get("secondary_facility")}

    tuned = RHUH / "label_field_tuned"
    arms = {
        "unscreened": dict(patients=_transported(tuned), label_dir=tuned),
        "gbm4d_recipe": dict(patients=[p for p in _transported(RHUH / "label_field_gbm4d")
                                       if p not in screened_out],
                             label_dir=RHUH / "label_field_gbm4d"),
        "alt_cavity": dict(patients=lock["evaluable"], label_dir=tuned, alt_cavity=True),
        "enh_necrosis": dict(patients=lock["evaluable"], label_dir=tuned,
                             positive=("enhancing", "necrosis")),
        "no_secondary": dict(patients=[p for p in lock["evaluable"] if p not in secondary],
                             label_dir=tuned),
    }
    freeze.open_once(OUT, sorted(arms), note="sensitivity arms of the completed primary")
    results = {}
    for arm, spec in arms.items():
        rows = {}
        for p in spec["patients"]:
            target = None
            if spec.get("alt_cavity"):
                import SimpleITK as sitk
                post = rhuh.files(p, "postop")
                seg = sitk.GetArrayFromImage(sitk.ReadImage(str(post["seg"])))
                brain = sitk.GetArrayFromImage(sitk.ReadImage(
                    str(spec["label_dir"] / p / "brain_mask.nii.gz"))) > 0
                core = rhuh.carried_preop_core(
                    p, mapping, stable_seed(p, "rhuh-core", master_seed=master))
                if rhuh.cavity_rule() is None:
                    # Primary: residual + the whole carried pre-op core. Alternative
                    # (D13, proposed): residual + only the fluid-dark part of that core,
                    # i.e. the cavity as the post-op image shows it.
                    t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(post["T1ce"])))
                    dark = t1 < float(config.get("rhuh.cavity_dark_t1ce_z"))
                    target, _ = rhuh.treatment_target(seg, brain, None, core & dark, mapping)
                else:
                    # Primary: enhancing + labelled cavity. Alternative: the carried-core rule.
                    target, _ = rhuh.treatment_target(seg, brain, None, core, mapping)
            inp = primary.patient_inputs(p, mapping, spec["label_dir"],
                                         spec.get("positive", ("enhancing",)), target)
            row, _, _ = confirm.score_patient(inp, models)
            rows[p] = row
        results[arm] = {"n_scored": len(rows),
                        "n_evaluable": sum(r["evaluable"] for r in rows.values()),
                        "delta_pr_auc_vs_C0": arm_delta(rows, challenger, master, n_boot, arm)}
        print(f"{arm}: {results[arm]}", flush=True)

    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(master_seed=master, challenger=challenger)
        (OUT / "sensitivity.json").write_text(json.dumps(
            {"challenger": challenger, "arms": results,
             "not_run": {"C1_plus_ADC": "exploratory; first in the plan's cut order"}},
            indent=2) + "\n", encoding="utf-8")
    freeze.mark_done(OUT, {"arms": sorted(results)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
