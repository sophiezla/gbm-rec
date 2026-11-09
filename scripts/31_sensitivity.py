#!/usr/bin/env python
"""Pre-registered sensitivity analyses of the primary (plan Steps 2.3, 4.5). Runs once, after 30_.

Each arm re-scores the frozen challenger and C0 with ``gbmc.confirm`` (the
same code as the primary) and reports the primary Δ PR-AUC with its BCa CI.
The primary itself is never recomputed or replaced.

  unscreened        every transported patient with recurrence outside the
                    target, registration screens not applied
  original_recipe   labels transported with the original recipe
                    (``21_ --recipe original``), same screens
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
import json
import sys
from pathlib import Path

import numpy as np

from gbmc import c5_import, config, confirm, freeze
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "confirm_sensitivity"
RELEASE = REPO_ROOT / "results" / "release"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    register = _json(RELEASE / "model_hashes.json")
    challenger = register["challenger"]
    freeze.verify_hashes(register["models"], RELEASE)
    models = confirm.load_models(RELEASE, register["models"])
    inv = _json(REPO_ROOT / "results/rhuh_inventory/inventory.json")
    mapping = {int(k): v for k, v in inv["mapping"]["decision"]["mapping"].items()}
    arms = confirm.sensitivity_arms(REPO_ROOT)
    c5_scores = {variant: c5_import.rhuh_scores(variant, patients,
                                                register["models"][c5_import.RELEASE_FILE])
                 for variant, patients in confirm.c5_variants(arms).items()}
    freeze.open_once(OUT, sorted(arms), note="sensitivity arms of the completed primary")
    results = {}
    for arm, spec in arms.items():
        rows = {}
        for p in spec["patients"]:
            target = (confirm.alt_cavity_target(p, mapping, spec["label_dir"], master)
                      if spec.get("alt_cavity") else None)
            inp = confirm.load_patient(p, mapping, spec["label_dir"],
                                       spec.get("positive", ("enhancing",)), target)
            row, _, _ = confirm.score_patient(inp, models, c5_scores[spec["c5"]][p])
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
