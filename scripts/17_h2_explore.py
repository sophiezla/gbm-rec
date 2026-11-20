#!/usr/bin/env python
"""Exploratory development of the H2 final model on LUMIERE (DECISIONS.md D19, deviation V6).

Six rank-average ensembles of already-registered models, fixed before any of them was
scored (2026-11-19), are scored exactly like the tournament (`02_`): per-patient PR-AUC on
the held-out maps in ``data/interim/oof/``, delta vs C0 with a patient-level BCa CI, and
coverage at each patient's 15 mm margin volume. LUMIERE is development data and these
ensembles were defined after the tournament was seen, so every number here is
exploratory (V6); they only guide the choice of the one H2 model.

Writes ``results/h2_explore/h2_explore.json``. Refuses a dirty tree.

    python scripts/17_h2_explore.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gbmc import candidates, config, metrics, tournament
from gbmc.lumiere import cohort
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "h2_explore"
OOF = REPO_ROOT / "data" / "interim" / "oof"
#: Fixed before scoring (2026-11-19). Non-GPU members only, plus C6 and C2 for reference.
SETS = {"E1": ["C0", "C2"], "E2": ["C0T", "C2"], "E3": ["C1", "C3", "C2"],
        "E4": ["C0", "C1", "C3", "C2"], "E5": ["C0T", "C1", "C3", "C2"],
        "E6": ["C0", "C0T", "C2"]}
REFERENCE = ["C0", "C2", "C6"]


def main() -> int:
    require_clean_tree()
    master = int(config.get("seeds.master"))
    n_boot = int(config.get("seeds.n_bootstrap"))
    rows = {}
    for patient in cohort():
        data = tournament.load(patient)
        if not data["truth"].any():
            continue          # undefined, never zero (D6 item 1)
        with np.load(OOF / f"{patient}.npz") as z:
            scores = {k: z[k] for k in z.files}
        for name, members in SETS.items():
            scores[name] = candidates.rank_average([scores[m] for m in members])
        rows[patient] = {n: metrics.patient_scores(scores[n], data["truth"], data["matched"])
                         for n in REFERENCE + list(SETS)}
    c0 = np.array([r["C0"]["pr_auc"] for r in rows.values()])
    results = {}
    for name in REFERENCE + list(SETS):
        values = np.array([r[name]["pr_auc"] for r in rows.values()])
        entry = {"members": SETS.get(name, [name]), "pr_auc_mean": float(values.mean()),
                 "coverage_margin15_mean": float(np.mean(
                     [r[name]["coverage_margin15"] for r in rows.values()]))}
        if name != "C0":
            entry["delta_vs_C0"] = mean_with_bca(
                values - c0, n_boot, stable_seed("h2_explore", name, master_seed=master)).as_dict()
        results[name] = entry
        print(name, json.dumps(entry))
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(master_seed=master, n_bootstrap=n_boot, n_evaluable=len(rows))
        (OUT / "h2_explore.json").write_text(json.dumps(
            {"label": "exploratory (V6): H2 development on LUMIERE; not a test result",
             "n_evaluable": len(rows), "sets": SETS, "results": results}, indent=2) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
