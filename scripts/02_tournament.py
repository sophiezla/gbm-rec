#!/usr/bin/env python
"""The LUMIERE tournament (plan Steps 1.5-1.6): every candidate on the same 7 folds.

Refuses to run unless the tournament rules are committed (a clean tree) and the
master seed is set: Step 1.1 requires the rules on record before the first
candidate is scored.

Writes ``results/tournament/``: ``folds.json``, ``per_patient.json``,
``leaderboard.json`` (means, Δ vs C0 with BCa CIs, the challenger) and the run
manifest; and every held-out voxel score map to ``data/interim/oof/`` (not in git).
C5 passed the Step 0.4 gate (D6 (g)). It is trained on the GPU computer, and its held-out
maps are imported into ``data/interim/c5_oof/``. This script refuses to run until they are
present, hash-checked and made on the same folds, so C6 always has its fixed members.

    python scripts/02_tournament.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gbmc import candidates, config, metrics
from gbmc.lumiere import cohort
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca
from gbmc.tournament import CACHE, c5_held_out, load, missing, training_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "tournament"
# Held-out voxel scores per patient, for Step 1.8 and the app (not in git).
OOF = REPO_ROOT / "data" / "interim" / "oof"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    require_clean_tree()
    master = int(config.get("seeds.master"))
    n_boot = int(config.get("seeds.n_bootstrap"))
    patients = cohort()
    if absent := missing(patients):
        raise SystemExit(f"features missing for {len(absent)} patients; run 01_ first")

    folds = candidates.patient_folds(patients, int(config.get("tournament.n_folds")),
                                     stable_seed("tournament_folds", master_seed=master))
    c5 = c5_held_out(patients, folds)
    train_cache = {p: training_rows(p, master) for p in patients}

    per_patient: dict[str, dict] = {}
    for k, held_out in enumerate(folds):
        train = [p for p in patients if p not in held_out]
        models = {}
        for name, cand in candidates.LEARNED.items():
            models[name] = candidates.fit(
                cand, [train_cache[p][0] for p in train], [train_cache[p][1] for p in train],
                stable_seed("tournament_fit", name, k, master_seed=master))
        for patient in held_out:
            data = load(patient)
            x, truth = data["matrix"], data["truth"]
            scores = {name: rule.score(x) for name, rule in candidates.RULES.items()}
            for name, cand in candidates.LEARNED.items():
                scores[name] = candidates.predict(cand, models[name], x)
            if c5[patient].shape != truth.shape:
                raise SystemExit(f"C5 map for {patient} has the wrong number of voxels")
            scores["C5"] = c5[patient]
            scores["C6"] = candidates.rank_average(
                [scores[m] for m in candidates.ENSEMBLE_MEMBERS])
            row = {"fold": k, "evaluable": bool(truth.any()),
                   "n_positive": int(truth.sum()), "prevalence": float(truth.mean()),
                   "matched_voxels": data["matched"]}
            per_ml = data["matched"]["volume50ml"] / 50.0
            for name, s in scores.items():
                row[name] = metrics.patient_scores(s, truth, data["matched"])
            if truth.any():
                row["curves"] = {name: metrics.coverage_curve(s, truth, per_ml)
                                 for name, s in scores.items()}
            per_patient[patient] = row
            OOF.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OOF / f"{patient}.npz",
                                **{n: s.astype(np.float32) for n, s in scores.items()})
        print(f"fold {k + 1}/{len(folds)} done", flush=True)

    names = [c for c in candidates.ORDER if c in per_patient[patients[0]]]
    metric_keys = list(per_patient[patients[0]]["C0"])  # "curves" is kept per patient only
    leaderboard = {}
    for name in names:
        entry = {}
        for m in metric_keys:
            values = np.array([per_patient[p][name][m] for p in patients])
            base = np.array([per_patient[p]["C0"][m] for p in patients])
            entry[m] = {"mean": float(np.nanmean(values)),
                        "n": int(np.isfinite(values).sum())}
            if name != "C0":
                ci = mean_with_bca(values - base, n_boot,
                                   stable_seed("tournament_bootstrap", name, m,
                                               master_seed=master))
                entry[m]["delta_vs_C0"] = ci.as_dict()
        leaderboard[name] = entry

    challenger = candidates.select_challenger(
        {n: leaderboard[n]["pr_auc"]["mean"] for n in names},
        float(config.get("tournament.tie_margin_pr_auc")))

    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as manifest:
        manifest.record(master_seed=master, n_bootstrap=n_boot, n_patients=len(patients),
                        n_evaluable=sum(per_patient[p]["evaluable"] for p in patients),
                        feature_cache=str(CACHE), challenger=challenger)
        (OUT / "folds.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
        (OUT / "per_patient.json").write_text(json.dumps(per_patient, indent=2) + "\n",
                                              encoding="utf-8")
        (OUT / "leaderboard.json").write_text(
            json.dumps({"order": names, "candidates": leaderboard, "challenger": challenger},
                       indent=2) + "\n", encoding="utf-8")
    print(f"challenger: {challenger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
