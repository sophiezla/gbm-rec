#!/usr/bin/env python
"""Exploratory: do the maps rank where the tumour spreads next? (plan Step 1.8)

For each patient with a transported 2nd or 3rd progression, the label is the
*new* enhancing tumour: enhancing at that progression, in baseline space, and
not enhancing at the first progression. It is scored inside the evaluation
region (brain minus target) with every candidate's held-out map from ``02_``.
First-progression enhancing voxels stay in the region as negatives,
unchanged, so every candidate is scored on the same voxels as in the tournament.

Only transports that gbm-4d's round-trip screen passes are used; the rest are
listed with their reason. Labelled exploratory everywhere.

Writes ``results/tournament/later_progression/later_progression.json``. Refuses a dirty tree.

    python scripts/06_later_progression_scores.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import candidates, config, lumiere, metrics
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "interim" / "features"
OOF = REPO_ROOT / "data" / "interim" / "oof"
LABELS = REPO_ROOT / "data" / "interim" / "label_field_later"
OUT = REPO_ROOT / "results" / "tournament"


def new_enhancing(patient: str, timepoint: str, evaluate: np.ndarray,
                  first: np.ndarray) -> tuple[np.ndarray | None, str]:
    folder = LABELS / timepoint / patient
    qc_path = folder / "qc.json"
    if not qc_path.is_file():
        return None, "not transported"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if not qc.get("ok") or not qc.get("label_written"):
        return None, f"transport failed: {qc.get('note', '')}"
    if (qc.get("roundtrip") or {}).get("verdict") != "pass":
        return None, "round-trip screen not passed"
    later = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / "recurrence_labels.nii.gz"),
                                                  sitk.sitkUInt8)) == lumiere.ENHANCING
    return (later & ~first)[evaluate], "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    require_clean_tree()
    master = int(config.get("seeds.master"))
    schedule = json.loads((LABELS / "schedule.json").read_text(encoding="utf-8"))

    rows, skipped = [], []
    for entry in schedule:
        patient = entry["patient"]
        if not entry["later"]:
            continue
        with np.load(CACHE / f"{patient}.npz") as z:
            shape = tuple(z["evaluate_shape"])
            evaluate = np.unpackbits(z["evaluate_packed"])[: int(np.prod(shape))
                                                           ].reshape(shape).astype(bool)
        first = lumiere.load_recurrence(patient)
        with np.load(OOF / f"{patient}.npz") as z:
            scores = {k: z[k] for k in z.files}
        for index, timepoint in enumerate(entry["later"], start=2):
            truth, why = new_enhancing(patient, timepoint, evaluate, first)
            if truth is None or not truth.any():
                skipped.append({"patient": patient, "timepoint": timepoint,
                                "reason": why if truth is None else "no new enhancing voxel"})
                continue
            row = {"patient": patient, "timepoint": timepoint, "progression": index,
                   "n_new": int(truth.sum())}
            for name, s in scores.items():
                row[name] = {"pr_auc": metrics.pr_auc(s, truth),
                             "lift": metrics.lift_over_chance(metrics.pr_auc(s, truth), truth)}
            rows.append(row)

    names = [c for c in candidates.ORDER if rows and c in rows[0]]
    summary = {}
    for index in (2, 3):
        subset = [r for r in rows if r["progression"] == index]
        summary[str(index)] = {"n": len(subset)}
        for name in names:
            values = np.array([r[name]["pr_auc"] for r in subset])
            base = np.array([r["C0"]["pr_auc"] for r in subset])
            summary[str(index)][name] = {
                "mean_pr_auc": float(np.nanmean(values)) if subset else None,
                "delta_vs_C0": mean_with_bca(
                    values - base, int(config.get("seeds.n_bootstrap")),
                    stable_seed("later_bootstrap", index, name, master_seed=master)).as_dict()
                if subset and name != "C0" else None}

    out = OUT / "later_progression"
    with run_manifest(out, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(master_seed=master, n_scored=len(rows), n_skipped=len(skipped))
        (out / "later_progression.json").write_text(json.dumps(
            {"label": "EXPLORATORY", "summary": summary, "per_timepoint": rows,
             "skipped": skipped}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
