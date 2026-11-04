#!/usr/bin/env python
"""Prediction vs reality, the centrepiece (plan Step 4.7), for the rule-chosen patient.

The patient is fixed by rule before any look at images: the evaluable
RHUH-GBM patient with the median per-patient Δ PR-AUC (lead model vs C0;
``gbmc.figures.demo_patients``). ``--which failure|best`` draws the other two
the app links to. The slice is the axial slice holding the most recurrence.

Panels: the 15 mm guideline margin (C0T's top voxels at the margin's own
volume, which is the margin), distance's prediction and the lead model's
prediction at that same volume, with coverage printed under each (read from
``results/confirm/per_patient.json``), and the actual recurrence.

    python scripts/51_reveal_figure.py
    python scripts/51_reveal_figure.py --which failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import figures, rhuh
from gbmc.provenance import require_clean_tree, run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRM = REPO_ROOT / "results" / "confirm"
LF = REPO_ROOT / "data" / "interim" / "rhuh" / "label_field_tuned"
OUT = REPO_ROOT / "results" / "figures_reveal"


def _arr(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def top_region(score_map: np.ndarray, evaluate: np.ndarray, n: int) -> np.ndarray:
    idx = np.flatnonzero(evaluate)
    order = np.argsort(-score_map.flat[idx], kind="stable")[:n]
    out = np.zeros(score_map.shape, bool)
    out.flat[idx[order]] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--which", choices=("demo", "failure", "best"), default="demo")
    args = ap.parse_args()
    require_clean_tree()
    confirm = json.loads((CONFIRM / "confirm.json").read_text(encoding="utf-8"))
    rows = json.loads((CONFIRM / "per_patient.json").read_text(encoding="utf-8"))
    lead = confirm["champion"] if confirm["champion"] != "C0" else confirm["challenger"]
    deltas = {p: r[lead]["pr_auc"] - r["C0"]["pr_auc"] for p, r in rows.items() if r["evaluable"]}
    patient = figures.demo_patients(deltas)[args.which]
    row = rows[patient]

    t1 = _arr(rhuh.files(patient, "postop")["T1ce"])
    brain = _arr(LF / patient / "brain_mask.nii.gz") > 0
    target = _arr(LF / patient / "target_mask.nii.gz") > 0
    evaluate = brain & ~target
    recurrence = (
        np.isin(
            _arr(LF / patient / "recurrence_labels.nii.gz"),
            [
                int(k)
                for k, v in json.loads(
                    (REPO_ROOT / "results/rhuh_inventory/inventory.json").read_text(
                        encoding="utf-8"
                    )
                )["mapping"]["decision"]["mapping"].items()
                if v == "enhancing"
            ],
        )
        & evaluate
    )
    n = int(row["matched_voxels"]["margin15"])
    maps = {
        c: _arr(CONFIRM / "maps" / patient / row["maps"][f"{c}_map"]) for c in ("C0T", "C0", lead)
    }
    margin = top_region(maps["C0T"], evaluate, n)
    dist = top_region(maps["C0"], evaluate, n)
    model = top_region(maps[lead], evaluate, n)
    z = int(np.argmax(recurrence.sum(axis=(1, 2))))
    coverages = {
        "margin": row["C0T"]["coverage_margin15"],
        "distance": row["C0"]["coverage_margin15"],
        "champion": row[lead]["coverage_margin15"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with run_manifest(OUT) as m:
        figures.prediction_vs_reality(
            t1[z],
            target[z],
            margin[z],
            dist[z],
            model[z],
            recurrence[z],
            coverages,
            lead,
            OUT / f"prediction_vs_reality_{args.which}",
        )
        m.record(patient=patient, rule=args.which, slice_z=z, lead=lead)
    print(f"{args.which}: {patient}, slice {z}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
