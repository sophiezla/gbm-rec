#!/usr/bin/env python
"""Compare recomputed feature caches with the earlier ones (deviation V3). Scores nothing.

Reads only feature columns and masks; no candidate is fitted and no metric is computed.

    python scripts/10_compare_feature_caches.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gbmc.features import COLUMNS
from gbmc.lumiere import cohort
from gbmc.provenance import require_clean_tree, run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD = REPO_ROOT / "data" / "interim" / "features_v1"
NEW = REPO_ROOT / "data" / "interim" / "features"
OUT = REPO_ROOT / "results" / "feature_recompute"


def compare(patient: str) -> dict:
    with np.load(OLD / f"{patient}.npz") as a, np.load(NEW / f"{patient}.npz") as b:
        if not np.array_equal(a["evaluate_packed"], b["evaluate_packed"]):
            raise ValueError(f"{patient}: the evaluation mask changed")
        if not np.array_equal(a["truth"], b["truth"]):
            raise ValueError(f"{patient}: the labels changed")
        delta = np.abs(a["matrix"] - b["matrix"])
        row = {"patient": patient, "voxels": int(a["truth"].size),
               "matched_voxels_old": dict(zip(a["matched_labels"].tolist(),
                                              a["matched_voxels"].tolist(), strict=True)),
               "matched_voxels_new": dict(zip(b["matched_labels"].tolist(),
                                              b["matched_voxels"].tolist(), strict=True)),
               "columns": {}}
    for index, name in enumerate(COLUMNS):
        d = delta[:, index]
        row["columns"][name] = {"max": float(d.max()), "mean": float(d.mean()),
                                "share_changed_over_0_01": float((d > 0.01).mean())}
    return row


def main() -> int:
    global OLD, NEW
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, default=OLD)
    ap.add_argument("--new", type=Path, default=NEW)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    OLD, NEW, out = args.old, args.new, args.out
    require_clean_tree()
    out.mkdir(parents=True, exist_ok=True)
    with run_manifest(out):
        rows = [compare(p) for p in cohort()]
        (out / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for name in COLUMNS:
        worst = max(r["columns"][name]["max"] for r in rows)
        moved = sum(r["columns"][name]["share_changed_over_0_01"] > 0 for r in rows)
        print(f"{name}: worst change {worst:.4f}; patients with any change over 0.01: {moved}")
    print("evaluation masks and labels identical for all", len(rows), "patients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
