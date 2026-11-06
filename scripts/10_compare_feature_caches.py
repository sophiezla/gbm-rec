#!/usr/bin/env python
"""Compare recomputed feature caches with the earlier ones (deviation V3). Scores nothing.

Reads only feature columns and masks; no candidate is fitted and no metric is computed.

    python scripts/10_compare_feature_caches.py
"""

from __future__ import annotations

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
        shape = tuple(int(n) for n in a["evaluate_shape"])
        old = np.unpackbits(a["evaluate_packed"])[:int(np.prod(shape))].astype(bool).reshape(shape)
        new = np.unpackbits(b["evaluate_packed"])[:int(np.prod(shape))].astype(bool).reshape(shape)
        keep = new[old]
        delta = np.abs(a["matrix"][keep] - b["matrix"])
        kept_truth = a["truth"][keep]
        row = {"patient": patient, "voxels_old": int(old.sum()), "voxels_new": int(new.sum()),
               "voxels_dropped": int((old & ~new).sum()),
               "positives_old": int(a["truth"].sum()), "positives_new": int(b["truth"].sum()),
               "labels_identical_on_shared_voxels": bool(np.array_equal(kept_truth, b["truth"])),
               "columns": {}}
    for index, name in enumerate(COLUMNS):
        d = delta[:, index]
        row["columns"][name] = {"max": float(d.max()), "mean": float(d.mean()),
                                "share_changed_over_0_01": float((d > 0.01).mean())}
    return row


def main() -> int:
    require_clean_tree()
    OUT.mkdir(parents=True, exist_ok=True)
    with run_manifest(OUT):
        rows = [compare(p) for p in cohort()]
        (OUT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for name in COLUMNS:
        worst = max(r["columns"][name]["max"] for r in rows)
        moved = sum(r["columns"][name]["share_changed_over_0_01"] > 0 for r in rows)
        print(f"{name}: worst change {worst:.4f}; patients with any change over 0.01: {moved}")
    print("voxels dropped, total:", sum(r["voxels_dropped"] for r in rows),
          "| positives dropped, total:", sum(r["positives_old"] - r["positives_new"] for r in rows),
          "| labels identical:", all(r["labels_identical_on_shared_voxels"] for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
