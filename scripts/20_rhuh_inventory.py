#!/usr/bin/env python
"""RHUH-GBM download record, inventory and label checks (plan Steps 3.1-3.2, rules from 2.2).

Refuses until the OSF timestamp and the verified file layout are recorded
(``config.yaml:data.rhuh_gbm``). Computes no model score and reads no
recurrence label beyond what the checks below need.

1. ``--record-download <archive>``: SHA-256 of the downloaded archive, to be
   committed with the download date (which must be after the OSF timestamp).
2. Inventory: early post-op (< 72 h) and recurrence scans present, each with
   T1ce, FLAIR and a segmentation; the secondary-facility covariate.
3. Label mapping re-derived from intensities on every scan carrying all three
   labels; ``gbmc.rhuh.check_mapping`` may only correct a swap.
4. Cavity inspection on ``rhuh.cavity_inspection_n`` seeded patients: label
   values in the fluid-dark T1ce voxels of the pre-op core, on the early post-op scan.

Writes ``results/rhuh_inventory/``.

    python scripts/20_rhuh_inventory.py --record-download RHUH-GBM.zip
    python scripts/20_rhuh_inventory.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, freeze, rhuh
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "rhuh_inventory"


def _array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record-download", type=Path, default=None)
    args = ap.parse_args()
    require_clean_tree()
    lay = rhuh.layout()  # raises until the layout is verified
    master = int(config.get("seeds.master"))

    if args.record_download:
        OUT.mkdir(parents=True, exist_ok=True)
        record = {
            "archive": args.record_download.name,
            "sha256": freeze.sha256(args.record_download),
            "bytes": args.record_download.stat().st_size,
            "download_date": "TODO: fill in by hand and commit with this file",
        }
        (OUT / "download.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        return 0

    import pandas as pd

    cols = lay["clinical_columns"]
    table = pd.read_csv(Path(lay["local_root"]) / lay["clinical_csv"]).set_index(cols["patient"])
    enrolled = rhuh.patients()
    rows, evidence = [], []
    for p in enrolled:
        present = {role: set(rhuh.files(p, role)) for role in rhuh.ROLES if rhuh.files(p, role)}
        hours = table.loc[p, cols["postop_hours"]] if cols.get("postop_hours") else None
        facility = (
            table.loc[p, cols["secondary_facility"]] if cols.get("secondary_facility") else None
        )
        row = rhuh.inventory_row(
            p,
            present,
            None if pd.isna(hours) else float(hours),
            None if facility is None or pd.isna(facility) else bool(facility),
        )
        rows.append(asdict(row) | {"usable": row.usable})
        for role in ("preop", "postop", "recurrence"):
            f = rhuh.files(p, role)
            if {"seg", "T1ce", "FLAIR"} <= set(f):
                t1 = _array(f["T1ce"])
                evidence.append(
                    rhuh.derive_mapping(
                        f"{p}:{role}", _array(f["seg"]), t1, _array(f["FLAIR"]), t1 != 0
                    )
                )

    decision = rhuh.check_mapping(evidence, float(config.get("rhuh.mapping_min_agreement")))

    usable = [r["patient"] for r in rows if r["usable"]]
    n = int(config.get("rhuh.cavity_inspection_n"))
    pick = sorted(
        np.random.default_rng(stable_seed("rhuh_cavity", master_seed=master))
        .choice(usable, size=min(n, len(usable)), replace=False)
        .tolist()
    )
    cavity = {}
    for p in pick:
        pre, post = rhuh.files(p, "preop"), rhuh.files(p, "postop")
        core = rhuh.preop_core(_array(pre["seg"]), decision.mapping)
        t1 = _array(post["T1ce"])
        # Both timepoints are in SRI24; the rule's rigid carry is applied in 21_.
        # For the inspection, the pre-op core is read in the shared atlas frame.
        cavity[p] = rhuh.cavity_label_counts(
            _array(post["seg"]), t1, core, t1 != 0, float(config.get("rhuh.cavity_dark_t1ce_z"))
        )

    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(n_enrolled=len(enrolled), n_usable=len(usable), master_seed=master)
        (OUT / "inventory.json").write_text(
            json.dumps(
                {
                    "enrolled": enrolled,
                    "rows": rows,
                    "mapping": {
                        "decision": asdict(decision),
                        "per_scan": [asdict(e) for e in evidence],
                    },
                    "cavity_inspection": {
                        "patients": pick,
                        "label_counts_in_dark_core": cavity,
                        "decision": "TODO: read the counts; record in DECISIONS.md",
                    },
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"enrolled {len(enrolled)}, usable {len(usable)}; mapping: {decision.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
