#!/usr/bin/env python
"""MU-Glioma-Post eligibility from its public clinical spreadsheet (D18, D19; registration
section 8). Reads no image and no label.

Applies the pre-registered rules (``gbmc.mu.eligibility``): IDH-wildtype glioblastoma, no
earlier brain tumour, imaging-confirmed first progression, a baseline MRI at least 56 days
before it, and a recurrence MRI from 14 days before to 60 days after it (the closest to the
progression date). Fixes the patient list and each patient's baseline and recurrence
timepoints before any MU image is downloaded.

Writes ``results/mu_eligibility/eligibility.json`` (rows, funnel, SHA-256 of the sheet).
Refuses a dirty tree.

    python scripts/25_mu_eligibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gbmc import config, mu
from gbmc.provenance import file_digest, require_clean_tree, run_manifest
from gbmc.xlsx import read_sheet

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "mu_eligibility"


def main() -> int:
    require_clean_tree()
    sheet = REPO_ROOT / config.get("data.mu_glioma_post.metadata_dir") / mu.CLINICAL_XLSX
    rows = mu.eligibility(read_sheet(sheet, mu.CLINICAL_SHEET))
    funnel = mu.funnel(rows)
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml", sheet]) as m:
        m.record(**{k: v for k, v in funnel.items() if k != "excluded_by_step"})
        (OUT / "eligibility.json").write_text(json.dumps(
            {"source": mu.CLINICAL_XLSX, "source_sha256": file_digest(sheet),
             "labels_read": False, "images_read": False, "funnel": funnel,
             "eligible": [r.patient for r in rows if r.eligible], "rows": mu.as_dicts(rows)},
            indent=2) + "\n", encoding="utf-8")
    print(json.dumps(funnel, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
