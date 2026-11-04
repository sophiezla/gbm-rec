#!/usr/bin/env python
"""Lock the RHUH-GBM cohort (plan Steps 3.5-3.6): CONSORT table, evaluable n, achieved MDE.

Each patient gets its first failing step, in the plan's order: missing
sequence or segmentation (20_), failed transport (21_), screen failure (22_),
empty target, no enhancing recurrence outside the target. The last is
reported as undefined, never scored 0.

Checking whether recurrence lies outside the target reads the transported
labels; that is a count, not a model score, and is the same check gbm-4d D18
made. Writes ``results/rhuh_lock/`` with ``consort.json``, ``consort.md`` and
the achieved MDE at the evaluable n (``gbmc.power`` seeded from C0's LUMIERE
scores, the same simulation as the pre-registered table).

After this, and before Phase 4: commit, and record the evaluable n and the
achieved MDE in HANDOFF.md.

    python scripts/23_rhuh_lock.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, consort, power, rhuh
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
LF = REPO_ROOT / "data" / "interim" / "rhuh" / "label_field_tuned"
OUT = REPO_ROOT / "results" / "rhuh_lock"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def outcomes(enrolled: list[str], mapping: dict[int, str]) -> list[consort.Outcome]:
    inv = {
        r["patient"]: r for r in _json(REPO_ROOT / "results/rhuh_inventory/inventory.json")["rows"]
    }
    screen = {
        r["patient"]: r for r in _json(REPO_ROOT / "results/rhuh_screens/verdict.json")["rows"]
    }
    enhancing = next(v for v, n in mapping.items() if n == "enhancing")
    out = []
    for p in enrolled:
        row = inv.get(p)
        if row is None or not row["usable"]:
            out.append(
                consort.Outcome(p, consort.STEPS[0], "; ".join(row["reasons"]) if row else "absent")
            )
            continue
        qc_path = LF / p / "qc.json"
        qc = _json(qc_path) if qc_path.is_file() else {}
        if not qc.get("ok"):
            out.append(consort.Outcome(p, consort.STEPS[1], qc.get("note", "not transported")))
            continue
        if screen.get(p, {}).get("status") == "EXCLUDE":
            out.append(consort.Outcome(p, consort.STEPS[2], ", ".join(screen[p]["flags"])))
            continue
        target = sitk.GetArrayFromImage(sitk.ReadImage(str(LF / p / "target_mask.nii.gz"))) > 0
        if not target.any():
            out.append(consort.Outcome(p, consort.STEPS[3], ""))
            continue
        brain = sitk.GetArrayFromImage(sitk.ReadImage(str(LF / p / "brain_mask.nii.gz"))) > 0
        rec = sitk.GetArrayFromImage(sitk.ReadImage(str(LF / p / "recurrence_labels.nii.gz")))
        if not ((rec == enhancing) & brain & ~target).any():
            out.append(consort.Outcome(p, consort.STEPS[4], "undefined, not zero"))
            continue
        out.append(consort.Outcome(p, None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args()
    require_clean_tree()
    rhuh.layout()
    master = int(config.get("seeds.master"))
    inv = _json(REPO_ROOT / "results/rhuh_inventory/inventory.json")
    mapping = {int(k): v for k, v in inv["mapping"]["decision"]["mapping"].items()}
    table = consort.flow(inv["enrolled"], outcomes(inv["enrolled"], mapping))

    per_patient = _json(REPO_ROOT / "results/tournament/per_patient.json")
    c0 = np.array([r["C0"]["pr_auc"] for r in per_patient.values() if r["evaluable"]])
    cfg = config.get("power")
    achieved = {}
    for rho in cfg["rho"]:
        mde, _ = power.minimum_detectable_effect(
            c0,
            table["n_evaluable"],
            float(rho),
            stable_seed("power", table["n_evaluable"], rho, master_seed=master),
            cfg["delta_grid"],
            target_power=float(cfg["target_power"]),
            n_sim=int(cfg["n_sim"]),
            n_boot=int(config.get("seeds.n_bootstrap")),
            alpha=float(cfg["alpha"]),
        )
        achieved[str(rho)] = mde

    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(n_evaluable=table["n_evaluable"], achieved_mde=achieved)
        (OUT / "consort.json").write_text(
            json.dumps(table | {"achieved_mde": achieved}, indent=2) + "\n", encoding="utf-8"
        )
        (OUT / "consort.md").write_text(consort.markdown(table), encoding="utf-8")
    print(consort.markdown(table))
    print(f"achieved MDE by rho: {achieved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
