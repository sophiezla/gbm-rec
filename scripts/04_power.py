#!/usr/bin/env python
"""Minimum detectable effect for the confirmatory primary (plan Step 2.1).

Seeded from C0's LUMIERE per-patient PR-AUC alone (``results/tournament/``),
under the real decision rule: BCa patient-bootstrap lower bound above zero
(``gbmc.power``). Writes ``results/power/power_table.json`` for the
pre-registration. If the MDE at the expected n is above 0.05, the plan says the
realistic headline is a bound, not a win.

    python scripts/04_power.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gbmc import config, power
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_PATIENT = REPO_ROOT / "results" / "tournament" / "per_patient.json"
OUT = REPO_ROOT / "results" / "power"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    require_clean_tree()
    master = int(config.get("seeds.master"))
    if not PER_PATIENT.is_file():
        raise SystemExit("no tournament results: run 02_ first")
    rows = json.loads(PER_PATIENT.read_text(encoding="utf-8"))
    c0 = np.array([r["C0"]["pr_auc"] for r in rows.values() if r["evaluable"]], dtype=float)

    cfg = config.get("power")
    table = []
    for n in cfg["n_evaluable"]:
        for rho in cfg["rho"]:
            mde, curve = power.minimum_detectable_effect(
                c0, int(n), float(rho), stable_seed("power", n, rho, master_seed=master),
                cfg["delta_grid"], target_power=float(cfg["target_power"]),
                n_sim=int(cfg["n_sim"]), n_boot=int(config.get("seeds.n_bootstrap")),
                alpha=float(cfg["alpha"]))
            table.append({"n": int(n), "rho": float(rho), "mde": mde,
                          "curve": [asdict(p) for p in curve]})
            print(f"n={n} rho={rho}: MDE {mde}", flush=True)

    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml", PER_PATIENT]) as manifest:
        manifest.record(master_seed=master, n_c0_patients=int(c0.size),
                        c0_mean=float(c0.mean()), c0_sd=float(c0.std(ddof=1)))
        (OUT / "power_table.json").write_text(json.dumps(
            {"decision_rule": "BCa lower bound > 0", "c0_n": int(c0.size), "table": table},
            indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
