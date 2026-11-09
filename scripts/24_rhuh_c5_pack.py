#!/usr/bin/env python
"""Pack C5's RHUH-GBM inputs for the GPU computer (D6 item 9). No label, no score, no PyTorch.

Runs on the Mac after the cohort lock (``23_``) and before the confirmatory run (``30_``).
For every patient the primary or a sensitivity arm will score, and for each C5 input
variant (``gbmc.c5_import.RHUH_VARIANTS``), writes
``data/interim/rhuh/c5_inputs/<variant>/<patient>.npz`` (not in git) with the same five
channels as on LUMIERE (``12_c5_pack.py``): T1ce and FLAIR z-scored inside the brain,
target mask, FLAIR mask (peritumoral label) and the f1 map, cropped to the brain.

The patient is loaded by ``gbmc.confirm.load_patient`` with ``with_recurrence=False``:
the recurrence file is never opened and the packed label is all False. The evaluation
region is ``brain & ~target`` from the same loader ``30_`` uses, so C5's scores line up
row for row with every other candidate's.

``manifest.json`` per variant records each file's SHA-256, its evaluation-voxel count and
the commit. The GPU computer scores these with the registered release model
(``14_c5_score_rhuh.py``); ``gbmc.c5_import.rhuh_scores`` refuses maps made from anything
else.

    python scripts/24_rhuh_c5_pack.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import c5, c5_import, config, confirm, features, rhuh
from gbmc.imaging import zscore_array
from gbmc.provenance import file_digest, git_state, require_clean_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


def pack_patient(patient: str, variant: str, spec: dict, mapping: dict[int, str],
                 master: int) -> dict:
    target = (confirm.alt_cavity_target(patient, mapping, spec["label_dir"], master)
              if variant == "alt_cavity" else None)
    inp = confirm.load_patient(patient, mapping, spec["label_dir"], target=target,
                               with_recurrence=False)
    post = rhuh.files(patient, "postop")
    t1ce = sitk.GetArrayFromImage(inp.reference)
    flair = sitk.GetArrayFromImage(sitk.ReadImage(str(post["FLAIR"]), sitk.sitkFloat32))
    evaluate = inp.brain & ~inp.target
    f1 = features.signed_distance_mm(inp.target, inp.reference)
    if f1 is None:
        raise SystemExit(f"{patient} ({variant}): empty target")
    return c5.pack(patient, zscore_array(t1ce, inp.brain), zscore_array(flair, inp.brain),
                   inp.target, inp.flair_abnormal, f1, evaluate,
                   np.zeros(int(evaluate.sum()), bool),
                   c5.lr_axis_from_direction(inp.reference.GetDirection()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(c5_import.RHUH_VARIANTS),
                    choices=c5_import.RHUH_VARIANTS)
    args = ap.parse_args()
    require_clean_tree()
    config.pin_itk_threads()
    rhuh.layout()
    master = int(config.get("seeds.master"))
    inv = json.loads((REPO_ROOT / "results/rhuh_inventory/inventory.json")
                     .read_text(encoding="utf-8"))
    mapping = {int(k): v for k, v in inv["mapping"]["decision"]["mapping"].items()}
    lock = json.loads((REPO_ROOT / "results/rhuh_lock/consort.json").read_text(encoding="utf-8"))
    arms = confirm.sensitivity_arms(REPO_ROOT)
    label_dirs = {spec["c5"]: spec["label_dir"] for spec in arms.values()}
    state = git_state()

    for variant, patients in confirm.c5_variants(arms, lock["evaluable"]).items():
        if variant not in args.variants:
            continue
        out = c5_import.RHUH_INPUTS / variant
        out.mkdir(parents=True, exist_ok=True)
        hashes, counts = {}, {}
        for patient in patients:
            packed = pack_patient(patient, variant, {"label_dir": label_dirs[variant]},
                                  mapping, master)
            path = out / f"{patient}.npz"
            np.savez_compressed(path, **packed)
            hashes[patient] = file_digest(path)
            counts[patient] = int(packed["evaluate"].sum())
            print(f"{variant} {patient}", flush=True)
        (out / "manifest.json").write_text(json.dumps(
            {"commit": state["commit"], "dirty": state["dirty"], "variant": variant,
             "label_dir": str(label_dirs[variant].relative_to(REPO_ROOT)),
             "labels": "none: the recurrence file is never opened; label is all False",
             "sha256": hashes, "n_evaluate": counts, "channels": list(c5.CHANNELS),
             "f1_scale_mm": c5.F1_SCALE_MM}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
