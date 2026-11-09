#!/usr/bin/env python
"""Train C5 on the GPU computer and write its held-out maps (D6 items 8-9).

Runs on the GPU computer only, from a clean commit, with the inputs from ``12_c5_pack.py``
copied to ``data/interim/c5_inputs/``. Uses the same 7 folds as every other candidate and
``stable_seed("C5", fold)``. For each fold it trains from scratch for exactly the fixed
number of steps, keeps the last model, and scores the held-out patients. Writes
``data/interim/c5_oof/`` (not in git): ``<patient>.npz`` (``score`` per evaluation voxel),
``models/fold<k>.pt`` and, once all folds are done, ``manifest.json`` with the folds, the
SHA-256 of every map and model, the commit, the GPU and the PyTorch version. Copy that folder
to the same path on the Mac; ``02_`` checks it before scoring.

Resumable: a fold whose record (``folds/fold<k>.json``) is from the same commit is skipped.

    python scripts/13_c5_train.py                 # all 7 folds (CUDA)
    python scripts/13_c5_train.py --release       # Phase 2: one model on all 63 patients
    python scripts/13_c5_train.py --smoke --device cpu   # tiny settings, separate folder,
                                                         # never importable
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA starts

from gbmc import c5, candidates, config  # noqa: E402
from gbmc.lumiere import cohort  # noqa: E402
from gbmc.provenance import file_digest, git_state, require_clean_tree  # noqa: E402
from gbmc.rng import stable_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS = REPO_ROOT / "data" / "interim" / "c5_inputs"
SMOKE = c5.Settings(patch=(32, 32, 32), batch=2, epochs=2, steps_per_epoch=3,
                    levels=3, base_channels=4)


def load_inputs(patients: list[str]) -> dict[str, c5.Case]:
    manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dirty", True):
        raise SystemExit("C5 inputs were packed from uncommitted code; refusing")
    cases = {}
    for patient in patients:
        path = INPUTS / f"{patient}.npz"
        if file_digest(path) != manifest["sha256"].get(patient):
            raise SystemExit(f"{patient}: packed input does not match its recorded hash")
        cases[patient] = c5.load_case(path)
    return cases


def device_info(device: str) -> dict:
    import torch

    info = {"torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        info |= {"gpu": props.name, "gpu_memory_gb": round(props.total_memory / 1e9, 1),
                 "cuda": torch.version.cuda}
    return info


def save_model(model, path: Path) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return file_digest(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--release", action="store_true", help="train one model on all patients")
    ap.add_argument("--smoke", action="store_true", help="tiny settings; not a C5 result")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="only these folds")
    ap.add_argument("--patients", nargs="+", default=None, help="smoke only: a subset")
    args = ap.parse_args()

    if args.smoke:
        settings, out = SMOKE, REPO_ROOT / "data" / "interim" / "c5_smoke"
    else:
        require_clean_tree()
        settings = c5.Settings.from_config()
        out = REPO_ROOT / "data" / "interim" / ("c5_release" if args.release else "c5_oof")
    if args.patients and not args.smoke:
        raise SystemExit("--patients is for smoke runs only")
    master = int(config.get("seeds.master"))
    patients = args.patients or cohort()
    folds = candidates.patient_folds(cohort(), int(config.get("tournament.n_folds")),
                                     stable_seed("tournament_folds", master_seed=master))
    if args.patients:
        folds = [[p for p in f if p in patients] for f in folds]
    cases = load_inputs(patients)
    state = git_state()
    info = device_info(args.device)
    print(json.dumps(info), flush=True)

    if args.release:
        model = c5.train(list(cases.values()), settings,
                         stable_seed("C5_release", master_seed=master), args.device)
        digest = save_model(model, out / "model.pt")
        (out / "manifest.json").write_text(json.dumps(
            {"commit": state["commit"], "dirty": state["dirty"], "smoke": args.smoke,
             "settings": dataclasses.asdict(settings), "patients": sorted(cases),
             "model_sha256": digest} | info,
            indent=2) + "\n", encoding="utf-8")
        return 0

    for k, held_out in enumerate(folds):
        if args.folds is not None and k not in args.folds:
            continue
        record_path = out / "folds" / f"fold{k}.json"
        if record_path.is_file() and not args.smoke:
            if json.loads(record_path.read_text())["commit"] == state["commit"]:
                print(f"fold {k}: done at this commit, skipped")
                continue
        train = [cases[p] for p in patients if p not in held_out]
        start = time.time()
        model = c5.train(train, settings, stable_seed("C5", k, master_seed=master), args.device,
                         log=lambda m, k=k: print(f"fold {k}: {m}", flush=True))
        record = {"commit": state["commit"], "dirty": state["dirty"], "fold": k,
                  "model_sha256": save_model(model, out / "models" / f"fold{k}.pt"),
                  "maps": {}, "seconds": round(time.time() - start)}
        for patient in held_out:
            score = c5.predict(model, cases[patient], settings, args.device)
            path = out / f"{patient}.npz"
            import numpy as np

            np.savez(path, score=score)
            record["maps"][patient] = file_digest(path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"fold {k}: {len(held_out)} maps written ({record['seconds']} s)", flush=True)

    records = [out / "folds" / f"fold{k}.json" for k in range(len(folds))]
    if all(r.is_file() for r in records):
        loaded = [json.loads(r.read_text()) for r in records]
        commits = {r["commit"] for r in loaded}
        if len(commits) != 1:
            raise SystemExit(f"folds were trained at different commits: {sorted(commits)}")
        (out / "manifest.json").write_text(json.dumps(
            {"commit": commits.pop(), "dirty": any(r["dirty"] for r in loaded),
             "smoke": args.smoke, "folds": folds,
             "settings": dataclasses.asdict(settings),
             "sha256": {p: d for r in loaded for p, d in r["maps"].items()},
             "model_sha256": {f"fold{r['fold']}": r["model_sha256"] for r in loaded}} | info,
            indent=2) + "\n", encoding="utf-8")
        print(f"manifest written: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
