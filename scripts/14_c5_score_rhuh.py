#!/usr/bin/env python
"""Score RHUH-GBM with the registered C5 release model (GPU computer; D6 item 9). Phase 4 input.

Needs PyTorch (``requirements-gpu.txt``). Runs from a clean commit that includes the
committed release register ``results/release/model_hashes.json`` (written by ``03_`` on
the Mac). Steps:

1. The model file (default ``data/interim/c5_release/model.pt``, the one this computer
   trained) must match the register's SHA-256 for ``models/C5.pt``; otherwise it stops.
2. Every packed input (``data/interim/rhuh/c5_inputs/<variant>/``, copied from the Mac's
   ``24_rhuh_c5_pack.py``) must match its manifest's SHA-256. The inputs hold no label.
3. Sliding-window prediction with the fixed settings (``config.yaml:c5``); one score per
   evaluation voxel, written to ``data/interim/rhuh/c5_maps/<variant>/<patient>.npz``.
4. ``manifest.json`` per variant: the model's SHA-256, the inputs manifest's hashes, every
   map's SHA-256, the commit and the device. Copy ``c5_maps/`` back to the Mac, where
   ``gbmc.c5_import.rhuh_scores`` checks all of it before ``30_``/``31_`` score anything.

Computes no metric and reads no label. Runs on CPU too (``--device cpu``), more slowly.

    python scripts/14_c5_score_rhuh.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA starts

import numpy as np  # noqa: E402

from gbmc import c5, c5_import  # noqa: E402
from gbmc.provenance import file_digest, git_state, require_clean_tree  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTER = REPO_ROOT / "results" / "release" / "model_hashes.json"


def load_model(path: Path, expected: str, settings: c5.Settings, device: str):
    import torch

    if file_digest(path) != expected:
        raise SystemExit(f"{path} does not match the registered C5 hash {expected}")
    model = c5.build_unet(settings).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", type=Path,
                    default=REPO_ROOT / "data" / "interim" / "c5_release" / "model.pt")
    ap.add_argument("--variants", nargs="+", default=list(c5_import.RHUH_VARIANTS),
                    choices=c5_import.RHUH_VARIANTS)
    args = ap.parse_args()
    require_clean_tree()
    if not REGISTER.is_file():
        raise SystemExit(f"{REGISTER} is missing: run 03_ on the Mac, commit, and git pull")
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    expected = register["models"][c5_import.RELEASE_FILE]
    settings = c5.Settings.from_config()
    c5.set_determinism(0)          # inference only; the flags matter, the seed does not
    model = load_model(args.model, expected, settings, args.device)
    state = git_state()

    for variant in args.variants:
        inputs = c5_import.RHUH_INPUTS / variant
        packed = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
        if packed.get("dirty", True):
            raise SystemExit(f"C5 inputs ({variant}) were packed from uncommitted code")
        out = c5_import.RHUH_MAPS / variant
        out.mkdir(parents=True, exist_ok=True)
        maps = {}
        for patient, digest in sorted(packed["sha256"].items()):
            path = inputs / f"{patient}.npz"
            if file_digest(path) != digest:
                raise SystemExit(f"{patient} ({variant}): input does not match its hash")
            score = c5.predict(model, c5.load_case(path), settings, args.device)
            if score.shape != (packed["n_evaluate"][patient],):
                raise SystemExit(f"{patient} ({variant}): wrong number of scores")
            target = out / f"{patient}.npz"
            np.savez(target, score=score)
            maps[patient] = file_digest(target)
            print(f"{variant} {patient}", flush=True)
        (out / "manifest.json").write_text(json.dumps(
            {"commit": state["commit"], "dirty": state["dirty"], "smoke": False,
             "variant": variant, "model_sha256": expected,
             "inputs_sha256": packed["sha256"], "sha256": maps, "device": args.device},
            indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
