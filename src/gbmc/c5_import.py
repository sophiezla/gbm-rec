"""Importing C5's files from the GPU computer onto the Mac (DECISIONS.md D6 item 9). No PyTorch.

The GPU computer trains C5 and runs its network; this Mac checks everything it brings back
before any of it touches a reportable number. Three imports, each refusing on any mismatch:

* held-out tournament maps: :func:`gbmc.tournament.c5_held_out` (Phase 1);
* the release model, trained on all 63 patients: :func:`import_release` (``03_``, Step 1.7).
  It must come from a clean, non-smoke run with today's fixed settings and the same
  ``gbmc/c5.py`` as the tournament maps, and its file is copied into the release register;
* RHUH-GBM maps: :func:`rhuh_scores` (``30_`` and ``31_``). They must be made by the
  registered release model from the inputs this Mac packed (``24_rhuh_c5_pack.py``).

Only the network's own code is compared between commits (``C5_CODE``), so documentation
commits between the tournament and the release do not block the import.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from gbmc.provenance import file_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
C5_CODE = ("src/gbmc/c5.py",)
RELEASE_FILE = "models/C5.pt"
RHUH_INPUTS = REPO_ROOT / "data" / "interim" / "rhuh" / "c5_inputs"
RHUH_MAPS = REPO_ROOT / "data" / "interim" / "rhuh" / "c5_maps"
#: The C5 input sets RHUH-GBM needs. ``tuned`` serves the primary and every arm that only
#: changes the patient list or the positive label; the other two change C5's own inputs.
RHUH_VARIANTS = ("tuned", "original", "alt_cavity")


def _read(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def same_code(commit_a: str, commit_b: str, paths=C5_CODE, repo: Path = REPO_ROOT) -> bool:
    """True when ``paths`` are byte-identical at both commits (both must exist locally)."""
    if commit_a == commit_b:
        return True
    for commit in (commit_a, commit_b):
        if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo,
                          capture_output=True).returncode != 0:
            raise SystemExit(f"commit {commit} is not in this clone; git pull first")
    return subprocess.run(["git", "diff", "--quiet", commit_a, commit_b, "--", *paths],
                          cwd=repo).returncode == 0


def check_run(manifest: dict, what: str) -> None:
    if manifest.get("dirty", True):
        raise SystemExit(f"{what} was made from uncommitted code; refusing")
    if manifest.get("smoke", True):
        raise SystemExit(f"{what} is from a smoke run, not the fixed settings; refusing")


def import_release(release_dir: Path, dest_root: Path, oof_manifest: Path, settings: dict,
                   patients: list[str], code_check=same_code) -> tuple[str, str, dict]:
    """Copy the checked release model to ``dest_root/models/C5.pt``.

    Returns ``(relative path, sha256, provenance)`` for the release register.
    """
    manifest = _read(release_dir / "manifest.json")
    check_run(manifest, "the C5 release model")
    if manifest.get("settings") != settings:
        raise SystemExit("the C5 release model was trained with other settings than "
                         "config.yaml:c5; refusing")
    if sorted(manifest.get("patients", [])) != sorted(patients):
        raise SystemExit("the C5 release model was not trained on exactly the locked cohort")
    oof = _read(oof_manifest)
    if not code_check(oof["commit"], manifest["commit"]):
        raise SystemExit("gbmc/c5.py differs between the tournament maps and the release "
                         "model; refusing")
    source = release_dir / "model.pt"
    if not source.is_file() or file_digest(source) != manifest.get("model_sha256"):
        raise SystemExit("the C5 release model does not match its recorded hash")
    dest = dest_root / RELEASE_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    digest = file_digest(dest)
    if digest != manifest["model_sha256"]:
        raise SystemExit("the C5 release model changed while it was copied")
    provenance = {k: manifest.get(k) for k in ("commit", "torch", "device", "gpu", "cuda")}
    return RELEASE_FILE, digest, provenance | {"tournament_maps_commit": oof["commit"]}


def rhuh_scores(variant: str, patients: list[str], register_sha256: str,
                inputs: Path = RHUH_INPUTS, maps: Path = RHUH_MAPS) -> dict[str, np.ndarray]:
    """C5's RHUH-GBM scores for one input variant, checked end to end.

    The maps manifest must name the registered release model's SHA-256 and the exact input
    files this Mac packed; every map must match its own hash and its input's voxel count.
    """
    if variant not in RHUH_VARIANTS:
        raise ValueError(f"unknown C5 input variant {variant!r}")
    packed = _read(inputs / variant / "manifest.json")
    made = _read(maps / variant / "manifest.json")
    check_run(packed | {"smoke": False}, f"C5 inputs ({variant})")
    check_run(made, f"C5 RHUH-GBM maps ({variant})")
    if made.get("model_sha256") != register_sha256:
        raise SystemExit("C5 RHUH-GBM maps were not made by the registered release model")
    if made.get("inputs_sha256") != packed["sha256"]:
        raise SystemExit("C5 RHUH-GBM maps were made from other inputs than this Mac packed")
    scores = {}
    for patient in patients:
        path = maps / variant / f"{patient}.npz"
        if not path.is_file():
            raise SystemExit(f"C5 map missing for {patient} ({variant})")
        if file_digest(path) != made.get("sha256", {}).get(patient):
            raise SystemExit(f"C5 map for {patient} ({variant}) does not match its hash")
        with np.load(path) as z:
            score = z["score"].astype(np.float64)
        if score.shape != (packed["n_evaluate"][patient],):
            raise SystemExit(f"C5 map for {patient} ({variant}) has the wrong voxel count")
        scores[patient] = score
    return scores


def settings_dict() -> dict:
    """``config.yaml:c5`` as the training script records it."""
    from gbmc import c5

    return json.loads(json.dumps(dataclasses.asdict(c5.Settings.from_config())))
