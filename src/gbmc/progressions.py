"""Later RANO progressions for the exploratory fourth-dimension analysis (plan Step 1.8).

gbm-4d transported only the FIRST segmented progression. The second and third
are selected here with gbm-4d's own rules (``gbm4d.cohort``): a timepoint rated
``PD``, after the first progression, carrying a DeepBraTumIA segmentation.

Also hosts :func:`frozen_gbm4d_script`, which loads one of gbm-4d's numbered
scripts from the checkout (scripts are not in the installed package) after
checking that the checkout is at the pinned commit, and redirects its output
roots into this repository so the frozen project is never written to.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from gbmc.lumiere import gbm4d_root, lumiere_root

#: The commit pyproject.toml pins (D4).
PINNED_GBM4D = "eeba7779804412b0a32d0b5ba0f23c3156448549"

#: Plan Step 1.8: "the second and third progression".
N_LATER = 2


def later_progressions(ratings, completeness, patient: str, first_progression: str,
                       n_later: int = N_LATER) -> list[str]:
    """PD timepoints after ``first_progression`` with a DeepBraTumIA segmentation."""
    from gbm4d.cohort import timepoint_sort_key

    first = timepoint_sort_key(first_progression)
    rows = ratings[(ratings["Patient"] == patient) & (ratings["Rating"] == "PD")]
    out = []
    for tp in sorted(rows["Timepoint"], key=timepoint_sort_key):
        if timepoint_sort_key(tp) <= first:
            continue
        key = (patient, tp)
        if key in completeness.index and bool(completeness.loc[key].get("DeepBraTumIA", False)):
            out.append(tp)
    return out[:n_later]


def schedule(patients: list[str]) -> list[dict]:
    """(patient, baseline, first progression, later progressions) for each patient."""
    from gbm4d import cohort

    from gbmc.lumiere import transport_qc

    md = lumiere_root() / "metadata"
    ratings = cohort.load_ratings(md)
    completeness = cohort.load_completeness(md).set_index(["Patient", "Timepoint"])
    out = []
    for p in patients:
        qc = transport_qc(p)
        out.append({"patient": p, "baseline": qc["baseline"], "first": qc["progression"],
                    "later": later_progressions(ratings, completeness, p, qc["progression"])})
    return out


def checkout_commit() -> str:
    return subprocess.run(["git", "-C", str(gbm4d_root()), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


#: Written into a vendored gbm-4d copy (second-machine bundle) in place of git.
VENDORED_MANIFEST = "VENDORED.json"


def vendored_files(root: Path) -> list[Path]:
    """The gbm-4d files whose integrity a vendored copy must prove."""
    return sorted([*(root / "scripts").glob("*.py"), *(root / "src" / "gbm4d").glob("*.py"),
                   root / "config.yaml"])


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_vendored_manifest(root: Path) -> None:
    """Record the hashes of a vendored copy, taken from the verified checkout."""
    files = {f.relative_to(root).as_posix(): _sha256(f) for f in vendored_files(root)}
    (root / VENDORED_MANIFEST).write_text(
        json.dumps({"commit": PINNED_GBM4D, "sha256": files}, indent=2) + "\n", encoding="utf-8")


def verify_frozen_gbm4d() -> str:
    """Prove the gbm-4d code about to run is the pinned code. Returns how it was proven.

    A git checkout must be at the pinned commit with ``scripts/`` and ``src/``
    unmodified. A vendored copy (no ``.git``) must match the hashes in its
    ``VENDORED.json``, which the bundle packer writes from a verified checkout.
    """
    root = gbm4d_root()
    if (root / ".git").exists():
        if (head := checkout_commit()) != PINNED_GBM4D:
            raise RuntimeError(f"gbm-4d checkout is at {head}, not the pinned {PINNED_GBM4D}")
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--",
                                "scripts", "src"], capture_output=True, text=True, check=True)
        if dirty.stdout.strip():
            raise RuntimeError(f"gbm-4d scripts/ or src/ modified:\n{dirty.stdout}")
        return "git"
    manifest = root / VENDORED_MANIFEST
    if not manifest.is_file():
        raise RuntimeError(f"{root} is neither a git checkout nor a vendored copy")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("commit") != PINNED_GBM4D:
        raise RuntimeError(f"vendored gbm-4d is {record.get('commit')}, not {PINNED_GBM4D}")
    for rel, digest in record["sha256"].items():
        if _sha256(root / rel) != digest:
            raise RuntimeError(f"vendored gbm-4d file modified: {rel}")
    return "vendored"


def frozen_gbm4d_script(stem: str, redirect: dict[str, Path]) -> ModuleType:
    """Load ``gbm-4d/scripts/<stem>.py`` with its output roots redirected.

    Refuses unless :func:`verify_frozen_gbm4d` passes, so the code that runs is
    the frozen code. gbm4d's config is pointed at the gbm-4d copy's own
    ``config.yaml``, which the installed package cannot find (D3), and its
    LUMIERE root at this project's, so both machines read the same layout.
    """
    import os

    verify_frozen_gbm4d()
    root = gbm4d_root()
    os.environ["GBM4D_LUMIERE_ROOT"] = str(lumiere_root())
    from gbm4d import config as g4config
    g4config.CONFIG_PATH = root / "config.yaml"
    g4config.load.cache_clear()

    spec = importlib.util.spec_from_file_location(f"_gbm4d_{stem}",
                                                  root / "scripts" / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, path in redirect.items():
        if not hasattr(module, name):
            raise AttributeError(f"{stem} has no {name} to redirect")
        setattr(module, name, Path(path))
    return module
