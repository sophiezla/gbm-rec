#!/usr/bin/env python
"""Merge a bundle_results_<pc>_<date>.zip from the second PC (gbm-wm 17_ pattern).

Never overwrites a unit already here. A unit present on both sides is compared
file by file, and any difference is reported and makes the exit code 1:
the runs are seeded and single-threaded, so the two machines should agree
bit for bit (gbm-4d's cross-machine check found 170 of 170 identical).

    python scripts/09_import_bundle.py <zip> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
UNIT_GLOBS = ("label_field_later/*/*", "roundtrip_tuned/*")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(root: Path, dry_run: bool) -> dict[str, list[str]]:
    """Merge every unit under ``root/data/interim`` into this repository."""
    out = {"added": [], "identical": [], "differing": []}
    for pattern in UNIT_GLOBS:
        for unit in sorted((root / "data" / "interim").glob(pattern)):
            if not (unit / "qc.json").is_file():
                continue
            rel = unit.relative_to(root / "data" / "interim")
            local = INTERIM / rel
            if (local / "qc.json").is_file():
                same = all((local / f.name).is_file() and digest(local / f.name) == digest(f)
                           for f in unit.iterdir() if f.is_file())
                out["identical" if same else "differing"].append(rel.as_posix())
                continue
            out["added"].append(rel.as_posix())
            if not dry_run:
                shutil.copytree(unit, local)
    schedule = root / "data" / "interim" / "label_field_later" / "schedule.json"
    if schedule.is_file() and not dry_run:
        (INTERIM / "label_field_later").mkdir(parents=True, exist_ok=True)
        shutil.copy2(schedule, INTERIM / "label_field_later" / "schedule.json")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="bundle_import_") as tmp:
        if args.source.suffix.lower() == ".zip":
            with zipfile.ZipFile(args.source) as z:
                z.extractall(tmp)
            root = Path(tmp)
        else:
            root = args.source
        result = merge(root, args.dry_run)
        if not args.dry_run and (root / "logs").is_dir():
            dest = REPO_ROOT / "logs" / "remote" / args.source.stem
            shutil.copytree(root / "logs", dest, dirs_exist_ok=True)
    print(f"added {len(result['added'])}, identical {len(result['identical'])}, "
          f"DIFFERING {len(result['differing'])} {result['differing']}")
    return 1 if result["differing"] else 0


if __name__ == "__main__":
    sys.exit(main())
