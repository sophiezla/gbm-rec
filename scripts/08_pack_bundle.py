#!/usr/bin/env python
# ruff: noqa: E501  (embedded .bat launchers cannot be wrapped)
"""Pack the long registration jobs into a folder another PC can run (gbm-4d 25_ / gbm-wm 16_ pattern).

Jobs:
  later      scripts/05_later_progressions.py  (Step 1.8 inputs; 83 timepoints)
  roundtrip  scripts/07_roundtrip_threshold.py (Step 3.3 threshold study; 63 pairs)

The bundle holds only what those two scripts read:

  repo/      this project's src/gbmc, the two scripts, config.yaml, pyproject.toml,
             requirements.lock
  gbm-4d/    a VENDORED copy of the pinned gbm-4d code (scripts, src, config.yaml,
             pyproject.toml) plus VENDORED.json with every file's SHA-256, written
             from this machine's verified checkout; the frozen loader checks it
             before running anything. Also the cohort lock, each patient's
             transport qc.json (baseline and first-progression names only), and the
             preprocessed baseline and first-progression folders.
  lumiere/   LUMIERE metadata, each baseline's native DeepBraTumIA segmentation,
             and the raw later-progression timepoints. All CC0.

No first-progression segmentation and no transported label is packed: the
threshold study reads no LUMIERE label, and the later-progression transport
reads only the later timepoints' own segmentations.

Launchers: RUN.bat <job> <workers> and PACK_RESULTS.bat. Import the returned
zip with scripts/09_import_bundle.py.

    python scripts/08_pack_bundle.py --zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

from gbmc import lumiere
from gbmc.progressions import PINNED_GBM4D, schedule, verify_frozen_gbm4d, write_vendored_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("05_later_progressions.py", "07_roundtrip_threshold.py")

RUN_BAT = r"""@echo off
setlocal
cd /d "%~dp0"
set JOB=%1
set WORKERS=%2
if "%JOB%"=="" set JOB=both
if "%WORKERS%"=="" set WORKERS=2
set GBMC_LUMIERE_ROOT=%~dp0lumiere
set GBMC_GBM4D_ROOT=%~dp0gbm-4d
call :ensure_env || exit /b 1
echo Each worker needs about 3 GB of RAM. Progress is saved after every unit;
echo closing this window stops it, and running this again resumes. Keep the PC awake.
if /i "%JOB%"=="later" goto later
if /i "%JOB%"=="roundtrip" goto roundtrip
:later
.venv\Scripts\python -u repo\scripts\05_later_progressions.py --workers %WORKERS% >> logs\05_later.log 2>&1
if /i "%JOB%"=="later" goto done
:roundtrip
.venv\Scripts\python -u repo\scripts\07_roundtrip_threshold.py --workers %WORKERS% >> logs\07_roundtrip.log 2>&1
:done
echo Finished or stopped. Run PACK_RESULTS.bat to package the results.
pause
exit /b 0

:ensure_env
if not exist logs mkdir logs
if exist .venv\Scripts\python.exe exit /b 0
echo First run: creating the Python environment (needs Python 3.13 and internet, once).
where py >nul 2>&1 && (py -3.13 -m venv .venv) || (python -m venv .venv)
if not exist .venv\Scripts\python.exe (
  echo Python 3.13 was not found. Install it from python.org, tick "Add python.exe to PATH", and retry.
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -r repo\requirements.lock || goto fail
.venv\Scripts\python -m pip install --quiet --no-deps .\gbm-4d || goto fail
.venv\Scripts\python -m pip install --quiet --no-deps -e .\repo || goto fail
exit /b 0
:fail
echo Package install failed. Check the internet connection and retry.
pause
exit /b 1
"""

PACK_BAT = r"""@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (.venv\Scripts\python pack_results.py) else (python pack_results.py)
pause
"""

PACK_RESULTS_PY = '''"""Zip finished units (qc.json + labels) and logs to send back."""
import json
import platform
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERIM = HERE / "repo" / "data" / "interim"
stamp = time.strftime("%Y%m%d-%H%M")
host = "".join(c for c in platform.node() if c.isalnum()) or "pc"
out = HERE / f"bundle_results_{host}_{stamp}.zip"
n = {"later": 0, "roundtrip": 0}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for qc in sorted((INTERIM / "label_field_later").glob("*/*/qc.json")):
        for f in ("qc.json", "recurrence_labels.nii.gz", "forward_transform.h5", "backward_transform.h5"):
            if (qc.parent / f).is_file():
                z.write(qc.parent / f, (qc.parent / f).relative_to(HERE / "repo").as_posix())
        n["later"] += 1
    sched = INTERIM / "label_field_later" / "schedule.json"
    if sched.is_file():
        z.write(sched, sched.relative_to(HERE / "repo").as_posix())
    for qc in sorted((INTERIM / "roundtrip_tuned").glob("*/qc.json")):
        z.write(qc, qc.relative_to(HERE / "repo").as_posix())
        n["roundtrip"] += 1
    for log in (HERE / "logs").glob("*.log"):
        z.write(log, f"logs/{log.name}")
    z.writestr("PACKED_ON.json", json.dumps({"host": platform.node(), "platform": platform.platform(),
                                            "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            "units": n}, indent=2))
print(f"Packed {n} -> {out}")
'''

README = """gbm-rec registration bundle
===========================

Runs two long registration jobs from the gbm-rec project on another PC:
  later      transport of the 2nd and 3rd LUMIERE progressions into baseline space
  roundtrip  the tuned-recipe round-trip study on the 63 LUMIERE pairs (no labels)

WINDOWS
  1. Install Python 3.13 from python.org if needed ("Add python.exe to PATH").
  2. Open a terminal in this folder and run, for example:
        RUN.bat both 3
     (job: later, roundtrip or both; the number is the worker count, about 3 GB RAM each).
     The first run builds a private environment (needs internet once).
  3. Keep the PC awake. Progress is saved after every unit; run the same command
     again to resume.
  4. Run PACK_RESULTS.bat and send back the bundle_results_<pc>_<date>.zip it writes.

BACK ON THE MAIN MACHINE
  .venv/Scripts/python scripts/09_import_bundle.py <zip>
"""


def _copy(src: Path, dst: Path) -> int:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        return sum(1 for p in dst.rglob("*") if p.is_file())
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT.parent / "gbmrec_bundle")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    if verify_frozen_gbm4d() != "git":
        raise SystemExit("pack from the machine with the gbm-4d git checkout")
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    g4, lum = lumiere.gbm4d_root(), lumiere.lumiere_root()
    start = time.time()

    # repo
    for rel in ("src/gbmc", "config.yaml", "pyproject.toml", "requirements.lock", "README.md"):
        _copy(REPO_ROOT / rel, out / "repo" / rel)
    for name in SCRIPTS:
        _copy(REPO_ROOT / "scripts" / name, out / "repo" / "scripts" / name)

    # vendored gbm-4d code, hashed
    vend = out / "gbm-4d"
    for rel in ("scripts", "src/gbm4d", "config.yaml", "pyproject.toml", "README.md",
                "results/cohort_lock/cohort_lock.json"):
        if (g4 / rel).exists():
            _copy(g4 / rel, vend / rel)
    for stray in (vend / "scripts").glob("*"):
        if stray.suffix != ".py":
            shutil.rmtree(stray) if stray.is_dir() else stray.unlink()
    write_vendored_manifest(vend)

    # data: per patient
    patients = lumiere.cohort()
    plan = schedule(patients)
    n_later = 0
    for row in plan:
        p, base, first = row["patient"], row["baseline"], row["first"]
        _copy(g4 / "data/interim/label_field" / p / "qc.json",
              vend / "data/interim/label_field" / p / "qc.json")
        for tp in (base, first):
            _copy(g4 / "data/interim/preprocessed" / p / tp, vend / "data/interim/preprocessed" / p / tp)
        _copy(lum / "Imaging" / p / base / "DeepBraTumIA-segmentation" / "native",
              out / "lumiere/Imaging" / p / base / "DeepBraTumIA-segmentation" / "native")
        for tp in row["later"]:
            _copy(lum / "Imaging" / p / tp, out / "lumiere/Imaging" / p / tp)
            n_later += 1
    _copy(lum / "metadata", out / "lumiere/metadata")

    # already-finished units, so the other PC skips them
    for rel in ("data/interim/label_field_later", "data/interim/preprocessed_later",
                "data/interim/roundtrip_tuned"):
        if (REPO_ROOT / rel).exists():
            _copy(REPO_ROOT / rel, out / "repo" / rel)

    (out / "logs").mkdir(exist_ok=True)
    (out / "RUN.bat").write_text(RUN_BAT, encoding="utf-8")
    (out / "PACK_RESULTS.bat").write_text(PACK_BAT, encoding="utf-8")
    (out / "pack_results.py").write_text(PACK_RESULTS_PY, encoding="utf-8")
    (out / "README.txt").write_text(README, encoding="utf-8")
    (out / "BUNDLE.json").write_text(json.dumps({
        "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "gbm4d_commit": PINNED_GBM4D,
        "patients": len(patients), "later_timepoints": n_later, "jobs": list(SCRIPTS)},
        indent=2) + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"bundle: {len(patients)} patients, {n_later} later timepoints, {size:.2f} GB -> {out} "
          f"({time.time() - start:.0f} s)")
    if args.zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for f in out.rglob("*"):
                if f.is_file():
                    z.write(f, f"{out.name}/{f.relative_to(out).as_posix()}")
        print(f"zip: {zpath.stat().st_size / 1e9:.2f} GB -> {zpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
