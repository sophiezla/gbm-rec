#!/usr/bin/env python
"""Run everything between the downloads and the C5 handoff, unattended (DECISIONS.md D20).

Stages, in order. Each is skipped when its output already exists, the first failure stops
the run, and each stage that writes ``results/`` is committed locally (never pushed):

  downloads        file counts, RHUH's checksum file, and the download record (must be later
                   than the registration timestamp)        -> results/downloads/
  rhuh_constants   timepoint check against the clinical tumour volumes, and the SRI24
                   midline, both by D20's written rules     -> results/rhuh_constants/, config
  rhuh_inventory   20_ (label-mapping check, cavity inspection); the cavity rule is set
                   by D20's written reading                 -> results/rhuh_inventory/, config
  rhuh_transport   21_ tuned and original recipes           -> data/interim/rhuh/
  rhuh_screens     22_ all screens, then the verdict        -> results/rhuh_screens/
  rhuh_lock        23_                                       -> results/rhuh_lock/
  rhuh_c5_pack     24_, then a copy of the C5 inputs to the shared folder ($GBMC_SHARE_DIR)
  mu_transport     26_                                       -> data/interim/mu/
  mu_screens       27_ all screens, then the verdict        -> results/mu_screens/
  mu_lock          28_                                       -> results/mu_lock/

It stops there. The partner then scores RHUH-GBM with C5 (``14_``, see
docs/PARTNER_NEXT.md). The run-once confirmatory scripts (``30_``, ``31_``, ``32_``) are
never started here: they are run by hand, on the user's go.

    GBMC_SHARE_DIR="<Drive>/gbm-rec-c5" python scripts/run_test_cohorts.py --workers 4
    python scripts/run_test_cohorts.py --only downloads      # one stage
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config, mu, rhuh
from gbmc.provenance import file_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CO_AUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EXPECTED_RHUH_FILES = 720          # TCIA package RHUH-GBM-nii-v1: 40 patients x 3 x 6
EXPECTED_MU_FILES = 2978           # TCIA package MU-Glioma-Post
ENHANCING = 3                      # published label value (RHUH-GBM paper, section 3.2)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def run(*args: str) -> None:
    log("run: " + " ".join(args))
    subprocess.run([PY, *args], cwd=REPO_ROOT, check=True)


def commit(paths: list[str], message: str) -> None:
    subprocess.run(["git", "add", "--", *paths], cwd=REPO_ROOT, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if staged.returncode == 0:
        log(f"nothing to commit for: {message}")
        return
    subprocess.run(["git", "commit", "-q", "-m", f"{message}\n\n{CO_AUTHOR}"],
                   cwd=REPO_ROOT, check=True)
    log(f"committed: {message}")


def set_config(key: str, value: str, comment: str) -> None:
    """Replace ``key: null`` (one exact line, under rhuh:) with a decided value."""
    path = REPO_ROOT / "config.yaml"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip().startswith(f"{key}: null")]
    if len(hits) != 1:
        raise SystemExit(f"config.yaml: expected one '{key}: null' line, found {len(hits)}")
    indent = lines[hits[0]][: len(lines[hits[0]]) - len(lines[hits[0]].lstrip())]
    lines[hits[0]] = f"{indent}{key}: {value}   # {comment}\n"
    path.write_text("".join(lines), encoding="utf-8")


def _done(path: Path) -> bool:
    return path.exists()


# ---------------------------------------------------------------- stages
def stage_downloads(_args) -> None:
    out = REPO_ROOT / "results" / "downloads" / "downloads.json"
    if _done(out):
        return
    rroot = rhuh.data_root()
    rfiles = sorted((rroot / "RHUH-GBM_nii_v1").rglob("*.nii.gz"))
    mroot = Path(config.get("data.mu_glioma_post.images_local_root"))
    mroot = mroot if mroot.is_absolute() else REPO_ROOT / mroot
    mfiles = sorted((mroot / "MU-Glioma-Post").rglob("*.nii.gz"))
    if len(rfiles) != EXPECTED_RHUH_FILES:
        raise SystemExit(f"RHUH-GBM: {len(rfiles)} .nii.gz files, expected {EXPECTED_RHUH_FILES}")
    if len(mfiles) != EXPECTED_MU_FILES:
        raise SystemExit(f"MU-Glioma-Post: {len(mfiles)} .nii.gz files, expected "
                         f"{EXPECTED_MU_FILES}")
    sums = rroot / "RHUH-GBM-nii-v1.sums"
    checked = None
    if sums.is_file():
        import hashlib
        checked = {"file": sums.name, "sha256": file_digest(sums), "ok": 0, "bad": [],
                   "unparsed_lines": 0}
        for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2 or len(parts[0]) not in (32, 40, 64):
                checked["unparsed_lines"] += 1
                continue
            algo = {32: "md5", 40: "sha1", 64: "sha256"}[len(parts[0])]
            target = rroot / parts[-1].lstrip("*").lstrip("./")
            if not target.is_file():
                target = next(rroot.rglob(Path(parts[-1]).name), None)
            if target is None:
                checked["bad"].append(f"missing {parts[-1]}")
                continue
            h = hashlib.new(algo, target.read_bytes()).hexdigest()
            if h == parts[0].lower():
                checked["ok"] += 1
            else:
                checked["bad"].append(parts[-1])
        if checked["bad"]:
            raise SystemExit(f"RHUH-GBM checksum mismatch: {checked['bad'][:5]}")
    osf_ts = str(config.get("osf.timestamp"))
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if now <= osf_ts:
        raise SystemExit("download record would not be later than the registration")

    def listing_digest(files, root):
        import hashlib
        h = hashlib.sha256()
        for f in files:
            h.update(f"{f.relative_to(root)}\t{f.stat().st_size}\n".encode())
        return h.hexdigest()

    elig = json.loads(mu.ELIGIBILITY.read_text(encoding="utf-8"))
    missing = []
    for row in elig["rows"]:
        if not row["eligible"]:
            continue
        for role in ("postop", "recurrence"):
            got = mu.files(row["patient"], role)
            if not {"T1ce", "seg"} <= set(got):
                missing.append(f"{row['patient']}:{role}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "recorded_utc": now, "registration_timestamp": osf_ts,
        "registration_url": config.get("osf.url"),
        "rhuh_gbm": {"source": "TCIA faspex package RHUH-GBM-nii-v1 (CC BY 4.0)",
                     "files": len(rfiles), "bytes": sum(f.stat().st_size for f in rfiles),
                     "listing_sha256": listing_digest(rfiles, rroot), "sums_check": checked},
        "mu_glioma_post": {"source": "TCIA faspex package MU-Glioma-Post (CC BY 4.0)",
                           "files": len(mfiles), "bytes": sum(f.stat().st_size for f in mfiles),
                           "listing_sha256": listing_digest(mfiles, mroot),
                           "eligible_scans_missing": missing}},
        indent=2) + "\n", encoding="utf-8")
    commit(["results/downloads"], f"Record the test-cohort downloads ({now}, after the "
           f"registration at {osf_ts})")


def stage_rhuh_constants(_args) -> None:
    out = REPO_ROOT / "results" / "rhuh_constants" / "constants.json"
    if _done(out):
        return
    import pandas as pd
    from scipy.stats import spearmanr

    lay = rhuh.layout()
    table = pd.read_csv(rhuh.data_root() / lay["clinical_csv"])
    pid = table.columns[0]
    pre_col = next(c for c in table.columns if c.strip().startswith("Preoperative  contrast"))
    post_col = next(c for c in table.columns if c.strip().startswith("Postoperative contrast"))
    rows, xs = [], []
    for _, r in table.iterrows():
        p = str(r[pid])
        pre, post = rhuh.files(p, "preop"), rhuh.files(p, "postop")
        vox = {}
        for name, f in (("preop", pre), ("postop", post)):
            seg = sitk.ReadImage(str(f["seg"]))
            sp = np.prod(seg.GetSpacing())
            vox[name] = float((sitk.GetArrayFromImage(seg) == ENHANCING).sum() * sp / 1000.0)
        img = sitk.ReadImage(str(pre["T1ce"]))
        idx = np.argwhere(sitk.GetArrayFromImage(img) != 0).mean(axis=0)[::-1]
        x = img.TransformContinuousIndexToPhysicalPoint([float(v) for v in idx])[0]
        xs.append(x)
        rows.append({"patient": p, "enhancing_cm3_tp_preop": vox["preop"],
                     "enhancing_cm3_tp_postop": vox["postop"],
                     "clinical_preop_ce_cm3": float(r[pre_col]),
                     "clinical_postop_residual_cm3": float(r[post_col]),
                     "brain_centroid_x_mm": float(x)})
    seg_pre = [r["enhancing_cm3_tp_preop"] for r in rows]
    seg_post = [r["enhancing_cm3_tp_postop"] for r in rows]
    rho = float(spearmanr(seg_pre, [r["clinical_preop_ce_cm3"] for r in rows]).statistic)
    ok = np.median(seg_pre) > np.median(seg_post) and rho >= 0.8
    midline = round(float(np.median(xs)), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rule": "D20: timepoint folders accepted if median enhancing volume at 'preop' > at "
                "'postop' and Spearman(preop enhancing, clinical preop CE volume) >= 0.8; "
                "SRI24 midline = median x of the pre-op brain-mask centroid",
        "timepoints_ok": bool(ok), "spearman_preop": rho,
        "median_enhancing_cm3": {"preop": float(np.median(seg_pre)),
                                 "postop": float(np.median(seg_post))},
        "sri24_midline_x_mm": midline, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit(f"timepoint names fail D20's check (rho {rho:.2f}); see {out}")
    set_config("sri24_midline_x_mm", str(midline),
               "D20 rule: median pre-op brain-mask centroid x (results/rhuh_constants/)")
    commit(["results/rhuh_constants", "config.yaml"],
           f"Set RHUH-GBM download-time constants by D20 (midline {midline} mm; timepoints "
           f"checked, Spearman {rho:.2f})")


def stage_rhuh_inventory(_args) -> None:
    out = REPO_ROOT / "results" / "rhuh_inventory" / "inventory.json"
    if not _done(out):
        run("scripts/20_rhuh_inventory.py")
        commit(["results/rhuh_inventory"], "RHUH-GBM inventory, label check and cavity "
               "inspection (20_)")
    rule = json.loads(out.read_text(encoding="utf-8"))["cavity_inspection"]["decision"]
    try:
        config.get("rhuh.cavity_rule")
    except ValueError:
        set_config("cavity_rule", f'"{rule}"', "D20 reading of 20_'s cavity inspection")
        commit(["config.yaml"], f"Set the RHUH-GBM cavity rule by D20: {rule}")


def stage_rhuh_transport(args) -> None:
    run("scripts/21_rhuh_transport.py", "--workers", str(args.workers))
    run("scripts/21_rhuh_transport.py", "--recipe", "original", "--workers", str(args.workers))


def stage_rhuh_screens(args) -> None:
    if _done(REPO_ROOT / "results" / "rhuh_screens" / "verdict.json"):
        return
    run("scripts/22_rhuh_screens.py", "--surface", "--arbiters", "--simulate",
        "--workers", str(args.workers))
    run("scripts/22_rhuh_screens.py", "--verdict")
    commit(["results/rhuh_screens"], "RHUH-GBM registration screens and verdict (22_)")


def stage_rhuh_lock(_args) -> None:
    if _done(REPO_ROOT / "results" / "rhuh_lock" / "consort.json"):
        return
    run("scripts/23_rhuh_lock.py")
    commit(["results/rhuh_lock"], "Lock the RHUH-GBM cohort (23_)")


def stage_rhuh_c5_pack(_args) -> None:
    from gbmc import c5_import

    if not _done(c5_import.RHUH_INPUTS / "tuned" / "manifest.json"):
        run("scripts/24_rhuh_c5_pack.py")
    share = os.environ.get("GBMC_SHARE_DIR")
    if share:
        dest = Path(share) / "rhuh" / "c5_inputs"
        if not dest.exists():
            shutil.copytree(c5_import.RHUH_INPUTS, dest)
            log(f"copied C5 inputs to {dest}")


def stage_mu_transport(args) -> None:
    run("scripts/26_mu_transport.py", "--workers", str(args.workers))


def stage_mu_screens(args) -> None:
    if _done(REPO_ROOT / "results" / "mu_screens" / "verdict.json"):
        return
    run("scripts/27_mu_screens.py", "--surface", "--arbiters", "--simulate",
        "--workers", str(args.workers))
    run("scripts/27_mu_screens.py", "--verdict")
    commit(["results/mu_screens"], "MU-Glioma-Post registration screens and verdict (27_)")


def stage_mu_lock(_args) -> None:
    if _done(REPO_ROOT / "results" / "mu_lock" / "consort.json"):
        return
    run("scripts/28_mu_lock.py")
    commit(["results/mu_lock"], "Lock the MU-Glioma-Post cohort (28_)")


STAGES = {"downloads": stage_downloads, "rhuh_constants": stage_rhuh_constants,
          "rhuh_inventory": stage_rhuh_inventory, "rhuh_transport": stage_rhuh_transport,
          "rhuh_screens": stage_rhuh_screens, "rhuh_lock": stage_rhuh_lock,
          "rhuh_c5_pack": stage_rhuh_c5_pack, "mu_transport": stage_mu_transport,
          "mu_screens": stage_mu_screens, "mu_lock": stage_mu_lock}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=list(STAGES), default=None)
    args = ap.parse_args()
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                      capture_output=True, text=True).stdout.strip():
        raise SystemExit("commit or stash local changes first: stages commit their results")
    for name, fn in STAGES.items():
        if args.only and name != args.only:
            continue
        start = time.time()
        log(f"== {name}")
        fn(args)
        log(f"== {name} done ({(time.time() - start) / 60:.1f} min)")
    log("all stages done; next: the partner's C5 scoring (docs/PARTNER_NEXT.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
