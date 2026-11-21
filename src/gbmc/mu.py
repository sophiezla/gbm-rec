"""MU-Glioma-Post: the sealed H2 test cohort (DECISIONS.md D18, D19; registration section 8).

Nothing here reads an MU image. Eligibility uses only the public clinical spreadsheet,
which was read before the registration (disclosed). Everything that depends on the image
files (folder layout, file names, sequence names) is null in ``config.yaml`` until the
download after the registration, and :func:`layout` raises until then, as for RHUH-GBM.

Pre-registered rules (approved 2026-11-19), applied by :func:`eligibility`:

* IDH-wildtype glioblastoma with no earlier brain tumour, and a first progression confirmed
  on imaging (progression types 1 or 3);
* baseline = the first post-treatment MRI at least ``BASELINE_GAP_DAYS`` before the first
  progression (MU has no early post-operative scan);
* recurrence scan = an MRI from 14 days before to 60 days after the first progression;
* target = enhancing tissue + resection cavity on the baseline; FLAIR region = SNFH;
  recurrence = enhancing tissue on the recurrence scan, carried onto the baseline grid.

Clarifications fixed before any image download (D18 item 4):

* "Glioblastoma" includes the one "Glioma w/ GBM features" diagnosis, as in the count of
  about 44 that the registration quotes; IDH-wildtype means IDH1 = 0 and IDH2 not mutated.
* If several MRIs fall in the recurrence window, the one closest to the progression date is
  used, the earlier one on a tie (user decision 2026-11-21). 14 of the 44 eligible patients
  have several; 38 of 44 have an MRI on the progression date itself, almost certainly the scan
  that documented it. Ordering is by days: the dataset's timepoint numbers are not always
  chronological (PatientID_0081).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gbmc import config

REPO_ROOT = Path(__file__).resolve().parents[2]
CLINICAL_XLSX = "MU-Glioma-Post_ClinicalData-July2025.xlsx"
CLINICAL_SHEET = "MU Glioma Post"
ELIGIBILITY = REPO_ROOT / "results" / "mu_eligibility" / "eligibility.json"

#: Label values as the dataset documents them (TCIA page; segmentation-volume sheet names).
#: Checked against the files on download, like RHUH-GBM's mapping check.
EXPECTED_LABELS = {1: "netc", 2: "snfh", 3: "enhancing", 4: "cavity"}
TARGET_LABELS = ("enhancing", "cavity")
FLAIR_LABEL = "snfh"
RECURRENCE_LABEL = "enhancing"

GBM_DIAGNOSES = ("GBM", "Glioma w/ GBM features")
IMAGING_PROGRESSION_TYPES = (1, 3)
BASELINE_GAP_DAYS = 56
RECURRENCE_WINDOW_DAYS = (-14, 60)
N_TIMEPOINTS = 6

#: Reasons, in the order they are checked; the first failing one is recorded.
STEPS = ("not glioblastoma", "IDH not wild-type or unknown", "earlier brain tumour",
         "no imaging-confirmed progression", "no progression date",
         "no baseline MRI at least 56 days before progression",
         "no MRI from 14 days before to 60 days after progression")


@dataclass
class Eligibility:
    patient: str
    eligible: bool
    reason: str | None
    baseline_timepoint: int | None = None      # 1-6, the dataset's Timepoint_<k>
    recurrence_timepoint: int | None = None
    baseline_days_before_progression: float | None = None
    recurrence_days_from_progression: float | None = None


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _col(header: list, prefix: str) -> int:
    hits = [i for i, h in enumerate(header) if isinstance(h, str) and h.strip().startswith(prefix)]
    if len(hits) != 1:
        raise KeyError(f"expected one column starting with {prefix!r}, found {len(hits)}")
    return hits[0]


def eligibility(rows: list[list]) -> list[Eligibility]:
    """Apply the pre-registered rules to the clinical sheet (header row first)."""
    header, body = rows[0], [r for r in rows[1:] if r and r[0] not in (None, "")]
    c = {k: _col(header, p) for k, p in {
        "id": "Patient_ID", "dx": "Primary Diagnosis", "idh1": "IDH1 mutation",
        "idh2": "IDH2 mutation", "prev": "Previous Brain Tumor", "prog": "Progression",
        "type": "Type of 1st Progression",
        "prog_day": "Number of days from Diagnosis to date of First Progression"}.items()}
    ordinals = ("1st", "2nd", "3rd", "4th", "5th", "6th")
    tp = [_col(header, f"Number of Days from Diagnosis to {o} MRI") for o in ordinals]
    out = []
    for r in body:
        pid = str(r[c["id"]]).strip()
        days = [_num(r[i]) for i in tp]
        prog_day = _num(r[c["prog_day"]])
        if str(r[c["dx"]] or "").strip() not in GBM_DIAGNOSES:
            out.append(Eligibility(pid, False, STEPS[0]))
        elif not (_num(r[c["idh1"]]) == 0 and _num(r[c["idh2"]]) != 1):
            out.append(Eligibility(pid, False, STEPS[1]))
        elif str(r[c["prev"]] or "").strip().lower() == "yes":
            out.append(Eligibility(pid, False, STEPS[2]))
        elif not (_num(r[c["prog"]]) == 1
                  and _num(r[c["type"]]) in IMAGING_PROGRESSION_TYPES):
            out.append(Eligibility(pid, False, STEPS[3]))
        elif not np.isfinite(prog_day):
            out.append(Eligibility(pid, False, STEPS[4]))
        else:
            base = [k for k, d in enumerate(days, 1)
                    if np.isfinite(d) and d <= prog_day - BASELINE_GAP_DAYS]
            lo, hi = RECURRENCE_WINDOW_DAYS
            rec = [k for k, d in enumerate(days, 1)
                   if np.isfinite(d) and lo <= d - prog_day <= hi]
            if not base:
                out.append(Eligibility(pid, False, STEPS[5]))
            elif not rec:
                out.append(Eligibility(pid, False, STEPS[6]))
            else:
                b = min(base, key=lambda k: days[k - 1])       # the first post-treatment MRI
                # closest to the progression date; ties go to the earlier scan
                e = min(rec, key=lambda k: (abs(days[k - 1] - prog_day), days[k - 1]))
                out.append(Eligibility(pid, True, None, b, e, prog_day - days[b - 1],
                                       days[e - 1] - prog_day))
    return out


def funnel(rows: list[Eligibility]) -> dict:
    counts = {step: sum(r.reason == step for r in rows) for step in STEPS}
    return {"enrolled": len(rows), "excluded_by_step": counts,
            "eligible": sum(r.eligible for r in rows)}


def as_dicts(rows: list[Eligibility]) -> list[dict]:
    return [asdict(r) for r in rows]


# ---------------------------------------------------------------- images (after download)
def layout() -> dict:
    """File-layout constants, verified on download. Raises while any is undecided."""
    return {key: config.get(f"data.mu_glioma_post.{key}")
            for key in ("images_local_root", "image_pattern", "seg_pattern", "sequence_names")}


ROLES = {"postop": "baseline_timepoint", "recurrence": "recurrence_timepoint"}


def files(patient: str, role: str, table: Path = ELIGIBILITY) -> dict[str, Path]:
    """Paths for one scan, with RHUH-GBM's roles: ``postop`` = the baseline, ``recurrence``.

    ``image_pattern`` and ``seg_pattern`` are format strings relative to
    ``images_local_root`` with ``{patient}``, ``{timepoint}`` (the number k of
    ``Timepoint_k``) and, for images, ``{sequence}``. Missing files are left out.
    """
    lay = layout()
    record = next(r for r in json.loads(table.read_text(encoding="utf-8"))["rows"]
                  if r["patient"] == patient)
    k = record[ROLES[role]]
    root = Path(lay["images_local_root"])
    root = root if root.is_absolute() else REPO_ROOT / root
    out = {}
    for ours, theirs in lay["sequence_names"].items():
        path = root / lay["image_pattern"].format(patient=patient, timepoint=k, sequence=theirs)
        if path.is_file():
            out[ours] = path
    seg = root / lay["seg_pattern"].format(patient=patient, timepoint=k)
    if seg.is_file():
        out["seg"] = seg
    return out


def label_values(names: tuple[str, ...], mapping: dict[int, str] = EXPECTED_LABELS) -> list[int]:
    return [v for v, n in mapping.items() if n in names]


def target(seg: np.ndarray, brain: np.ndarray,
           mapping: dict[int, str] = EXPECTED_LABELS) -> np.ndarray:
    """Pre-registered MU target: enhancing tissue + resection cavity on the baseline."""
    return np.isin(seg, label_values(TARGET_LABELS, mapping)) & brain
