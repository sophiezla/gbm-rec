"""RHUH-GBM cohort logic (plan Steps 2.2, 3.1-3.6). Nothing here reads RHUH-GBM files yet.

What the plan states about the dataset, used here only as EXPECTATIONS:

* labels 1 = necrosis, 2 = peritumoral signal alteration (oedema + non-enhancing
  tumour), 3 = enhancing tumour (Cepeda et al. 2023, via plan Step 2.2);
* each timepoint is registered separately to the SRI24 atlas;
* timepoints: pre-operative, early post-operative (< 72 h), recurrence.

Everything that depends on the files themselves (names, folder layout, clinical
columns) is in ``config.yaml:data.rhuh_gbm`` and stays null until the download,
which the plan forbids before the OSF timestamp. :func:`layout` raises until
then, so no script can run on a guessed layout.

Two pre-registered rules live here as code (plan Step 2.2):

* :func:`check_mapping`: the label mapping is re-derived from intensities, and
  that check "may only correct a swapped mapping, never choose between
  alternatives";
* :func:`treatment_target`: if the early post-op cavity carries no label, the
  target is post-op enhancing residual + the pre-operative core (1 + 3),
  carried rigidly into post-op space and clipped to the brain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gbmc import config

EXPECTED_LABELS = {1: "necrosis", 2: "peritumoral", 3: "enhancing"}
EARLY_POSTOP_MAX_HOURS = 72


def layout() -> dict:
    """File-layout constants, verified on download. Raises while any is undecided."""
    return {key: config.get(f"data.rhuh_gbm.{key}")
            for key in ("local_root", "image_pattern", "seg_pattern", "clinical_csv",
                        "clinical_columns", "timepoint_names", "sequence_names")}


ROLES = ("preop", "postop", "recurrence")


def files(patient: str, role: str) -> dict[str, Path]:
    """Paths of one timepoint's images and segmentation, from the verified layout.

    ``image_pattern`` and ``seg_pattern`` are format strings relative to
    ``local_root`` with ``{patient}``, ``{timepoint}`` and (images only)
    ``{sequence}``; ``timepoint_names`` maps each role to the dataset's own
    timepoint name and ``sequence_names`` maps T1ce/FLAIR/T1/T2/ADC to the
    dataset's sequence names. Missing files are simply absent from the result.
    """
    lay = layout()
    root = Path(lay["local_root"])
    tp = lay["timepoint_names"][role]
    out = {}
    for ours, theirs in lay["sequence_names"].items():
        path = root / lay["image_pattern"].format(patient=patient, timepoint=tp, sequence=theirs)
        if path.is_file():
            out[ours] = path
    seg = root / lay["seg_pattern"].format(patient=patient, timepoint=tp)
    if seg.is_file():
        out["seg"] = seg
    return out


def patients() -> list[str]:
    """Patient IDs from the clinical table (the enrolled list, 40 per the dataset paper)."""
    import pandas as pd

    lay = layout()
    table = pd.read_csv(Path(lay["local_root"]) / lay["clinical_csv"])
    return sorted(table[lay["clinical_columns"]["patient"]].astype(str))


# ---------------------------------------------------------------- label mapping
@dataclass
class LabelEvidence:
    """Per-label intensity signature on one patient's scan."""

    patient: str
    t1ce: dict[int, float]      # mean z-scored T1ce per label
    flair: dict[int, float]     # mean z-scored FLAIR per label
    derived: dict[int, str] = field(default_factory=dict)


def derive_mapping(patient: str, seg: np.ndarray, t1ce: np.ndarray, flair: np.ndarray,
                   brain: np.ndarray) -> LabelEvidence:
    """Name each present label from intensity alone, without looking at its value.

    * enhancing: the label with the brightest mean T1ce;
    * necrosis: of the rest, the darkest mean T1ce (necrosis and fluid do not enhance);
    * peritumoral: the remaining one, which should also be the brightest on FLAIR.

    Needs all three labels present; returns an empty ``derived`` otherwise, so a
    scan missing a compartment votes for nothing.
    """
    labels = sorted(int(v) for v in np.unique(seg[brain]) if v != 0)
    t1 = {lab: float(t1ce[(seg == lab) & brain].mean()) for lab in labels}
    fl = {lab: float(flair[(seg == lab) & brain].mean()) for lab in labels}
    evidence = LabelEvidence(patient, t1, fl)
    if len(labels) != 3:
        return evidence
    enhancing = max(labels, key=lambda lab: t1[lab])
    rest = [lab for lab in labels if lab != enhancing]
    necrosis = min(rest, key=lambda lab: t1[lab])
    peritumoral = next(lab for lab in rest if lab != necrosis)
    evidence.derived = {enhancing: "enhancing", necrosis: "necrosis", peritumoral: "peritumoral"}
    return evidence


@dataclass
class MappingDecision:
    mapping: dict[int, str]
    corrected: bool
    agreement: float
    n_voting: int
    note: str


def check_mapping(evidence: list[LabelEvidence], min_agreement: float) -> MappingDecision:
    """Apply the Step 2.2 rule to the per-patient derivations.

    The modal derived mapping across voting scans is compared with
    :data:`EXPECTED_LABELS`. If they agree, the expected mapping stands. If the
    modal mapping is a PERMUTATION of the expected names reached with at least
    ``min_agreement`` of voting scans, it replaces the expected one (a swapped
    mapping corrected). Anything else (weak agreement, or a mapping that is not
    a permutation of the same three names) raises: the check may not choose
    between alternatives.
    """
    votes = [tuple(sorted(e.derived.items())) for e in evidence if e.derived]
    if not votes:
        raise ValueError("no scan carried all three labels; the mapping cannot be checked")
    from collections import Counter

    top, count = Counter(votes).most_common(1)[0]
    modal = dict(top)
    agreement = count / len(votes)
    if modal == EXPECTED_LABELS:
        return MappingDecision(dict(EXPECTED_LABELS), False, agreement, len(votes),
                               "derived mapping equals the published one")
    if agreement < min_agreement:
        raise ValueError(f"derived mapping disagrees with the published one and only "
                         f"{agreement:.0%} of scans agree on an alternative: not decidable")
    if sorted(modal.values()) != sorted(EXPECTED_LABELS.values()):
        raise ValueError(f"derived mapping {modal} is not a relabelling of the published one")
    return MappingDecision(modal, True, agreement, len(votes),
                           "published mapping swapped; corrected to the derived one")


# ---------------------------------------------------------------- target
def treatment_target(postop_seg: np.ndarray, brain: np.ndarray,
                     cavity_label: int | None,
                     preop_core_in_postop: np.ndarray | None,
                     mapping: dict[int, str] = EXPECTED_LABELS) -> tuple[np.ndarray, str]:
    """The treatment target on the early post-op grid, by the pre-registered rule.

    ``cavity_label`` is the label value found inside the cavity by the Step 3.2
    inspection, or None if the cavity carries no label. In the first case the
    target follows gbm-4d (enhancing + cavity); in the second, the fallback
    rule applies and needs ``preop_core_in_postop``.
    """
    enhancing = next(v for v, n in mapping.items() if n == "enhancing")
    residual = postop_seg == enhancing
    if cavity_label is not None:
        return ((residual | (postop_seg == cavity_label)) & brain), "enhancing + labelled cavity"
    if preop_core_in_postop is None:
        raise ValueError("cavity carries no label: the pre-op core in post-op space is required")
    return ((residual | preop_core_in_postop) & brain), \
        "post-op enhancing + pre-op core (1+3) carried rigidly, clipped to brain"


def preop_core(preop_seg: np.ndarray, mapping: dict[int, str] = EXPECTED_LABELS) -> np.ndarray:
    keep = [v for v, n in mapping.items() if n in ("necrosis", "enhancing")]
    return np.isin(preop_seg, keep)


def cavity_rule() -> int | None:
    """The recorded Step 2.2 outcome: the cavity's label value, or None if unlabelled.

    ``config.yaml:rhuh.cavity_rule`` is null until 20_'s inspection is read and
    the result recorded; reading it before then raises.
    """
    rule = str(config.get("rhuh.cavity_rule"))
    if rule == "unlabelled":
        return None
    if rule.startswith("labelled:"):
        return int(rule.split(":", 1)[1])
    raise ValueError(f"rhuh.cavity_rule must be 'unlabelled' or 'labelled:<value>', not {rule!r}")


def carried_preop_core(patient: str, mapping: dict[int, str], seed: int) -> np.ndarray:
    """Pre-op core (1 + 3) carried RIGIDLY onto the early post-op grid (Step 2.2 rule)."""
    import SimpleITK as sitk
    from gbm4d import preprocess, transport

    pre, post = files(patient, "preop"), files(patient, "postop")
    fixed = sitk.ReadImage(str(post["T1ce"]), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(pre["T1ce"]), sitk.sitkFloat32)
    rigid, _ = preprocess.register_rigid(fixed, moving, seed)
    seg = sitk.ReadImage(str(pre["seg"]), sitk.sitkUInt8)
    core = sitk.GetImageFromArray(preop_core(sitk.GetArrayFromImage(seg), mapping).astype(np.uint8))
    core.CopyInformation(seg)
    return sitk.GetArrayFromImage(transport.warp_label(core, fixed, rigid)) > 0


def cavity_label_counts(postop_seg: np.ndarray, postop_t1ce: np.ndarray, core: np.ndarray,
                        brain: np.ndarray, dark_z: float) -> dict[str, int]:
    """Label values found in the dark (fluid-like) T1ce voxels inside the pre-op core.

    The Step 3.2 inspection on 5 seeded patients: if most of this region is
    label 0, the cavity is unlabelled. Reported as counts; the decision is read
    off them and recorded, never inferred silently.
    """
    region = core & brain & (postop_t1ce < dark_z)
    values, counts = np.unique(postop_seg[region], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts, strict=True)}


# ---------------------------------------------------------------- inventory
@dataclass
class InventoryRow:
    patient: str
    has_preop: bool
    has_postop: bool
    has_recurrence: bool
    postop_hours: float | None
    sequences_ok: bool
    secondary_facility: bool | None
    reasons: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.reasons


def inventory_row(patient: str, timepoints: dict[str, set[str]], postop_hours: float | None,
                  secondary_facility: bool | None,
                  required: tuple[str, ...] = ("T1ce", "FLAIR", "seg")) -> InventoryRow:
    """Step 3.2 checks for one patient. ``timepoints`` maps role -> files present."""
    row = InventoryRow(patient, "preop" in timepoints, "postop" in timepoints,
                       "recurrence" in timepoints, postop_hours,
                       all(set(required) <= timepoints.get(r, set())
                           for r in ("postop", "recurrence")),
                       secondary_facility)
    for role in ("postop", "recurrence"):
        if role not in timepoints:
            row.reasons.append(f"missing {role} scan")
        elif not set(required) <= timepoints[role]:
            row.reasons.append(f"{role}: missing {sorted(set(required) - timepoints[role])}")
    if postop_hours is not None and postop_hours >= EARLY_POSTOP_MAX_HOURS:
        row.reasons.append(
            f"post-op scan at {postop_hours:.0f} h, not < {EARLY_POSTOP_MAX_HOURS} h")
    return row
