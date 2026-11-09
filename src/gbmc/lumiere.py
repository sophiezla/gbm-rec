"""LUMIERE inputs for the tournament, read from gbm-4d's frozen derived data.

Nothing here re-runs gbm-4d. Preprocessed baselines, the transported recurrence
labels and the cohort lock are read exactly as gbm-4d left them (plan Step 1.2:
"in gbm-4d's baseline space, with gbm-4d's transported labels").

Where things live, first hit wins for each root:

* ``$GBMC_LUMIERE_ROOT``: the raw
  LUMIERE archive (``Imaging/``, ``metadata/``), for the native DeepBraTumIA
  segmentation and the shipped atlas transform.
* ``$GBMC_DERIVED_ROOT``: a folder holding the preprocessed baselines and first-progression
  labels under ``data/interim/{preprocessed,label_field}``.

The cohort lock is in this repository: ``results/cohort_lock/cohort_lock.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

COHORT_LOCK = Path(__file__).resolve().parents[2] / "results" / "cohort_lock" / "cohort_lock.json"
BASELINES = COHORT_LOCK.with_name("baselines.json")

#: DeepBraTumIA label values, as gbm-4d's 02_verify_labels.py derived them from
#: the shipped volumes. Label 2 is what
#: gbm-4d treats as the resection cavity.
ENHANCING, NECROTIC_NONENHANCING, OEDEMA = 1, 2, 3

#: gbm-4d D16: the locked cohort size.
N_LOCKED = 63

DBT = "DeepBraTumIA-segmentation"
NATIVE_SEG = f"{DBT}/native/segmentation/ct1_seg_mask.nii.gz"


def _root(env_var: str) -> Path:
    if env := os.environ.get(env_var):
        return Path(env)
    raise ValueError(f"set ${env_var} to this computer's data folder (see docs/SETUP.md)")


def lumiere_root() -> Path:
    return _root("GBMC_LUMIERE_ROOT")


def derived_root() -> Path:
    return _root("GBMC_DERIVED_ROOT")


def timepoint_sort_key(timepoint: str) -> tuple[int, int]:
    """Order ``week-000-1`` < ``week-000-2`` < ``week-044``; string order would not."""
    match = re.fullmatch(r"week-(\d+)(?:-(\d+))?", timepoint.strip())
    if not match:
        raise ValueError(f"unparseable timepoint {timepoint!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def load_ratings(metadata: Path) -> pd.DataFrame:
    """RANO ratings per patient and timepoint, in columns Patient, Timepoint, Rating."""
    table = pd.read_csv(metadata / "LUMIERE-ExpertRating-v202211.csv")
    rating = next(c for c in table.columns if c.startswith("Rating (according to RANO"))
    table = table.rename(columns={rating: "Rating", "Date": "Timepoint"})
    table["Rating"] = table["Rating"].astype("string").str.strip()
    return table


def load_completeness(metadata: Path) -> pd.DataFrame:
    """Which sequences and segmentations each timepoint has, as booleans ("x" = present)."""
    table = pd.read_csv(metadata / "LUMIERE-datacompleteness.csv")
    for column in table.columns.difference(["Patient", "Timepoint"]):
        cells = table[column].astype("string").str.strip().str.lower()
        table[column] = cells.eq("x").fillna(False)
    return table


def cohort() -> list[str]:
    """The 63 locked patients (gbm-4d D16), training and opened test set together.

    LUMIERE is development data only in this project (deviation V1), so the
    gbm-4d train/test split is deliberately not carried over.
    """
    lock = json.loads(COHORT_LOCK.read_text(encoding="utf-8"))
    patients = sorted(lock["cohort"])
    if set(patients) != set(lock["train"]) | set(lock["test"]):
        raise ValueError("cohort lock: train + test does not equal the cohort list")
    if len(patients) != N_LOCKED:
        raise ValueError(f"cohort lock lists {len(patients)} patients, expected {N_LOCKED}")
    return patients


@dataclass
class BaselineCase:
    """One patient's baseline, on gbm-4d's preprocessed CT1 grid."""

    patient: str
    baseline: str
    reference: sitk.Image      # CT1, the grid every array below lives on
    brain: np.ndarray          # bool
    t1: np.ndarray             # z-scored T1 (for the falx's white-matter test)
    ct1: np.ndarray            # z-scored CT1 (for the ventricle segmentation)
    seg: np.ndarray            # DeepBraTumIA labels, nearest-neighbour resampled
    to_atlas: sitk.Transform   # native -> symmetric MNI152, direction verified

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        sx, sy, sz = self.reference.GetSpacing()
        return (sz, sy, sx)

    @property
    def target(self) -> np.ndarray:
        """Enhancing tumour + cavity, as gbm-4d's Arm 0 defined it."""
        return (self.seg == ENHANCING) | (self.seg == NECROTIC_NONENHANCING)

    @property
    def flair_abnormal(self) -> np.ndarray:
        return (self.seg == OEDEMA) & self.brain


def _array(path: Path, pixel=sitk.sitkFloat32) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path), pixel))


def _dice(a: sitk.Image, b: sitk.Image) -> float:
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(sitk.Cast(a > 0, sitk.sitkUInt8), sitk.Cast(b > 0, sitk.sitkUInt8))
    return f.GetDiceCoefficient()


def atlas_transform(patient: str, session: str) -> sitk.Transform:
    """native -> atlas transform, with its direction settled per patient by evidence.

    Copied from gbm-4d ``scripts/11_independent_agreement.py:verify_direction``
    (scripts are not part of the installed package). The stored ``ct1.tfm`` is
    applied both ways and the one that reproduces the shipped atlas segmentation
    wins; gbm-4d measured Dice 0.000 as stored against 0.994-0.999 inverted.
    """
    root = lumiere_root() / "Imaging" / patient / session / DBT
    native = sitk.ReadImage(str(root / "native/segmentation/ct1_seg_mask.nii.gz"),
                            sitk.sitkUInt8)
    atlas = sitk.ReadImage(str(root / "atlas/segmentation/seg_mask.nii.gz"), sitk.sitkUInt8)
    stored = sitk.ReadTransform(str(root / "native/transformation/ct1.tfm"))
    as_stored = _dice(atlas, sitk.Resample(native, atlas, stored, sitk.sitkNearestNeighbor, 0))
    inverted = _dice(atlas, sitk.Resample(native, atlas, stored.GetInverse(),
                                          sitk.sitkNearestNeighbor, 0))
    # Resample pulls with an atlas -> native map, so "works only when inverted"
    # means the stored transform is native -> atlas.
    return stored if inverted > as_stored else stored.GetInverse()


#: Committed table of each patient's first-progression timepoint, built by
#: ``scripts/15_first_progressions.py`` from gbm-4d's transport records (the timepoints
#: whose labels are in the feature cache). It replaces reading ``label_field/<p>/qc.json``,
#: which is not on this computer.
FIRST_PROGRESSIONS = COHORT_LOCK.with_name("first_progressions.json")


def first_progression(patient: str, table: Path = FIRST_PROGRESSIONS) -> str:
    """The first segmented RANO progression timepoint whose label is the endpoint."""
    record = json.loads(table.read_text(encoding="utf-8"))
    if patient not in record["first_progression"]:
        raise KeyError(f"{patient}: first-progression timepoint not recorded in {table.name} "
                       "(its gbm-4d transport record is missing; see 15_first_progressions.py)")
    return record["first_progression"][patient]


def baseline_name(patient: str) -> str:
    """The post-operative baseline timepoint, from the committed list."""
    return json.loads(BASELINES.read_text(encoding="utf-8"))[patient]


def load_baseline(patient: str) -> BaselineCase:
    baseline = baseline_name(patient)
    base = derived_root() / "data" / "interim" / "preprocessed" / patient / baseline
    reference = sitk.ReadImage(str(base / "CT1.nii.gz"), sitk.sitkFloat32)
    native = sitk.ReadImage(str(lumiere_root() / "Imaging" / patient / baseline / NATIVE_SEG),
                            sitk.sitkUInt8)
    seg = sitk.Resample(native, reference, sitk.Transform(3, sitk.sitkIdentity),
                        sitk.sitkNearestNeighbor, 0, native.GetPixelID())
    return BaselineCase(
        patient=patient, baseline=baseline, reference=reference,
        brain=_array(base / "brain_mask.nii.gz", sitk.sitkUInt8) > 0,
        t1=_array(base / "T1.nii.gz"),
        ct1=sitk.GetArrayFromImage(reference),
        seg=sitk.GetArrayFromImage(seg),
        to_atlas=atlas_transform(patient, baseline),
    )


def load_recurrence(patient: str) -> np.ndarray:
    """Enhancing tumour at the first segmented RANO progression, in baseline space.

    gbm-4d D18: only ``Enhancing_Core`` counts as recurrence, because the
    necrotic/non-enhancing label at progression sits largely on the old cavity.
    """
    fields = derived_root() / "data" / "interim" / "label_field"
    path = fields / patient / "recurrence_labels.nii.gz"
    return _array(path, sitk.sitkUInt8) == ENHANCING
