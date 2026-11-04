"""LUMIERE inputs for the tournament, read from gbm-4d's frozen derived data.

Nothing here re-runs gbm-4d. Preprocessed baselines, the transported recurrence
labels and the cohort lock are read exactly as gbm-4d left them (plan Step 1.2:
"in gbm-4d's baseline space, with gbm-4d's transported labels").

Where things live, first hit wins for each root:

* ``$GBMC_LUMIERE_ROOT`` or ``config.yaml:data.lumiere.local_root``: the raw
  LUMIERE archive (``Imaging/``, ``metadata/``), for the native DeepBraTumIA
  segmentation and the shipped atlas transform.
* ``$GBMC_GBM4D_ROOT`` or ``config.yaml:data.lumiere.gbm4d_root``: the gbm-4d
  checkout, for ``data/interim/{preprocessed,label_field}`` and
  ``results/cohort_lock/cohort_lock.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import config

#: DeepBraTumIA label values, as gbm-4d's 02_verify_labels.py derived them from
#: the shipped volumes (gbm4d.labels.EXPECTED_DEEPBRATUMIA). Label 2 is what
#: gbm-4d treats as the resection cavity.
ENHANCING, NECROTIC_NONENHANCING, OEDEMA = 1, 2, 3

#: gbm-4d D16: the locked cohort size.
N_LOCKED = 63

DBT = "DeepBraTumIA-segmentation"
NATIVE_SEG = f"{DBT}/native/segmentation/ct1_seg_mask.nii.gz"


def _root(env_var: str, key: str) -> Path:
    if env := os.environ.get(env_var):
        return Path(env)
    return Path(config.get(key))


def lumiere_root() -> Path:
    return _root("GBMC_LUMIERE_ROOT", "data.lumiere.local_root")


def gbm4d_root() -> Path:
    return _root("GBMC_GBM4D_ROOT", "data.lumiere.gbm4d_root")


def cohort() -> list[str]:
    """The 63 locked patients (gbm-4d D16), training and opened test set together.

    LUMIERE is development data only in this project (deviation V1), so the
    gbm-4d train/test split is deliberately not carried over.
    """
    lock = json.loads((gbm4d_root() / "results" / "cohort_lock" / "cohort_lock.json")
                      .read_text(encoding="utf-8"))
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


def transport_qc(patient: str) -> dict:
    path = gbm4d_root() / "data" / "interim" / "label_field" / patient / "qc.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(patient: str) -> BaselineCase:
    baseline = transport_qc(patient)["baseline"]
    base = gbm4d_root() / "data" / "interim" / "preprocessed" / patient / baseline
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
    path = gbm4d_root() / "data" / "interim" / "label_field" / patient / "recurrence_labels.nii.gz"
    return _array(path, sitk.sitkUInt8) == ENHANCING
