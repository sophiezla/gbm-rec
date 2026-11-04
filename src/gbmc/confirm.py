"""Scoring the frozen candidates on RHUH-GBM (plan Steps 4.1-4.6).

Shared by ``scripts/30_confirm.py`` (the primary, once) and
``scripts/31_sensitivity.py`` (the pre-registered sensitivity arms), so every
arm scores patients with exactly the same code.

Per patient: features on the early post-op grid (``gbmc.rhuh_features``),
every candidate's voxel scores from the hash-checked models, and the metrics
of Step 1.5 at the patient's own matched volumes. Also the descriptive
"where distance fails" numbers (Step 2.3): the share of recurrence beyond the
15 mm and 20 mm guideline margins, and the share of it each candidate's
matched-volume region catches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import candidates, config, freeze, metrics, rhuh_features
from gbmc.features import COLUMNS


@dataclass
class PatientInputs:
    patient: str
    reference: sitk.Image         # early post-op T1ce
    brain: np.ndarray
    target: np.ndarray
    flair_abnormal: np.ndarray    # post-op peritumoral label
    t1_z: np.ndarray              # for the falx white-matter test
    lesion: np.ndarray            # any tumour label, excluded from the ventricle search
    recurrence: np.ndarray        # transported recurrence label(s) counted as positive


def load_models(release_dir: Path, register: dict[str, str]) -> dict:
    """Every learned candidate, each loaded only after its SHA-256 matches (Step 4.2)."""
    return {name: freeze.load_model(release_dir / f"models/{name}.joblib",
                                    register[f"models/{name}.joblib"])
            for name in candidates.LEARNED}


def score_patient(inp: PatientInputs, models: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Metrics per candidate, voxel scores per candidate, and feature QC."""
    vols, qc = rhuh_features.build(inp.reference, inp.brain, inp.target, inp.flair_abnormal,
                                   inp.t1_z, inp.lesion)
    evaluate = vols.evaluate
    x = vols.matrix()
    truth = inp.recurrence[evaluate]
    sx, sy, sz = inp.reference.GetSpacing()
    c0t = x[:, COLUMNS.index("c0t")]
    f1 = x[:, COLUMNS.index("f1")]
    matched = {f"margin{int(r)}": int((c0t <= float(r)).sum())
               for r in config.get("tournament.margin_mm")}
    matched |= {f"volume{int(v)}ml": metrics.voxels_for_volume(float(v), (sz, sy, sx))
                for v in config.get("tournament.fixed_volume_ml")}

    scores = {name: rule.score(x) for name, rule in candidates.RULES.items()}
    for name, cand in candidates.LEARNED.items():
        scores[name] = candidates.predict(cand, models[name], x)
    scores["C6"] = candidates.rank_average([scores[m] for m in candidates.ENSEMBLE_MEMBERS])

    escape = f1 > 20.0   # plan Step 2.3: the escape region, > 20 mm from the target
    row: dict = {"evaluable": bool(truth.any()), "n_positive": int(truth.sum()),
                 "n_positive_in_target": int((inp.recurrence & inp.target).sum()),
                 "prevalence": float(truth.mean()), "matched_voxels": matched,
                 "escape_positive": int((truth & escape).sum())}
    positives = max(int(truth.sum()), 1)
    row["share_beyond_15mm"] = float((truth & (c0t > 15)).sum() / positives)
    row["share_beyond_20mm"] = float((truth & (c0t > 20)).sum() / positives)
    for name, s in scores.items():
        m = metrics.patient_scores(s, truth, matched)
        m["pr_auc_escape"] = (metrics.pr_auc(s[escape], truth[escape])
                              if (truth & escape).any() else float("nan"))
        # Of the recurrence beyond the 15 mm margin, the share inside this
        # candidate's matched-volume region (descriptive, no test).
        beyond = truth & (c0t > 15)
        if beyond.any():
            order = np.argsort(-s, kind="stable")[: matched["margin15"]]
            top = np.zeros(s.size, bool)
            top[order] = True
            m["catches_beyond_15mm"] = float((top & beyond).sum() / beyond.sum())
        row[name] = m
    if truth.any():
        per_ml = matched["volume50ml"] / 50.0
        row["curves"] = {name: metrics.coverage_curve(s, truth, per_ml)
                         for name, s in scores.items()}
    return row, scores, qc


def write_maps(out_dir: Path, patient: str, reference: sitk.Image, evaluate: np.ndarray,
               scores: dict[str, np.ndarray], volumes_ml=(50, 100)) -> dict[str, str]:
    """Step 4.6: probability map and top-ranked masks per candidate.

    The plan names float16. SimpleITK cannot write float16 NIfTI, so values are
    rounded to float16 precision and stored as float32 (the size cost is
    compression-dependent and small; the app export downsamples anyway).
    """
    sx, sy, sz = reference.GetSpacing()
    written = {}
    idx = np.flatnonzero(evaluate)
    for name, s in scores.items():
        vol = np.zeros(evaluate.shape, np.float16)
        vol.flat[idx] = s.astype(np.float16)
        img = sitk.GetImageFromArray(vol.astype(np.float32))
        img.CopyInformation(reference)
        path = out_dir / patient / f"{name}_map.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.Cast(img, sitk.sitkFloat32), str(path), useCompression=True)
        written[f"{name}_map"] = path.name
        order = np.argsort(-s, kind="stable")
        for v in volumes_ml:
            n = metrics.voxels_for_volume(float(v), (sz, sy, sx))
            mask = np.zeros(evaluate.shape, np.uint8)
            mask.flat[idx[order[:n]]] = 1
            mimg = sitk.GetImageFromArray(mask)
            mimg.CopyInformation(reference)
            mpath = out_dir / patient / f"{name}_top{v}ml.nii.gz"
            sitk.WriteImage(mimg, str(mpath), useCompression=True)
            written[f"{name}_top{v}ml"] = mpath.name
    return written
