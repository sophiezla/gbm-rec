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
from gbmc.imaging import zscore_array


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


REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_FIELD = REPO_ROOT / "data" / "interim" / "rhuh" / "label_field_tuned"


def load_patient(patient: str, mapping: dict[int, str], label_dir: Path = LABEL_FIELD,
                 positive: tuple[str, ...] = ("enhancing",), target: np.ndarray | None = None,
                 with_recurrence: bool = True) -> PatientInputs:
    """One RHUH-GBM patient on the early post-op grid.

    ``positive`` names the label(s) counted as recurrence. With ``with_recurrence=False``
    the recurrence file is never opened and ``recurrence`` is all False: that is how C5's
    inputs are packed before the confirmatory run (``24_rhuh_c5_pack.py``).

    T1 for the falx white-matter test is z-scored inside the brain, because the test's
    threshold (``anatomy.WHITE_MATTER_Z``) is on the z scale, as on LUMIERE's preprocessed
    images. The plan says RHUH-GBM's own pipeline z-scores its images, but over which region
    is not verified (no file downloaded); z-scoring again inside this brain mask puts it on
    LUMIERE's scale either way, and is a no-op if the files are already z-scored there.
    """
    from gbmc import rhuh

    post = rhuh.files(patient, "postop")
    reference = sitk.ReadImage(str(post["T1ce"]), sitk.sitkFloat32)
    arr = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(str(p)))  # noqa: E731
    seg = arr(post["seg"])
    brain = arr(label_dir / patient / "brain_mask.nii.gz") > 0
    if target is None:
        target = arr(label_dir / patient / "target_mask.nii.gz") > 0
    peri = next(v for v, n in mapping.items() if n == "peritumoral")
    if with_recurrence:
        rec = arr(label_dir / patient / "recurrence_labels.nii.gz")
        pos_values = [v for v, n in mapping.items() if n in positive]
        recurrence = np.isin(rec, pos_values) & brain
    else:
        recurrence = np.zeros(brain.shape, bool)
    t1 = arr(post[config.get("rhuh.falx_t1_sequence")]).astype(np.float32)
    return PatientInputs(
        patient=patient, reference=reference, brain=brain, target=target,
        flair_abnormal=(seg == peri) & brain, t1_z=zscore_array(t1, brain),
        lesion=seg > 0, recurrence=recurrence)


def alt_cavity_target(patient: str, mapping: dict[int, str], label_dir: Path,
                      master: int) -> np.ndarray:
    """The other branch of the Step 2.2 cavity rule (sensitivity arm ``alt_cavity``, D13)."""
    from gbmc import rhuh
    from gbmc.rng import stable_seed

    post = rhuh.files(patient, "postop")
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(post["seg"])))
    brain = sitk.GetArrayFromImage(sitk.ReadImage(
        str(label_dir / patient / "brain_mask.nii.gz"))) > 0
    core = rhuh.carried_preop_core(patient, mapping,
                                   stable_seed(patient, "rhuh-core", master_seed=master))
    if rhuh.cavity_rule() is None:
        # Primary: residual + the whole carried pre-op core. Alternative (D13, proposed):
        # residual + only the fluid-dark part of that core, the cavity as the post-op
        # image shows it.
        # The threshold is on the z scale, z-scored over non-zero voxels exactly as in
        # 20_'s cavity inspection, which is where the threshold is chosen.
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(post["T1ce"]))).astype(np.float32)
        dark = zscore_array(t1, t1 != 0) < float(config.get("rhuh.cavity_dark_t1ce_z"))
        target, _ = rhuh.treatment_target(seg, brain, None, core & dark, mapping)
    else:
        # Primary: enhancing + labelled cavity. Alternative: the carried-core rule.
        target, _ = rhuh.treatment_target(seg, brain, None, core, mapping)
    return target


def _json(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def transported(label_dir: Path) -> list[str]:
    """Patients whose transport finished with ``qc.json`` ok."""
    return sorted(p.name for p in label_dir.iterdir()
                  if (p / "qc.json").is_file() and _json(p / "qc.json").get("ok"))


def sensitivity_arms(repo_root: Path = REPO_ROOT) -> dict[str, dict]:
    """The pre-registered sensitivity arms (``31_``), from the Phase 3 records.

    Each arm names its patients, its label folder, and the C5 input variant whose maps it
    scores with (``gbmc.c5_import.RHUH_VARIANTS``): C5's inputs change only when the
    target or the transported label folder changes.
    """
    rhuh_dir = repo_root / "data" / "interim" / "rhuh"
    inv = _json(repo_root / "results/rhuh_inventory/inventory.json")
    lock = _json(repo_root / "results/rhuh_lock/consort.json")
    screened_out = {r["patient"] for r in
                    _json(repo_root / "results/rhuh_screens/verdict.json")["rows"]
                    if r["status"] == "EXCLUDE"}
    secondary = {r["patient"] for r in inv["rows"] if r.get("secondary_facility")}
    tuned, original = rhuh_dir / "label_field_tuned", rhuh_dir / "label_field_original"
    return {
        "unscreened": dict(patients=transported(tuned), label_dir=tuned, c5="tuned"),
        "original_recipe": dict(patients=[p for p in transported(original)
                                          if p not in screened_out],
                                label_dir=original, c5="original"),
        "alt_cavity": dict(patients=lock["evaluable"], label_dir=tuned, alt_cavity=True,
                           c5="alt_cavity"),
        "enh_necrosis": dict(patients=lock["evaluable"], label_dir=tuned,
                             positive=("enhancing", "necrosis"), c5="tuned"),
        "no_secondary": dict(patients=[p for p in lock["evaluable"] if p not in secondary],
                             label_dir=tuned, c5="tuned"),
    }


def c5_variants(arms: dict[str, dict], primary: list[str] | None = None) -> dict[str, list[str]]:
    """Patients per C5 input variant: every patient any arm (or the primary) scores with it."""
    out: dict[str, set[str]] = {}
    for spec in arms.values():
        out.setdefault(spec["c5"], set()).update(spec["patients"])
    if primary is not None:
        out.setdefault("tuned", set()).update(primary)
    return {variant: sorted(patients) for variant, patients in out.items()}


def load_models(release_dir: Path, register: dict[str, str]) -> dict:
    """Every learned candidate, each loaded only after its SHA-256 matches (Step 4.2)."""
    return {name: freeze.load_model(release_dir / f"models/{name}.joblib",
                                    register[f"models/{name}.joblib"])
            for name in candidates.LEARNED}


def score_patient(inp: PatientInputs, models: dict, c5_score: np.ndarray | None = None
                  ) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Metrics per candidate, voxel scores per candidate, and feature QC.

    ``c5_score`` is C5's imported map for this patient (``gbmc.c5_import.rhuh_scores``),
    one value per evaluation voxel in C order; C5 runs on the GPU computer (D6 item 9).
    """
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
    if c5_score is None:
        raise RuntimeError("C5 scores missing: C6 cannot be formed with its fixed members")
    if c5_score.shape != (int(evaluate.sum()),):
        raise RuntimeError(f"{inp.patient}: {c5_score.size} C5 scores for "
                           f"{int(evaluate.sum())} evaluation voxels")
    scores["C5"] = np.asarray(c5_score, dtype=np.float64)
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
