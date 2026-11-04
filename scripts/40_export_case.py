#!/usr/bin/env python
"""Export LUMIERE cases for the explainer app (plan Step 5.4).

For each case in ``config.yaml:app.lumiere_cases`` (gbm-4d's hero patients),
writes ``app/public/cases/<case>/``:

* one GLB per layer per timepoint (brain, target, FLAIR region, 15 and 20 mm
  guideline margins, recurrence at each transported progression, and each
  candidate's top-ranked region at the 15 mm margin's volume when the held-out
  maps from ``02_`` exist);
* ``t1ce.nii.gz`` and ``flair.nii.gz``: 2 mm windowed slice volumes (baseline);
* ``case.json`` (schema ``gbmc/case/1``): cohort, licence and credit, the
  timepoints with their week interval, layer files with colour roles, and the
  SHA-256 of every source file.

Fails a case above 5 MB. Recurrence is shown only as a real label, never as a
prediction, and no score is computed here: per-model numbers enter case.json
only from ``results/``.

    python scripts/40_export_case.py
    python scripts/40_export_case.py --patients Patient-038
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gbmc import candidates, config, export, lumiere
from gbmc.progressions import schedule

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "app" / "public" / "cases"
FEATURES = REPO_ROOT / "data" / "interim" / "features"
OOF = REPO_ROOT / "data" / "interim" / "oof"
PDE = REPO_ROOT / "data" / "interim" / "pde_frames"
PDE_FACES = 6_000   # per frame; nine frames must share the 5 MB case budget
LATER = REPO_ROOT / "data" / "interim" / "label_field_later"
CREDIT = ("LUMIERE (Suter et al., Scientific Data 2022, doi:10.1038/s41597-022-01881-7), "
          "CC0")


def _week(timepoint: str) -> int:
    return lumiere.timepoint_sort_key(timepoint)[0]


def _source_name(path: Path) -> str:
    for label, root in (("derived", lumiere.derived_root()), ("gbm-rec", REPO_ROOT)):
        try:
            return f"{label}:{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return path.name


def _centroid_ras(mask: np.ndarray, reference: sitk.Image) -> list[float] | None:
    """Centroid in RAS mm (NIfTI/NiiVue convention); ITK physical space is LPS."""
    if not mask.any():
        return None
    z, y, x = np.argwhere(mask).mean(axis=0)
    lps = reference.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
    return [round(-lps[0], 2), round(-lps[1], 2), round(lps[2], 2)]


def _scores(patient: str) -> dict | None:
    """This case's per-candidate numbers from the tournament, labelled as development."""
    path = REPO_ROOT / "results" / "tournament" / "per_patient.json"
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8")).get(patient)
    if row is None or not row["evaluable"]:
        return None
    keep = ("pr_auc", "lift", "coverage_margin15")
    return {"status": "development (LUMIERE, held-out fold)",
            "coverage_denominator": "recurrence outside the target",
            "candidates": {c: {k: row[c][k] for k in keep}
                           for c in candidates.ORDER if c in row}}


def results_summary() -> dict:
    """app/public/results.json: what the app may say, read only from results/."""
    def read(rel):
        path = REPO_ROOT / "results" / rel
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    board, confirm = read("tournament/leaderboard.json"), read("confirm/confirm.json")
    doc = {"label": export.FIXED_LABEL,
           "status": "validation" if confirm else ("development" if board else "pending"),
           "order": list(candidates.ORDER)}
    if board:
        doc["development"] = {
            "cohort": "LUMIERE", "challenger": board["challenger"],
            "delta_pr_auc_vs_C0": {c: v["pr_auc"].get("delta_vs_C0")
                                   for c, v in board["candidates"].items()}}
    if confirm:
        doc["validation"] = {
            "cohort": "RHUH-GBM", "n_evaluable": confirm["n_evaluable"],
            "challenger": confirm["challenger"], "champion": confirm["champion"],
            "outcome": confirm["outcome"], "sentence": confirm["sentence"],
            "delta_pr_auc_vs_C0": {c: v["pr_auc"].get("delta_vs_C0")
                                   for c, v in confirm["candidates"].items()
                                   if not c.startswith("_")},
            "hero_coverage_margin15": {c: confirm["candidates"][c]["coverage_margin15"]["mean"]
                                       for c in ("C0T", "C0", confirm["challenger"])},
            "coverage_denominator": "recurrence outside the target"}
    return doc


def export_case(patient: str) -> dict:
    case = lumiere.load_baseline(patient)
    ref = case.reference
    out = CASES / patient
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with np.load(FEATURES / f"{patient}.npz") as z:
        shape = tuple(z["evaluate_shape"])
        evaluate = np.unpackbits(z["evaluate_packed"])[: int(np.prod(shape))
                                                       ].reshape(shape).astype(bool)
        c0t = np.zeros(shape, np.float32)
        c0t[evaluate] = z["matrix"][:, list(candidates.COLUMNS).index("c0t")]
        matched = dict(zip(z["matched_labels"].tolist(), z["matched_voxels"].tolist(),
                           strict=True))

    layers = {
        "brain": ("brain", case.brain),
        "target": ("target", case.target),
        "flair": ("flair_region", case.flair_abnormal),
        "margin15": ("guideline_margin", (evaluate & (c0t <= 15)) | case.target),
        "margin20": ("guideline_margin_20", (evaluate & (c0t <= 20)) | case.target),
    }
    row = schedule([patient])[0]
    timepoints = [{"id": row["baseline"], "role": "post-op baseline", "weeks_from_baseline": 0}]
    first = lumiere.load_recurrence(patient)
    labelled = [(row["first"], first, "first progression")]
    for index, tp in enumerate(row["later"], start=2):
        path = LATER / tp / patient / "recurrence_labels.nii.gz"
        if path.is_file():
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkUInt8))
            labelled.append((tp, arr == lumiere.ENHANCING, f"progression {index}"))
    for tp, mask, role in labelled:
        timepoints.append({"id": tp, "role": role,
                           "weeks_from_baseline": _week(tp) - _week(row["baseline"]),
                           "recurrence_centroid_ras_mm": _centroid_ras(mask & case.brain, ref)})
        layers[f"recurrence_{tp}"] = ("recurrence", mask & case.brain)

    oof = OOF / f"{patient}.npz"
    if oof.is_file():
        with np.load(oof) as z:
            for name in z.files:
                scores = z[name]
                n = int(matched["margin15"])
                top = np.zeros(shape, bool)
                top[np.flatnonzero(evaluate)[np.argsort(-scores, kind="stable")[:n]]] = True
                layers[f"model_{name}"] = ("model_prediction", top)

    files = []
    for key, (role, mask) in layers.items():
        budget = export.FACE_BUDGET.get(key, export.DEFAULT_FACE_BUDGET)
        mesh = export.surface(mask, ref, budget)
        if mesh is None:
            continue
        export.write_glb(mesh, out / f"{key}.glb")
        files.append({"layer": key, "role": role, "file": f"{key}.glb",
                      "faces": int(len(mesh.faces))})

    pde = None
    frames_path = PDE / f"{patient}.npz"
    if frames_path.is_file():
        threshold = float(config.get("pde.c_density_flair"))
        with np.load(frames_path) as z:
            box = tuple(slice(int(a), int(b)) for a, b in z["box"])
            pde = {"label": "what a diffusion model draws", "model": "Fisher-KPP",
                   "d_white_mm2_per_day": float(z["params"][0]),
                   "rho_per_day": float(z["params"][1]),
                   "surface": f"density >= {threshold} (FLAIR detection threshold)",
                   "frames": []}
            for frame, day in zip(z["frames"], z["days"], strict=True):
                full = np.zeros(shape, np.float32)
                full[box] = frame
                mesh = export.surface((full >= threshold) & case.brain, ref, PDE_FACES)
                if mesh is None:
                    continue
                name = f"pde_day_{int(round(float(day))):03d}.glb"
                export.write_glb(mesh, out / name)
                pde["frames"].append({"day": float(day), "file": name})
        sources_extra = [frames_path]
    else:
        sources_extra = []

    base = lumiere.derived_root() / "data" / "interim" / "preprocessed" / patient / row["baseline"]
    export.slice_volume(ref, out / "t1ce.nii.gz", case.brain)
    export.slice_volume(sitk.ReadImage(str(base / "FLAIR.nii.gz"), sitk.sitkFloat32),
                        out / "flair.nii.gz", case.brain)
    overlays = {}
    for tp, mask, _ in labelled:
        name = f"recurrence_{tp}.nii.gz"
        export.mask_volume(mask & case.brain, ref, out / name)
        overlays[tp] = name

    sources = [base / "CT1.nii.gz", base / "FLAIR.nii.gz", base / "brain_mask.nii.gz",
               lumiere.derived_root() / "data" / "interim" / "label_field" / patient
               / "recurrence_labels.nii.gz", FEATURES / f"{patient}.npz"]
    if oof.is_file():
        sources.append(oof)
    sources += sources_extra
    doc = {
        "schema": export.SCHEMA, "case": patient, "cohort": "LUMIERE",
        "licence": "CC0", "credit": CREDIT, "label": export.FIXED_LABEL,
        "frame": "physical mm (ITK LPS) of the baseline CT1 grid",
        "timepoints": timepoints, "layers": files,
        "slices": {"t1ce": "t1ce.nii.gz", "flair": "flair.nii.gz", "spacing_mm": 2.0,
                   "recurrence_overlays": overlays},
        "pde": pde,
        "scores": _scores(patient),
        "sources": {_source_name(p): export.sha256(p) for p in sources},
    }
    (out / "case.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    size = export.case_bytes(out)
    if size > export.CASE_BYTES_LIMIT:
        raise SystemExit(f"{patient}: case is {size / 2**20:.1f} MB, over the 5 MB budget")
    return {"patient": patient, "megabytes": round(size / 2 ** 20, 2), "layers": len(files)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    args = ap.parse_args()
    config.pin_itk_threads()
    patients = args.patients or config.get("app.lumiere_cases")
    for patient in patients:
        info = export_case(patient)
        print(f"{patient}: {info['layers']} layers, {info['megabytes']} MB", flush=True)
    (CASES.parent / "results.json").write_text(json.dumps(results_summary(), indent=2) + "\n",
                                              encoding="utf-8")
    existing = sorted(p.name for p in CASES.iterdir() if (p / "case.json").is_file())
    (CASES / "index.json").write_text(json.dumps({"cases": existing}, indent=2) + "\n",
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
