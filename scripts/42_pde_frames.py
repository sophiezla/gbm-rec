#!/usr/bin/env python
"""Frames of gbm-4d's Fisher-KPP simulation for the app's PDE mode (plan Step 5.3).

"Plays the Fisher-KPP simulation from day 0 to day 206, labelled 'what a
diffusion model draws'." gbm-4d saved only the final field, so the frozen
simulation is re-run, not re-fitted:

* inputs and fields come from gbm-4d ``scripts/23_arm1_fkpp.py`` itself
  (``load_patient``, ``build_fields``), loaded through the verified frozen
  loader, at gbm-4d's population choice D = 0.5 mm^2/day, rho = 0.002/day
  (its 27_/28_, read from its ``results/test_set/test_set.json``);
* the horizon is split into frames with the SAME time step gbm-4d's single
  run used (dt = horizon / n_steps from its CFL rule), so the step sequence is
  identical, and the final frame is checked against gbm-4d's saved density.

Writes ``data/interim/pde_frames/<patient>.npz`` (float32 frames on the
brain box, the box, the days). ``40_`` turns them into meshes.

    python scripts/42_pde_frames.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gbmc import config, lumiere
from gbmc.progressions import frozen_gbm4d_script

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "interim" / "pde_frames"
N_FRAMES = 9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    args = ap.parse_args()
    config.pin_itk_threads()
    g4 = lumiere.gbm4d_root()
    arm1 = frozen_gbm4d_script("23_arm1_fkpp", {})
    from gbm4d import config as g4config
    from gbm4d import fkpp

    point = json.loads((g4 / "results/test_set/test_set.json").read_text(encoding="utf-8"))[
        "arm1_population_point"]
    horizon = float(g4config.get("fkpp.horizon_days"))
    ratio = float(g4config.get("fkpp.white_to_grey_diffusivity_ratio"))
    safety = float(g4config.get("fkpp.cfl_safety_factor"))
    from gbm4d.barriers import DEFAULT_WHITE_MATTER_Z

    OUT.mkdir(parents=True, exist_ok=True)
    for patient in args.patients or config.get("app.lumiere_cases"):
        # gbm-4d cached the opened test patients' features separately.
        for cache in ("features", "features_test"):
            arm1.FEATURE_CACHE = g4 / "data" / "interim" / cache
            data = arm1.load_patient(patient)
            if data is not None:
                break
        if data is None:
            print(f"{patient}: no gbm-4d inputs", flush=True)
            continue
        d_field, rho, c0 = arm1.build_fields(data, point["d_white_mm2_per_day"],
                                             point["rho_per_day"], ratio, DEFAULT_WHITE_MATTER_Z)
        box = data["box"]
        dt_cfl = fkpp.cfl_dt(1.0, float(np.max(d_field)), safety=safety, ndim=3)
        n_steps = int(np.ceil(horizon / dt_cfl))
        dt = horizon / n_steps
        cuts = np.unique(np.round(np.linspace(0, n_steps, N_FRAMES)).astype(int))
        c = c0[box].astype(float)
        frames, days = [c.astype(np.float32)], [0.0]
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            c = fkpp.simulate_fast(c, d_field[box], rho[box], (b - a) * dt,
                                   data["brain"][box], spacing_mm=1.0, dt_days=dt).density
            frames.append(c.astype(np.float32))
            days.append(round(b * dt, 1))
        saved = g4 / "results/test_set/arm1_fields" / f"{patient}_density.nii.gz"
        if not saved.is_file():
            saved = g4 / "data/interim/arm1_fields" / f"{patient}_density.nii.gz"
        check = None
        if saved.is_file():
            import SimpleITK as sitk
            ref = sitk.GetArrayFromImage(sitk.ReadImage(str(saved)))[box]
            check = float(np.abs(ref - frames[-1]).max())
        np.savez_compressed(OUT / f"{patient}.npz", frames=np.stack(frames),
                            days=np.array(days), box=np.array([[s.start, s.stop] for s in box]),
                            params=np.array([point["d_white_mm2_per_day"], point["rho_per_day"]]))
        print(f"{patient}: {len(frames)} frames to day {days[-1]}; max |final - gbm-4d saved| = "
              f"{check}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
