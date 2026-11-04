#!/usr/bin/env python
"""Frames of a Fisher-KPP simulation for the app's PDE mode (plan Step 5.3).

"Plays the Fisher-KPP simulation from day 0 to day 206, labelled 'what a diffusion model
draws'." Parameters are in ``config.yaml:pde``. Nothing is fitted here.

Writes ``data/interim/pde_frames/<patient>.npz`` (float32 frames on the brain box, the box,
the days). ``40_`` turns them into meshes.

    python scripts/42_pde_frames.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gbmc import anatomy, config, lumiere, pde

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "interim" / "pde_frames"
N_FRAMES = 9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None)
    args = ap.parse_args()
    config.pin_itk_threads()
    d_white = float(config.get("pde.d_white_mm2_per_day"))
    rho_value = float(config.get("pde.rho_per_day"))
    horizon = float(config.get("pde.horizon_days"))

    OUT.mkdir(parents=True, exist_ok=True)
    for patient in args.patients or config.get("app.lumiere_cases"):
        case = lumiere.load_baseline(patient)
        d_field, rho, c0 = pde.build_fields(
            case.brain, case.t1, case.seg == lumiere.ENHANCING, case.seg == lumiere.OEDEMA,
            case.seg == lumiere.NECROTIC_NONENHANCING, d_white, rho_value,
            float(config.get("pde.white_to_grey_ratio")), anatomy.WHITE_MATTER_Z,
            float(config.get("pde.c_density_t1c")), float(config.get("pde.c_density_flair")))
        where = np.argwhere(case.brain)
        box = tuple(slice(int(low), int(high) + 1)
                    for low, high in zip(where.min(0), where.max(0), strict=True))
        limit = pde.stable_dt(1.0, float(d_field.max()), float(config.get("pde.cfl_safety")))
        n_steps = int(np.ceil(horizon / limit))
        dt = horizon / n_steps
        cuts = np.unique(np.round(np.linspace(0, n_steps, N_FRAMES)).astype(int))
        c = c0[box]
        frames, days = [c.astype(np.float32)], [0.0]
        for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
            c = pde.advance(c, d_field[box], rho[box], case.brain[box], dt, int(stop - start))
            frames.append(c.astype(np.float32))
            days.append(round(float(stop) * dt, 1))
        np.savez_compressed(OUT / f"{patient}.npz", frames=np.stack(frames), days=np.array(days),
                            box=np.array([[s.start, s.stop] for s in box]),
                            params=np.array([d_white, rho_value]))
        print(f"{patient}: {len(frames)} frames to day {days[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
