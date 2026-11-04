"""Fisher-KPP simulation for the app's PDE mode (plan Step 5.3). Not a tournament candidate.

    dc/dt = div(D grad c) + rho c (1 - c)

Written in flux form so mass is conserved, with no flux across any face that touches a
non-brain voxel. Stays in float64: Fisher fronts are set by their tiny leading edge, and
float32 roundoff there makes the front run too fast with no visible warning.
"""

from __future__ import annotations

import numpy as np


def stable_dt(spacing_mm: float, d_max: float, safety: float = 0.5, ndim: int = 3) -> float:
    """A stable explicit time step in days: ``safety * dx^2 / (2 * ndim * D_max)``."""
    if d_max <= 0:
        raise ValueError(f"d_max must be positive, got {d_max}")
    if not 0 < safety <= 1:
        raise ValueError(f"safety must be in (0, 1], got {safety}")
    return safety * spacing_mm**2 / (2 * ndim * d_max)


def advance(c: np.ndarray, d_field: np.ndarray, rho: np.ndarray, brain: np.ndarray, dt: float,
            n_steps: int, spacing_mm: float = 1.0) -> np.ndarray:
    """Return ``c`` after ``n_steps`` explicit Euler steps of ``dt`` days."""
    c = np.array(c, dtype=np.float64)
    d_field = np.asarray(d_field, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    brain = np.asarray(brain, dtype=bool)
    axes = []
    for axis in range(c.ndim):
        low = tuple(slice(None, -1) if a == axis else slice(None) for a in range(c.ndim))
        high = tuple(slice(1, None) if a == axis else slice(None) for a in range(c.ndim))
        conductance = 0.5 * (d_field[low] + d_field[high]) * (brain[low] & brain[high])
        axes.append((low, high, conductance))
    for _ in range(n_steps):
        change = rho * c * (1.0 - c)
        for low, high, conductance in axes:
            flux = conductance * (c[high] - c[low]) / spacing_mm**2
            change[low] += flux
            change[high] -= flux
        c += dt * change
        np.clip(c, 0.0, 1.0, out=c)
        c[~brain] = 0.0
    return c


def build_fields(brain: np.ndarray, t1_z: np.ndarray, enhancing: np.ndarray, oedema: np.ndarray,
                 cavity: np.ndarray, d_white: float, rho_value: float, ratio: float,
                 white_matter_z: float, c_t1c: float, c_flair: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D(x) (``d_white`` in white matter, ``d_white / ratio`` in grey), uniform rho and c0."""
    d_field = np.where(brain, d_white / ratio, 0.0)
    d_field[brain & (t1_z > white_matter_z)] = d_white
    rho = np.where(brain, rho_value, 0.0)
    c0 = np.zeros(brain.shape, dtype=np.float64)
    c0[enhancing] = c_t1c
    c0[oedema] = c_flair
    c0[cavity | ~brain] = 0.0
    return d_field, rho, c0
