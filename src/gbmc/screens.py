"""Registration screens (plan Step 3.4). They can exclude a patient, never certify one.

Ported from gbm-wm ``scripts/14_mugp_registration_screens.py`` (itself ported
from gbm-4d 10_ and 12_), as functions that take arrays, so they are testable
without data:

* :func:`consensus` -- multi-arbiter: the transport recipe re-driven by other
  sequences plus the shipped atlas alignment; per-voxel median consensus; each
  estimate's deviation from it; the dissenter;
* :func:`synthetic_field` -- a known smooth deformation (RMS 3 mm, sigma 8 mm,
  capped, tapered at the brain edge) for the simulated-recovery screen;
* :func:`recovery_error` -- recovered vs true displacement, off the rim;
* :func:`verdict` -- combine per-patient screen results into EXCLUDE / REVIEW /
  SCREENED with gbm-wm's rules. Thresholds are arguments: they are fixed in
  Phase 2 (``config.yaml:screens``), never inside this module.

The brain-surface screen itself is gbm-4d's ``ventricles.surface_distance_mm``
applied after and before transport, and needs no new code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


def consensus(fields: dict[str, np.ndarray], mask: np.ndarray,
              peritumoral: np.ndarray | None = None, ours: str = "T1c") -> dict:
    """Median consensus of displacement fields (z, y, x, 3) and each one's deviation."""
    if len(fields) < 3:
        return {"verdict": "undetermined", "estimates": sorted(fields)}
    names = sorted(fields)
    stack = np.stack([fields[k] for k in names], axis=0)
    centre = np.median(stack, axis=0)
    dev = {k: float(np.median(np.linalg.norm(fields[k] - centre, axis=-1)[mask])) for k in names}
    dev_peri = None
    if peritumoral is not None and int(peritumoral.sum()) > 50:
        dev_peri = {k: float(np.median(np.linalg.norm(fields[k] - centre, axis=-1)[peritumoral]))
                    for k in names}
    pairwise = {f"{a}|{b}": float(np.median(np.linalg.norm(fields[a] - fields[b], axis=-1)[mask]))
                for a, b in itertools.combinations(names, 2)}
    dissenter = max(dev, key=dev.get)
    others = [v for k, v in dev.items() if k != dissenter]
    return {"estimates": names, "deviation_from_consensus_mm": dev,
            "deviation_peritumoral_mm": dev_peri, "pairwise_median_mm": pairwise,
            "dissenter": dissenter, "dissenter_mm": dev[dissenter],
            "our_deviation_mm": dev.get(ours), "consensus_spread_mm": float(np.median(others))}


def synthetic_field(shape: tuple[int, int, int], spacing_zyx, brain: np.ndarray, seed: int,
                    rms_mm: float = 3.0, sigma_mm: float = 8.0, cap_mm: float = 8.0,
                    taper_mm: float = 10.0) -> np.ndarray:
    """Smooth random displacement (z, y, x, 3) in mm, zero outside the brain (gbm-wm 15_)."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(spacing_zyx, dtype=float)
    f = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape), sigma_mm / sp)
                  for _ in range(3)], -1)
    rms = np.sqrt((np.linalg.norm(f[brain], axis=-1) ** 2).mean())
    f *= rms_mm / max(rms, 1e-9)
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    f = np.where(norm > cap_mm, f * cap_mm / np.maximum(norm, 1e-9), f)
    depth = ndimage.distance_transform_edt(brain) * sp.min()
    f = f * np.clip(depth / taper_mm, 0.0, 1.0)[..., None]
    f[~brain] = 0.0
    return f


def recovery_error(recovered: np.ndarray, truth: np.ndarray, brain: np.ndarray,
                   spacing_mm: float, rim_mm: float = 4.0) -> dict:
    """Residual |recovered - truth| inside the brain, excluding the outer rim."""
    mask = brain & (ndimage.distance_transform_edt(brain) * spacing_mm >= rim_mm)
    err = np.linalg.norm(recovered - truth, axis=-1)[mask]
    true = np.linalg.norm(truth, axis=-1)[mask]
    return {"true_median_mm": float(np.median(true)), "error_median_mm": float(np.median(err)),
            "error_p95_mm": float(np.percentile(err, 95)),
            "recovered_fraction": float(1 - np.median(err) / max(np.median(true), 1e-9))}


@dataclass
class ScreenResult:
    patient: str
    roundtrip_median_mm: float | None
    surface_after_mm: float | None
    surface_before_mm: float | None
    dissenter: str | None = None
    our_deviation_mm: float | None = None
    flags: list[str] = field(default_factory=list)


def verdict(r: ScreenResult, roundtrip_max_mm: float, surface_review_mm: float,
            worse_margin_mm: float, ours: str = "T1c") -> str:
    """gbm-wm's combination rule (its D10/D16 lineage), with Phase-2 thresholds.

    EXCLUDE when the round trip exceeds its threshold, or when a surface flag is
    corroborated by our transform being the arbiters' dissenter. REVIEW for an
    uncorroborated flag. SCREENED otherwise -- never "licensed".
    """
    r.flags = []
    if r.roundtrip_median_mm is not None and r.roundtrip_median_mm > roundtrip_max_mm:
        r.flags.append(f"roundtrip>{roundtrip_max_mm}mm")
    surface_flag = False
    if r.surface_after_mm is not None and r.surface_before_mm is not None:
        if r.surface_after_mm > r.surface_before_mm + worse_margin_mm:
            r.flags.append("surface worse than identity")
            surface_flag = True
        elif r.surface_after_mm > surface_review_mm:
            r.flags.append(f"surface {r.surface_after_mm:.1f}mm>review")
            surface_flag = True
    dissents = r.dissenter == ours
    if dissents:
        r.flags.append("our transform dissents")
    if any(f.startswith("roundtrip") for f in r.flags) or (surface_flag and dissents):
        return "EXCLUDE"
    return "REVIEW" if r.flags else "SCREENED"
