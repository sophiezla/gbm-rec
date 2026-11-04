"""Case export for the explainer app (plan Step 5.4).

Masks become GLB surfaces: marching cubes at the voxel grid, Taubin smoothing
(no shrinkage, unlike plain Laplacian), then a face budget. Vertices are placed
in the reference image's physical frame (mm), shared by every layer and
timepoint of a case.

Face budget without a new dependency: trimesh's quadric decimation needs the
``fast_simplification`` package, which is not in the lock. Instead the
marching-cubes step size is raised until the surface fits the budget, which
coarsens the mesh uniformly (D10, proposed).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SCHEMA = "gbmc/case/1"
FIXED_LABEL = ("Research visualization of a pre-registered method comparison on public data. "
               "Not a treatment prediction.")
#: Plan Step 5.4: brain <= 60k faces, other layers <= 20k.
FACE_BUDGET = {"brain": 60_000}
DEFAULT_FACE_BUDGET = 20_000
#: Plan Step 5.4: about 5 MB per case; the export fails above it.
CASE_BYTES_LIMIT = 5 * 1024 * 1024


def index_to_physical(reference: sitk.Image) -> np.ndarray:
    """4x4 matrix mapping (z, y, x, 1) array indices to physical (x, y, z, 1) mm."""
    direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    origin = np.asarray(reference.GetOrigin(), dtype=float)
    m = np.eye(4)
    # physical = origin + D S (x, y, z); array index order is (z, y, x)
    m[:3, :3] = (direction * spacing)[:, ::-1]
    m[:3, 3] = origin
    return m


def surface(mask: np.ndarray, reference: sitk.Image, max_faces: int,
            smooth_iterations: int = 10):
    """A smoothed trimesh surface of ``mask`` within ``max_faces``; None if empty."""
    import trimesh
    from skimage.measure import marching_cubes

    if mask.sum() < 8:
        return None
    padded = np.pad(mask.astype(np.float32), 1)
    for step in range(1, 9):
        verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step,
                                            allow_degenerate=False)
        if len(faces) <= max_faces:
            break
    else:
        raise ValueError(f"cannot fit surface in {max_faces} faces even at step 8")
    verts = verts - 1.0                                   # undo the padding
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iterations)
    homogeneous = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    mesh.vertices = (homogeneous @ index_to_physical(reference).T)[:, :3]
    mesh.fix_normals()
    return mesh


def write_glb(mesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mesh.export(file_type="glb"))


def slice_volume(image: sitk.Image, path: Path, brain: np.ndarray | None = None,
                 spacing_mm: float = 2.0, window=(-3.0, 5.0)) -> None:
    """Downsampled, windowed uint8 copy of a z-scored image, as compressed NIfTI.

    Outside ``brain`` the value is 0 (black). Without the mask, background 0 in
    z-score units would land mid-window and draw grey.
    """
    if brain is not None:
        arr = sitk.GetArrayFromImage(image)
        masked = sitk.GetImageFromArray(np.where(brain, arr, window[0]).astype(np.float32))
        masked.CopyInformation(image)
        image = masked
    size = [int(round(sz * sp / spacing_mm)) for sz, sp in zip(image.GetSize(),
                                                               image.GetSpacing(), strict=True)]
    resampled = sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                              [spacing_mm] * 3, image.GetDirection(), float(window[0]),
                              sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(resampled)
    lo, hi = window
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
    out = sitk.GetImageFromArray(scaled.astype(np.uint8))
    out.CopyInformation(resampled)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path), useCompression=True)


def mask_volume(mask: np.ndarray, reference: sitk.Image, path: Path,
                spacing_mm: float = 2.0) -> None:
    """A binary mask on the slice grid, for NiiVue overlays.

    A 2 mm voxel is on when any 1 mm voxel within its footprint is on: the
    mask is dilated by one voxel, then sampled at the 2 mm centres (which
    coincide with 1 mm centres, so plain interpolation would average nothing).
    A recurrence of a few millimetres therefore still shows. Display only:
    every score is computed on the 1 mm grid.
    """
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    image = sitk.BinaryDilate(image, [1, 1, 1], sitk.sitkBox)
    size = [int(round(sz * sp / spacing_mm)) for sz, sp in zip(reference.GetSize(),
                                                               reference.GetSpacing(), strict=True)]
    resampled = sitk.Resample(image, size, sitk.Transform(), sitk.sitkNearestNeighbor,
                              reference.GetOrigin(), [spacing_mm] * 3, reference.GetDirection(),
                              0, sitk.sitkUInt8)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(path), useCompression=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())
