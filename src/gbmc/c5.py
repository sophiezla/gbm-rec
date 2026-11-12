"""C5: a small 3D U-Net that ranks voxels by recurrence (DECISIONS.md D6 items 8-9).

Every setting comes from ``config.yaml:c5``, fixed before training and never tuned.
PyTorch is imported inside the functions that need it, so the main environment (and every
other candidate) runs without it; C5 trains on the GPU computer (``scripts/13_c5_train.py``).

Inputs per voxel (plan Step 1.4): z-scored T1c and FLAIR, the target mask, the FLAIR mask
(DeepBraTumIA oedema, the f2 region) and the f1 map. Two details the approved settings
leave open, fixed here before any training (D6 item 8):

* downsampling uses stride-2 convolutions, not max-pooling, because CUDA's max-pool
  backward has no deterministic implementation;
* the learning rate decays linearly to 0 over all steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CHANNELS = ("t1c", "flair", "target", "flair_mask", "f1")
#: f1 is divided by this, so the distance channel is of order 1 like the others.
F1_SCALE_MM = 100.0


@dataclass(frozen=True)
class Settings:
    patch: tuple[int, int, int] = (96, 96, 96)
    batch: int = 2
    positive_share: float = 0.5
    levels: int = 4
    base_channels: int = 16
    pos_weight_cap: float = 100.0
    learning_rate: float = 1e-3
    epochs: int = 200
    steps_per_epoch: int = 100
    max_rotation_deg: float = 10.0
    overlap: float = 0.5

    @classmethod
    def from_config(cls) -> Settings:
        from gbmc import config

        return cls(patch=tuple(int(v) for v in config.get("c5.patch_size")),
                   batch=int(config.get("c5.batch_size")),
                   positive_share=float(config.get("c5.positive_patch_share")),
                   levels=int(config.get("c5.unet_levels")),
                   base_channels=int(config.get("c5.base_channels")),
                   pos_weight_cap=float(config.get("c5.pos_weight_cap")),
                   learning_rate=float(config.get("c5.learning_rate")),
                   epochs=int(config.get("c5.epochs")),
                   steps_per_epoch=int(config.get("c5.steps_per_epoch")),
                   max_rotation_deg=float(config.get("c5.max_rotation_deg")),
                   overlap=float(config.get("c5.inference_overlap")))

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


@dataclass
class Case:
    """One patient's packed inputs, cropped to the brain."""

    patient: str
    image: np.ndarray        # (5, z, y, x) float32, channels in CHANNELS order
    evaluate: np.ndarray     # (z, y, x) bool: brain minus target, where scores count
    label: np.ndarray        # (z, y, x) bool: recurrence, only ever True inside evaluate
    lr_axis: int             # array axis (0-2) that runs left-right
    positives: np.ndarray = field(init=False)
    inside: np.ndarray = field(init=False)

    def __post_init__(self):
        self.positives = np.argwhere(self.label)
        self.inside = np.argwhere(self.evaluate)

    def padded(self, patch) -> Case:
        """Zero-pad so every axis is at least one patch long (padding is never scored)."""
        pads = [(0, max(0, p - s)) for p, s in zip(patch, self.evaluate.shape, strict=True)]
        if not any(after for _, after in pads):
            return self
        return Case(self.patient, np.pad(self.image, [(0, 0), *pads]),
                    np.pad(self.evaluate, pads), np.pad(self.label, pads), self.lr_axis)


def pack(patient: str, t1c, flair, target, flair_mask, f1, evaluate, truth, lr_axis) -> dict:
    """Crop one patient's volumes to the brain box; arrays for ``np.savez_compressed``.

    ``truth`` is the feature cache's label vector, in C order over ``evaluate``. Cropping
    keeps that order, so C5's scores line up with every other candidate's row for row.
    """
    if truth.shape != (int(evaluate.sum()),):
        raise ValueError(f"{patient}: {truth.size} labels for {int(evaluate.sum())} voxels")
    brain = evaluate | target
    lo = np.argwhere(brain).min(axis=0)
    hi = np.argwhere(brain).max(axis=0) + 1
    box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    label = np.zeros(evaluate.shape, dtype=bool)
    label[evaluate] = truth
    f1 = np.clip(f1, -300.0, 300.0) / F1_SCALE_MM
    image = np.stack([t1c, flair, target, flair_mask, f1]).astype(np.float32)
    image[:, ~brain] = 0.0
    return {"image": image[(slice(None), *box)].astype(np.float16),
            "evaluate": evaluate[box], "label": label[box],
            "lr_axis": np.int64(lr_axis), "offset": lo, "shape": np.array(evaluate.shape)}


def load_case(path: Path) -> Case:
    with np.load(path) as z:
        return Case(Path(path).stem, z["image"].astype(np.float32), z["evaluate"].astype(bool),
                    z["label"].astype(bool), int(z["lr_axis"]))


def lr_axis_from_direction(direction) -> int:
    """Array axis (z, y, x order) most aligned with the patient's left-right (physical x)."""
    d = np.asarray(direction, dtype=float).reshape(3, 3)
    index_axis = int(np.argmax(np.abs(d[0])))    # SimpleITK index order is (x, y, z)
    return 2 - index_axis


def pos_weight(cases: list[Case], cap: float) -> float:
    """Training fold's negative-to-positive ratio inside the evaluation region, capped."""
    positives = sum(int(c.label.sum()) for c in cases)
    negatives = sum(int(c.evaluate.sum()) for c in cases) - positives
    if positives == 0:
        raise ValueError("no positive voxels in the training fold")
    return float(min(cap, negatives / positives))


def patch_starts(shape, patch, centre) -> tuple[int, ...]:
    return tuple(int(np.clip(c - p // 2, 0, s - p))
                 for c, p, s in zip(centre, patch, shape, strict=True))


def sample_batch(cases: list[Case], rng: np.random.Generator, settings: Settings, step: int):
    """One batch; patches alternate positive-centred and random, so exactly half are positive."""
    with_pos = [c for c in cases if len(c.positives)]
    xs, ys, ws, axes = [], [], [], []
    for b in range(settings.batch):
        positive = (step * settings.batch + b) % 2 == 0 if settings.positive_share == 0.5 \
            else rng.random() < settings.positive_share
        case = with_pos[rng.integers(len(with_pos))] if positive \
            else cases[rng.integers(len(cases))]
        pool = case.positives if positive else case.inside
        start = patch_starts(case.evaluate.shape, settings.patch, pool[rng.integers(len(pool))])
        box = tuple(slice(s, s + p) for s, p in zip(start, settings.patch, strict=True))
        xs.append(case.image[(slice(None), *box)])
        ys.append(case.label[box])
        ws.append(case.evaluate[box])
        axes.append(case.lr_axis)
    return np.stack(xs), np.stack(ys), np.stack(ws), axes


def rotation_matrix(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """A rotation by up to ``max_deg`` about a random axis."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def set_determinism(seed: int) -> None:
    import os
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)          # numpy randomness uses explicit Generators only
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_unet(settings: Settings):
    """Small 3D U-Net: ``levels`` resolutions, ``base_channels`` doubling at each."""
    import torch
    from torch import nn

    def block(c_in, c_out):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1), nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1), nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.01, inplace=True))

    widths = [settings.base_channels * 2**i for i in range(settings.levels)]

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList([block(len(CHANNELS), widths[0])] +
                                     [block(widths[i], widths[i]) for i in range(1, len(widths))])
            self.down = nn.ModuleList([nn.Conv3d(widths[i], widths[i + 1], 2, stride=2)
                                       for i in range(len(widths) - 1)])
            self.up = nn.ModuleList([nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2)
                                     for i in range(len(widths) - 1)])
            self.dec = nn.ModuleList([block(2 * widths[i], widths[i])
                                      for i in range(len(widths) - 1)])
            self.head = nn.Conv3d(widths[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, enc in enumerate(self.enc):
                x = enc(x)
                if i < len(self.down):
                    skips.append(x)
                    x = self.down[i](x)
            for i in reversed(range(len(self.up))):
                x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
            return self.head(x)[:, 0]

    return UNet()


def augment(x, y, w, axes, rng: np.random.Generator, max_deg: float):
    """Left-right flips and rotations up to ``max_deg`` only (plan Step 1.4)."""
    import torch
    import torch.nn.functional as F

    x, y, w = x.clone(), y.clone(), w.clone()
    for b, axis in enumerate(axes):
        if rng.random() < 0.5:
            x[b] = torch.flip(x[b], dims=[axis + 1])
            y[b] = torch.flip(y[b], dims=[axis])
            w[b] = torch.flip(w[b], dims=[axis])
    theta = torch.zeros((x.shape[0], 3, 4), dtype=x.dtype, device=x.device)
    for b in range(x.shape[0]):
        theta[b, :, :3] = torch.as_tensor(rotation_matrix(rng, max_deg), dtype=x.dtype)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    smooth = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
    crisp = F.grid_sample(x, grid, mode="nearest", align_corners=False)
    masks = [CHANNELS.index("target"), CHANNELS.index("flair_mask")]
    smooth[:, masks] = crisp[:, masks]
    yw = F.grid_sample(torch.stack([y, w], dim=1), grid, mode="nearest", align_corners=False)
    return smooth, yw[:, 0], yw[:, 1]


def _checkpoint_key(cases: list[Case], settings: Settings, seed: int, tag: str) -> str:
    return json.dumps({"tag": tag, "seed": seed, "settings": asdict(settings),
                       "patients": [c.patient for c in cases]}, sort_keys=True)


def _save_checkpoint(path: Path, key: str, step: int, model, optimiser, schedule,
                     rng: np.random.Generator) -> None:
    """Write the full training state atomically: a crash mid-write keeps the old file."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"key": key, "step": step, "model": model.state_dict(),
                "optimiser": optimiser.state_dict(), "schedule": schedule.state_dict(),
                "numpy_rng": json.dumps(rng.bit_generator.state),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                else None}, tmp)
    os.replace(tmp, path)


def train(cases: list[Case], settings: Settings, seed: int, device: str = "cpu",
          log=print, checkpoint: Path | None = None, checkpoint_every: int = 0,
          checkpoint_tag: str = ""):
    """Train from scratch for exactly ``settings.total_steps``; returns the last model.

    With ``checkpoint``, the full training state (model, optimiser, schedule, step and the
    random-number states) is saved there every ``checkpoint_every`` steps, and a call that
    finds a checkpoint for the same run (cases, settings, seed and ``checkpoint_tag``)
    continues from it. A resumed run ends with the same model as an uninterrupted one,
    bit for bit; checkpointing changes no setting and no step (tests/test_c5.py).
    """
    import torch
    import torch.nn.functional as F

    set_determinism(seed)
    rng = np.random.default_rng(seed)
    key = _checkpoint_key(cases, settings, seed, checkpoint_tag)
    cases = [c.padded(settings.patch) for c in cases]
    weight = torch.tensor(pos_weight(cases, settings.pos_weight_cap), device=device)
    model = build_unet(settings).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    total = settings.total_steps
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 1.0 - s / total)
    first = 0
    if checkpoint is not None and checkpoint.is_file():
        # Load onto the CPU: the random-number states must stay CPU tensors; the model and
        # optimiser states are copied to ``device`` by their load_state_dict methods.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["key"] != key:
            raise SystemExit(f"{checkpoint} is from a different run (commit, fold, settings "
                             "or seed); delete it to start this fold from scratch")
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        schedule.load_state_dict(saved["schedule"])
        rng.bit_generator.state = json.loads(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first = int(saved["step"])
        log(f"resumed from checkpoint at epoch {first // settings.steps_per_epoch}/"
            f"{settings.epochs}")
    model.train()
    for step in range(first, total):
        xb, yb, wb, axes = sample_batch(cases, rng, settings, step)
        x = torch.as_tensor(xb, device=device)
        y = torch.as_tensor(yb, dtype=torch.float32, device=device)
        w = torch.as_tensor(wb, dtype=torch.float32, device=device)
        x, y, w = augment(x, y, w, axes, rng, settings.max_rotation_deg)
        logits = model(x)
        loss_map = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight,
                                                      reduction="none")
        loss = (loss_map * w).sum() / w.sum().clamp(min=1.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if (checkpoint is not None and checkpoint_every > 0
                and (step + 1) % checkpoint_every == 0 and step + 1 < total):
            _save_checkpoint(checkpoint, key, step + 1, model, optimiser, schedule, rng)
        if (step + 1) % settings.steps_per_epoch == 0:
            log(f"epoch {(step + 1) // settings.steps_per_epoch}/{settings.epochs} "
                f"loss {loss.item():.4f}")
    return model


def window_starts(size: int, patch: int, overlap: float) -> list[int]:
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if starts[-1] != max(size - patch, 0):
        starts.append(max(size - patch, 0))
    return starts


def predict(model, case: Case, settings: Settings, device: str = "cpu") -> np.ndarray:
    """Sliding-window probabilities, averaged over overlaps; one score per evaluation voxel."""
    import itertools

    import torch

    padded = case.padded(settings.patch)
    shape = padded.evaluate.shape
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in itertools.product(*(window_starts(s, p, settings.overlap)
                                         for s, p in zip(shape, settings.patch, strict=True))):
            box = tuple(slice(s, s + p) for s, p in zip(start, settings.patch, strict=True))
            x = torch.as_tensor(padded.image[(slice(None), *box)][None], device=device)
            total[box] += torch.sigmoid(model(x))[0].double().cpu().numpy()
            count[box] += 1.0
    prob = total / count
    original = tuple(slice(0, s) for s in case.evaluate.shape)
    return prob[original][case.evaluate].astype(np.float32)
