"""Deterministic seeds that do not depend on the process, machine or platform.

Python's ``hash()`` is salted per process, so a seed built from it changes between
workers and between runs. Seeds here come from BLAKE2b instead (plan section 13).
Any test of this must spawn a real subprocess: ``hash()`` passes in-process checks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np

__all__ = [
    "as_seed_sequence",
    "check_no_global_rng_dependence",
    "rng",
    "spawn_seeds",
    "stable_seed",
]


def stable_seed(*parts: Any, master_seed: int = 0) -> int:
    """A seed in ``[0, 2**32)`` from ``parts``; ``("a", "b")`` and ``("ab",)`` differ."""
    body = b"\x00".join(repr(part).encode("utf-8") for part in parts)
    message = f"{int(master_seed)!r}".encode() + b"\xff" + body
    return int.from_bytes(hashlib.blake2b(message, digest_size=4).digest(), "big")


def rng(*parts: Any, master_seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts, master_seed=master_seed))


def spawn_seeds(n: int, *parts: Any, master_seed: int = 0) -> list[int]:
    return [stable_seed(*parts, index, master_seed=master_seed) for index in range(n)]


def as_seed_sequence(parts: Iterable[Any], master_seed: int = 0) -> np.random.SeedSequence:
    return np.random.SeedSequence(stable_seed(*tuple(parts), master_seed=master_seed))


def check_no_global_rng_dependence(fn, *args, **kwargs) -> bool:
    """True if ``fn`` returns the same output under two different global numpy states."""
    results = []
    for state in (1, 2):
        np.random.seed(state)  # noqa: NPY002  (perturbing the legacy state is the point)
        results.append(np.asarray(fn(*args, **kwargs)))
    return bool(np.array_equal(results[0], results[1]))
