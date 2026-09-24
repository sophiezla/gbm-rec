"""Deterministic seeding, reused unchanged from gbm-4d.

The seeds must match gbm-4d's bit for bit, so this module re-exports the frozen
implementation instead of copying it. See ``gbm4d.rng`` for why seeds come from
BLAKE2b and never from ``hash()``.
"""

from gbm4d.rng import (
    as_seed_sequence,
    check_no_global_rng_dependence,
    rng,
    spawn_seeds,
    stable_seed,
)

__all__ = [
    "as_seed_sequence",
    "check_no_global_rng_dependence",
    "rng",
    "spawn_seeds",
    "stable_seed",
]
