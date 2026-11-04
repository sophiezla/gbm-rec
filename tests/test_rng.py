"""Seeds must be stable across processes and pinned.

The subprocess test is the point of this file: an in-process equality check
also passes for ``hash()``, which is exactly the failure it guards against.
"""

from __future__ import annotations

import os
import subprocess
import sys

from gbmc.rng import stable_seed

SNIPPET = "from gbmc.rng import stable_seed; print(stable_seed('patient-001', 'fold', 3))"


def _seed_in_subprocess(hashseed: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        capture_output=True, text=True, check=True,
        # Inherit the full environment (a stripped one makes Windows write caches
        # under a literal "%SystemDrive%" folder); only the hash salt changes.
        env={**os.environ, "PYTHONHASHSEED": hashseed},
    )
    return out.stdout.strip()


def test_seed_is_stable_across_processes_and_hash_salts():
    seeds = {_seed_in_subprocess(salt) for salt in ("0", "1", "12345")}
    assert len(seeds) == 1, f"seed varied with PYTHONHASHSEED: {seeds}"
    assert seeds.pop() == str(stable_seed("patient-001", "fold", 3))


def test_seed_values_are_pinned():
    # Values from the implementation these seeds were first drawn with. A change here
    # would silently redraw every fold and bootstrap.
    assert stable_seed("patient-001", "fold", 3) == 4026506090
    assert stable_seed("a", "b", master_seed=7) == 693144360
    assert stable_seed("ab", master_seed=7) != 693144360
