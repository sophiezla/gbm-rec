"""The installed environment must be the one the lockfile and the plan describe."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_version():
    assert sys.version_info[:3] == (3, 13, 14)


def test_installed_packages_match_lockfile():
    mismatches = []
    for line in (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, pinned = line.split("==")
        installed = metadata.version(name)
        if installed != pinned:
            mismatches.append(f"{name}: lock {pinned}, installed {installed}")
    assert not mismatches, "\n".join(mismatches)
