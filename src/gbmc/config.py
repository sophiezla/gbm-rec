"""Single source of truth for study constants.

``config.yaml`` sits at the repository root. Machine-specific data paths are not
stored there: they come from environment variables (see ``gbmc.lumiere``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load(path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed ``config.yaml``."""
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get(dotted: str, *, path: str | Path | None = None) -> Any:
    """Return the value at a dotted path, e.g. ``"seeds.master"``.

    Raises if the key is missing or still ``null``: an undecided constant must
    never silently become a default.
    """
    node: Any = load(path)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config.yaml has no key {dotted!r} (missing at {part!r})")
        node = node[part]
    if node is None:
        raise ValueError(
            f"config.yaml:{dotted} is null, so this constant is undecided. "
            "Setting it is a decision: record it in DECISIONS.md in the same commit."
        )
    return node


def pin_itk_threads() -> int:
    """Pin ITK's global thread count to ``preprocessing.itk_threads``.

    Returns the previous setting. Call once per process, including inside every
    worker, because the setting does not cross a process boundary (gbm-4d D9).
    """
    from gbmc.imaging import set_itk_threads

    return set_itk_threads(int(get("preprocessing.itk_threads")))
