"""Freeze, hash-check and run-once machinery (plan Steps 1.7, 4.2, 4.3).

* :func:`save_model` writes a fitted model and returns its SHA-256, which goes
  into the run manifest and the OSF pre-registration.
* :func:`verify_hashes` recomputes every model's SHA-256 before scoring and
  stops on any difference from the registered value.
* :func:`open_once` / :func:`mark_done` implement the irreversible marker:
  the confirmatory run writes ``OPENED.json`` before it reads any label and
  ``DONE.json`` when it finishes; once ``DONE.json`` exists it refuses to run.
  An interrupted run may resume, and every resumed attempt is logged (the
  gbm-4d ``30_test_set.py`` pattern).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gbmc.provenance import git_state

OPENED = "OPENED.json"
DONE = "DONE.json"


class HashMismatchError(RuntimeError):
    """A model on disk is not the one that was registered."""


class AlreadyRunError(RuntimeError):
    """The run-once step has already completed."""


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model: Any, path: str | Path) -> str:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return sha256(path)


def load_model(path: str | Path, expected_sha256: str) -> Any:
    """Load a model only after its hash matches the registered one."""
    import joblib

    actual = sha256(path)
    if actual != expected_sha256:
        raise HashMismatchError(f"{path}: sha256 {actual} != registered {expected_sha256}")
    return joblib.load(path)


def verify_hashes(register: dict[str, str], root: str | Path) -> None:
    """Check every ``relative path -> sha256`` entry; raise on the first difference."""
    root = Path(root)
    for rel, expected in sorted(register.items()):
        path = root / rel
        if not path.is_file():
            raise HashMismatchError(f"{path}: registered model is missing")
        actual = sha256(path)
        if actual != expected:
            raise HashMismatchError(f"{path}: sha256 {actual} != registered {expected}")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_once(out_dir: str | Path, subjects: list[str], note: str = "") -> dict:
    """Write (or extend) the marker BEFORE any label is read; refuse if already done."""
    out = Path(out_dir)
    done = out / DONE
    if done.exists():
        raise AlreadyRunError(
            f"{done} exists: this step ran once and completed. There is no flag to bypass it.\n"
            + done.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    opened = out / OPENED
    if opened.exists():
        state = json.loads(opened.read_text(encoding="utf-8"))
        state.setdefault("resumed_attempts", []).append({"at": _now(), "git": git_state()})
    else:
        state = {"opened_at": _now(), "git": git_state(), "subjects": subjects, "note": note}
    opened.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def mark_done(out_dir: str | Path, summary: dict) -> None:
    out = Path(out_dir)
    if not (out / OPENED).exists():
        raise RuntimeError("mark_done without open_once: the marker was never written")
    (out / DONE).write_text(json.dumps({"completed_at": _now(), **summary}, indent=2) + "\n",
                            encoding="utf-8")
