"""Run manifests: make every file in ``results/`` traceable to its cause.

Adapted from ``gbm4d.provenance`` with two changes:

* git state is read from this repository. gbm4d's version resolves the repo
  from its own file location, which after installation is site-packages.
* every manifest also records the gbm4d commit that was installed, so a result
  can be tied to the exact frozen predecessor code.

Usage
-----
>>> from gbmc.provenance import run_manifest
>>> with run_manifest("results/tournament", inputs=["config.yaml"]) as m:
...     m.record(seed=123)
...     ...  # write results/tournament/*.json
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from gbm4d.provenance import env_state, file_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "run.manifest.json"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def git_state() -> dict[str, Any]:
    """Return commit, branch and dirty-file list for this repository."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": sorted(status.splitlines()) if status else [],
        "remote": _git("config", "--get", "remote.origin.url"),
    }


def gbm4d_source() -> dict[str, Any]:
    """Return where the installed gbm4d came from (URL and commit), if recorded."""
    try:
        raw = metadata.distribution("gbm4d").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return {"installed": False}
    info = json.loads(raw) if raw else {}
    return {
        "installed": True,
        "version": metadata.version("gbm4d"),
        "url": info.get("url"),
        "commit": info.get("vcs_info", {}).get("commit_id"),
    }


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


@dataclass
class Manifest:
    """Accumulates provenance for one script run."""

    out_dir: Path
    inputs: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def record(self, **kwargs: Any) -> None:
        """Attach key/values such as seeds, counts and resolved data paths."""
        self.values.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "gbmc/manifest/1",
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "argv": sys.argv,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "duration_s": round(time.time() - self.started, 3),
            "git": git_state(),
            "gbm4d": gbm4d_source(),
            "environment": env_state(),
            "inputs": [
                {"path": str(p), "sha256": file_digest(p), "bytes": Path(p).stat().st_size}
                for p in self.inputs if Path(p).is_file()
            ],
            "outputs": [
                {"path": _repo_relative(p), "sha256": file_digest(p)}
                for p in sorted(self.out_dir.rglob("*"))
                if p.is_file() and p.name != MANIFEST_NAME
            ],
            "values": self.values,
        }


@contextlib.contextmanager
def run_manifest(out_dir: str | Path, inputs: Iterable[str | Path] = ()):
    """Write ``<out_dir>/run.manifest.json`` when the block exits.

    Output digests are computed after the body runs, so the manifest describes
    the files that actually landed on disk.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir=out, inputs=[str(i) for i in inputs])
    try:
        yield manifest
    finally:
        (out / MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Abort unless the working tree is committed.

    A result produced from uncommitted code cannot be reproduced from any
    commit. ``allow_dirty`` is for throwaway exploratory runs only, never for a
    number that enters the poster, report or app.
    """
    state = git_state()
    if state["commit"] is None:
        raise SystemExit("Refusing to run: not inside a git repository with a commit.")
    if state["dirty"] and not allow_dirty:
        files = "\n  ".join(state["dirty_files"])
        raise SystemExit(
            "Refusing to run: the working tree has uncommitted changes, so this "
            "result could not be reproduced from any commit.\n"
            f"  {files}\n"
            "Commit them, or pass allow_dirty=True for a throwaway exploratory run."
        )
