from __future__ import annotations

import json
import subprocess

import pytest

from gbmc import provenance


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repository standing in for this one."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    return tmp_path


def test_manifest_records_git_gbm4d_and_outputs(repo):
    out = repo / "results" / "stage"
    with provenance.run_manifest(out) as m:
        m.record(seed=5)
        (out / "x.json").write_text("{}", encoding="utf-8")
    manifest = json.loads((out / provenance.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == "gbmc/manifest/1"
    assert manifest["git"]["commit"]
    assert manifest["gbm4d"]["installed"] is True
    assert manifest["values"] == {"seed": 5}
    assert [o["path"] for o in manifest["outputs"]] == ["results/stage/x.json"]


def test_require_clean_tree_passes_on_clean_repo(repo):
    provenance.require_clean_tree()


def test_require_clean_tree_refuses_dirty_repo(repo):
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(SystemExit, match="uncommitted"):
        provenance.require_clean_tree()
    provenance.require_clean_tree(allow_dirty=True)
