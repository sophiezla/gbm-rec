"""Guards on the repository's own rules (plan section 13), not on the science."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel_path], capture_output=True
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    ["data/raw/rhuh/case.nii.gz", "data/lumiere/x.csv", "results/confirm/maps/p1.nii.gz"],
)
def test_data_and_images_stay_out_of_git(path):
    assert _is_ignored(path)


@pytest.mark.parametrize(
    "path", ["results/tournament/leaderboard.json", "results/tournament/run.manifest.json"]
)
def test_results_json_and_manifests_are_committable(path):
    assert not _is_ignored(path)
