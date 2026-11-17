from __future__ import annotations

import pytest

from gbmc import config


def _write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_null_constant_raises_on_read(tmp_path):
    path = _write(tmp_path, "seeds:\n  master: null\n")
    with pytest.raises(ValueError, match="undecided"):
        config.get("seeds.master", path=path)


def test_missing_key_raises(tmp_path):
    path = _write(tmp_path, "seeds:\n  master: 1\n")
    with pytest.raises(KeyError):
        config.get("seeds.nope", path=path)


def test_set_constant_is_returned(tmp_path):
    path = _write(tmp_path, "seeds:\n  master: 7\n")
    assert config.get("seeds.master", path=path) == 7


def test_rhuh_gbm_path_stays_unset_before_preregistration():
    """RHUH-GBM may not be downloaded before the OSF timestamp (plan Phase 2).

    Delete this test in the same commit that records the OSF timestamp.
    """
    with pytest.raises(ValueError):
        config.get("data.rhuh_gbm.local_root")


def test_mu_glioma_post_stays_sealed():
    """MU-Glioma-Post is the follow-up study's sealed test (DECISIONS.md D18).

    No image or segmentation may exist on this computer, and the image path stays unset,
    until the follow-up's own pre-registration is posted. Only the clinical and volume
    spreadsheets are allowed. Change this test in the commit that records that posting.
    """
    from pathlib import Path

    with pytest.raises(ValueError):
        config.get("data.mu_glioma_post.images_local_root")
    root = Path(__file__).resolve().parents[1] / config.get("data.mu_glioma_post.metadata_dir")
    if root.is_dir():
        files = [p for p in root.rglob("*") if p.is_file()]
        sealed = (".nii", ".gz", ".dcm", ".zip", ".npz", ".nrrd", ".mha")
        assert not [p.name for p in files if p.name.lower().endswith(sealed)]
        assert all(p.suffix.lower() == ".xlsx" for p in files)


def test_itk_threads_pinned_to_one():
    import SimpleITK as sitk

    previous = config.pin_itk_threads()
    try:
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 1
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)
