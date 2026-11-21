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


def test_mu_glioma_post_metadata_folder_holds_only_spreadsheets():
    """MU-Glioma-Post's metadata folder holds only the two spreadsheets (D18, D19).

    The registration was posted on 2026-11-20 (config.yaml:osf), so the images may now be
    downloaded, into ``images_local_root``, never into the metadata folder.
    """
    from pathlib import Path

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
