from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_import_bundle.py"


def _module(interim: Path):
    spec = importlib.util.spec_from_file_location("_import_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INTERIM = interim
    return module


def _unit(root: Path, rel: str, text: str) -> None:
    d = root / "data" / "interim" / rel
    d.mkdir(parents=True)
    (d / "qc.json").write_text(text)


def test_merge_adds_new_keeps_local_and_flags_differences(tmp_path):
    remote, local = tmp_path / "remote", tmp_path / "local"
    _unit(remote, "roundtrip_tuned/P1", '{"ok": true}')
    _unit(remote, "roundtrip_tuned/P2", '{"ok": true}')
    _unit(remote, "label_field_later/week-010/P3", '{"ok": false}')
    (local / "roundtrip_tuned" / "P2").mkdir(parents=True)
    (local / "roundtrip_tuned" / "P2" / "qc.json").write_text('{"ok": false}')
    mod = _module(local)
    out = mod.merge(remote, dry_run=False)
    assert out["added"] == ["label_field_later/week-010/P3", "roundtrip_tuned/P1"]
    assert out["differing"] == ["roundtrip_tuned/P2"]
    assert (local / "roundtrip_tuned" / "P2" / "qc.json").read_text() == '{"ok": false}'
    again = mod.merge(remote, dry_run=True)
    assert sorted(again["identical"]) == ["label_field_later/week-010/P3", "roundtrip_tuned/P1"]
