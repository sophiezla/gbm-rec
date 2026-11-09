"""The committed first-progression table: built from gbm-4d records, merged, never overwritten."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gbmc import lumiere


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "15_first_progressions.py"
    spec = importlib.util.spec_from_file_location("first_progressions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(folder: Path, patient: str, baseline: str, progression: str):
    (folder / patient).mkdir(parents=True)
    (folder / patient / "qc.json").write_text(json.dumps(
        {"patient": patient, "baseline": baseline, "progression": progression}))


def test_collect_checks_baseline_and_merge_refuses_conflicts(tmp_path, monkeypatch):
    fp = _script()
    monkeypatch.setattr(lumiere, "baseline_name", lambda p: "week-000-1")
    _record(tmp_path / "a", "P1", "week-000-1", "week-044")
    names, digests = fp.collect(tmp_path / "a", ["P1", "P2"])
    table = fp.merge({}, names, digests, "a", ["P1", "P2"])
    assert table["first_progression"] == {"P1": "week-044"} and table["missing"] == ["P2"]

    _record(tmp_path / "b", "P2", "week-000-1", "week-030")
    table = fp.merge(table, *fp.collect(tmp_path / "b", ["P1", "P2"]), "b", ["P1", "P2"])
    assert table["missing"] == [] and table["source_folders"] == ["a", "b"]

    _record(tmp_path / "c", "P1", "week-000-1", "week-050")
    with pytest.raises(SystemExit, match="already recorded"):
        fp.merge(table, *fp.collect(tmp_path / "c", ["P1"]), "c", ["P1"])

    _record(tmp_path / "d", "P1", "week-000-9", "week-044")
    with pytest.raises(SystemExit, match="baseline"):
        fp.collect(tmp_path / "d", ["P1"])


def test_first_progression_reads_the_table(tmp_path):
    table = tmp_path / "first_progressions.json"
    table.write_text(json.dumps({"first_progression": {"P1": "week-044"}}))
    assert lumiere.first_progression("P1", table) == "week-044"
    with pytest.raises(KeyError, match="not recorded"):
        lumiere.first_progression("P2", table)


def test_committed_table_covers_only_the_locked_cohort():
    table = json.loads(lumiere.FIRST_PROGRESSIONS.read_text(encoding="utf-8"))
    cohort = set(lumiere.cohort())
    assert set(table["first_progression"]) | set(table["missing"]) == cohort
    assert not set(table["first_progression"]) & set(table["missing"])
