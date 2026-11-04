from __future__ import annotations

import json

from gbmc import audit


def _results(tmp_path):
    d = tmp_path / "results" / "tournament"
    d.mkdir(parents=True)
    (d / "leaderboard.json").write_text(json.dumps(
        {"C0": {"pr_auc": 0.07431, "coverage": 0.4518, "n": 53}, "flag": True}))
    (d / "run.manifest.json").write_text(json.dumps({"duration_s": 9.99}))
    return audit.result_values(tmp_path / "results")


def test_matching_numbers_pass(tmp_path):
    values = _results(tmp_path)
    text = "C0 scored 0.0743 (0.074 rounded), and covered 45.2% of recurrence in [[53]] patients."
    assert audit.check_text(text, values) == []


def test_unbacked_numbers_are_reported(tmp_path):
    values = _results(tmp_path)
    found = audit.check_text("Line one.\nC0 scored 0.0751 and 9.99 s, in [[54]] patients.",
                             values)
    assert [(f.line, f.quoted) for f in found] == [(2, "0.0751"), (2, "9.99"), (2, "[[54]]")]


def test_versions_and_licences_are_not_results(tmp_path):
    values = _results(tmp_path)
    text = "RHUH-GBM is CC BY 4.0; built with Python 3.13.14 and v1.2.3, lock 2.5.6."
    assert audit.check_text(text, values) == []


def test_ignore_marker_and_plain_integers(tmp_path):
    values = _results(tmp_path)
    text = "The 2023 guideline uses 15 mm.\nRR 7.29 in gbm-wm (audit: ignore)"
    assert audit.check_text(text, values) == []


def test_lists_of_plan_steps_are_not_results(tmp_path):
    values = _results(tmp_path)
    assert audit.check_text("Expert review (plan Steps 0.7, 6.6 and 6.7).", values) == []
