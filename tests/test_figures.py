from __future__ import annotations

import numpy as np
import pytest

from gbmc import figures


def test_demo_patient_rule():
    picks = figures.demo_patients({"a": 0.3, "b": -0.2, "c": 0.0, "d": 0.1, "e": float("nan")})
    assert picks == {"demo": "c", "failure": "b", "best": "a"}


def _ci(m):
    return {"mean": m, "ci_low": m - 0.01, "ci_high": m + 0.01}


@pytest.fixture
def out(tmp_path):
    return tmp_path


def test_every_figure_renders(out):
    dev = {c: _ci(v) for c, v in zip(("C0T", "C4", "C1", "C3", "C2", "C6"),
                                     (0.001, -0.02, 0.003, 0.002, -0.004, 0.001), strict=True)}
    val = {c: _ci(d["mean"] - 0.002) for c, d in dev.items()}
    paths = []
    paths += figures.complexity_forest(None, dev, "C1", out / "forest_dev")
    paths += figures.complexity_forest(val, dev, "C1", out / "forest", champion="C0")
    paths += figures.hero_coverage(
        {"Guideline margin": {"role": "guideline_margin", **_ci(0.44)},
         "Distance": {"role": "distance", **_ci(0.45)},
         "C1": {"role": "challenger", **_ci(0.47)}},
        "recurrence outside the target", out / "hero")
    paths += figures.coverage_curve(
        {"C0": {5: 0.1, 50: 0.5, 200: 0.8}, "C1": {5: 0.12, 50: 0.52, 200: 0.81}},
        {"C0": "distance", "C1": "challenger"}, "recurrence outside the target", out / "curve",
        margin_ml=60)
    rng = np.random.default_rng(0)
    deltas = {f"P{i:02d}": float(rng.normal(0, 0.02)) for i in range(25)}
    paths += figures.per_patient_strip(deltas, out / "strip", "C1 vs distance")
    rows = {p: {"share_beyond_15mm": 0.3, "share_beyond_20mm": 0.1,
                "C0": {"catches_beyond_15mm": 0.0}, "C1": {"catches_beyond_15mm": 0.2}}
            for p in deltas}
    paths += figures.escape_panel(rows, ["C0", "C1"], {"C0": "distance", "C1": "challenger"},
                                  out / "escape")
    paths += figures.later_progression({"2": {"n": 30, "C0": {"mean_pr_auc": 0.05},
                                              "C1": {"mean_pr_auc": 0.06}}}, out / "later")
    paths += figures.protocol_strip([("63 patients", "LUMIERE"), ("tournament", "7 folds"),
                                     ("freeze + hash", ""), ("OSF", "date"),
                                     ("40 patients", "RHUH-GBM"), ("one run", "")],
                                    out / "protocol")
    z, y = np.indices((64, 64))
    disc = lambda c, r: (z - c[0]) ** 2 + (y - c[1]) ** 2 < r ** 2  # noqa: E731
    paths += figures.prediction_vs_reality(
        rng.normal(size=(64, 64)), disc((32, 30), 6), disc((32, 30), 18) & ~disc((32, 30), 6),
        disc((32, 32), 16), disc((34, 36), 16), disc((40, 44), 5),
        {"margin": 0.4, "distance": 0.42, "champion": 0.5}, "C1", out / "reveal")
    assert len(paths) == 18 and all(p.stat().st_size > 1000 for p in paths)


def test_per_patient_figures_from_result_rows(out):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_fig50", Path(__file__).resolve().parents[1] / "scripts" / "50_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = out
    rng = np.random.default_rng(3)
    rows = {}
    for i in range(12):
        row = {"evaluable": i != 0, "matched_voxels": {"margin15": 60000, "volume50ml": 50000},
               "curves": {}}
        for c in ("C0", "C0T", "C1"):
            row[c] = {"pr_auc": float(rng.uniform(0.02, 0.1)),
                      "coverage_margin15": float(rng.uniform(0.3, 0.6))}
            row["curves"][c] = {"5": 0.1, "50": 0.5, "200": 0.8}
        rows[f"P{i:02d}"] = row
    written = []
    mod.per_patient_figures(rows, "C1", "LUMIERE", 7, 200, written)
    assert len(written) == 6
