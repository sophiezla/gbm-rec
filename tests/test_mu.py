"""MU-Glioma-Post eligibility (D18, D19) and the stdlib .xlsx reader, on synthetic sheets."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from gbmc import mu
from gbmc.xlsx import read_sheet

HEADER = ["Patient_ID", "Primary Diagnosis", "IDH1 mutation", "IDH2 mutation",
          "Previous Brain Tumor", "Progression", "Type of 1st Progression",
          "Number of days from Diagnosis to date of First Progression"] + [
    f"Number of Days from Diagnosis to {o} MRI (Timepoint_{k}) "
    for k, o in enumerate(("1st", "2nd", "3rd", "4th", "5th", "6th"), 1)]


def _row(pid, dx="GBM", idh1=0, idh2=0, prev="No", prog=1, ptype=1, pday=300, mris=()):
    days = list(mris) + [None] * (6 - len(mris))
    return [pid, dx, idh1, idh2, prev, prog, ptype, pday, *days]


def test_eligibility_rules_and_reasons():
    rows = [HEADER,
            _row("ok", mris=(100, 300, 330)),
            _row("astro", dx="Astrocytoma", mris=(100, 300)),
            _row("idh_mut", idh1=1, mris=(100, 300)),
            _row("idh_unknown", idh1=2, mris=(100, 300)),
            _row("idh2_unknown_ok", idh2=2, mris=(100, 300)),
            _row("prior", prev="Yes ", mris=(100, 300)),
            _row("clinical_only", ptype=2, mris=(100, 300)),
            _row("no_date", pday=None, mris=(100, 300)),
            _row("baseline_too_late", mris=(260, 300)),
            _row("no_rec_scan", mris=(100, 380)),
            [None] * len(HEADER)]                      # blank row, skipped
    got = {r.patient: r for r in mu.eligibility(rows)}
    assert len(got) == 10
    assert got["ok"].eligible and got["ok"].baseline_timepoint == 1
    assert got["ok"].recurrence_timepoint == 2 and got["ok"].recurrence_days_from_progression == 0
    assert got["idh2_unknown_ok"].eligible
    assert got["astro"].reason == mu.STEPS[0]
    assert got["idh_mut"].reason == got["idh_unknown"].reason == mu.STEPS[1]
    assert got["prior"].reason == mu.STEPS[2]
    assert got["clinical_only"].reason == mu.STEPS[3]
    assert got["no_date"].reason == mu.STEPS[4]
    assert got["baseline_too_late"].reason == mu.STEPS[5]
    assert got["no_rec_scan"].reason == mu.STEPS[6]
    assert mu.funnel(list(got.values()))["eligible"] == 2


def test_recurrence_scan_is_closest_to_progression_and_ordering_is_by_days():
    # Timepoints out of date order (as for PatientID_0081): Timepoint_1 is the latest scan.
    rows = [HEADER, _row("p", pday=300, mris=(310, 100, 290))]
    r = mu.eligibility(rows)[0]
    assert r.baseline_timepoint == 2                 # day 100, the first post-treatment MRI
    assert r.recurrence_timepoint == 3               # day 290 and 310 tie at 10 days: the earlier
    rows = [HEADER, _row("q", pday=300, mris=(100, 288, 305))]
    assert mu.eligibility(rows)[0].recurrence_timepoint == 3     # 5 days beats 12


def test_target_is_enhancing_plus_cavity():
    seg = np.array([0, 1, 2, 3, 4])
    assert mu.target(seg, np.ones(5, bool)).tolist() == [False, False, False, True, True]


def test_layout_resolves_the_package_names():
    """The layout read from TCIA's package listing (D20) builds the expected file names."""
    lay = mu.layout()
    name = lay["image_pattern"].format(patient="PatientID_0003", timepoint=1,
                                       sequence=lay["sequence_names"]["T1ce"])
    assert name == ("MU-Glioma-Post/PatientID_0003/Timepoint_1/"
                    "PatientID_0003_Timepoint_1_brain_t1c.nii.gz")
    assert lay["seg_pattern"].format(patient="PatientID_0003", timepoint=1).endswith(
        "PatientID_0003_Timepoint_1_tumorMask.nii.gz")


def _xlsx(path, rows):
    shared = sorted({v for r in rows for v in r if isinstance(v, str)})
    def cell(ref, v):
        if v is None:
            return ""
        if isinstance(v, str):
            return f'<c r="{ref}" t="s"><v>{shared.index(v)}</v></c>'
        return f'<c r="{ref}"><v>{v}</v></c>'
    body = "".join(
        f'<row r="{i}">' + "".join(cell(f"{chr(65 + j)}{i}", v) for j, v in enumerate(r)) + "</row>"
        for i, r in enumerate(rows, 1))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rns}><sheets>'
                   '<sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                   'relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
                   '</Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst {ns}>' + "".join(
            f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
        z.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{body}"
                   "</sheetData></worksheet>")


def test_xlsx_reader_keeps_gaps_and_types(tmp_path):
    path = tmp_path / "t.xlsx"
    _xlsx(path, [["id", "n", "x"], ["a", 3, None], ["b", None, 2.5]])
    rows = read_sheet(path, "S")
    assert rows == [["id", "n", "x"], ["a", 3, None], ["b", None, 2.5]]
    with pytest.raises(KeyError):
        read_sheet(path, "missing")


def test_mu_h2_path_end_to_end_on_a_synthetic_patient(tmp_path, monkeypatch):
    """32_'s patient loader + scoring without C5 + the H2 summary, on synthetic files."""
    import importlib.util
    from pathlib import Path

    import SimpleITK as sitk

    from gbmc import candidates, confirm, rhuh_features

    shape = (40, 40, 40)
    z, y, x = np.indices(shape)
    r2 = lambda c: (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2  # noqa: E731
    brain = r2((20, 20, 20)) < 17 ** 2
    seg = np.zeros(shape, np.uint8)
    seg[(r2((20, 20, 14)) < 8 ** 2) & brain] = 2          # SNFH
    seg[r2((20, 20, 14)) < 4 ** 2] = 3                    # enhancing
    seg[r2((20, 20, 14)) < 2 ** 2] = 4                    # cavity
    rec = np.zeros(shape, np.uint8)
    rec[(r2((20, 20, 21)) < 3 ** 2) | (r2((20, 10, 28)) < 2 ** 2)] = 3
    rng = np.random.default_rng(0)
    ref = sitk.GetImageFromArray((rng.normal(size=shape) * brain).astype(np.float32))

    def save(arr, path):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(path))
        return path

    noise = lambda: (rng.normal(size=shape) * brain).astype(np.float32)  # noqa: E731
    files = {"T1ce": save(noise(), tmp_path / "t1c.nii.gz"),
             "T1": save(noise(), tmp_path / "t1.nii.gz"),
             "seg": save(seg, tmp_path / "seg.nii.gz")}
    lf = tmp_path / "lf" / "P1"
    lf.mkdir(parents=True)
    save(brain.astype(np.uint8), lf / "brain_mask.nii.gz")
    save(mu.target(seg, brain).astype(np.uint8), lf / "target_mask.nii.gz")
    save(rec, lf / "recurrence_labels.nii.gz")

    path = Path(__file__).resolve().parents[1] / "scripts" / "32_mu_confirm.py"
    spec = importlib.util.spec_from_file_location("mu_confirm", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "LF", tmp_path / "lf")
    monkeypatch.setattr(mu, "files", lambda p, role: files)
    monkeypatch.setattr(rhuh_features, "falx", lambda r, b, t: np.zeros(b.shape, bool))

    inp = script.patient_inputs("P1")
    assert inp.target.sum() == mu.target(seg, brain).sum()
    assert inp.flair_abnormal.sum() == ((seg == 2) & brain).sum()
    xr = rng.normal(size=(400, 6)).astype(np.float32)
    models = {n: candidates.fit(c, [xr], [xr[:, 0] < -1], seed=1)
              for n, c in candidates.LEARNED.items()}
    row, scores, _ = confirm.score_patient(inp, models, include_c5=False)
    assert row["evaluable"] and "E1" in scores and "C6" not in scores
    out = confirm.h2_summary({"P1": row, "P2": row, "P3": row}, "MU-Glioma-Post", 0.04,
                             master=7, n_boot=50)
    assert out["n_evaluable"] == 3 and out["outcome"] in (
        "Final model wins", "Bounded null", "Inconclusive", "Final model worse")
