#!/usr/bin/env python
"""Record each patient's first-progression timepoint from gbm-4d's transport records.

The endpoint labels in the feature cache came from gbm-4d's transport of the first
segmented RANO progression (D5). ``05_`` and ``07_`` need that timepoint's name, which
was only in ``label_field/<patient>/qc.json``; those records are not on this computer.
This copies the names (nothing else) into the committed
``results/cohort_lock/first_progressions.json``, with the SHA-256 of each source record.

Each record's ``baseline`` must equal the committed ``baselines.json``. Running again with
another folder adds patients; a patient already recorded with a different timepoint stops
the script. Reads no image and no label.

    python scripts/15_first_progressions.py --from <gbm-4d>/data/interim/label_field
    python scripts/15_first_progressions.py --from-ratings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gbmc import lumiere
from gbmc.provenance import file_digest


def collect(source: Path, patients: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    names, digests = {}, {}
    for patient in patients:
        path = source / patient / "qc.json"
        if not path.is_file():
            continue
        qc = json.loads(path.read_text(encoding="utf-8"))
        if qc.get("patient", patient) != patient:
            raise SystemExit(f"{path}: record is for {qc['patient']}")
        if qc["baseline"] != lumiere.baseline_name(patient):
            raise SystemExit(f"{patient}: record baseline {qc['baseline']} differs from "
                             f"baselines.json ({lumiere.baseline_name(patient)})")
        names[patient], digests[patient] = qc["progression"], file_digest(path)
    return names, digests


def merge(table: dict, names: dict[str, str], digests: dict[str, str], source: str,
          patients: list[str]) -> dict:
    known = dict(table.get("first_progression", {}))
    sources = dict(table.get("source_sha256", {}))
    for patient, name in names.items():
        if patient in known and known[patient] != name:
            raise SystemExit(f"{patient}: already recorded as {known[patient]}, record says {name}")
        known[patient] = name
        sources[patient] = digests[patient]
    folders = sorted(set(table.get("source_folders", [])) | {source})
    return {"note": "First segmented RANO progression per patient. Copied from gbm-4d's "
                    "transport records (label_field/<patient>/qc.json), or, for the patients "
                    "in derived_from_ratings, derived from LUMIERE's rating sheets by a rule "
                    "that reproduces every copied name. Names only.",
            "first_progression": dict(sorted(known.items())),
            "missing": [p for p in patients if p not in known],
            "source_sha256": dict(sorted(sources.items())), "source_folders": folders}


def from_ratings(table: dict, patients: list[str]) -> tuple[dict, dict, str]:
    """Names for the patients not yet recorded, from LUMIERE's rating sheets.

    The rule (``gbmc.progressions.first_progression_from_ratings``) must reproduce every
    name already in the table first; any disagreement stops the script.
    """
    from gbmc.progressions import first_progression_from_ratings

    md = lumiere.lumiere_root() / "metadata"
    sheet = md / "LUMIERE-ExpertRating-v202211.csv"
    ratings = lumiere.load_ratings(md)
    completeness = lumiere.load_completeness(md).set_index(["Patient", "Timepoint"])
    known = table.get("first_progression", {})
    derived = {p: first_progression_from_ratings(ratings, completeness, p,
                                                 lumiere.baseline_name(p))
               for p in patients}
    wrong = {p: (known[p], derived[p]) for p in known if derived[p] != known[p]}
    if wrong:
        raise SystemExit(f"the rule disagrees with recorded names (recorded, derived): {wrong}")
    print(f"rule reproduces all {len(known)} recorded names")
    new = {p: tp for p, tp in derived.items() if p not in known and tp is not None}
    digest = file_digest(sheet)
    return new, dict.fromkeys(new, digest), f"rule on {sheet.name} + LUMIERE-datacompleteness.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="source", type=Path,
                     help="a gbm-4d data/interim/label_field folder")
    src.add_argument("--from-ratings", action="store_true",
                     help="derive the names from LUMIERE's rating sheets "
                          "($GBMC_LUMIERE_ROOT/metadata); refuses unless every name already "
                          "recorded is reproduced")
    ap.add_argument("--source-name", default=None,
                    help="how to describe the folder in the record (default: its path)")
    args = ap.parse_args()
    patients = lumiere.cohort()
    table_path = lumiere.FIRST_PROGRESSIONS
    table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.is_file() else {}
    if args.from_ratings:
        names, digests, source = from_ratings(table, patients)
    else:
        names, digests = collect(args.source, patients)
        source = args.source_name or str(args.source)
    if not names:
        raise SystemExit("no new names for the locked cohort")
    record = merge(table, names, digests, source, patients)
    if args.from_ratings:
        record["derived_from_ratings"] = sorted(
            set(table.get("derived_from_ratings", [])) | set(names))
    table_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{len(record['first_progression'])} of {len(patients)} recorded; "
          f"missing {len(record['missing'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
