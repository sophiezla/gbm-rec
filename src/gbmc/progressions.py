"""Later RANO progressions for the exploratory fourth-dimension analysis (plan Step 1.8).

A later progression is a timepoint rated ``PD``, after the first progression, that has a
DeepBraTumIA segmentation.
"""

from __future__ import annotations

from gbmc.lumiere import (
    baseline_name,
    first_progression,
    load_completeness,
    load_ratings,
    lumiere_root,
    timepoint_sort_key,
)

#: Plan Step 1.8: "the second and third progression".
N_LATER = 2


def later_progressions(ratings, completeness, patient: str, first_progression: str,
                       n_later: int = N_LATER) -> list[str]:
    """PD timepoints after ``first_progression`` with a DeepBraTumIA segmentation."""
    first = timepoint_sort_key(first_progression)
    rows = ratings[(ratings["Patient"] == patient) & (ratings["Rating"] == "PD")]
    out = []
    for tp in sorted(rows["Timepoint"], key=timepoint_sort_key):
        if timepoint_sort_key(tp) <= first:
            continue
        key = (patient, tp)
        if key in completeness.index and bool(completeness.loc[key].get("DeepBraTumIA", False)):
            out.append(tp)
    return out[:n_later]


def schedule(patients: list[str]) -> list[dict]:
    """(patient, baseline, first progression, later progressions) for each patient."""
    md = lumiere_root() / "metadata"
    ratings = load_ratings(md)
    completeness = load_completeness(md).set_index(["Patient", "Timepoint"])
    out = []
    for p in patients:
        first = first_progression(p)
        out.append({"patient": p, "baseline": baseline_name(p), "first": first,
                    "later": later_progressions(ratings, completeness, p, first)})
    return out
