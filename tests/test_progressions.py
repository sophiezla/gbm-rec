from __future__ import annotations

import pandas as pd

from gbmc import progressions


def _tables():
    ratings = pd.DataFrame({
        "Patient": ["P"] * 6,
        "Timepoint": ["week-000-2", "week-010", "week-020", "week-030", "week-044", "week-060"],
        "Rating": ["Post-Op", "PD", "SD", "PD", "PD", "PD"],
    })
    completeness = pd.DataFrame({
        "Patient": ["P"] * 6,
        "Timepoint": ratings["Timepoint"],
        "DeepBraTumIA": [True, True, True, False, True, True],
    }).set_index(["Patient", "Timepoint"])
    return ratings, completeness


def test_later_progressions_skip_unsegmented_and_cap_at_two():
    ratings, completeness = _tables()
    out = progressions.later_progressions(ratings, completeness, "P", "week-010")
    assert out == ["week-044", "week-060"]      # week-030 has no segmentation; SD skipped


def test_later_progressions_orders_by_week_not_string():
    ratings, completeness = _tables()
    out = progressions.later_progressions(ratings, completeness, "P", "week-000-2", n_later=5)
    assert out == ["week-010", "week-044", "week-060"]
