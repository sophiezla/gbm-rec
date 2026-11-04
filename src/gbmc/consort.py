"""CONSORT flow for the RHUH-GBM cohort lock (plan Steps 3.5-3.6).

Each patient carries the FIRST exclusion step it fails, in the plan's order,
so the counts add up from enrolled to evaluable. Exclusions are logged with
their reason; "no enhancing recurrence outside the target" is a separate,
final category (evaluable = no, reported as undefined, never scored 0).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Plan Step 3.5 order.
STEPS = ("missing sequence or segmentation", "failed transport", "screen failure",
         "empty target", "no enhancing recurrence outside the target")


@dataclass(frozen=True)
class Outcome:
    patient: str
    step: str | None          # None = evaluable
    reason: str = ""


def flow(enrolled: list[str], outcomes: list[Outcome]) -> dict:
    """Counts at every step, from enrolled down to evaluable."""
    by_patient = {o.patient: o for o in outcomes}
    missing = sorted(set(enrolled) - set(by_patient))
    if missing:
        raise ValueError(f"no outcome recorded for {missing}")
    for o in outcomes:
        if o.step is not None and o.step not in STEPS:
            raise ValueError(f"{o.patient}: unknown exclusion step {o.step!r}")
    remaining = len(enrolled)
    rows = [{"stage": "enrolled", "n": remaining}]
    for step in STEPS:
        excluded = sorted((o for o in outcomes if o.step == step), key=lambda o: o.patient)
        remaining -= len(excluded)
        rows.append({"excluded_for": step, "n_excluded": len(excluded),
                     "patients": [{"patient": o.patient, "reason": o.reason} for o in excluded],
                     "remaining": remaining})
    evaluable = sorted(o.patient for o in outcomes if o.step is None)
    assert remaining == len(evaluable)
    return {"enrolled": len(enrolled), "rows": rows, "evaluable": evaluable,
            "n_evaluable": len(evaluable)}


def markdown(table: dict) -> str:
    lines = ["| Step | Excluded | Remaining |", "|---|---|---|",
             f"| Enrolled | | {table['enrolled']} |"]
    for row in table["rows"][1:]:
        lines.append(f"| {row['excluded_for']} | {row['n_excluded']} | {row['remaining']} |")
    lines.append(f"| **Evaluable** | | **{table['n_evaluable']}** |")
    return "\n".join(lines) + "\n"
