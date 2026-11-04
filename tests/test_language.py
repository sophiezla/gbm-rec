from __future__ import annotations

import pytest

from gbmc import language


@pytest.mark.parametrize("text, champion, rule", [
    ("An AI model predicts recurrence.", None, "ai"),
    ("LUMIERE validation showed a PR-AUC of 0.07.", None, "validation"),
    ("The champion is C1.", None, "champion"),
    ("C1 improves on distance.", "C1", "improves"),
    ("The TRE was 1.4 mm.", None, "tre"),
    ("Distance covered 45.2% of the recurrence.", None, "coverage"),
    ("Baseline FLAIR had RR 7.29 in gbm-wm.", None, "flair_lead"),
])
def test_rule_violations_are_flagged(text, champion, rule):
    assert [f.rule for f in language.check(text, champion)] == [rule]


@pytest.mark.parametrize("text, champion", [
    ("C5, a 3D CNN, is the only AI candidate.", None),
    ("LUMIERE gives development results; RHUH-GBM gives validation results.", None),
    ("The title reads: [Champion] finds more of it than distance.", None),
    ("Distance holds: the champion is C0.", "C0"),
    ("C1 improves on distance by 0.02 PR-AUC (95% CI 0.01 to 0.03).", "C1"),
    ("No registration number is a TRE; TRE was never measured, and the screens gave 1.4 mm.",
     None),
    ("Distance covered 45.2% of recurrence outside the target.", None),
    ("FLAIR had RR 7.29 in gbm-wm, an association on 9-13 patients.", None),
    ("Quoted: RR 7.29 for FLAIR (audit: ignore)", None),
])
def test_compliant_sentences_pass(text, champion):
    assert language.check(text, champion) == []


def test_flair_rule_is_per_clause():
    ok = "Background: the guideline's FLAIR sentence, quoted; gbm-4d and gbm-wm (plan §2)."
    assert language.check(ok, None) == []
