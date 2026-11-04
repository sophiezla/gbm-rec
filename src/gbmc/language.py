"""Language rules for every deliverable (plan §13), checked mechanically.

Each rule looks at one sentence at a time and returns a reason when the
sentence breaks it. The checks are conservative pattern matches: they catch
the common slips; they do not prove a text is compliant. A line containing
``audit: ignore`` is skipped (for quoting a source in its own words).

Rules:
  ai            "AI" only for C5 (3D CNN) and C6 (ensemble)
  validation    LUMIERE numbers are development results, never validation
  champion      "champion" only once the verdict exists (the "[Champion]"
                placeholder, the outcome name "Challenger champion" and
                sentences stating the rule are allowed before it)
  improves      "improves on distance" only if a challenger is champion, with its Δ and CI
  tre           no registration number is called a TRE
  coverage      a coverage percentage states its denominator
  flair_lead    the gbm-wm FLAIR result is tagged as an association on 9-13 patients
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE = re.compile(r"(?<=[.!?])\s+")
PERCENT = re.compile(r"\d+(\.\d+)?\s?%(?!\s?CI)")
DENOMINATOR = re.compile(r"outside the target|all recurrence|of all|denominator")


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    text: str


def _sentences(text: str):
    for number, line in enumerate(text.splitlines(), start=1):
        if "audit: ignore" in line:
            continue
        for sentence in SENTENCE.split(line):
            if sentence.strip():
                yield number, sentence


def broken_rules(s: str, champion: str | None) -> list[str]:
    """Names of the rules one sentence breaks."""
    low = s.lower()
    broken = []
    bare_ai = re.sub(r"(TRIPOD|PROBAST)\+AI", "", s)      # checklist names, not claims
    if re.search(r"\bAI\b", bare_ai) and not re.search(r"\bC5\b|\bC6\b|CNN|ensemble|neural",
                                                       s, re.I):
        broken.append("ai")
    if any("lumiere" in clause and re.search(r"\bvalidat", clause)
           for clause in re.split(r"[;,]", low)):
        broken.append("validation")
    describes_rule = re.search(r"\brule\b|only if|\[champion\]|challenger champion", low)
    if re.search(r"\bchampion", low) and champion is None and not describes_rule:
        broken.append("champion")
    if "improves on distance" in low and (champion in (None, "C0")
                                          or not re.search(r"CI|confidence", s)):
        broken.append("improves")
    if (re.search(r"\bTRE\b|target registration error", s, re.I) and re.search(r"\d", s)
            and not re.search(r"never measured|not measured|no TRE", s, re.I)):
        broken.append("tre")
    if re.search(r"\bcover", low) and PERCENT.search(s) and not DENOMINATOR.search(low):
        broken.append("coverage")
    # The caveat may sit anywhere in the sentence, but the claim is looked for per clause.
    if any("flair" in clause and re.search(r"gbm-wm|\bRR\b|rate ratio|7\.29|3\.22", clause,
                                           re.I)
           for clause in re.split(r";", s.lower())) and not (
            "association" in low and re.search(r"9.{0,3}13", s)):
        broken.append("flair_lead")
    return broken


def check(text: str, champion: str | None) -> list[Finding]:
    """``champion``: the verdict's champion ID (e.g. "C0" or "C1"), or None before Phase 4."""
    return [Finding(line, rule, s.strip())
            for line, s in _sentences(text) for rule in broken_rules(s, champion)]
