"""Number audit (plan Step 4.8): every number quoted in text must exist in results/.

Like gbm-wm's ``21_check_manuscript.py``, but generic: it collects every
numeric leaf of the JSON files under ``results/``, extracts every decimal
number from a text file, and reports each quoted number that no result
reproduces at the precision it was quoted to.

Rules
-----
* A quoted number matches a result value when the value, rounded to the
  quoted number of decimals, equals it. Percentages ("45.2%") also match a
  fraction (0.452).
* Integers are checked only when marked. Bare integers (years, "Step 2",
  "15 mm", "n = 40") are too often not results; mark a result integer by
  writing it as ``[[63]]`` in the text source, or keep it out of the audit.
* A line containing ``audit: ignore`` is skipped, for quoted literature values.
* Licence and software versions ("CC BY 4.0", "Python 3.13.14", "v1.2.3") are
  identifiers, not results, and are removed before matching.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])([-−]?\d+\.\d+)(%?)")
MARKED_INT = re.compile(r"\[\[([-−]?\d+)\]\]")
IDENTIFIERS = re.compile(
    r"CC[- ]BY(?:[- ][A-Z]{2})*[- ]\d+\.\d+"        # licences
    r"|\bdoi:\s?\S+|\b10\.\d{4,}/\S+"                # DOIs
    r"|\b(?:Python|v|version)\s?\d+(?:\.\d+)+"      # software versions
    r"|\b\d+\.\d+\.\d+(?:\.\d+)*\b"                 # any dotted version triple
    r"|\b(?:Steps?|§)\s?\d+(?:\.\d+)?(?:\s?(?:[-–,]|and)\s?\d+(?:\.\d+)?)*",  # plan steps
    re.IGNORECASE)


def result_values(results_root: str | Path) -> list[float]:
    values: list[float] = []

    def walk(node) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            values.append(float(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for path in sorted(Path(results_root).rglob("*.json")):
        if path.name == "run.manifest.json":
            continue
        walk(json.loads(path.read_text(encoding="utf-8")))
    return values


def _round(value: float, decimals: int) -> Decimal:
    q = Decimal(1).scaleb(-decimals)
    return Decimal(repr(value)).quantize(q, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Finding:
    line: int
    quoted: str
    context: str


def check_text(text: str, values: list[float]) -> list[Finding]:
    """Quoted numbers with no matching result value."""
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        if "audit: ignore" in line:
            continue
        line = IDENTIFIERS.sub(" ", line)
        for match in NUMBER.finditer(line):
            quoted, percent = match.group(1).replace("−", "-"), match.group(2)
            decimals = len(quoted.split(".")[1])
            target = Decimal(quoted)
            candidates = values + ([v * 100 for v in values] if percent else [])
            if not any(_round(v, decimals) == target for v in candidates):
                findings.append(Finding(number, match.group(0), line.strip()))
        for match in MARKED_INT.finditer(line):
            target = Decimal(match.group(1).replace("−", "-"))
            if not any(_round(v, 0) == target and float(v).is_integer() for v in values):
                findings.append(Finding(number, match.group(0), line.strip()))
    return findings
