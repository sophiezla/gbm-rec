#!/usr/bin/env python
"""Audit deliverable text (plan Step 4.8 and §13): numbers against results/, and the language rules.

1. Every number quoted must exist in results/ (``gbmc.audit``: decimals
   always; integers only when written ``[[n]]``).
2. Every sentence must keep the language rules (``gbmc.language``). The
   champion comes from ``results/confirm/confirm.json``; before that file
   exists, the word "champion" is only allowed as the "[Champion]" placeholder.

Lines with ``audit: ignore`` are skipped by both. Exits non-zero on any finding.

    python scripts/41_audit_numbers.py docs/poster.md docs/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gbmc import audit, language

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    values = audit.result_values(args.results)
    verdict = args.results / "confirm" / "confirm.json"
    champion = (json.loads(verdict.read_text(encoding="utf-8"))["champion"]
                if verdict.is_file() else None)
    failed = 0
    for path in args.files:
        text = path.read_text(encoding="utf-8")
        numbers = audit.check_text(text, values)
        words = language.check(text, champion)
        for f in numbers:
            print(f"{path}:{f.line}: number {f.quoted} not in results/  | {f.context}")
        for f in words:
            print(f"{path}:{f.line}: language rule '{f.rule}'  | {f.text}")
        failed += len(numbers) + len(words)
        state = "PASS" if not (numbers or words) else f"FAIL ({len(numbers)} numbers, " \
                                                      f"{len(words)} language)"
        print(f"{path}: {state}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
