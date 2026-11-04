#!/usr/bin/env python
"""Install the pre-commit hook that regenerates HANDOFF.md's status block.

Writes ``.git/hooks/pre-commit`` (refusing to overwrite a different hook) that
runs ``scripts/update_handoff.py`` with this repository's Python and stages
HANDOFF.md. Run once by hand; it is not run automatically.

    python scripts/install_hooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".git" / "hooks" / "pre-commit"
MARKER = "# gbm-rec: regenerate HANDOFF.md status"
PYTHON = Path(sys.executable).as_posix()
UPDATE = (REPO_ROOT / "scripts" / "update_handoff.py").as_posix()
SCRIPT = f"""#!/bin/sh
{MARKER}
"{PYTHON}" "{UPDATE}" || exit 1
git add HANDOFF.md
"""


def main() -> int:
    if HOOK.exists() and MARKER not in HOOK.read_text(encoding="utf-8"):
        raise SystemExit(f"{HOOK} exists and is not ours; not overwriting it")
    HOOK.write_text(SCRIPT, encoding="utf-8", newline="\n")
    HOOK.chmod(0o755)
    print(f"installed {HOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
