#!/usr/bin/env python
"""Refit every learned candidate on all 63 patients and hash the weights (plan Step 1.7).

Same settings and training rows as the tournament (``02_``). The app shows
every candidate, so all learned candidates are refit, not only the challenger.
C6 has no weights of its own: it is rebuilt from the refit members, and the
register lists their hashes under ``C6_members``. C5 is not refit here: its release
model is trained on the GPU computer (``13_c5_train.py --release``, copied to
``data/interim/c5_release/``) and imported by ``gbmc.c5_import.import_release``, which
checks its hash, settings, cohort and code before copying it to ``models/C5.pt``.

Writes ``results/release/models/<candidate>.joblib`` and
``results/release/model_hashes.json`` (the register the confirmatory run
checks, and the OSF pre-registration quotes), plus the run manifest.

Refuses a dirty tree and a null master seed, like ``02_``, and refuses to run
before the tournament has named a challenger.

    python scripts/03_refit_release.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gbmc import c5_import, candidates, config, freeze, tournament
from gbmc.lumiere import cohort
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.tournament import missing, training_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "release"
C5_RELEASE = REPO_ROOT / "data" / "interim" / "c5_release"
LEADERBOARD = REPO_ROOT / "results" / "tournament" / "leaderboard.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    require_clean_tree()
    master = int(config.get("seeds.master"))
    if not LEADERBOARD.is_file():
        raise SystemExit("no tournament leaderboard: run 02_ first")
    challenger = json.loads(LEADERBOARD.read_text(encoding="utf-8"))["challenger"]
    patients = cohort()
    if absent := missing(patients):
        raise SystemExit(f"features missing for {len(absent)} patients; run 01_ first")

    rows = [training_rows(p, master) for p in patients]
    register: dict[str, str] = {}
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml", LEADERBOARD]) as manifest:
        for name, cand in candidates.LEARNED.items():
            model = candidates.fit(cand, [r[0] for r in rows], [r[1] for r in rows],
                                   stable_seed("release_fit", name, master_seed=master))
            rel = f"models/{name}.joblib"
            register[rel] = freeze.save_model(model, OUT / rel)
            print(f"{name}: {register[rel]}")
        # C5 is trained on the GPU computer (D6 item 9); its model is checked and copied in.
        rel, digest, c5_provenance = c5_import.import_release(
            C5_RELEASE, OUT, tournament.C5_MAPS / "manifest.json", c5_import.settings_dict(),
            patients)
        register[rel] = digest
        print(f"C5: {digest} (imported, trained at {c5_provenance['commit']})")
        files = {name: f"models/{name}.joblib" for name in candidates.LEARNED}
        files["C5"] = rel
        members = {m: register[files[m]] for m in candidates.ENSEMBLE_MEMBERS}
        (OUT / "model_hashes.json").write_text(json.dumps(
            {"challenger": challenger, "models": register, "C6_members": members,
             "C5": c5_provenance}, indent=2) + "\n", encoding="utf-8")
        manifest.record(master_seed=master, n_patients=len(patients), challenger=challenger,
                        model_sha256=register, c5=c5_provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
