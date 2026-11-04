#!/usr/bin/env python
"""Every Step 4.7 figure, drawn only from results/ (the plan requires it).

Draws whatever the existing results support and says what it skipped:

* after the tournament (02_): the complexity forest (LUMIERE development
  points only), matched-volume coverage, coverage curve and per-patient strip
  on LUMIERE, all titled as development results;
* after 06_: the later-progression figure (exploratory);
* after the confirmatory run (30_): every figure on RHUH-GBM, the forest with
  LUMIERE drawn faint beside it, the escape panel, the protocol strip with the
  OSF and download dates, and the four-panel prediction-vs-reality figure for
  the rule-chosen patient (median per-patient Δ).

Writes ``results/figures/*.png|svg`` and a manifest. Refuses a dirty tree.

    python scripts/50_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gbmc import config, figures
from gbmc.provenance import require_clean_tree, run_manifest
from gbmc.rng import stable_seed
from gbmc.stats import mean_with_bca

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "figures"
DENOMINATOR = "recurrence outside the target"


def _json(rel: str) -> dict | None:
    path = RESULTS / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _mean_ci(values, seed_parts, master, n_boot) -> dict:
    ci = mean_with_bca(
        np.asarray(values, float), n_boot, stable_seed("figure", *seed_parts, master_seed=master)
    )
    return {"mean": ci.mean, "ci_low": ci.low, "ci_high": ci.high}


def _deltas(board: dict) -> dict:
    return {
        c: {
            "mean": v["pr_auc"]["delta_vs_C0"]["mean"],
            "ci_low": v["pr_auc"]["delta_vs_C0"]["ci_low"],
            "ci_high": v["pr_auc"]["delta_vs_C0"]["ci_high"],
        }
        for c, v in board.items()
        if isinstance(v, dict) and "delta_vs_C0" in v.get("pr_auc", {})
    }


def per_patient_figures(
    rows: dict, lead: str, cohort: str, master: int, n_boot: int, written: list
) -> None:
    rows = {p: r for p, r in rows.items() if r["evaluable"]}
    bars = {
        "Guideline margin": ("C0T", "guideline_margin"),
        "Distance": ("C0", "distance"),
        lead: (lead, "challenger"),
    }
    hero = {
        label: {
            "role": role,
            **_mean_ci(
                [r[c]["coverage_margin15"] for r in rows.values()],
                (cohort, "hero", c),
                master,
                n_boot,
            ),
        }
        for label, (c, role) in bars.items()
    }
    written += figures.hero_coverage(hero, DENOMINATOR, OUT / f"hero_coverage_{cohort}")
    curves = {}
    for c in ("C0", lead):
        keys = sorted({k for r in rows.values() for k in r["curves"][c]}, key=float)
        curves[c] = {
            float(k): float(np.nanmean([r["curves"][c].get(k, np.nan) for r in rows.values()]))
            for k in keys
        }
    # median volume of the patients' own 15 mm margins, in ml
    margin_ml = float(
        np.median(
            [
                50.0 * r["matched_voxels"]["margin15"] / r["matched_voxels"]["volume50ml"]
                for r in rows.values()
            ]
        )
    )
    written += figures.coverage_curve(
        curves,
        {"C0": "distance", lead: "challenger"},
        DENOMINATOR,
        OUT / f"coverage_curve_{cohort}",
        margin_ml,
    )
    deltas = {p: r[lead]["pr_auc"] - r["C0"]["pr_auc"] for p, r in rows.items()}
    written += figures.per_patient_strip(
        deltas, OUT / f"per_patient_{cohort}", f"{lead} vs distance ({cohort})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args()
    require_clean_tree()
    master = int(config.get("seeds.master"))
    n_boot = int(config.get("seeds.n_bootstrap"))
    written, skipped = [], []

    board = _json("tournament/leaderboard.json")
    confirm = _json("confirm/confirm.json")
    if board is None:
        skipped.append("all tournament figures: results/tournament/ missing (02_ not run)")
    else:
        dev = _deltas(board["candidates"])
        val = _deltas(confirm["candidates"]) if confirm else None
        champion = confirm["champion"] if confirm else None
        written += figures.complexity_forest(
            val, dev, board["challenger"], OUT / "complexity_forest", champion
        )
        per_patient_figures(
            _json("tournament/per_patient.json"),
            board["challenger"],
            "LUMIERE",
            master,
            n_boot,
            written,
        )

    later = _json("tournament/later_progression/later_progression.json")
    if later:
        written += figures.later_progression(later["summary"], OUT / "later_progression")
    else:
        skipped.append("later progression: 06_ not run")

    if confirm is None:
        skipped.append("RHUH-GBM figures, escape panel, protocol strip, reveal: 30_ not run")
    else:
        rows = _json("confirm/per_patient.json")
        lead = confirm["champion"] if confirm["champion"] != "C0" else confirm["challenger"]
        per_patient_figures(rows, lead, "RHUH-GBM", master, n_boot, written)
        written += figures.escape_panel(
            rows, ["C0", lead], {"C0": "distance", lead: "challenger"}, OUT / "escape_RHUH-GBM"
        )
        download = _json("rhuh_inventory/download.json") or {}
        lock = _json("rhuh_lock/consort.json") or {}
        written += figures.protocol_strip(
            [
                ("63 patients", "LUMIERE"),
                ("tournament", "7 folds"),
                ("freeze + hash", "SHA-256"),
                ("OSF", str(config.get("osf.timestamp"))[:10]),
                ("RHUH-GBM", f"downloaded {download.get('download_date', '?')}"),
                (f"{lock.get('n_evaluable', '?')} evaluable", "one run"),
                ("reveal", confirm["outcome"]),
            ],
            OUT / "protocol_strip",
        )
        skipped.append(
            "prediction vs reality: drawn by scripts/51_reveal_figure.py "
            "(needs the RHUH-GBM images)"
        )
    with run_manifest(OUT, inputs=[REPO_ROOT / "config.yaml"]) as m:
        m.record(written=[p.name for p in written], skipped=skipped)
    for line in skipped:
        print(f"skipped: {line}")
    print(f"wrote {len(written)} files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
