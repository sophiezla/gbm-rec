# Changelog: gbm-rec

## [Unreleased: Phase 1 start, not committed]

### Changed
- 2026-11-06: rebuilt the 63 LUMIERE baselines from raw data with the corrected alignment and recomputed f1-f5 and C0T, keeping the original evaluation masks and labels (`00_`, `01_ --keep-from`, `10_`; results in `results/feature_recompute/`; deviation V3).
- 2026-11-03: gbm-rec is standalone (deviation V3). New modules `gbmc.imaging`, `transport`, `anatomy`, `pde` replace the gbm4d package and scripts; `results/cohort_lock/cohort_lock.json` is committed; data paths now come only from `$GBMC_LUMIERE_ROOT` and `$GBMC_DERIVED_ROOT`; rigid resampling direction corrected; `--recipe gbm4d` renamed `original`.

### Added
- 2026-10-29: `docs/plan/project_plan_lacsef.md`, a team build guide for LACSEF derived from the final plan (no timeline, no individual role split). Shared copies: Claude doc https://claude.ai/artifact/7jD9jT6FFKRUhkA4Ljft7g and Google Doc https://docs.google.com/document/d/1Dqy4ejZ648nCc8FEF9tCRtQ8wHXmTVFzzuxGZhdR2q0/edit. `final_plan.docx` stays authoritative.
- `src/gbmc/lumiere.py` (D5), `features.py` (f1-f5, C0T distance), `metrics.py` (PR-AUC, lift, tie-aware matched-volume coverage), `stats.py` (BCa, from gbm-wm), `candidates.py` (C0-C6 except C5, folds, selection rule).
- `scripts/01_lumiere_features.py`, `scripts/02_tournament.py` (not run on the cohort).
- `tests/test_features.py`, `tests/test_tournament_parts.py`.
- `DECISIONS.md` D5-D8; D6 = tournament rules, DRAFT with open points (a)-(g).

- `src/gbmc/`: `tournament.py` (shared feature-cache access), `freeze.py` (model hashing, hash check, run-once marker), `power.py` (MDE under the BCa rule), `progressions.py` (later-progression selection, frozen gbm-4d script loader), `audit.py` (number audit), `export.py` (meshes, slices).
- `scripts/03_refit_release.py`, `04_power.py`, `05_later_progressions.py`, `06_later_progression_scores.py`, `40_export_case.py`, `41_audit_numbers.py`.
- `app/`: Vite + TypeScript + three.js + NiiVue Explore screen (D11).
- Tests: `test_lumiere`, `test_freeze`, `test_power`, `test_progressions`, `test_audit`, `test_export` (55 in total).
- `DECISIONS.md` D9-D11 (proposed); D6 open points (h), (i).
- `docs/preregistration.md`: Step 2.5 checklist mapped to sources; Step 2.4 sentences; title bands.

### Changed
- `config.yaml`: LUMIERE and gbm-4d roots; draft `tournament:`, `power:`, `later_progression:` and `app:` sections.
- `scripts/02_tournament.py`: saves held-out score maps to `data/interim/oof/`.
- `.gitignore`: `app/public/cases/`, `app/node_modules/`, `app/dist/`.

### Added (round 3: everything buildable before the rules are committed)
- Long jobs:
  - `05_` gains `--workers`.
  - Second-machine bundle: `08_pack_bundle.py` / `09_import_bundle.py`. The gbm-4d code is vendored with SHA-256 checks (`gbmc.progressions.verify_frozen_gbm4d`).
- Step 3.3 threshold study: `gbmc.registration` (gbm-wm tuned SyN) and `07_roundtrip_threshold.py`.
- RHUH-GBM Phase 3:
  - Modules: `gbmc.rhuh`, `gbmc.consort`, `gbmc.screens`.
  - Scripts: `20_rhuh_inventory`, `21_rhuh_transport`, `22_rhuh_screens`, `23_rhuh_lock`.
- Phase 4:
  - Modules: `gbmc.verdict`, `gbmc.confirm`, `gbmc.rhuh_features`.
  - Scripts: `30_confirm` (once), `31_sensitivity` (once).
- Figures: `gbmc.figures`, `50_figures.py`, `51_reveal_figure.py`. Coverage curves are recorded per patient in `02_` and `30_`.
- Language rules: `gbmc.language`, wired into `41_audit_numbers.py`.
- App:
  - Story, Compare, Results and Learn screens, and PDE mode in Explore.
  - `app/src/content/*.html`, audited by `41_`.
  - `42_pde_frames.py` builds the PDE frames.
  - `40_` also writes `app/public/results.json` and the PDE meshes.
- `update_handoff.py` + `install_hooks.py` (not installed).
- `docs/report.md` (TRIPOD+AI skeleton), `docs/probast_ai.md`.
- `DECISIONS.md` D12–D15.

### Fixed
- `gbmc.lumiere.cohort()` read the cohort lock's `cohort` field as a count; it is the patient list.
- `docs/protocol.md`: Phase 1 script table.

## [2026-09-24]

### Added
- `docs/literature/search_protocol.md` and `extraction_form.csv`: Step 0.6 drafts (not yet committed; nothing searched or read).

### Changed
- `PROJECT_STATUS.md`: Step 0.4 candidate pretrained weights (web search only, nothing downloaded).

## [2026-09-23]

### Added
- Project documentation: `README.md`, `PROJECT_STATUS.md`, `HANDOFF.md`, `DECISIONS.md` (D0–D4), `CHANGELOG.md`.
- `docs/plan/final_plan.md`: the project plan (D1).
- `docs/protocol.md`, `docs/deviations.md` (V1), `docs/preregistration.md`.
- Package `gbmc` (`src/gbmc/`: `rng`, `config`, `provenance`) carrying over gbm-4d's reproducibility contract (D3).
- `pyproject.toml`, `requirements.lock` (Python 3.13.14; gbm-4d lock + scikit-image + trimesh, D4), `config.yaml`.
- Test suite (18 tests) and `.gitignore` rules keeping data and images out of git.
