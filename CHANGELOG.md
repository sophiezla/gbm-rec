# Changelog: gbm-rec

## [Unreleased: Phase 1 start, not committed]

### Added
- 2026-11-12: C5 checkpoints within a fold (user request after a Colab disconnect restarted fold 0). `c5.train` saves the full training state every 5 epochs and resumes from it; `13_c5_train.py` keeps one checkpoint per fold (and one for the release model) and deletes it when the fold is done. No setting, seed or step changes:
  - `tests/test_c5.py`: an interrupted-and-resumed run ends with the same weights as an uninterrupted one, bit for bit; a checkpoint from another run is refused.
  - One-off check (CPU, PyTorch 2.8, not the pinned environment): the new code with checkpoints and the code at `940e8ea` train bit-identical models at two network sizes.
  - Folds trained at `940e8ea` do not carry over: the fold records are per commit, so all 7 folds are trained at the new commit.
- 2026-11-10: rebuilt the missing LUMIERE inputs from the public release (user decision: "Rebuild from public"):
  - Downloaded `LUMIERE-ExpertRating-v202211.csv` and `LUMIERE-datacompleteness.csv` from figshare collection 5904905, MD5 matching, into `$GBMC_LUMIERE_ROOT/metadata/` (not in git).
  - Extracted the 146 first- and later-progression timepoints (8.1 GB) from the local `Imaging-v202211.zip`. Its size matches figshare's to the byte.
  - Added `gbmc.progressions.first_progression_from_ratings` and `15_ --from-ratings`. The rule reproduces all 50 copied names exactly and refuses on any disagreement.
  - The 13 missing names are now recorded: 63 of 63, listed in `derived_from_ratings`.
  - The schedule recounts as 83 later timepoints in 46 patients, matching D9.
- 2026-11-10: D17, the pre-registration constants (user decisions):
  - mapping agreement 0.8;
  - cavity dark z -1.0 (from LUMIERE ventricle and tissue intensities);
  - round-trip rule `max`;
  - surface review 3.0 mm and worse-margin 1.0 mm;
  - `mde_rho` 0.7.

  They are written into `config.yaml` and `docs/preregistration.md`. `07_ --summary` now applies the rule and reports `threshold_mm`.
- 2026-11-10: `docs/C5_COLAB_FOR_PARTNER.md`: instructions for the partner's AI to build their own Colab setup for C5. It says to train on the shared `c5_inputs/`, never a re-pack from their own LUMIERE copy, because the labels and masks inside come from gbm-4d's lost transport and a re-pack would not line up voxel for voxel. The inputs are shared as Google Drive folder `gbm-rec-c5/` (63 files, hashes verified after copying).
- 2026-11-10: `docs/preregistration.md` filled from the plan (Steps 2.2–2.5, 3.3–3.6, 4.4) and the locked D6: candidates and features, the cavity and mapping rules, the registration and exclusions, the primary and verdict with outcome classes, all secondary, sensitivity and exploratory lists, and the deviations. The open constants are listed. `41_` passes on it.
- 2026-11-10: C5 is trained on Google Colab by the user's partner, with their own notebook. `13_` now records the device and GPU per fold (`device_info`) and lists them in the manifest (`fold_devices`), because Colab GPUs can differ between sessions. Runbook section 0 lists what any Colab notebook must respect; D6 (g) and item 9 record the switch.
- 2026-11-08: the C5 imports that were TODO in D6 item 9:
  - `gbmc.c5_import`:
    - `import_release` (used by `03_`) checks the clean tree, the smoke flag, the settings, the cohort, the hash, and that `gbmc/c5.py` is the same as for the tournament maps. It registers `models/C5.pt`.
    - `rhuh_scores` (used by `30_` and `31_`) checks the registered model hash, the packed inputs, each map's hash and its voxel count.
  - New scripts: `24_rhuh_c5_pack.py` (Mac; label-free RHUH-GBM inputs per variant: tuned, original, alt_cavity) and `14_c5_score_rhuh.py` (GPU computer).
  - `13_ --release` records its patient list.
  - Tests: `tests/test_c5_release_rhuh.py`; runbook sections 5–6 in `docs/C5_GPU.md`.
- 2026-11-08: `scripts/15_first_progressions.py` and the committed `results/cohort_lock/first_progressions.json`:
  - The table holds the first-progression timepoint names, copied from the gbm-4d bundle's transport records (50 of 63; the 13 gbm-4d test patients are missing).
  - `gbmc.lumiere.first_progression` replaces `transport_qc`, whose `label_field/<p>/qc.json` is not on this computer.
  - `07_` now preprocesses the first-progression scan itself when it is missing, with the same function and seed as `00_`/`05_`.
  - Tests: `tests/test_first_progressions.py`.
- 2026-11-08: PDE frames re-made from the V3 baselines for the three hero cases (`42_`, `data/interim/pde_frames/`, not in git). 93 s in all; 9 frames from day 0 to 206 each; the peak density falls from 0.80 to 0.12–0.24. gbm-4d's saved densities are not on this computer, so the exact-match check of D15 was not repeated. The app cases (`40_`) cannot be re-exported yet: they need the first-progression labels and names for these three gbm-4d test patients.
- 2026-11-07: C5 code: `src/gbmc/c5.py` (3D U-Net, patch sampling, augmentation, training, sliding-window prediction; PyTorch imported lazily), `scripts/12_c5_pack.py`, `scripts/13_c5_train.py`, `tests/test_c5.py`, `docs/C5_GPU.md`, `requirements-gpu.txt`. The C5 import also refuses smoke runs. Inputs packed for all 63; a CPU smoke run passed on 4 patients.
- 2026-11-07: tournament rules locked (D6): `seeds.master` = 20261107; C5 gate passed, settings in `config.yaml:c5`; C5 joins the C6 members. `gbmc.tournament.c5_held_out` imports C5 maps with hash and fold checks (`tests/test_c5_import.py`). `02_` refuses without them, and `03_` and the confirmatory scoring refuse until C5's imports are built.
- 2026-11-07: ventricle rule V4 in `gbmc.anatomy.segment_ventricles` (the old rule is kept as `segment_ventricles_otsu`); `tests/test_ventricles.py`; `--old/--new/--out` options on `10_` and `--out` on `11_`; `results/feature_recompute_v4/` and `results/f5_cap_survey_v4/`. f5 and C0T are recomputed for all 63 (D16, deviation V4).
- 2026-11-06: `scripts/11_f5_cap_survey.py` and `results/f5_cap_survey/`: anatomy-only survey of voxels at the f5 cap. No sealed-off tissue; found unreliable ventricle masks (`DECISIONS.md` D6 (k), open). Nothing changed in the features.

### Fixed
- 2026-11-10: `gbmc.provenance.run_manifest` read the git state when a run ended, so a clean run's own new outputs made its manifest say `dirty: true`. That is why manifests had to be regenerated after committing. It now captures the state when the run starts. Test: `test_manifest_git_state_is_taken_before_the_run_writes`. The committed manifests were all regenerated clean, so none is affected.
- 2026-11-08, RHUH-GBM intensity scale (found in code review; no RHUH-GBM file exists, nothing scored):
  - **Falx test.** `30_` passed raw post-op T1 as `t1_z` to the falx white-matter test, whose threshold (`WHITE_MATTER_Z` = 0) is on the z scale. If RHUH-GBM's files are not z-scored, that would have left it with no falx barrier. The plan says they are z-scored by the dataset's own pipeline, over a region not yet verified (no file downloaded; corrected 2026-11-10), so the effect may be small. Either way, z-scoring again inside the brain mask matches LUMIERE. T1 is now z-scored inside the brain, as on LUMIERE's preprocessed images.
  - **Cavity threshold.** The cavity inspection (`20_`) and the alternative-cavity target compared raw T1ce with `rhuh.cavity_dark_t1ce_z`. Both now z-score T1ce over non-zero voxels. The threshold is still null.
  - The ventricle rule needs no change: its threshold is built from a percentile and the median, so the intensity scale does not matter.
- 2026-11-08: the RHUH-GBM patient loader and the sensitivity-arm definitions moved from `30_`/`31_` into `gbmc.confirm` (`load_patient`, `alt_cavity_target`, `sensitivity_arms`, `c5_variants`), so the C5 packer builds exactly the patients the scoring builds. `confirm.score_patient` now takes C5's scores explicitly and checks their length. `gbmc.imaging.zscore_array` is new.

### Changed
- 2026-11-08: `DECISIONS.md` D9 and D15 reworded for the V3 rewrite (no gbm-4d loader); D6 item 9 records the built imports.
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
