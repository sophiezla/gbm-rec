# Project Status: gbm-rec ("Where Does Glioblastoma Come Back?")

_Last updated: 2026-11-08_

## Current Objective
Carry out the final plan (`docs/plan/final_plan.md`, dated 2026-09-22) as written. **Phase 1 (tournament on LUMIERE)**: the rules are locked (D6, 2026-11-07), and C5 is waiting to be trained on the GPU computer (`docs/C5_GPU.md`). Nothing is scored. The user reports Phase 0 is complete; its checklist boxes below are not ticked individually.

### Current blockers (2026-11-08)
1. **C5 training.** The partner trains the 7 folds on Google Colab with their own notebook (from 2026-11-10; runbook section 0). Then copy `c5_oof/` here, to `data/interim/c5_oof/`. `02_` refuses until then.
2. **LightGBM cannot load on this Mac** (`libomp` missing, no Homebrew), so `02_` cannot fit C2 here. Fix: install `libomp` (`docs/SETUP.md`). Not done: it is a system install, for the user. With PyTorch's bundled `libomp` on `DYLD_FALLBACK_LIBRARY_PATH` in `.venv-c5`, all 148 tests pass, so the code itself works.
3. **Data for `05_`, `07_` and `40_`: mostly resolved 2026-11-10.**
   - The rating sheets were downloaded from figshare (MD5 checked).
   - The 146 progression timepoints were extracted from `Imaging-v202211.zip`.
   - All 63 first-progression names are recorded.
   - Still missing: gbm-4d's `label_field/` recurrence labels. Only the app export (`40_`) needs them, for the hero cases; re-transporting them with the current code is the way to rebuild them.

## Phase Gates (plan §3)
| Phase | Weeks | Gate | State |
|---|---|---|---|
| 0 Setup + paperwork | 1 | ISEF forms signed; SRC contacted; repo scaffolded, builds, tests pass; CNN hardware + starting-weights decisions recorded; literature search string committed; expert emails sent | **Open.** Repo, build and tests done |
| 1 Tournament on LUMIERE | 1–2 | All candidates scored on identical folds; challenger named by rule; weights hashed | **Started.** Rules locked (D6); code and tests done, C5 included; waiting on C5 training; nothing scored |
| 2 Freeze + OSF | 2 | OSF timestamp precedes the RHUH-GBM download date | Not started |
| 3 RHUH-GBM cohort lock | 3 | Cohort locked; CONSORT committed; achieved MDE recorded; no model score computed | Not started |
| 4 Confirmatory run, once | 4–5 | Verdict sentence filled; figures from results/; audit passes | Not started |
| 5 App | 2–7 | Runs offline; five-person usability check passed | Not started |
| 6 Communication | 6–8 | Poster final; report drafted; expert review; mock judging; booth kit | Not started |

## Phase 0 Checklist
- [ ] 0.1 ISEF forms 1, 1A, Research Plan/Addendum, 1B, 2A, 7; SRC email sent
- [x] 0.2 Repository scaffolded; gbm4d pinned at `eeba777` (see Blockers)
- [x] 0.3 Environment: Python 3.13.14, gbm-4d lock + scikit-image + trimesh; ITK threads pinned to 1
- [ ] 0.4 CNN gates: hardware and starting weights (facts so far below)
- [x] 0.5 Reproducibility contract carried over (`gbmc.rng`, `gbmc.config`, `gbmc.provenance`, tests)
- [ ] 0.6 PubMed search string and extraction form committed (drafted 2026-09-24 in `docs/literature/`, not committed; the cap rule needs approval)
- [ ] 0.7 Expert outreach emails sent

### Step 0.4 facts (no decision yet)
- This laptop: Intel Arc Graphics, about 2 GB adapter memory, no CUDA. It does not meet the ≥ 8 GB CUDA requirement.
- Second machine: not checked yet. Its GPU and memory decide whether C5 runs.
- Pretrained 3D brain-tumour weights: searched 2026-09-24 (web pages only; nothing downloaded, because Phase 0 bars file downloads until the forms are signed). Candidates, not verified:
  - nnU-Net v2, BRATS 2021, Zenodo DOI 10.5281/zenodo.11582627 (BAMF Health, v1.0.0, 2024-06-11), licence CC BY 4.0 per the record page; `Dataset002_BRATS19.zip`, 1.2 GB, MD5 `23a3f55dead4a6642271a08d1a503bbb`.
  - MONAI model zoo `brats_mri_segmentation` (SegResNet, BraTS 2018), bundle licence Apache 2.0; MONAI says users must check the training data's own terms.
  - Rejected: BrainSegFounder (Hugging Face `smilelab/BrainSegFounder`), licence is the UK Biobank MTA.
  - Open issues: all BraTS models expect 4 channels (T1, T1c, T2, FLAIR), while C5's plan input is T1c, FLAIR, target mask, FLAIR mask and f1 map, so the first layer would need adapting; the BraTS data-use terms behind the weights are not checked yet. `TODO: verify` both before deciding.

## Completed
- 2026-09-23: Documentation scaffold; the plan is stored in `docs/plan/`.
- 2026-09-23: Steps 0.2, 0.3 and 0.5. `pytest`: 18 passed. `ruff`: clean.

## Phase 1 Checklist
- [x] 1.1 Rules committed and locked (D6, 2026-11-07; seed 20261107)
- [x] 1.2 Cohort and endpoint wired to gbm-4d's frozen data (D5); Patient-001 matches gbm-4d exactly
- [x] 1.3 Features for all 63 patients (`data/interim/features/`), rebuilt under V3 and V4; 53 evaluable
- [ ] 1.4 Candidates: all eight coded. C5 is built and its inputs are packed; training on the GPU computer is pending
- [ ] 1.5 / 1.6 Folds, metrics, BCa and selection rule coded and unit-tested; `02_` refuses until the C5 maps are imported
- [ ] 1.7 Refit and hash: `03_` built, including the C5 release import (2026-11-08); not run
- [ ] 1.8 Later progressions: `05_` and `06_` built. Under V3 the old smoke unit must be redone. Blocked on data not on this Mac (see Current blockers)

## Other work done ahead of its phase (no scoring, no RHUH-GBM data)
- **Phase 2 (Steps 2.1, 2.4, 2.5):**
  - Power simulation: `gbmc.power` + `04_`, which needs C0 scores from `02_`.
  - Pre-registration draft (`docs/preregistration.md`).
  - Threshold study `07_` (D12): built; smoke run on 1 of 63 pairs in progress.
- **Phase 3 (Steps 3.1–3.6):** `20_`–`23_` and `gbmc.rhuh` / `consort` / `screens` are built and unit-tested. They refuse until the OSF record and the verified layout are in `config.yaml` (D13).
- **Phase 4 (Steps 4.1–4.8):**
  - `30_confirm` (hash check, run-once marker, verdict, BH-FDR, maps) and `31_sensitivity`.
  - Figures: `50_` / `51_`, rendered from synthetic data and inspected.
  - `41_` audit: numbers plus the language rules.
- **Long jobs:**
  - `05_` has workers.
  - Second-machine bundle `08_` / `09_`: packing tested at 6 GB, and vendored-code verification tested, including tamper detection.
- **Phase 5 app:**
  - Five screens: Story, Explore, Compare, Results, Learn.
  - PDE mode (D15): the final frame matches gbm-4d's saved field exactly for all three hero cases.
  - Each case is 2.3–3.2 MB (budget 5 MB).
  - Browser-checked: Story steps, PDE play, pending states, and the Results charts with synthetic data (since removed).
- **Phase 6:** `docs/report.md` (TRIPOD+AI skeleton), `docs/probast_ai.md`.
- **Tooling:** `update_handoff.py` + `install_hooks.py` (hook not installed).

## Next Steps
1. Share `c5_inputs/` in a Drive folder `gbm-rec-c5/`. The partner trains on Colab (runbook section 0) and shares back `c5_oof/`.
2. This Mac: install `libomp`, then run `02_`.
3. Copy from the Windows PC (or LUMIERE's public release) the data listed under Current blockers, item 3. Then run `15_first_progressions.py` for the 13 missing patients, and run `05_`, `07_` and `40_`.
4. Pre-registration constants decided 2026-11-10 (D17). Still open: the round-trip threshold value, which comes from `07_ --summary`.
5. Phase 0 leftovers, status not recorded here: 0.6 (commit the PubMed search string) and 0.7 (expert emails).

## Known Issues
- Resolved 2026-11-03: the gbm-4d dependency is gone (deviation V3).
- Resolved 2026-11-07: the unreliable ventricle masks are replaced by rule V4 (D16, deviation V4), and f5 and C0T are recomputed. Patient-074 now has ventricles, which resolves D6 (j).
- 2026-11-07: the tournament rules are locked (D6, Step 1.1), with seed 20261107 and C5 in. Nothing is scored yet. The C5 code is built and its inputs are packed. Blocking the tournament: training C5 on the GPU computer (`docs/C5_GPU.md`).

## Blockers
- See "Current blockers" at the top. ISEF forms: the user reports Phase 0 complete; whether the forms are signed is not recorded here (TODO: verify).

## Items Requiring Approval
- The fair date. The plan assumes the week of Nov 16 but says the date is not recorded yet.

## Recent Changes
See `CHANGELOG.md`.
