# Project Status: gbm-rec ("Where Does Glioblastoma Come Back?")

_Last updated: 2026-09-24_

## Current Objective
Carry out the final plan (`docs/plan/final_plan.md`, dated 2026-09-22) as written. The user reports Phase 0 is complete, and **Phase 1 (tournament on LUMIERE)** has started. Phase 0 checklist boxes below are not yet ticked individually. Tick them, with their dates, when committing.

## Phase Gates (plan §3)
| Phase | Weeks | Gate | State |
|---|---|---|---|
| 0 Setup + paperwork | 1 | ISEF forms signed; SRC contacted; repo scaffolded, builds, tests pass; CNN hardware + starting-weights decisions recorded; literature search string committed; expert emails sent | **Open.** Repo, build and tests done |
| 1 Tournament on LUMIERE | 1–2 | All candidates scored on identical folds; challenger named by rule; weights hashed | **Started.** Code and tests done; rules drafted (D6); nothing scored |
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
- [ ] 1.1 Rules committed: D6 drafted; open points (a)–(j) await approval
- [x] 1.2 Cohort and endpoint wired to gbm-4d's frozen data (D5); Patient-001 matches gbm-4d exactly
- [x] 1.3 Features extracted for all 63 patients (`data/interim/features/`). 53 are evaluable, matching gbm-4d's 42 + 11. Patient-074 has no ventricle barrier (D6 j)
- [ ] 1.4 Candidates: C0, C0T, C4, C1, C3, C2, C6 coded; C5 waits on Step 0.4
- [ ] 1.5 / 1.6 Folds, metrics, BCa and selection rule coded and unit-tested; `02_` not run (needs committed rules + seed)
- [ ] 1.7 Refit and hash: `03_` and `gbmc.freeze` built and tested; not run
- [ ] 1.8 Later progressions: `05_` (transport) smoke-tested on 1 of 83 timepoints; `06_` (scoring) built, not run

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
1. Step 0.1: sign the forms and email the SRC. **No model training and no data download before this.**
2. Step 0.4: check the second machine's GPU; search for pretrained weights; record both in DECISIONS.md by Sep 28.
3. Step 0.6: commit the PubMed search string and extraction form before reading.
4. Step 0.7: send the expert emails.

## Known Issues
- Resolved 2026-11-03: the gbm-4d dependency is gone (deviation V3).
- Open 2026-11-06: ventricle masks are unreliable, and Patient-002's covers 52.8% of the brain. They feed f5, C0T and the matched volume. Decision needed before the rules lock: `DECISIONS.md` D6 (k).

## Blockers
- ISEF forms not signed (blocks Phase 1).

## Items Requiring Approval
- The fair date. The plan assumes the week of Nov 16 but says the date is not recorded yet.

## Recent Changes
See `CHANGELOG.md`.
