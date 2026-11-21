# Project Handoff

_Last updated: 2026-11-18_

## Latest (2026-11-18): C5 release model trained (Step 1.7)
- The user ran `13_c5_train.py --release --device cuda` on Colab at `bcadfaa` (`src/gbmc/c5.py` identical to `43a41a0`'s, checked), log `gbm-rec-c5/logs/release_20261118_050954.txt`.
- `gbm-rec-c5/c5_release/manifest.json` (as printed on Colab): clean, not smoke, settings equal `config.yaml:c5`, 63 patients (the same set as `results/tournament/folds.json`, checked), NVIDIA L4, torch 2.15.0+cu132 (as in `requirements-gpu.lock`), `model_sha256` `285cd2857d2e5b409f913b0737a629dbc1903641742370f51b54905c1746d687`.
- Not yet imported. Next (computer with `data/`): copy `gbm-rec-c5/c5_release/` to `data/interim/c5_release/` and run `03_`; it rechecks the hash and registers it in `results/release/model_hashes.json`.

## Latest (2026-11-12/13): the user trained C5 on Colab, with checkpoints
- **Who:** the user (Jack) trains C5 on their own Colab (T4, paid compute units), not the partner. The partner was told. Inputs: the shared `gbm-rec-c5/c5_inputs/` (63 files, hashes checked by `13_`).
- **Pinned commit: `43a41a0`** (`gbm-rec-c5/PINNED_COMMIT.txt`). Do not change it until all 7 folds are done; a new commit restarts every fold.
- **Checkpoints within a fold** (`939be80`, GPU fix `43a41a0`): every 5 epochs to `c5_oof/checkpoints/fold<k>.pt`; a rerun resumes. Resume after a disconnect: Colab cells mount → clone pinned → env → link → train.
- **Verified on the T4:** fold 0 was resumed 3 times (epochs 5, 55, 105) and its model is bit-for-bit identical to the uninterrupted fold 0 at `940e8ea` (SHA-256 `f2b7e840…fe77f`); all 186 compared epoch losses match.
- **Progress: all 7 folds done 2026-11-13** (user report: the Colab watcher printed `all 7 folds done: True`, so `gbm-rec-c5/c5_oof/manifest.json` exists on Drive; the manifest step only writes it when all 7 folds share one commit). Not yet checked on a Mac. Next: copy `gbm-rec-c5/c5_oof/` to `data/interim/c5_oof/`, plus `requirements-gpu.lock`, on the computer that holds `data/` (Jack's clone has no `data/` folder, no LUMIERE copies and no `.venv`; TODO: verify which computer runs `02_`).
- **GPU switch 2026-11-13 (deviation V5):** from fold 1 epoch 81 the run continues on an NVIDIA L4 (TF32 convolutions) to save time; fold 0 and fold 1 epochs 1–80 ran on the T4.
- **Folds 0–1 retrained on the L4 (user decision 2026-11-15, before any score):** set aside the T4-era fold 0/1 files and the manifest into `gbm-rec-c5/c5_oof_T4_folds01/`, then rerun `13_c5_train.py` at `43a41a0` on an L4; it retrains folds 0–1, skips 2–6 and writes a new manifest. Done 2026-11-15/16: new manifest written (all 7 folds at `43a41a0`). Verified: all 7 fold records name the NVIDIA L4, 9 maps each. Next: copy `c5_oof/` to `data/interim/c5_oof/` on the computer with `data/` and run `02_`.
- **Set aside on Drive (not results):** `c5_oof_940e8ea/` (the old fold 0, for the hash check) and `old_checkpoints_939be80/`.

## Latest (2026-11-21): registration public
- https://osf.io/2yuqn/ has been public since 2026-11-20T16:42:57Z (checked through the OSF API: accepted, not embargoed). It contains commit `51d777b` and all four model hashes. Recorded in `config.yaml:osf`.
- The RHUH-GBM pre-registration gate test was deleted, as its docstring said. The MU test now checks only that its metadata folder holds just the spreadsheets.
- **Next:** download RHUH-GBM and MU-Glioma-Post, and commit each download date (after the timestamp). Then the RHUH-GBM layout and `20_`, and the MU layout and `26_`.

## Earlier (2026-11-21): registration submitted; H2 and MU pipeline built
- **Registration:** OSF https://osf.io/2yuqn/, submitted 2026-11-20, pending approval and expected public about 24 hours later. Once it is public, record `osf.url` and `osf.timestamp` in `config.yaml`, after checking the date through OSF's API. Only then download RHUH-GBM and MU-Glioma-Post.
- **H2 in `30_`:** E1 vs C0, outside the H1 family.
- **MU pipeline:** `25_`–`28_`, `32_`. All refuse until the MU file layout is recorded after download.
- **Next:** run `25_mu_eligibility.py` from a clean commit (metadata only). It fixes the 44 patients now, before any image.
- **Deviation V7:** MU eligibility details.

## Earlier (2026-11-19): pre-registration numbers ready
- `03_`, `04_`, `07_ --summary` and `06_` have run from clean commit `657dc32`.
  - Hashes: `results/release/`. MDE at rho 0.7: 0.04–0.05. Round-trip threshold: 3.54 mm, set in `config.yaml`.
- Proposed H2 model: E1, the rank average of C0 and C2 (`scripts/17_h2_explore.py`; rerun from a clean commit).
- Copy-paste text for every registration field: https://claude.ai/artifact/Thr2abwHJHFVm8Sv1kXbJj
- Open before posting: the H2 choice, the MU rules, D12/D13 approval, and title/authors. Then commit, push, rerun `17_`, and post.

## Earlier (2026-11-19): H2 final model in the same registration (D19, V6)
- A final model is built on LUMIERE only, frozen and hashed, and pre-registered as H2 (vs distance) next to the unchanged primary (C1 vs C0). It is tested once on RHUH-GBM and once on MU-Glioma-Post.
- Who does what: `docs/team_plan.md`. Registration platform: OSF Registries, with fallbacks (`docs/preregistration.md`).

## Earlier (2026-11-16): MU-Glioma-Post reserved (D18)
- For a later follow-up study, MU-Glioma-Post is the **sealed test**. Tweaking happens only on LUMIERE and, after the verdict, RHUH-GBM.
- Do not download any MU image until the follow-up is pre-registered. A test enforces this.
- Only its clinical and volume spreadsheets are on disk (`data/raw/mu_glioma_post/`).
- The roadmap is in `docs/future_cohorts.md`. Nothing in the current study changes.

## Tournament (2026-11-16): challenger C1
- `02_` ran once from clean commit `e04c555`. The challenger is C1 by rule 7: C6 had the highest mean PR-AUC, and C1 is within the 0.005 margin by 0.00011. Full table in `results/tournament/leaderboard.json`.
- On LUMIERE (development data):
  - no candidate's Δ PR-AUC vs C0 has a 95% CI above 0;
  - C4 and C5 are below C0 (CIs entirely below 0);
  - C1's Δ is −0.0029 [−0.0079, +0.0013].

  The challenger goes to Phase 4 whatever its Δ (rule 7).
- Next:
  - `03_`: refit and hash. It needs the C5 release model: the partner runs `13_c5_train.py --release` at a commit whose `src/gbmc/c5.py` equals `43a41a0`'s.
  - `04_`: power.
  - `07_ --summary`: the round-trip threshold.

## Earlier (2026-11-11): long jobs running; app cases re-exported
- `07_` is running detached with 3 workers (`logs/07_roundtrip.log`), at about 20 min per pair per worker. When it finishes, run `07_ --summary` from a clean tree: it gives the D17 threshold.
- `05_` is running detached with 2 workers (`logs/05_later.log`), 83 timepoints. Rerunning either job resumes it.
- App hero cases are re-exported. The first-progression layer comes from the feature cache (outside the target, as scored). Node.js is not installed here, so the app was not built or viewed.

## Earlier (2026-11-10): LUMIERE inputs rebuilt from the public release
- The rating sheets were downloaded (MD5 checked), and the 146 progression timepoints were extracted from `~/Downloads/Imaging-v202211.zip` into `data/raw/lumiere/Imaging/`.
- All 63 first-progression names are recorded (13 derived by a rule that reproduces the other 50).
- `05_` and `07_` can now run here. The app export still needs re-transported first-progression labels.

## Earlier (2026-11-10): C5 on Google Colab (superseded 2026-11-12: the user trains it)
**Partner's AI:** read `docs/C5_COLAB_FOR_PARTNER.md` before building the Colab setup. Train on the shared `c5_inputs/`, not a re-pack.

The user's partner trains C5 on Colab with their own notebook; `docs/C5_GPU.md` section 0 lists what it must respect.
- The Mac owner shares a Drive folder `gbm-rec-c5/` with `c5_inputs/` (434 MB) in it.
- The partner trains all 7 folds at one pinned commit, from a clean clone.
- The partner shares back `c5_oof/` and `requirements-gpu.lock`. Copy `c5_oof/` to `data/interim/c5_oof/` and commit the lockfile.

Each fold records its GPU. Not yet run on Colab.

## Earlier (2026-11-08): work that does not need the GPU
- **C5 imports built** (`gbmc.c5_import`, runbook `docs/C5_GPU.md` sections 5–6):
  - `03_` imports and registers the release model with full checks.
  - RHUH-GBM C5 maps flow Mac (`24_`, label-free inputs) → GPU computer (`14_`, registered model only) → Mac (`30_`/`31_` check everything before the run-once marker).
- **Bug fixed before any RHUH-GBM file exists.** On RHUH-GBM, the falx test and the cavity threshold assumed z-scaled intensities without making sure. Both now z-score inside the mask. Whether the shipped files are already z-scored is unverified (the plan says the dataset's pipeline z-scores).
- **First-progression names** are committed in `results/cohort_lock/first_progressions.json`: 50 of 63, from the gbm-4d bundle on Google Drive. The 13 gbm-4d test patients are missing, including all three app hero cases.
- **D9 and D15** are reworded for V3.
- **PDE frames** rebuilt from the V3 baselines (`data/interim/pde_frames/`); see the CHANGELOG for the result.
- **Not on this Mac**, so `05_`, `07_` and `40_` cannot run here:
  - progression scans;
  - LUMIERE rating sheets;
  - `label_field/` recurrence labels;
  - the 13 missing names.
  
  The old "data on disk" list below described the old Windows computer.
- **LightGBM** cannot load here without `libomp`. `02_` needs it, so install it before the tournament.
- **Tests:** 148 passed in `.venv-c5` with PyTorch's `libomp`; 143 passed, 2 libomp failures and 3 PyTorch skips in `.venv`. `ruff` clean.

## Current State (Phase 1, as of 2026-11-07)
All of this work was committed and pushed to the public `sophiezla/gbm-rec` on 2026-11-03, so both partners work from that repo; clone it outside Google Drive. No dates were filled in. **No candidate has been scored, and no RHUH-GBM file exists.**
- **Feature recompute (2026-11-06, deviation V3).** `data/interim/features/` now holds the corrected caches, and `data/interim/features_v1/` the earlier ones. Masks, labels and f1-f4 are unchanged; f5 and C0T changed. Neither is in git, and the Google Drive backup still has only the earlier caches.
- **Open before the tournament rules lock (2026-11-06):**
  - **Ventricle masks: resolved 2026-11-07 (D16, deviation V4).** The f5 cap check found no sealed-off tissue, but the old ventricle masks were unreliable (Patient-002's was 771 ml). Rule V4 replaces them for every patient. All 63 pass QC at 6.2–61.7 ml, and f5 and C0T are recomputed in `data/interim/features/` (V3 caches in `features_v3/`). Known weaknesses: thin ventricles are under-segmented (Patient-028), and some deep sulci remain (Patient-018). The rule is now frozen.
  - PDE frames and app cases were built from the earlier data, so they need re-exporting.
  - Transport names for `05_` and `07_` come from `label_field/<patient>/qc.json`, which is lost; those scripts need the first-progression timepoints from another source.
- **Standalone (2026-11-03, deviation V3).** gbm-rec no longer imports or installs gbm-4d; the code is rewritten in `gbmc`, and the rigid resampling direction is corrected. Earlier sections below that mention `eeba777`, or the old `C:/` paths, are historical. Setup is in `docs/SETUP.md`.
- **Team (2026-10-29).** The project is now a LACSEF team project. The team build guide is `docs/plan/project_plan_lacsef.md` (also a Google Doc: https://docs.google.com/document/d/1Dqy4ejZ648nCc8FEF9tCRtQ8wHXmTVFzzuxGZhdR2q0/edit). It adds no methodology; `final_plan.docx` still governs. The LACSEF SRC questions (team forms, continuation with a new partner, who may test the app under V2) were resolved by the user on 2026-10-30; the outcomes are not recorded here.
- **Rules: LOCKED 2026-11-07 (`DECISIONS.md` D6).** The seed is 20261107. C5 is in (NVIDIA RTX on the GPU computer, trains from scratch), and its fixed settings are D6 item 8 and `config.yaml:c5`. Machine split (item 9): the GPU computer only trains C5, while this Mac computes every reportable number. No candidate has been scored. `02_` refuses until C5's held-out maps are imported to `data/interim/c5_oof/` (hash-checked, same folds). **C5 is built (2026-11-07):** `12_c5_pack.py` has packed all 63 patients into `data/interim/c5_inputs/` (434 MB, clean). A CPU smoke run on 4 real patients worked end to end (maps match the cache lengths; no metric computed). **Next:** on the GPU computer, follow `docs/C5_GPU.md`: set it up, copy `c5_inputs/`, run the smoke test, then train the 7 folds and copy `c5_oof/` back. Then run `02_` here. GPU memory is unmeasured: one full-size step used about 9 GB of RAM on CPU, and a rough activation count says about 3 GB on the GPU. D9–D15 are still proposed; D9 and D15 still describe the gbm-4d loaders replaced by V3 and need rewording.
- **Data on disk (not in git):**
  - Features for all 63 patients (53 evaluable).
  - Later-progression transport: 1 of 83 timepoints done.
  - PDE frames for the three hero cases.
  - The app's exported cases.
- **Phase 2–4 code.** It is built ahead of time and tested on synthetic data. The RHUH-GBM scripts refuse until the OSF record, the verified file layout and the Phase 2 constants are in `config.yaml`; the list is in D13.
- **Long jobs.**
  - `05_` needs about 25 h on one thread and `07_` about 49 min per pair. This machine has about 2 GB of RAM free, so use the second PC: `scripts/08_pack_bundle.py --zip`, then `RUN.bat both 3` there, then `scripts/09_import_bundle.py`.
  - The `07_` smoke run on Patient-001 finished: ok, round-trip median 0.77 mm, no label written, 2,932 s. That puts all 63 pairs at about 51 h on one thread.
- **Validation.**
  - `pytest` 104 passed; `ruff` clean; `tsc` and `vite build` clean.
  - `41_` passes on every deliverable text written so far (README excepted: it flags the forecast "distance remains champion").
  - Browser check of all five app screens.

## Earlier state (Phase 0)
Phase 0 (week 1) of the plan. The repository, the Python environment and the reproducibility helpers are in place, and the tests pass. There is no data and there are no results. The ISEF forms are not signed yet, so no training or downloads are allowed (plan Step 0.1).

## What Was Changed
- `pyproject.toml`, `requirements.lock`, `config.yaml`: package `gbmc`, pinned environment, study constants (D3, D4).
- `src/gbmc/`: `rng.py` (re-exports gbm4d), `config.py`, `provenance.py` (adapted from gbm4d, D3).
- `tests/`: `test_rng`, `test_config`, `test_provenance`, `test_environment`, `test_repo_hygiene`.
- `docs/`: `plan/final_plan.md`, `protocol.md`, `deviations.md` (V1), `preregistration.md`.
- Skeleton folders: `scripts/`, `results/`, `app/`.

## Why It Was Changed
Plan Steps 0.2 (repository), 0.3 (environment) and 0.5 (reproducibility contract).

## Current Results
No study results. Validation:
- `pytest`: 18 passed.
- `ruff check src tests`: clean.

## Known Issues
- gbm-4d commit `eeba777` is not on GitHub (local gbm-4d is 48 commits ahead of `origin/main`). A fresh clone cannot install gbm4d from the URL in `pyproject.toml` until gbm-4d is pushed. The local install used `git+file:///C:/Documents/Projects/gbm-4d@<commit>`.
- Plan quirk, no action needed: Step 6.5 (booth kit) appears after 6.6 and 6.7 in the document.

## Uncommitted work (2026-09-24)
- Step 0.6 drafts in `docs/literature/`. They must be committed before the PubMed search is run. The 25-paper cap rule is a proposal awaiting approval.
- Step 0.4 weight candidates recorded in `PROJECT_STATUS.md` (not verified, not downloaded).

## Next Recommended Step
Resolve D6 open points (a)–(j), D9–D15 and Step 0.4 (C5). Commit, with the dates filled in. Then run `02_`, `03_`, `04_`, `50_`. Pack the bundle (`08_`) and start `05_` + `07_` on the second machine; `06_` follows `05_`. Before OSF: set the round-trip threshold rule from `07_ --summary`, and the D13 constants that do not need the RHUH-GBM files.

## Previous next step (Phase 0)
Sign the ISEF forms and email the SRC (Step 0.1). Then finish Step 0.4 before Sep 28: check the second machine's GPU and search for pretrained 3D tumour-segmentation weights. This laptop has no CUDA GPU.

## Do Not Change Without Approval
- Everything in the plan: the question, both cohorts and their roles, candidates C0–C6, features f1–f5, 7-fold patient-grouped folds, PR-AUC primary, matched-volume coverage headline, selection rule (0.005 tie margin), verdict rule, BCa bootstrap (1000), BH-FDR q = 0.05, the one-shot confirmatory run.
- **RHUH-GBM must not be downloaded before the OSF timestamp.** `config.yaml:data.rhuh_gbm.local_root` stays null until then, and a test enforces this.
- gbm-4d and gbm-wm are frozen: reuse their code and re-fit nothing. gbm4d is pinned at `eeba777`.
- `seeds.master` is null on purpose. It is set with the tournament rules (Step 1.1).

## Files to Inspect First
1. `docs/plan/final_plan.md` (§1, §3, then the current phase)
2. `PROJECT_STATUS.md`
3. `DECISIONS.md`
4. `config.yaml`

## Commands Used
```bash
# environment (Python 3.13.14; the other 3.13.5 install on this machine is the wrong one)
"<python 3.13.14>" -m venv .venv
.venv/Scripts/python -m pip install -r ../gbm-4d/requirements.lock
.venv/Scripts/python -m pip install scikit-image trimesh
.venv/Scripts/python -m pip install --no-deps "gbm4d @ git+file:///C:/Documents/Projects/gbm-4d@eeba7779804412b0a32d0b5ba0f23c3156448549"
.venv/Scripts/python -m pip freeze --exclude gbm4d > requirements.lock
.venv/Scripts/python -m pip install --no-deps -e .

# checks
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests
```

## Auto-generated status

<!-- AUTO:status BEGIN -->

_Auto-generated by `scripts/update_handoff.py`; do not edit inside._

### Results on disk

| Stage | Commit | Dirty | Started (UTC) |
|---|---|---|---|
| (none) | | | |

### Long jobs (finished units)

- LUMIERE feature caches: 63 of 63
- Later-progression transports: 1 of 83
- Round-trip threshold study: 1 of 63
- PDE frame sets: 3
- RHUH-GBM transports: 0

### Recent commits

```
c02aa57 Set up repository, environment and reproducibility helpers (Phase 0)
5eb3e21 Initial commit
```

<!-- AUTO:status END -->
