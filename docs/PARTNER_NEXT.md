# For the partner and his AI: everything left, start to finish

_Written 2026-11-21. Read `CLAUDE.md` first, then this file, then the files it names. Since deviation V9 (user decision), all test-cohort work runs on the partner's computer; C5's scoring step runs on a Colab NVIDIA L4._

## Where things stand
- **Registered and frozen.** The pre-registration is public: https://osf.io/2yuqn/, registered 2026-11-20T16:42:57Z. Nothing in it may change.
- **Development is finished.** The LUMIERE tournament ran, the challenger is C1, and the H2 final model is E1 (the rank average of C0 and C2). Every model is frozen, with its SHA-256 in `results/release/model_hashes.json` and in the registration.
- **Prepared in code.** Every rule and constant for the test cohorts is fixed (`DECISIONS.md` D17–D20). The 44 eligible MU-Glioma-Post patients are already listed in `results/mu_eligibility/`.
- **Left to do:**
  1. download the two test cohorts;
  2. one unattended run;
  3. C5 scoring on an L4;
  4. three one-time confirmatory runs.

## Ground rules (stop and tell the user if any cannot be kept)
- **Edit nothing.** Change no code, setting, threshold or file in the repository, and retrain nothing. The scripts commit their own results; never commit anything else.
- **Never work around a refusal.** If a script refuses or fails (dirty tree, hash mismatch, missing file, failed check), stop and send the user the exact error.
- **One machine runs the final tests.** Only this computer runs `30_`, `31_` and `32_`, each once and only after the user says go. Push straight after each one.
- **Keep data out of git.** Images, labels and models stay in the git-ignored `data/` and `results/release/models/`.
- **Report results as written.** In any summary, follow `CLAUDE.md`'s language rules: "champion" only as the verdict rule assigns it, coverage always with its denominator, RHUH-GBM and MU-Glioma-Post results called validation, LUMIERE never.

## Step 1: set up this computer (once)
1. Clone `https://github.com/sophiezla/gbm-rec`, outside any synced folder (Google Drive, OneDrive, iCloud), and `git pull` to the latest `main`.
2. Create the environment with uv:
   ```bash
   uv venv .venv --python 3.13.14
   uv pip install --python .venv -r requirements.lock
   uv pip install --python .venv --no-deps -e .
   ```
   - On Windows, the interpreter is `.venv\Scripts\python`; elsewhere it's `.venv/bin/python`. This guide writes `python` for it.
   - **Mac:** LightGBM needs OpenMP: `brew install libomp`.
   - **Windows and Linux:** the LightGBM wheel brings its own OpenMP.
3. Check the environment: `python -m pytest` should end with every test passed and 5 skipped. The skips are PyTorch tests; PyTorch isn't needed on this computer. If any test fails, stop.
4. Copy the four frozen models from the shared Drive folder `gbm-rec-c5/release_models/models/` into `results/release/models/`. Verify them:
   ```bash
   python -c "import json; from gbmc import freeze; r=json.load(open('results/release/model_hashes.json')); freeze.verify_hashes(r['models'], 'results/release'); print('models OK')"
   ```

## Step 2: download the two test cohorts
Both image sets are on TCIA, CC BY 4.0, and need the **IBM Aspera client**. TCIA's package page offers it under "Install now".
1. **RHUH-GBM** (2.9 GB, NIfTI). Open the NIfTI package from https://www.cancerimagingarchive.net/collection/rhuh-gbm/ (the row "Brain-extracted Images, Segmentations… NIfTI"). Click **Download all** and save into `data/raw/rhuh_gbm/` in the repo, so that `data/raw/rhuh_gbm/RHUH-GBM_nii_v1/` and `RHUH-GBM-nii-v1.sums` exist. Do not download the DICOM item.
2. **RHUH-GBM clinical table** (6.9 KB): download https://www.cancerimagingarchive.net/wp-content/uploads/clinical_data_TCIA_RHUH-GBM.csv into `data/raw/rhuh_gbm/`. Its SHA-256 must be `32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728`.
3. **MU-Glioma-Post** (11 GB, NIfTI). Open the package from https://www.cancerimagingarchive.net/collection/mu-glioma-post/. Click **Download all** and save into `data/raw/mu_glioma_post_images/`, so that `data/raw/mu_glioma_post_images/MU-Glioma-Post/` exists.

The paths are in `config.yaml` (repo-relative). If Aspera wraps a download in an extra folder, move the contents so the paths above are exact.

## Step 3: the unattended run (`scripts/run_test_cohorts.py`)
Run from the repo root, with a clean tree. Set `GBMC_SHARE_DIR` to the local path of the shared Drive folder `gbm-rec-c5`, so C5's inputs get copied there for Colab:
```bash
GBMC_SHARE_DIR="<path to the synced gbm-rec-c5 folder>" python scripts/run_test_cohorts.py --workers 4
```
Pick `--workers` as roughly the smaller of (CPU cores ÷ 2) and (free RAM in GB ÷ 3). It runs these stages, each resumable, committing its results locally as it goes:
1. **downloads:** file counts (720 RHUH-GBM files, 2978 MU files), RHUH-GBM's checksum file, and the download record. The record must be later than the registration.
2. **rhuh_constants:** checks that RHUH folders 0/1/2 are pre-op, post-op and recurrence, and sets the SRI24 midline (D20 rules).
3. **rhuh_inventory** (`20_`): the label-mapping check and the cavity inspection; it sets the cavity rule by D20.
4. **rhuh_transport** (`21_`, two recipes), **rhuh_screens** (`22_`) and **rhuh_lock** (`23_`).
5. **rhuh_c5_pack** (`24_`): C5's RHUH-GBM inputs, with no labels, copied to `gbm-rec-c5/rhuh/c5_inputs/`.
6. **mu_transport** (`26_`), **mu_screens** (`27_`) and **mu_lock** (`28_`).

It takes many hours; registration is the slow part (on the order of 20 minutes per patient pair per worker on the Mac; not measured on this computer). If it stops, fix nothing yourself: report the error to the user. If it was only interrupted (power, sleep), rerun the same command; finished stages are skipped. **When it finishes: `git push`.**

## Step 4: C5 scores on a Colab NVIDIA L4 (`scripts/14_c5_score_rhuh.py`)
1. In Colab (L4 only; if another GPU is assigned, stop and tell the user), set up as for training: `docs/C5_GPU.md` section 0, `docs/C5_COLAB_FOR_PARTNER.md`, `requirements-gpu.lock`. Check out the commit you just pushed. `git status --porcelain` must print nothing.
2. Link these into the clone's git-ignored `data/`:
   - `gbm-rec-c5/rhuh/c5_inputs/` as `data/interim/rhuh/c5_inputs/`;
   - `gbm-rec-c5/c5_release/` as `data/interim/c5_release/`;
   - a new `gbm-rec-c5/rhuh/c5_maps/` as `data/interim/rhuh/c5_maps/`.
3. Run `python scripts/14_c5_score_rhuh.py --device cuda`. It checks that `model.pt` is the registered C5 (`285cd2857d2e…`) and that every input matches its manifest. Inference only, probably under an hour (not measured).
4. Copy `gbm-rec-c5/rhuh/c5_maps/` (all three variants and their manifests) to `data/interim/rhuh/c5_maps/` on this computer.

## Step 5: the three confirmatory runs, each once, on the user's go
Ask the user before starting, then run in this order. Each refuses on any check failure and **can never be run again**:
1. `python scripts/30_confirm.py`: the primary (C1 vs C0 on RHUH-GBM) and H2a (E1 vs C0). Then immediately:
   ```bash
   git add results/confirm
   git commit -m "Confirmatory run on RHUH-GBM (30_)"
   git push
   ```
2. `python scripts/31_sensitivity.py`: the sensitivity arms. Then commit `results/confirm_sensitivity` and push.
3. `python scripts/32_mu_confirm.py`: H2b (E1 vs C0 on MU-Glioma-Post). Then commit `results/mu_confirm` and push.

Send the user the printed verdict lines and sentences exactly as the scripts print them. Do not reword, interpret or round them.

## What the user does meanwhile
Nothing is needed. She reviews each pushed result, and afterwards makes the figures, app cases and poster.
