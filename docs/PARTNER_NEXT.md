# For the partner's AI: the last GPU job (C5 on RHUH-GBM)

_Written 2026-11-21. Read `CLAUDE.md`, then this file. This is the only remaining step on the partner's side._

## Where the project stands
- **Registered.** The pre-registration is public on OSF (https://osf.io/2yuqn/, registered 2026-11-20T16:42:57Z). Nothing in it may change.
- **Frozen models.** The tournament has run, and the challenger is C1. All models are frozen, and their SHA-256 values are in `results/release/model_hashes.json`. C5's release model is the `c5_release/model.pt` you trained (`285cd285…`).
- **On the Mac:** RHUH-GBM and MU-Glioma-Post are downloaded and processed. `scripts/run_test_cohorts.py` ran the label transport, the screens and the cohort locks, then packed C5's RHUH-GBM inputs (no labels inside) into the shared Drive folder.
- **Still missing:** the primary confirmatory run (`30_`) needs C5's scores on RHUH-GBM, and only a GPU runtime should make them, as D6 item 9 requires.

## Your task
Run `scripts/14_c5_score_rhuh.py` once on a Colab **NVIDIA L4**, the GPU type every C5 fold and the release model used (deviation V5), and put the maps back on Drive. It computes no metric and reads no label.

1. Clone or pull `https://github.com/sophiezla/gbm-rec`. Check out the commit the user names; it is the latest `main` after "Lock the RHUH-GBM cohort (23_)". Pin it for the whole run.
2. Set up the environment as before (`docs/C5_GPU.md` section 0, and `docs/C5_COLAB_FOR_PARTNER.md`). Use `requirements-gpu.lock`, which pins torch 2.15.0+cu132. `git status --porcelain` must print nothing.
3. Put the inputs in place without writing inside the clone. Link or copy these into the clone's ignored `data/` folder:
   - Drive `gbm-rec-c5/rhuh/c5_inputs/`, holding `tuned/`, `original/` and `alt_cavity/`, each with a `manifest.json`, as `data/interim/rhuh/c5_inputs/`;
   - Drive `gbm-rec-c5/c5_release/model.pt` and its `manifest.json`, as `data/interim/c5_release/`;
   - Drive `gbm-rec-c5/rhuh/c5_maps/`, which you create, as `data/interim/rhuh/c5_maps/`, so the output lands on Drive.
4. Run:
   ```bash
   python scripts/14_c5_score_rhuh.py --device cuda
   ```
   - The script refuses unless `model.pt` matches the registered hash (`285cd2857d2e…`). It also refuses any input whose hash differs from its manifest.
   - It writes one `.npz` per patient and a `manifest.json` per variant. That's roughly 120 patients over the three variants, all inference, so it should take well under an hour on an L4 (not measured).
5. Check that `data/interim/rhuh/c5_maps/{tuned,original,alt_cavity}/manifest.json` all exist, then tell the user it is done. Nothing else is needed: no commit and no push from Colab.

## Rules
- **Do not change** any code, setting or file in the repository, and do not retrain anything. If a check refuses, stop and send the user the exact error; do not work around it.
- **Do not open, download or look at** RHUH-GBM or MU-Glioma-Post images or labels outside this script. The inputs contain no labels, and the script reads none.
- **Do not run** `30_`, `31_` or `32_`. They are one-time confirmatory runs, done on the Mac only, on the user's go.
- If Colab gives a GPU other than an L4, stop and tell the user before running. Using another GPU would need its own logged deviation.

## What happens after (Mac, user side, not yours)
- The C5 maps get copied to the Mac, and `gbmc.c5_import.rhuh_scores` checks every hash.
- Then, on the user's go: `30_` (primary, plus H2a on RHUH-GBM), `31_` (sensitivity) and `32_` (H2b on MU-Glioma-Post), each run once.
