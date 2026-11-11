# Training C5 on the GPU computer

C5 is the only candidate trained away from the Mac (`DECISIONS.md` D6 items 8–9). The GPU computer only trains C5 and writes its maps, models and hashes. The Mac imports them and computes every reportable number.

## 0. Google Colab (the partner's route, from 2026-11-10)

The partner trains C5 on Colab with a notebook of their own. Instructions for building it: `docs/C5_COLAB_FOR_PARTNER.md`. Whatever it looks like, it must:
- run `scripts/13_c5_train.py` unchanged, from a clone at **one pinned commit** for all 7 folds. A re-clone after a disconnect must check out that same commit, not the newest `main`;
- leave the clone **clean**. Any file written inside it (a lockfile, a notebook, outputs) makes `git status` non-empty, and the Mac refuses the maps. `data/` is git-ignored, so `data/interim/c5_*` can be links to persistent storage;
- keep `data/interim/c5_oof/` on storage that outlives the runtime (Colab wipes its disk). Finished folds are skipped on a rerun; an interrupted fold restarts from scratch;
- install the pinned environment (sections 1.2–1.3; `uv pip install --torch-backend auto -r requirements-gpu.txt` picks the CUDA build) and save `requirements-gpu.lock` outside the clone.

Each fold records its GPU (`device_info`), because Colab can hand out a different GPU per session.

## 1. Set up the GPU computer (once)

You need an NVIDIA GPU with CUDA and at least 8 GB of memory (the plan's Step 0.4 gate).

1. Record the card and its memory, and put them in `config.yaml:c5.gpu` (commit on the Mac):
   ```bash
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
   ```
2. Clone the repo **outside any synced folder** (Google Drive, OneDrive), then:
   ```bash
   git clone https://github.com/sophiezla/gbm-rec.git
   cd gbm-rec
   uv venv .venv --python 3.13.14
   uv pip install --python .venv -r requirements.lock
   uv pip install --python .venv -r requirements-gpu.txt
   uv pip install --python .venv --no-deps -e .
   ```
   On Windows the interpreter is `.venv\Scripts\python`; on Linux, `.venv/bin/python`.
3. Pin what was actually installed and commit it **from the Mac** (copy the file over):
   ```bash
   uv pip freeze --python .venv > requirements-gpu.lock
   ```
4. Check that PyTorch sees the GPU:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## 2. Copy the inputs (each time they change)

On the Mac, `scripts/12_c5_pack.py` writes `data/interim/c5_inputs/` (63 files plus `manifest.json`). Copy the whole folder to the same path on the GPU computer. These are derived from LUMIERE (CC0). They are never committed.

## 3. Train

Both computers must be at the **same commit** with a clean tree (`git pull`, `git status`).

```bash
python scripts/13_c5_train.py --smoke --device cuda --patients Patient-001 Patient-002 Patient-003
python scripts/13_c5_train.py
```

- The smoke run uses tiny settings and writes to `data/interim/c5_smoke/`. Its maps are marked `smoke` and `02_` refuses them.
- The real run trains the 7 folds with the fixed settings (`config.yaml:c5`) and writes `data/interim/c5_oof/`. It resumes: a fold already finished at the same commit is skipped. The manifest is written once all 7 folds are done.
- Speed on the GPU is not known yet (TODO: record the first fold's time).

## 4. Bring the maps back

Copy `data/interim/c5_oof/` (maps, `models/`, `folds/`, `manifest.json`) to the same path on the Mac. On the Mac, `scripts/02_tournament.py` checks the folds, the clean-tree flag, the smoke flag and every hash before it scores anything.

Do not `git pull` while the folds are training. A fold finished at a different commit makes the manifest step refuse ("folds were trained at different commits"), and those folds then have to be retrained.

## 5. Release model (Phase 1, Step 1.7, after the tournament)

1. GPU computer: `python scripts/13_c5_train.py --release`. This trains one model on all 63 patients into `data/interim/c5_release/` (`model.pt` + `manifest.json`).
2. Copy `data/interim/c5_release/` to the same path on the Mac.
3. Mac: `python scripts/03_refit_release.py`. `gbmc.c5_import.import_release` refuses unless the release run:
   - was clean and not a smoke run;
   - used today's `config.yaml:c5`;
   - trained on exactly the locked cohort;
   - used the same `src/gbmc/c5.py` as the tournament maps;
   - matches its own hash.

   It then copies the model to `results/release/models/C5.pt` and registers its SHA-256 in `results/release/model_hashes.json`.
4. Commit the register on the Mac and push. It goes into the OSF pre-registration.

## 6. RHUH-GBM maps (Phase 3–4, after the cohort lock, before `30_`)

1. Mac: `python scripts/24_rhuh_c5_pack.py`. This writes `data/interim/rhuh/c5_inputs/<variant>/` for the primary and every sensitivity arm. No label is read: the recurrence file is never opened.
2. Copy `data/interim/rhuh/c5_inputs/` to the GPU computer, then `git pull` there, so the committed register is present.
3. GPU computer: `python scripts/14_c5_score_rhuh.py`. It loads `c5_release/model.pt` only if it matches the registered hash, checks every input, and writes `data/interim/rhuh/c5_maps/<variant>/`.
4. Copy `data/interim/rhuh/c5_maps/` back to the Mac. `30_` and `31_` check the model hash, the inputs and every map before they write the run-once marker.

## Rules that must not bend

- Nothing in `config.yaml:c5` changes after any C5 score is seen. A change is a deviation.
- No early stopping and no picking epochs: the last epoch is the model.
- Never rerun a fold to get a different result. A rerun at a new commit replaces the old fold, and the reason goes in `CHANGELOG.md`.
