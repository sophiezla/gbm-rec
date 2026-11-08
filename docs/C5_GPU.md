# Training C5 on the GPU computer

C5 is the only candidate trained away from the Mac (`DECISIONS.md` D6 items 8–9). The GPU computer only trains C5 and writes its maps, models and hashes. The Mac imports them and computes every reportable number.

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

## Phase 2 (after the tournament)

`python scripts/13_c5_train.py --release` trains one model on all 63 patients into `data/interim/c5_release/`. Importing it into `03_`, and scoring RHUH-GBM with it for `30_`, are not built yet.

## Rules that must not bend

- Nothing in `config.yaml:c5` changes after any C5 score is seen. A change is a deviation.
- No early stopping and no picking epochs: the last epoch is the model.
- Never rerun a fold to get a different result. A rerun at a new commit replaces the old fold, and the reason goes in `CHANGELOG.md`.
