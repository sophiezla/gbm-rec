# Note for the partner's AI assistant: build the C5 Colab setup

_Written 2026-11-10 at the user's request. Read `CLAUDE.md`, `HANDOFF.md`, `docs/C5_GPU.md` and `DECISIONS.md` D6 (items 8–9) first._

## Your task
Build a Google Colab notebook (or an equivalent setup) that trains candidate C5 with this repository's own script, `scripts/13_c5_train.py`. The partner keeps the notebook in their own Drive or Colab, **not in this repository**. Committing it would change the commit that the folds are pinned to, and writing it inside the clone makes the tree dirty.

The network, its settings and the training script are already written and locked (D6 item 8, `config.yaml:c5`, `src/gbmc/c5.py`). Do not change them, tune anything, or add early stopping. Any change after a C5 score is seen is a deviation.

## Inputs: use the shared Drive folder, not your own copy of LUMIERE
The user shares a Google Drive folder, `gbm-rec-c5/`, with `c5_inputs/` inside: 63 `.npz` files plus `manifest.json`, 434 MB. They were packed by `scripts/12_c5_pack.py` at commit `dabe702` from a clean tree. **Train on these files. Do not re-pack them from a local copy of the dataset.** Re-packing is not just slower; it would be wrong:
- **Labels you cannot rebuild.** Each input holds the evaluation mask and the recurrence labels taken from the Mac's feature cache. Those came from gbm-4d's first-progression label transport, which raw LUMIERE cannot reproduce: the transport outputs are lost (`HANDOFF.md`, deviation V3).
- **Masks that differ by computer.** Rebuilding the baselines on another computer changed 15 of 63 brain masks (deviation V3), so the voxels would no longer line up.
- **Voxel alignment is enforced.** C5 must score exactly the voxels every other candidate scores, in the same order. `13_` refuses inputs that do not match `manifest.json`, and the Mac refuses maps whose voxel counts or hashes do not match.

In Colab, make the Drive folder appear at `<clone>/data/interim/c5_inputs/`. A symlink is fine; `data/` is git-ignored.

## What the notebook must do (`docs/C5_GPU.md` section 0)
1. **GPU.** Use an NVIDIA GPU with at least 8 GB. Print `nvidia-smi`.
2. **Pinned commit.** Clone `https://github.com/sophiezla/gbm-rec`, check out one commit, and use that same commit for all 7 folds, including after disconnects. For example, save the commit hash on Drive at the first run and check it out on every later run. If folds come from different commits, the manifest step stops with "folds were trained at different commits".
3. **Environment.**
   - Create the venv with `uv venv --python 3.13.14`.
   - Install `requirements.lock`.
   - Install `requirements-gpu.txt` with `uv pip install --torch-backend auto`; this picks the CUDA build of torch 2.15.0.
   - Install the repo itself with `uv pip install --no-deps -e .`.
   - Save `uv pip freeze` as `requirements-gpu.lock` on Drive, **outside the clone**.
4. **Clean tree.** `git status --porcelain` must print nothing before training. The scripts check this, and so does the Mac. Any file written inside the clone (the notebook, logs, the lockfile) breaks it.
5. **Persistent output.** Colab wipes its disk, so make `data/interim/c5_smoke`, `c5_oof` and `c5_release` links to Drive folders.
6. **Smoke test.** Run `python scripts/13_c5_train.py --smoke --device cuda --patients Patient-001 Patient-002 Patient-003`. It only checks that the code runs on this GPU; its output is never used.
7. **Training.** Run `python scripts/13_c5_train.py --device cuda`. After a disconnect, run it again: finished folds are skipped, and an interrupted fold restarts from scratch, which gives the same result. `--folds 0` trains one fold at a time. Each fold records its own GPU and time in `c5_oof/folds/fold<k>.json`.
8. **Hand-back.** When `c5_oof/manifest.json` exists, give the user the whole `c5_oof/` folder (maps, `models/`, `folds/`, `manifest.json`) and `requirements-gpu.lock`.

## Do not
- Do not commit data, maps, models or the notebook to this public repository (plan §13: data never goes into git).
- Do not run `--release` or `14_c5_score_rhuh.py` until the user says so. Those come after the tournament has run on the Mac.
- Do not compute any metric. The Mac computes every reportable number (D6 item 9).

## Unknowns
- Time per fold on a Colab GPU is not measured. One full-size training step took 10.6 s on the Mac's CPU. The GPU memory need is a rough estimate of about 3 GB, not measured.
- Nothing here has run on Colab yet. The smoke test will show whether deterministic mode works on that GPU.
