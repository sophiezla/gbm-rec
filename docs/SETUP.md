# Setup

Clone outside Google Drive or any other sync folder; syncing can corrupt `.git`.

## Install (Mac or Linux)
```bash
git clone https://github.com/sophiezla/gbm-rec.git
cd gbm-rec
uv venv --python 3.13.14 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest
```
On a Mac, LightGBM (candidate C2) also needs `brew install libomp`. Without it, two tests fail: `test_score_patient_every_candidate` and `test_learned_candidates_fit_and_predict[C2]`.

On Windows, use `.venv\\Scripts\\python` in place of `.venv/bin/python`.

## The app
```bash
cd app
npm install
npm run dev
```

## Data (never in git)
| Variable or folder | Holds | Needed for |
|---|---|---|
| `data/interim/features/` | Corrected feature caches for all 63 LUMIERE patients | The tournament |
| `$GBMC_LUMIERE_ROOT` | Raw LUMIERE archive (`Imaging/`, `metadata/`; CC0) | `01_`, `05_`, `07_`, `40_`, `42_` |
| `$GBMC_DERIVED_ROOT` | `data/interim/preprocessed/` and `data/interim/label_field/` | `01_`, `05_`, `06_`, `07_`, `40_`, `42_` |

## libomp on this Mac (2026-11-15)

Homebrew is not installed, and installing it needs the administrator password, so `brew install libomp` was not run. Instead, LightGBM finds `libomp.dylib` in the uv Python's own `lib/` folder, where it is a symlink:

```
~/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/lib/libomp.dylib
  -> ~/Projects/gbm-rec/.venv-c5/lib/python3.13/site-packages/torch/lib/libomp.dylib
```

That file is the LLVM OpenMP runtime shipped in the official torch 2.15.0 wheel (SHA-256 `6256bee09e93c28d71c65711cc69224d69994c6965648b628b70a22772fe98d4`).
- It is a symlink, not a copy, because two OpenMP runtimes in one process (torch's and LightGBM's) segfault in `.venv-c5`.
- Checks: `.venv` 152 passed and 5 skipped (the torch tests); `.venv-c5` 157 passed.
- **Do not delete `.venv-c5`**, or LightGBM stops loading.
- To switch to Homebrew later, install it (from brew.sh), run `brew install libomp`, then delete the symlink.

## LUMIERE metadata

`$GBMC_LUMIERE_ROOT/metadata/` holds `LUMIERE-ExpertRating-v202211.csv` and `LUMIERE-datacompleteness.csv` from LUMIERE's figshare collection 5904905. Downloads: https://ndownloader.figshare.com/files/38369741 (MD5 `3f196e118aa5390952ff8397aa3d25ed`) and https://ndownloader.figshare.com/files/37575523 (MD5 `3be2baf9b7fe060cca03b7965ae5513f`). The scans come from `Imaging-v202211.zip` (https://ndownloader.figshare.com/files/38249697, MD5 `b2708d297badd245c8c8cbbb91fdda40`, 32.6 GB).
