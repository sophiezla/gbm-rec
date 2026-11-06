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
