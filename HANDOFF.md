# Project Handoff

_Last updated: 2026-09-23_

## Current State
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

## Next Recommended Step
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
