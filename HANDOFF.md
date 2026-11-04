# Project Handoff

_Last updated: 2026-09-24_

## Current State (Phase 1, uncommitted)
All of this work was committed and pushed to the public `sophiezla/gbm-rec` on 2026-11-03, so both partners work from that repo; clone it outside Google Drive. No dates were filled in. **No candidate has been scored, and no RHUH-GBM file exists.**
- **Team (2026-10-29).** The project is now a LACSEF team project. The team build guide is `docs/plan/project_plan_lacsef.md` (also a Google Doc: https://docs.google.com/document/d/1Dqy4ejZ648nCc8FEF9tCRtQ8wHXmTVFzzuxGZhdR2q0/edit). It adds no methodology; `final_plan.docx` still governs. The LACSEF SRC questions (team forms, continuation with a new partner, who may test the app under V2) were resolved by the user on 2026-10-30; the outcomes are not recorded here.
- **Rules.** The tournament rules are drafted as `DECISIONS.md` D6, with open points (a)–(j); D9–D15 are proposed. Every script that produces a reportable number refuses a dirty tree, and the ones that need it refuse a null `seeds.master`.
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
