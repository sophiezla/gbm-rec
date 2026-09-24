# Where Does Glioblastoma Come Back? Can MRI-based models beat distance?

> **Status (2026-09-23): Phase 0, week 1.** The repository, environment and reproducibility helpers are set up. There is no data and there are no results yet. The plan is [`docs/plan/final_plan.md`](docs/plan/final_plan.md).
>
> *Research visualization of a pre-registered method comparison on public data. Not a treatment prediction.*

## Description
The 2023 ESTRO-EANO radiotherapy guideline for glioblastoma recommends a uniform 15 mm margin around the tumour, trimmed only at anatomical barriers. It also says FLAIR abnormality "may represent non-enhancing tumour and should be considered for inclusion". This project asks whether any of eight models, ranging from a one-line rule to a 3D convolutional network, predicts where glioblastoma recurs better than distance from the tumour. The models are chosen in a pre-registered tournament on LUMIERE (63 patients). The winner is then tested once against distance on RHUH-GBM (40 patients no model has seen).

## Purpose / Motivation
If recurrence follows the baseline FLAIR abnormality, a margin shaped by that abnormality would cover more recurrence at the same treated volume. If it does not, the uniform margin rests on better evidence. The project builds on two finished, frozen studies, gbm-4d and gbm-wm (plan §2). In gbm-4d, distance from the target was the model to beat.

## Key Components (planned; none built yet)
- **Unit of prediction:** every 1 mm voxel of brain outside the treatment target gets a recurrence score.
- **Primary metric:** per-patient PR-AUC, averaged across patients. **Headline metric:** coverage at matched volume, meaning the share of recurrence caught when a model treats exactly the volume of that patient's own 15 mm guideline margin.
- **Eight candidates**, from simplest to most complex: C0 distance, C0T trimmed distance, C4 FLAIR edge, C1 logistic regression, C3 logistic regression plus anatomy, C2 LightGBM, C5 3D CNN (only if its hardware gate passes), C6 rank-average ensemble.
- **Champion rule:** the challenger becomes champion only if its ΔPR-AUC over C0 on RHUH-GBM has a 95% BCa CI lower bound above 0.
- **3D/4D explainer app:** a static website (Vite + TypeScript, three.js, NiiVue) with offline and phone builds.
- **Fair deliverables:** poster, a TRIPOD+AI report, 30 s / 3 min / 10 min pitches, and a booth kit.

## Project Structure
```text
gbm-rec/
├── README.md            this file
├── PROJECT_STATUS.md    current phase, gates, blockers
├── HANDOFF.md           session-to-session continuity
├── DECISIONS.md         decision register (D-numbers)
├── CHANGELOG.md
├── config.yaml          every study constant; undecided ones are null and raise when read
├── pyproject.toml       package gbmc; pins gbm4d at eeba777
├── requirements.lock    exact package versions
├── src/gbmc/            library: rng, config, provenance (features, candidates, scoring, export to come)
├── scripts/             numbered stages 01_ ... 40_ (none yet)
├── tests/               pytest suite
├── results/<stage>/     JSON results, each with run.manifest.json (none yet)
├── app/                 the explainer app (Phase 5; empty)
└── docs/
    ├── plan/final_plan.md   the plan
    ├── protocol.md          operational details, filled in per phase
    ├── deviations.md        V-numbered departures from the plan
    └── preregistration.md   OSF record (Phase 2)
```

## Installation / Setup
Windows, Python **3.13.14** (the interpreter gbm-4d used).
```bash
python -m venv .venv                       # must report Python 3.13.14
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install --no-deps "gbm4d @ git+https://github.com/sophiezla/gbm-4d.git@eeba7779804412b0a32d0b5ba0f23c3156448549"
.venv/Scripts/python -m pip install --no-deps -e .
```
**Known issue:** commit `eeba777` is not yet on GitHub (local gbm-4d is 48 commits ahead of `origin/main`), so the second command fails until gbm-4d is pushed. Until then, install from a sibling checkout: `git+file:///C:/Documents/Projects/gbm-4d@eeba7779804412b0a32d0b5ba0f23c3156448549`.

## Dependencies
`requirements.lock` is gbm-4d's lockfile unchanged, plus scikit-image 0.26.0 and trimesh 5.1.0 (plan Step 0.3) and their dependencies (ImageIO, lazy-loader, networkx, tifffile). PyTorch will be added only if the CNN gate passes (Step 0.4).

## How to Run / Example Usage / Inputs & Outputs
TODO: no analysis scripts yet.

## Testing
```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests
```
The tests check the following:
- seeds are stable across processes and identical to gbm-4d's
- null constants raise when read
- ITK is pinned to one thread
- manifests record git state and the gbm4d commit
- `require_clean_tree` refuses a dirty tree
- the installed packages match the lockfile
- data and images are git-ignored
- the RHUH-GBM path stays unset before pre-registration

## Data
| Cohort | Role | Licence | Status |
|---|---|---|---|
| LUMIERE, 63 locked patients (via gbm-4d) | Development only (deviation V1) | CC0 | Not yet wired in |
| RHUH-GBM, 40 patients, open NIfTI + clinical CSV only | Confirmatory, analysed **once** | CC BY 4.0 | Not downloaded; download only after the OSF pre-registration |

## Limitations (known in advance, plan §10 and §12)
The registration gate cannot be fully discharged. The cohort is small, with an expected evaluable n of about 20–40. Labels and preprocessing shift between the two cohorts. The most likely outcome is that distance remains champion; the plan treats that as a complete result.

## Reproducibility
The rules carry over from gbm-4d and gbm-wm (plan §13):
- BLAKE2b `stable_seed`
- a run manifest for every result
- `require_clean_tree()` before any reportable number is written
- null constants that raise when read
- D- and V-numbered registers
- a one-shot confirmatory run guarded by an irreversible marker
- a model hash check before scoring
- a number-audit script
- no data in git

## Current Status
Phase 0 (setup and paperwork). See `PROJECT_STATUS.md`.
