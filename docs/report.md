# Where does glioblastoma come back? Can MRI-based models beat distance?

Research visualization of a pre-registered method comparison on public data. Not a treatment prediction.

> Skeleton (plan Step 6.2). Sections follow the TRIPOD+AI reporting checklist's order. TODO: verify the section order and item numbers against the published checklist (Collins et al., BMJ 2024) before drafting. Every number is filled in from `results/` and checked with `scripts/41_audit_numbers.py docs/report.md`, which also enforces the language rules. Nothing below is a result.

## Title and abstract
- Title: the band chosen after the verdict (`docs/preregistration.md`, title bands).
- Abstract: question; cohorts (LUMIERE development, RHUH-GBM validation); candidates; primary endpoint and verdict rule; the filled result sentence (`results/confirm/sentence.txt`); limits.

## Introduction
- Background: the ESTRO-EANO 2023 margin and its FLAIR sentence, quoted in the guideline's own words; gbm-4d and gbm-wm (plan §2). The gbm-wm FLAIR finding is an association on 9-13 patients.
- Objective: the question, as `config.yaml:study.question`.

## Methods
### Data sources and participants
- LUMIERE: 63 locked patients (gbm-4d D16), development only (deviation V1). CONSORT-style numbers from gbm-4d's cohort lock.
- RHUH-GBM: enrolled, inventory and exclusions (`results/rhuh_inventory/`, `results/rhuh_lock/consort.md`).
- Download date after the OSF timestamp (`docs/preregistration.md`).

### Data preparation
- LUMIERE: gbm-4d preprocessing and label transport (reused frozen, D5).
- RHUH-GBM: shipped preprocessing (SynthStrip, SRI24, z-score); label mapping check; cavity rule (Step 2.2); transport with the tuned recipe (Step 3.3); screens (Step 3.4).

### Outcome
- Enhancing tumour at the first segmented progression, outside the treatment target; undefined patients reported, never scored 0 (DECISIONS.md D6 item 1).

### Predictors
- f1-f5 and the C0T distance (D6 item 2; `src/gbmc/features.py`). Native spacing excluded (D7).

### Sample size
- Power table and achieved MDE (`results/power/power_table.json`, `results/rhuh_lock/consort.json`).

### Missing data
- TODO: state how missing sequences and failed transports were handled (CONSORT steps).

### Analysis
- Candidates, folds, metrics, BCa bootstrap, selection rule (D6); refit and hashing (`results/release/model_hashes.json`).
- Primary, verdict rule, secondaries with BH-FDR, sensitivity arms (`scripts/30_confirm.py`, `scripts/31_sensitivity.py`).
- Class imbalance: negative subsampling in training only (D6 item 3); scoring on every evaluation voxel.

### Fairness
- TODO: what the cohorts allow to be said about subgroups (sex, age, scanner site; the secondary-facility covariate).

### Model output and training vs evaluation
- Voxel scores ranked within patient; PR-AUC, lift, matched-volume coverage (denominator: recurrence outside the target).

## Open science
- Pre-registration: OSF URL and timestamp.
- Code: repository and commit of every result (run manifests).
- Data: both cohorts public (LUMIERE CC0; RHUH-GBM CC BY 4.0).
- Funding and conflicts: TODO.

## Patient and public involvement
- Expert review (plan Steps 0.7, 6.6): `docs/expert_review.md`. TODO.

## Results
- Participants: CONSORT table.
- Development: tournament leaderboard (LUMIERE, development results).
- Validation: the primary, the verdict and the filled sentence; hero secondary; complexity forest plot; secondaries; sensitivity arms; where distance fails (descriptive).
- Exploratory: later progressions (LUMIERE only).
- Figures: `results/figures/`, `results/figures_reveal/`.

## Discussion
- Interpretation: evidence about where recurrence occurs; margin recommendations remain the guideline's.
- Limitations: registration screened, never measured against landmarks (no TRE); cohort size; label and preprocessing shift between cohorts; the recipe differs from the one that made the LUMIERE labels.
- Usability: the app's fixed label.
- Future work: plan §14.

## Appendix: PROBAST+AI self-audit
See `docs/probast_ai.md`.
