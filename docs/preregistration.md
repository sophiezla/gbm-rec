# Pre-registration (OSF)

Not posted. This is the Phase 2 draft, built from the plan's checklist in Step 2.5. Every blank is filled only from `results/` or a committed decision. Text marked *plan* is copied from `docs/plan/final_plan.md`. Text marked *implementation* says how this repository carries it out, with its source.

- OSF URL: TODO
- OSF timestamp: TODO
- RHUH-GBM download date: TODO (must be later than the OSF timestamp)

## Checklist (plan Step 2.5) and where each item comes from

| Item | Source in this repository | State (2026-11-10) |
|---|---|---|
| Tournament leaderboard and the named challenger, with the commit hash | `results/tournament/leaderboard.json` + its `run.manifest.json` (`scripts/02_tournament.py`) | not run: waiting on C5 training |
| SHA-256 of every refit model | `results/release/model_hashes.json` (`scripts/03_refit_release.py`; C5 imported, D6 item 9) | not run |
| Feature definitions and the label mapping for each cohort | section 1 below | drafted |
| Cavity rule and label-verification rule | section 2 below | drafted; constants decided (D17) |
| Registration recipe, screen thresholds, exclusion rules | section 3 below | drafted; the round-trip value waits on `07_` |
| Power table | `results/power/power_table.json` (`scripts/04_power.py`) | not run: needs C0's LUMIERE scores from `02_` |
| Primary, verdict rule, secondaries, sensitivity and exploratory lists | section 4 below | drafted |
| Result sentences | section 5 below (plan Step 2.4) | drafted |
| What was seen before posting | section 7 below | TODO at posting |

## 1. Candidates and features (locked 2026-11-07, `DECISIONS.md` D6)

Eight candidates, in complexity order:
- **C0:** distance from the treatment target (−f1).
- **C0T:** distance through tissue, falx closed, at most 5 mm into the ventricles.
- **C4:** distance to the FLAIR region (−f3).
- **C1:** logistic regression on f1–f4.
- **C3:** logistic regression on f1–f5.
- **C2:** LightGBM on f1–f4.
- **C5:** a small 3D U-Net on T1c, FLAIR, target mask, FLAIR mask and the f1 map, trained from scratch.
- **C6:** the rank average of C1, C3, C2 and C5.

All are trained on LUMIERE (63 patients, 53 evaluable) on one shared 7-fold split. The master seed is 20261107. The full rules, including C5's fixed settings, are D6 items 1–8.

Features:
- f1: signed Euclidean distance to the target.
- f2: FLAIR region.
- f3: distance to the FLAIR region.
- f4: f1 × f2.
- f5: geodesic distance with the falx and ventricles closed, unreachable voxels capped at 300 mm.

Label mappings:
- **LUMIERE:** DeepBraTumIA labels. Target = labels 1 + 2. Recurrence = enhancing tumour at the first segmented progression (D5, D6 item 1).
- **RHUH-GBM** (*plan* Step 2.2, from the dataset paper): 1 = necrosis, 2 = peritumoral signal alteration, 3 = enhancing tumour. Recurrence = enhancing tumour (label 3) outside the target. The mapping is re-checked on download (section 2).

Barriers on RHUH-GBM (*plan* Step 4.1; implementation `gbmc.rhuh_features`, D13):
- **Falx:** derived as on LUMIERE and adapted from MNI to SRI24 by a translation to the SRI24 midline. `rhuh.sri24_midline_x_mm` and the T1 sequence used are recorded before scoring.
- **Ventricles:** rule V4 (D16), as on LUMIERE.
- **Intensity scale:** T1 and T1ce are z-scored inside the brain mask before any fixed intensity threshold is applied, as on LUMIERE.

**C5 on RHUH-GBM** (implementation, D6 item 9): the Mac packs label-free inputs (`24_`); the GPU runtime scores them only with the registered model (`14_`); the confirmatory run checks every hash before it reads a label.

*Plan* Step 3.6: RHUH-GBM's preprocessing (SynthStrip, SRI24, z-score) differs from LUMIERE's (N4, rigid to T1c, z-score in the brain mask). C0–C4 depend only on masks, but C5 reads intensities, so this is a known source of shift for C5.

## 2. Cavity rule and label-verification rule (*plan* Step 2.2)

- **Cavity.** *Plan:* "On download, inspect label values inside the cavity on 5 seeded patients. If the cavity has no label, the target = post-op enhancing residual + the preoperative tumour core (labels 1 + 3), carried rigidly into post-op space and clipped to the brain mask. Record this as a deviation from gbm-4d's target definition."
  - *Implementation* (`gbmc.rhuh.treatment_target`, `cavity_rule`; `20_` does the inspection): the 5 patients are drawn with `stable_seed("rhuh_cavity")`. The cavity is read as the fluid-dark part of the pre-op core, where T1ce, z-scored over non-zero voxels, is below `rhuh.cavity_dark_t1ce_z`. The rule recorded is either "unlabelled" or "labelled:<value>".
  - Threshold: **-1.0** (D17). It is the midpoint of LUMIERE ventricle-fluid and normal-tissue intensities (D17; `scripts/16_cavity_dark_evidence.py` → `results/cavity_dark_evidence/`).
- **Label mapping.** *Plan:* "Re-derive the mapping from intensities, as gbm-4d's 02_verify_labels.py did. This check may only correct a swapped mapping, never choose between alternatives."
  - *Implementation* (`gbmc.rhuh.derive_mapping`, `check_mapping`): on each scan carrying all three labels, the brightest mean T1ce is enhancing, the darkest of the rest is necrosis, and the remaining label is peritumoral. A pure relabelling is applied only when at least `rhuh.mapping_min_agreement` of scans agree; otherwise the check stops the pipeline.
  - Agreement required: **80%** of voting scans (D17).

## 3. Registration, screens and exclusions (*plan* Steps 3.3–3.5)

- **Recipe.** *Primary:* gbm-wm's tuned SyN, driven by T1ce with the cavity masked out of the metric (D12, `gbmc.registration`). *Sensitivity:* the original recipe that made the LUMIERE labels (rewritten in `gbmc`, deviation V3). The mismatch with the LUMIERE training labels is recorded as a deviation, and the sensitivity arm measures its effect.
- **Round-trip exclusion threshold.** Fixed against the tuned recipe's round-trip distribution on the 63 LUMIERE baseline→first-progression pairs (`07_`, registration only, no LUMIERE label read).
  - Rule: **the maximum** of LUMIERE's per-patient round-trip medians (D17). **Open before posting:** the value it gives (`07_ --summary`).
  - `07_` has not run: the progression scans are not on the analysis computer.
- **Screens** (*plan* Step 3.4). They can exclude a patient but cannot certify one:
  - round-trip error (peritumoral median and maximum), Jacobian statistics and the folded-voxel count;
  - brain-surface distance against the shipped brain masks, with the identity transform as a null;
  - multi-arbiter consensus (recipe re-driven by T2 and FLAIR, plus the shipped atlas alignment);
  - simulated deformation recovery on 10 seeded patients.
  - Thresholds: review above **3.0 mm**, or when **1.0 mm** worse than the identity transform (D17). These are gbm-wm's values, chosen post hoc there and fixed here before any RHUH-GBM image.
- **Exclusions** (*plan* Step 3.5), each logged with its reason:
  - a missing sequence or segmentation;
  - failed transport;
  - screen failure;
  - an empty target;
  - no enhancing recurrence outside the target. Such a patient is *undefined*, never scored 0.

  A CONSORT table runs from 40 enrolled down to the evaluable count, and the achieved MDE is recorded before any score (*plan* Step 3.6).

## 4. Primary, verdict, secondaries, sensitivity and exploratory analyses (*plan* Step 2.3)

- **Primary.** Challenger vs C0, Δ mean PR-AUC on evaluable RHUH-GBM patients, with a patient-level bootstrap 95% BCa CI. *Implementation:* 1000 resamples, seeded with `stable_seed`.
- **Champion verdict.** The challenger is champion if the CI's lower bound is above 0. Otherwise C0 is champion. <!-- audit: ignore (the plan's verdict rule itself, which defines the word) -->
- **Outcome classes** (*plan* Step 4.4):

  | Outcome | Condition |
  |---|---|
  | Challenger champion | CI lower bound > 0 |
  | Bounded null | CI contains 0; upper bound < MDE |
  | Inconclusive | CI contains 0; upper bound ≥ MDE |
  | Challenger worse | CI upper bound < 0 |

  The MDE used is the achieved MDE at an assumed arm correlation of **rho = 0.7** (D17). The achieved MDE at 0.5 and 0.9 is reported alongside it.
- **Hero secondary**, shown first on every deliverable: challenger vs C0 on coverage at matched volume, i.e. each patient's own 15 mm guideline-margin volume. Coverage is always stated with its denominator: recurrence outside the target.
- **Complexity trend:** every candidate vs C0, in complexity order, reported as one forest plot.
- **Other secondaries, under BH-FDR (q = 0.05):**
  - every other candidate vs C0;
  - challenger vs C0T;
  - challenger vs C0 in the escape region (> 20 mm from the target);
  - lift over chance;
  - coverage at the matched 20 mm volume and at 50 ml.
- **Where distance fails, descriptive only:** per patient, the share of recurrence beyond the 15 mm margin and beyond 20 mm, and the share of it each model catches. No test is run.
- **Sensitivity:**
  - the unscreened arm;
  - the original registration recipe;
  - the alternative cavity definition;
  - recurrence label = enhancing + necrosis;
  - the 11 patients scanned partly at a secondary facility removed.
- **Exploratory:**
  - C1 + baseline ADC, fitted leave-one-patient-out inside RHUH-GBM. *Not built yet:* it is first in the plan's cut order.
  - The LUMIERE later-progression analysis (Step 1.8), quoted as exploratory.

The confirmatory run happens once: an irreversible marker is written before the first RHUH-GBM label is read (*plan* Step 4.3, `gbmc.freeze`).

## 5. Result sentences (*plan* Step 2.4, verbatim, blanks unfilled)

- *Challenger wins:* "On an independent cohort, [challenger] adds [Δ] PR-AUC (95% CI [...]) over distance from the treatment target, and at the volume of the 15 mm guideline margin captures [x]% of recurrence against [y]% for distance."
- *Distance holds:* "On an independent cohort, the best challenger from an eight-model tournament adds less than [upper limit] PR-AUC over distance, and at the volume of the 15 mm guideline margin captures [x]% of recurrence against [y]% for distance; distance remains the best available predictor."
- *The two metrics disagree:* "The verdict follows the pre-registered primary, PR-AUC, which [does / does not] favour [challenger]. At the guideline margin's volume alone, coverage [does / does not] differ ([x]% vs [y]%, 95% CI [...]); the difference is reported as a secondary result."

"Eight-model" is correct: C5 passed the Step 0.4 gate (D6 (g)), so all eight candidates run.

## 6. Deviations from the plan, disclosed (`docs/deviations.md`)

- **V1:** LUMIERE is development data only. Only RHUH-GBM numbers are called validation.
- **V3:**
  - The gbm-4d code is rewritten in `gbmc`, because the pinned commit was never published.
  - A rigid resampling direction is corrected.
  - The LUMIERE baselines are rebuilt, while the original evaluation masks and labels are kept.
- **V4:** the ventricle segmentation is replaced for every patient (D16), before any score was seen.
- **C5 on Google Colab** (D6 (g), 2026-11-10): the settings, seeds and folds are unchanged, and each fold records its GPU.
- **V2:** no usability check with other people (plan Step 5.8 dropped).
- TODO: any later deviations, copied at posting.

## 7. What was seen before posting

TODO at posting:
- all LUMIERE results (the leaderboard, the power table, the round-trip distribution);
- both predecessor studies (gbm-4d, gbm-wm);
- the RHUH-GBM dataset paper;
- no RHUH-GBM image.

## Title bands (plan Step 6.1, written in week 2)

- *Challenger champion:* "Where does glioblastoma come back? [Champion] finds more of it than distance, on patients it has never seen."
- *Distance holds:* "Where does glioblastoma come back? Eight models, one sealed test: distance still wins."
