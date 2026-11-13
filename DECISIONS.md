# Decision Log (D-numbers)

This is the plan's `decisions.md` register (plan §13). Tournament rules (plan Step 1.1) are written here **before the first candidate is scored**. Departures from the plan are logged in `docs/deviations.md` (V-numbers, with the direction of effect).

## D1: The final plan is the authoritative project plan

Date: 2026-09-23
Status: Active

Decision:
"Where Does Glioblastoma Come Back? Final plan", dated 2026-09-22, is the plan to follow. The original .docx (SHA-256 `71295409a58878a338c6f407d9da315cb67e23150406f93ad7a08eed7ab05c6d`) is kept locally and not committed. `docs/plan/final_plan.md` is its text, converted with pandoc. Links to unpublished write-ups are reduced to their titles; nothing else is edited.

Reason:
The project follows one fixed plan written before any new data are touched.

Alternatives Considered:
None.

Consequences:
Every departure from the plan is logged as a deviation with its direction of effect. Changes to the question, cohorts, candidates, metrics, selection rule, verdict rule or timeline need explicit approval.

Do Not Revisit Unless:
A revised plan is issued.

## D2: Repository name and documentation layout

Date: 2026-09-23
Status: Active

Decision:
- The repository is `sophiezla/gbm-rec`. This name replaces the plan's working name `gbm-champion` (Step 0.2). The library keeps the plan's name, `gbmc`.
- The decision register is this root-level `DECISIONS.md`, not `docs/decisions.md`. `docs/protocol.md`, `docs/deviations.md` and `docs/preregistration.md` live in `docs/` as the plan specifies.
- `HANDOFF.md` is maintained by hand for now. The plan's pre-commit hook that regenerates it is deferred until there is state worth generating.

Reason:
The status documents (README, PROJECT_STATUS, HANDOFF, DECISIONS, CHANGELOG) sit together at the root, and keeping one register avoids two competing logs.

Alternatives Considered:
Moving the register to `docs/decisions.md` to match the plan literally.

Consequences:
Wherever the plan says `decisions.md`, read `DECISIONS.md`.

Do Not Revisit Unless:
The plan's literal layout is preferred.

## D3: gbmc keeps its own config and provenance; seeding is imported from gbm4d

Date: 2026-09-23
Status: Active

Decision:
`gbmc.rng` re-exports `gbm4d.rng` unchanged. `gbmc.config` and `gbmc.provenance` are adapted copies of gbm4d's modules. They differ in two ways: they resolve the repository root to *this* repository, and every manifest also records the installed gbm4d commit.

Reason:
gbm4d's `config` and `provenance` find the repo root from their own file location. Once gbm4d is installed as a package, that location is site-packages, so they would read the wrong `config.yaml` and record the wrong git state. Seeding has no such dependence, and it must match gbm-4d bit for bit.

Alternatives Considered:
Importing all three modules from gbm4d. Rejected because the manifests would be wrong.

Consequences:
`tests/test_rng.py` asserts that `gbmc.rng.stable_seed` *is* gbm4d's function. `tests/test_provenance.py` covers the manifest and the clean-tree guard.

Do Not Revisit Unless:
gbm4d gains a way to set its repo root.

## D4: Python 3.13.14 and a lockfile extended from gbm-4d's

Date: 2026-09-23
Status: Active

Decision:
The environment uses the same Python 3.13.14 interpreter as gbm-4d. `requirements.lock` is gbm-4d's lockfile with no version changed, plus scikit-image 0.26.0, trimesh 5.1.0 and their dependencies (ImageIO 2.37.4, lazy-loader 0.6, networkx 3.7, tifffile 2026.9.20). gbm4d itself is pinned by full commit hash in `pyproject.toml` and installed with `--no-deps`.

Reason:
Plan Step 0.3. Adding the two packages did not change any version from gbm-4d's lock (checked by diffing the two lockfiles).

Alternatives Considered:
The other Python install on this machine (3.13.5). Rejected because it does not match gbm-4d.

Consequences:
`tests/test_environment.py` fails if the Python version, the gbm4d commit or any locked version drifts.

Do Not Revisit Unless:
PyTorch is added (CNN gate, Step 0.4) and needs a different version of something.

## D5: LUMIERE inputs come from gbm-4d's frozen derived data

Date: (set at commit)
Status: Active

Decision:
`gbmc.lumiere` reads the 63 locked patients from gbm-4d's `results/cohort_lock/cohort_lock.json` (train and test together, V1), the baselines from gbm-4d `data/interim/preprocessed/`, the recurrence labels from `data/interim/label_field/` (Enhancing_Core only, gbm-4d D18), and the native DeepBraTumIA segmentation and atlas transform from the shared LUMIERE copy. Both roots are set in `config.yaml:data.lumiere` and can be overridden with `$GBMC_LUMIERE_ROOT` / `$GBMC_GBM4D_ROOT`. Nothing is re-fitted or re-transported.

Reason:
Plan Step 1.2: "in gbm-4d's baseline space, with gbm-4d's transported labels".

Consequences:
Checked on Patient-001: the evaluation mask and the 3,303 positive voxels are identical to gbm-4d's cached features.

## D6: Tournament rules (plan Step 1.1). LOCKED

Date: 2026-11-07
Status: **Locked** (user approved every open point on 2026-11-07). Committed with `seeds.master` = 20261107 before any candidate was scored. Any change from here on is a deviation, logged with its direction of effect. `scripts/02_tournament.py` refuses a dirty tree, a null seed, or missing C5 maps.

Decision:
1. **Cohort and endpoint.** 63 locked LUMIERE patients; label = enhancing tumour at the first segmented RANO progression; evaluation region = brain minus target (DeepBraTumIA labels 1 + 2). A patient with no positive voxel there is *undefined* for every metric and is left out of every mean and Δ, never scored 0. Undefined patients still give training negatives (20,000 random voxels, gbm-4d arm2).
2. **Features** (`gbmc.features`), baseline only:
   - f1: signed Euclidean distance to the target surface.
   - f2: DeepBraTumIA label 3 (Edema_Compartment).
   - f3: signed Euclidean distance to the f2 surface, capped at 300 mm when f2 is empty.
   - f4: f1 × f2.
   - f5: fast-marching geodesic distance from the target inside the brain, with the falx (`gbmc.anatomy.falx_barrier`, rewritten from gbm-4d, deviation V3) and the ventricles (rule V4, D16; lesion excluded) closed.
   - Unreachable voxels are capped at 300 mm.
3. **Candidates** (`gbmc.candidates`), in complexity order C0, C0T, C4, C1, C3, C2, C5, C6:
   - C0 = −f1.
   - C0T = −(geodesic distance with the falx closed, entering the ventricles at most 5 mm); a ventricle voxel inherits its nearest wall voxel's distance plus its depth below the wall.
   - C4 = −f3.
   - C1 = L2 logistic regression (gbm-4d's standardise + C = 1 pipeline) on f1–f4, with patient-balanced sample weights.
   - C3 = the same on f1–f5, with the same weights.
   - C2 = LightGBM with gbm-4d's frozen `LIGHTGBM_PARAMS` on f1–f4, unweighted.
   - C5 = small 3D U-Net on images, trained from scratch on the GPU computer; settings in item 8.
   - C6 = unweighted mean of within-patient percentile ranks of C1, C3, C2 and C5, from their held-out predictions.
   - Training rows for C1, C3 and C2: every positive plus 50 negatives per positive, seeded per patient and the same in every fold (gbm-4d arm2).
4. **Folds.** One 7-fold split (9 patients each) drawn from the sorted IDs with `stable_seed("tournament_folds")`, shared by every candidate.
5. **Metrics per held-out patient.**
   - PR-AUC (average precision, gbm-4d's function).
   - Lift = PR-AUC ÷ prevalence.
   - Coverage (denominator: recurrence outside the target) at matched volume. The matched volume is the number of evaluation voxels inside the patient's own C0T margin at 15 mm (headline) and at 20 mm, plus 50 ml. At the cut, tied scores share the remaining treated volume in proportion, so coverage is the expected value under a random tie-break.
6. **Aggregation.** Mean over evaluable patients. Δ vs C0 with a patient-level BCa bootstrap 95% CI, 1000 resamples, seeded per candidate and metric.
7. **Selection rule.** From C4, C1, C3, C2, C5, C6, take the highest mean held-out PR-AUC. If any simpler candidate is within 0.005 of it, the simplest such one is the challenger. The challenger goes to Phase 4 whatever its Δ vs C0.
8. **C5** (plan Step 1.4 C5 rules; all fixed before training, never tuned; `config.yaml:c5`):
   - Inputs: baseline T1c and FLAIR (z-scored), target mask, FLAIR mask and the f1 map. No pretrained weights: none were verified in week 1, so C5 trains from scratch as the plan directs.
   - Network: small 3D U-Net, 4 levels, 16 base channels.
   - Training: 96³ patches, batch 2, with half the patches centred on recurrence voxels.
   - Loss: weighted binary cross-entropy. Positives are weighted by the training fold's negative-to-positive ratio, capped at 100.
   - Optimiser: Adam at a learning rate of 1e-3, decaying to 0.
   - Length: exactly 200 epochs × 100 steps. The last epoch is kept; there is no early stopping.
   - Augmentation: left–right flips and rotations up to 10° only.
   - Seeds: one per fold, `stable_seed("C5", fold)`, with PyTorch's deterministic flags on.
   - Prediction: sliding window with 50% overlap. A score is kept for each evaluation voxel, in the feature cache's order.
   - Folds: the same 7 as every other candidate.
   - Implementation details the approved settings left open, fixed in `gbmc.c5` on 2026-11-07 before any C5 training or score:
     - Downsampling uses stride-2 convolutions, not max-pooling, because CUDA max-pool has no deterministic backward.
     - The learning rate decays linearly to 0.
     - The f1 channel is divided by 100 mm, and every input is 0 outside the brain.
     - The loss counts only evaluation voxels (brain minus target).
     - "Half the patches" alternates exactly: positive, then random.
     - The overlapping sliding windows are averaged with equal weight.
9. **Machines** (user decision, 2026-11-07):
   - The GPU computer only trains C5 and writes its held-out maps, model files and their SHA-256 hashes. It runs from a clean, recorded commit with the same folds.
   - This Mac imports those files with hash checks (`gbmc.tournament.c5_held_out`) and computes every reportable number, C6 and the confirmatory run included.
   - From 2026-11-10 the "GPU computer" is a Google Colab runtime run by the user's partner with their own notebook (runbook section 0). The same rules apply: a pinned clean commit, the same folds, and all output on Drive.
   - PyTorch is approved for C5. It is pinned in a separate GPU lockfile; the main environment is unchanged.
   - Built 2026-11-07: `scripts/12_c5_pack.py` (Mac), `scripts/13_c5_train.py` (GPU computer) and `docs/C5_GPU.md`.
   - Built 2026-11-08 (`gbmc.c5_import`, runbook sections 5–6):
     - **Release model.** `03_` imports the GPU computer's release model. It checks the clean tree, the smoke flag, the settings, the cohort, the SHA-256, and that `src/gbmc/c5.py` is the same as for the tournament maps. The model is copied to `results/release/models/C5.pt` and registered.
     - **RHUH-GBM maps.** The Mac packs label-free inputs (`24_rhuh_c5_pack.py`). The GPU computer scores them only with the model whose hash is in the committed register (`14_c5_score_rhuh.py`). `30_` and `31_` import the maps after checking the model hash, the inputs and every map, before the run-once marker.
     - The GPU computer therefore also runs C5's network on RHUH-GBM. It reads no RHUH-GBM label and computes no metric.

Open points: the user decided all of them on 2026-11-07. (a) is 20261107; (b), (c), (d), (e), (f), (h) and (i) are approved as written below; (g) is approved with C5 in (item 8).
- (a) `seeds.master` value. **Decided: 20261107** (the date the rules were set).
- (b) f1 follows the plan's text (Euclidean). gbm-4d's Arm 0 used a brain-constrained geodesic, so C0 will not reproduce gbm-4d's 0.0743 exactly. On Patient-001 the two correlate at 0.9999, with a median difference of 0.95 mm.
- (c) The C0T ventricle rule (item 3) is one reading of "cut to 5 mm at the ventricles".
- (d) f5 closes the ventricles fully. The plan's f5 names the falx, ventricles and skull, and the plan does not give the 5 mm allowance to f5.
- (e) Rule 7 applies the 0.005 margin to every pair. The plan spells it out only for C6, but also says "a simple model wins ties".
- (f) C3 uses patient-balanced weights like C1. The plan states the weights only for C1; using them for C3 too makes C3 vs C1 isolate f5.
- (g) C5 is out until Step 0.4 is recorded (GPU, starting weights). **Decided 2026-11-07: the gate passes.** The user reports an NVIDIA RTX GPU on another computer that meets the gate (CUDA, at least 8 GB). The exact model and memory are not recorded yet (TODO: verify with `nvidia-smi` before C5 training). Starting weights: none were verified in week 1, so C5 trains from scratch (plan Step 0.4). Settings are in item 8; the machine split is in item 9. **Update 2026-11-10 (user):** the user's partner trains C5 on Google Colab, not on the RTX computer. Every Colab GPU meets the gate (CUDA, at least 8 GB), but the model can differ from session to session, so each fold records its GPU (`device_info`) and the manifest lists them (`fold_devices`). The settings, seeds and folds are unchanged. A different GPU can change the last digits of the weights; the run stays deterministic for a given GPU. The direction of effect on any score is unknown, and none has been seen. **Update 2026-11-13 (user):** the user trains C5 on their own Colab. Fold 0 and fold 1 epochs 1–80 ran on a T4; from fold 1 epoch 81 the run continues on an NVIDIA L4, whose convolutions use TF32 by default (deviation V5).
- (h) Power (Step 2.1): a simulated study "detects" when the BCa lower bound is above 0, which is how the verdict rule reads the CI. gbm-4d used a two-sided percentile test. The assumed arm correlation rho is reported at 0.5, 0.7 and 0.9, with 0.5 as the pessimistic headline (`config.yaml:power`).
- (j) **Resolved 2026-11-07 by D16:** the V4 rule finds Patient-074's ventricles (10 ml), so all 63 have both barriers. Original note: gbm-4d's ventricle segmentation found no central component ("segmentation not trusted"), so this patient's f5 and C0T had the falx barrier but no ventricle barrier.
- (k) **Resolved 2026-11-07 by D16 (user: "Fix for all patients"; deviation V4).** Ventricle masks (found 2026-11-06, no labels read, `results/f5_cap_survey/`). The segmenter applies Otsu to T1c in the deep brain. Confirmed: Patient-002's mask is 771 ml (52.8% of the brain) with threshold 1.08, against a cohort median of 49 ml and -0.18, and it passes the QC because the QC has no upper limit. In 27 of 62 masks under half the voxels pass a rough CSF screen (T2 z > 1 and FLAIR z < 0; not validated), and 12 are under a quarter. Overlays of the 9 masks checked by eye show several spreading into deep tissue. Lower CSF share goes with a higher threshold (r = -0.87), which suggests Otsu is sometimes splitting grey from white matter rather than CSF from tissue (hypothesis, not proven). The masks feed f5, C0T and so the matched volume: C0T's 15 mm margin holds a median 6.4% fewer voxels than the Euclidean shell, up to 18.5%, though falx trimming is part of that. Direction of effect on any score: unknown, nothing scored. The rewrite matched the original code, so this is inherited, and plan Step 4.1 says to derive barriers "the same way gbm-4d did", so any change is a deviation. Options: fix the segmenter for all patients before scoring; fix or exclude only Patient-002; or keep and report. Any fix must also work on RHUH-GBM (its sequences: TODO verify from the plan, no download).
- (i) Step 1.8 label: the *new* enhancing tumour at progressions 2 and 3 (enhancing then, not enhancing at the first progression). It is scored in the same evaluation region with each candidate's held-out map from `02_`. Only transports that pass gbm-4d's 2 mm round-trip screen are used. It is labelled exploratory.

Reason:
Plan Step 1.1: rules on record before the first score. Any change after a score is seen is a deviation.

## D7: Native voxel spacing is not a feature

Date: (set at commit)
Status: Active (plan Step 1.3 asks for this to be recorded)

Decision:
gbm-4d's spacing covariate is left out of every candidate.

Reason:
RHUH-GBM's open files are resampled to 1 mm, and their native spacing is only in the controlled DICOM.

## D8: No weighted stacking

Date: (set at commit)
Status: Active (plan Step 1.4 asks for this to be recorded)

Decision:
C6 is an unweighted rank average; no second-level model is fitted.

Reason:
Learning member weights would need nested cross-validation, and with about 54 training patients per fold the weights would be noisy.

## D9: Later progressions are transported with the rewritten original recipe (proposed)

Date: (set at commit; reworded 2026-11-08 for deviation V3)
Status: Proposed

Decision:
- `scripts/05_later_progressions.py` selects the second and third segmented PD timepoints after the first progression (`gbmc.progressions`, the gbm-4d rules: rated PD, with a DeepBraTumIA segmentation).
- Each later timepoint is preprocessed with `gbmc.imaging.preprocess_timepoint`, the same function as the baselines (`00_`). It is then transported with `gbmc.registration.transport_pair` using the original recipe (`recipe={}`).
- That code is the `gbmc` rewrite of gbm-4d's `03_` and `04_` (deviation V3). It replaces the earlier plan to load the frozen gbm-4d scripts from a pinned checkout, which no longer exists.
- The seed is gbm-4d's master seed, 20260819 (`config.yaml:later_progression.transport_master_seed`), with `stable_seed(patient, timepoint, step)`.
- Each patient's first-progression timepoint comes from the committed `results/cohort_lock/first_progressions.json` (`15_first_progressions.py`, 2026-11-08), because the gbm-4d transport records it was read from are not on this computer. 50 of 63 were copied from the gbm-4d bundle. The 13 gbm-4d test patients were derived on 2026-11-10 from the rating sheets (`15_ --from-ratings`), with the rule `gbmc.progressions.first_progression_from_ratings`, which reproduces all 50 copied names exactly.

Reason:
Plan Step 1.8: later progressions are "carried into baseline space by the same transport". gbm-4d transported only the first progression.

Consequences:
- 83 later timepoints in 46 of 63 patients, of which 37 have both a 2nd and a 3rd. Recounted on 2026-11-10 from the downloaded rating sheets: the same 83 / 46 / 37.
- The earlier smoke test on Patient-002 (week-047) ran the old gbm-4d code. Under V3 it must be redone.
- Resolved 2026-11-10: the inputs `05_` needs are now here.
  - The rating and completeness sheets were downloaded from LUMIERE's figshare release (`LUMIERE-ExpertRating-v202211.csv` MD5 `3f196e11…`, `LUMIERE-datacompleteness.csv` MD5 `3be2baf9…`, both matching figshare).
  - The 146 needed timepoints (first plus later progressions) were extracted from the already-downloaded `Imaging-v202211.zip`.
  - The 13 missing first-progression names were derived from the ratings by a rule that reproduces all 50 copied names.
- At about 1,100 s per timepoint on one ITK thread (old code), all 83 need about 25 hours sequentially.

## D10: Mesh face budget by marching-cubes step, not quadric decimation (proposed)

Date: (set at commit)
Status: Proposed

Decision:
`gbmc.export.surface` raises the marching-cubes step size until a layer fits its face budget (brain ≤ 60k, others ≤ 20k; plan Step 5.4). Taubin smoothing follows.

Reason:
trimesh's decimation needs `fast_simplification`, which is not in the lock and would be a new dependency.

Do Not Revisit Unless:
The app needs finer meshes within budget. Then approve `fast_simplification`.

## D11: App stack is Vite + TypeScript + three.js + NiiVue, without React (proposed)

Date: (set at commit)
Status: Proposed. The plan leaves React vs plain TypeScript to the user ("choose once in week 2").

Decision:
The app uses plain TypeScript. Its dependencies are three 0.186, @niivue/niivue 0.69, vite 8 and typescript 7, recorded in `app/package-lock.json`. The build uses a relative base, so the same output works on GitHub Pages and from disk offline.

Open:
Exported cases (`app/public/cases/`) are kept out of git for now. Publishing them to GitHub Pages means committing meshes derived from LUMIERE (CC0) and later RHUH-GBM (CC BY 4.0) images, which needs a decision.

## D12: RHUH-GBM transport uses gbm-wm's tuned recipe; the threshold is set on LUMIERE first (proposed)

Date: (set at commit)
Status: Proposed

Decision:
`gbmc.registration` copies gbm-wm's `register_syn` and its D11 settings (`TUNED_SYN`). `transport_pair` follows gbm-wm's `12_` step for step: rigid pre-alignment; SyN forward and backward with the cavity masked; round trip in the peritumoral region; Jacobian QC; nearest-neighbour warp. The seed is `config.yaml:registration.master_seed` (20260819, gbm-4d's, as in D9).

`scripts/07_roundtrip_threshold.py` runs this recipe on the 63 LUMIERE baseline→first-progression pairs with **registration only**. The progression segmentation is never read (plan Step 3.3: "a check that uses no LUMIERE labels").

Open, to decide before OSF:
- `registration.roundtrip_threshold_rule`: how that distribution becomes the RHUH-GBM threshold, e.g. its p95 or its max.
- `registration.roundtrip_threshold_mm`: the value that rule gives.

Consequences:
Smoke test on Patient-001: registration ok, round-trip median 0.77 mm (p95 1.86 mm), no label written, 2,932 s on one ITK thread.

Reason:
gbm-wm lost 21 of 72 patients when a threshold fixed for one recipe met another (plan §2, row 7).

## D13: RHUH-GBM pipeline built before download; every file-dependent constant is null (proposed)

Date: (set at commit)
Status: Proposed

Decision:
`gbmc.rhuh`, `gbmc.consort`, `gbmc.screens`, `gbmc.rhuh_features`, `gbmc.confirm`, `gbmc.verdict` and scripts `20_`–`23_`, `30_`, `31_` are written and unit-tested on synthetic data. They use only what the plan states about the dataset. Everything that depends on the files is null in `config.yaml` until the download and refuses to run: the layout, the clinical columns, the timepoint and sequence names, the cavity rule, the SRI24 midline, the falx T1 sequence, the mapping agreement, the cavity dark threshold, the screen thresholds, `verdict.mde_rho`, `osf.url` and `osf.timestamp`.

Rules implemented as code (plan Step 2.2):
- The label-mapping check may only correct a pure relabelling agreed by `rhuh.mapping_min_agreement` of scans. Otherwise it raises.
- The cavity rule: "labelled:<value>" gives enhancing + that label. "unlabelled" gives post-op enhancing + the pre-op core (1+3) carried rigidly and clipped to the brain.

Proposed, not in the plan:
- **Alternative-cavity sensitivity arm** (`31_`). If the primary is "unlabelled", the alternative is residual + only the fluid-dark (T1ce < `rhuh.cavity_dark_t1ce_z`) part of the carried core. If the primary is labelled, the alternative is the carried-core rule.
- **Falx in SRI24.** gbm-4d's `falx_barrier` is applied through a translation to the SRI24 midline, instead of LUMIERE's shipped MNI transform (Step 4.1 asks for this adaptation to be recorded before scoring).
- **Probability maps (Step 4.6).** They are rounded to float16 precision and stored as float32, because SimpleITK cannot write float16 NIfTI.
- **Not built:** the exploratory C1 + ADC arm, which is first in the plan's cut order.

## D14: Figures and colour roles (proposed)

Date: (set at commit)
Status: Proposed

Decision:
All Step 4.7 figures come from `gbmc.figures`, fed only from `results/` by `scripts/50_figures.py` and `51_reveal_figure.py`. The colour roles are the same in the figures and the app:

| Role | Colour |
|---|---|
| target | #e69f00 |
| FLAIR region | #56b4e9 |
| guideline margin | #0072b2 |
| distance | #cc79a7 |
| challenger / champion | #009e73 |
| recurrence | #d55e00 |

The dataviz skill's validator passes every check on a light surface, all pairs. Distance vs challenger is in the 6–8 CVD band, which is legal only with secondary encoding, so both are always direct-labelled. The rule-chosen demo patient is the median per-patient Δ; with an even count, the lower middle one.

Open:
- No dark-surface palette passes yet: orange vs vermillion collides in dark mode.
- In the lit 3D view the target and the recurrence look alike (seen in the browser check).

## D15: PDE mode re-runs gbm-4d's Fisher–KPP simulation, rewritten in gbmc (proposed)

Date: (set at commit; reworded 2026-11-08 for deviation V3)
Status: Proposed

Decision:
- `scripts/42_pde_frames.py` builds the fields with `gbmc.pde.build_fields` from the patient's baseline (`gbmc.lumiere.load_baseline`).
- It runs at gbm-4d's population point: D 0.5 mm²/day, ρ 0.002/day, white-to-grey ratio 10 (`config.yaml:pde`).
- It advances the solver in chunks with gbm-4d's time-step rule (`pde.stable_dt`, CFL safety 0.5) and keeps nine frames from day 0 to 206. Nothing is fitted.
- `gbmc.pde` is the V3 rewrite of gbm-4d `23_arm1_fkpp.py`'s `load_patient` and `build_fields`, which the first version called through a loader that no longer exists.

Consequences:
- The first version, on gbm-4d's code and data, matched gbm-4d's saved density exactly (maximum absolute difference 0.0, Patients 038, 039 and 090).
- The rewrite matched the old solver to within 3.5e-18 over the full 206-day run on synthetic inputs (deviation V3).
- The three hero cases' frames on disk were made before the V3 baseline rebuild, so they must be re-made from the rebuilt baselines (and the app cases re-exported).

## D16: Ventricle rule V4 for f5 and C0T

Date: 2026-11-07
Status: Active (user decision: "Fix for all patients")

Decision:
`gbmc.anatomy.segment_ventricles` segments the lateral ventricles on T1c as follows, for every LUMIERE and RHUH-GBM patient:
1. Deep brain = brain mask eroded by 15 mm, minus the lesion dilated by 5 mm.
2. Dark = T1c ≤ (1st percentile + median of the deep brain) / 2.
3. Dark components ≥ 1,500 mm³.
4. Cores = what survives a 1 mm opening; keep cores ≥ 500 mm³ whose centre lies within 55 mm of the deep-brain centre.
5. Grow the kept cores through dark voxels by at most 10 mm, then close by 2 mm.
6. QC fails below 1,500 mm³ or above 150,000 mm³ (the old QC had no upper limit).

Reason:
The old Otsu rule produced implausible masks (D6 (k); deviation V4). Each step addresses one failure seen on LUMIERE:
- The midpoint threshold stops Otsu splitting grey from white matter.
- Without the old opening and the "a quarter of the largest" filter, thin ventricles survive.
- The depth limit, the core test and the limited growth keep sulci out.

Alternatives Considered:
Nine variants, compared on volumes, a T2/FLAIR CSF screen and overlays, using anatomy only:
- 3-class Otsu on T1c: still took in grey matter in about 10 patients.
- Midpoint threshold with the old filters: masks far too small.
- T1c and FLAIR jointly: too small, and 4 patients failed.
- No core test: took in sulci.
- 3 mm growth: masks too small.
- 8 mm depth: took in sulci.
Fixing only Patient-002, or keeping the old rule, were rejected by the user.

Consequences:
- All 63 LUMIERE masks pass QC, at 6.2–61.7 ml. This also resolves D6 (j).
- f5, C0T and the matched volume are recomputed (deviation V4).
- The rule is frozen: any change after a score is seen is a deviation.

Do Not Revisit Unless:
A failure is found on anatomy alone, before any candidate is scored.

## D17: Pre-registration constants (plan Steps 2.2, 3.3, 3.4, 4.4)

Date: 2026-11-10
Status: Active (user decisions, 2026-11-10). Fixed before any RHUH-GBM file is downloaded and before any candidate score is seen; any change after the OSF posting is a deviation.

Decision (each value is in `config.yaml`):
1. **Label-mapping check** (`rhuh.mapping_min_agreement` = 0.8). A swapped RHUH-GBM label mapping is corrected only when at least 80% of voting scans derive the same alternative. Otherwise the pipeline stops. Recommended and chosen: a real swap is systematic, while intensity-based derivations are noisy on some scans.
2. **Cavity darkness** (`rhuh.cavity_dark_t1ce_z` = -1.0). T1ce, z-scored over non-zero voxels, below -1.0 counts as fluid-dark. This applies to the cavity inspection (`20_`) and the alternative-cavity sensitivity arm.
   - Evidence: 12 LUMIERE baselines drawn with seed 20261107. Only baseline images and baseline labels were read; no outcome.
   - Ventricle fluid had a median z of about -1.9 (per-patient medians -1.0 to -2.3); normal tissue about +0.1.
   - -1.0 is roughly the midpoint, the same logic as ventricle rule V4.
   - Recorded by `scripts/16_cavity_dark_evidence.py` in `results/cavity_dark_evidence/`, run from a clean commit:
     - ventricle fluid median z -1.92; tissue +0.15; exact midpoint -0.89;
     - per patient, a median 85% of ventricle voxels and 10% of tissue voxels fall below -1.0.
     - So -1.0 is the midpoint rounded slightly toward the darker, stricter side.
   - Caveat: RHUH-GBM's preprocessing differs, so the scale is not verified on its files. Recommended and chosen.
3. **Round-trip exclusion rule** (`registration.roundtrip_threshold_rule` = `max`). The RHUH-GBM threshold on each patient's median round-trip error is the maximum of the per-patient medians from `07_ --summary`. That run uses the tuned recipe on the 63 LUMIERE baseline→first-progression pairs and reads no label.
   - The user chose the maximum over the recommended 95th percentile. It is the more lenient rule: it excludes only registrations worse than LUMIERE's single worst pair.
   - The value (`registration.roundtrip_threshold_mm`) is filled from `07_ --summary` before posting, and `07_` still has to run.
4. **Brain-surface screen** (`screens.surface_review_mm` = 3.0, `screens.surface_worse_margin_mm` = 1.0). These are gbm-wm's values, carried over unchanged. They were chosen post hoc in gbm-wm, but here they are fixed before any RHUH-GBM image. A surface flag excludes a patient only when the arbiters also dissent from our transform. Recommended and chosen.
5. **Verdict MDE** (`verdict.mde_rho` = 0.7). The Step 4.4 outcome ("Bounded null" vs "Inconclusive") uses the achieved MDE at an assumed arm correlation of 0.7.
   - The user chose the middle value over the recommended pessimistic 0.5.
   - The MDE at 0.7 is smaller than at 0.5, so "Bounded null" is easier to reach.
   - The achieved MDE at every rho (0.5, 0.7, 0.9) is still recorded by `23_`.

Reason:
Plan Steps 2.2–2.5: these rules must be on record in the OSF pre-registration, before the RHUH-GBM download.

## D0: Documentation structure

Date: 2026-09-23
Status: Active

Decision:
The project maintains README, PROJECT_STATUS, HANDOFF, DECISIONS and CHANGELOG at the repository root.

Reason:
A new session must be able to pick up the project from the repository alone.

Alternatives Considered:
None.

Consequences:
Every substantial session reads and updates these files.

Do Not Revisit Unless:
The documentation protocol changes.
