# Where Does Glioblastoma Come Back? Final plan

Sep 22, 2026 · \@sophie

**Can MRI-based models beat distance?** The 2023 ESTRO-EANO radiotherapy guideline for glioblastoma recommends a 15 mm margin around the tumour, trimmed only at anatomical barriers; it advises against including oedema, yet says FLAIR abnormality \"may represent non-enhancing tumour and should be considered for inclusion\". This project asks whether any of eight models, from a one-line rule to a 3D neural network, predicts where glioblastoma recurs better than distance from the tumour, on 40 patients no model has seen. The headline test is simple: when a model and the guideline margin treat exactly the same volume of brain, which catches more of the actual recurrence? Each phase below is a step-by-step guide with a gate that must pass before the next phase starts.

## 1. Project at a glance

The champion is whichever candidate wins a pre-registered tournament on LUMIERE and then beats distance from the treatment target on RHUH-GBM, a cohort no model has seen. If no challenger beats distance there, distance is the champion, and the project reports by how much every challenger fell short.

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Item                                Decision
  ----------------------------------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Working title                       Where does glioblastoma come back? Can MRI-based models beat distance?

  Question                            Do MRI-based models add predictive information about where glioblastoma recurs beyond distance from the tumour, on an independent cohort, and does more model complexity buy more? Every analysis, figure and screen points back to this question

  Why it matters                      The guideline margin is the same 15 mm in every direction except at anatomical barriers. If recurrence follows the baseline FLAIR abnormality, a margin shaped by it would cover more recurrence at the same treated volume. If it does not, the uniform margin stands on better evidence

  Unit of prediction                  Each 1 mm voxel of brain outside the treatment target gets a recurrence score. Together the scores form a 3D map

  Primary score                       PR-AUC (area under the precision--recall curve), computed per patient and then averaged. Suited to recurrence, which occupies a small minority of voxels

  Headline number for judges          Coverage at matched volume: the share of recurrence each model captures when it treats exactly the volume of that patient\'s own 15 mm guideline margin

  Development cohort                  LUMIERE, 63 locked patients (CC0), already used by gbm-4d

  Confirmatory cohort                 RHUH-GBM, 40 patients (open NIfTI files, CC BY 4.0), analysed once

  Candidates                          Two distance baselines, a FLAIR-edge rule, distance + baseline FLAIR (logistic and gradient-boosted), distance + FLAIR + anatomy, a gated 3D convolutional network, and a rank-average ensemble: eight in all (Step 1.4)

  Champion rule                       The best non-distance candidate on LUMIERE becomes the challenger. It is champion only if its ΔPR-AUC over distance on RHUH-GBM has a 95% CI lower bound above zero

  The fourth dimension                The app steps through each patient\'s real scans over time, and an exploratory analysis asks whether the maps also rank where the tumour spreads at later progressions (Step 1.8)

  Public deliverable                  A 3D/4D explainer app, built as a static website that also opens on a phone: real public cases, each model\'s predicted map beside the actual recurrence

  Fair deliverables                   Poster, written report (TRIPOD+AI template), 30 s / 3 min / 10 min pitches, the app at the booth with recorded and printed fallbacks

  Duration                            Eight weeks, 2026-09-22 to mid-November
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Why a tournament and a separate confirmation.** Choosing the best of several models and scoring it on the same data inflates its score, because the winner is partly the model that got lucky. Choosing on LUMIERE and scoring once on RHUH-GBM removes that inflation. A judge can check the sequence from the OSF timestamp and the download date committed to the repository.

**What a judge should see in the first minute.**

1.  **One question.** Can MRI-based models beat distance? Every panel and screen answers it.

2.  **A clinical hook.** The guideline\'s 15 mm margin, and its open question about FLAIR abnormality, which most of the challengers are built on.

3.  **A number anyone understands.** At exactly the guideline margin\'s volume, what share of the recurrence does each model catch?

4.  **Prediction, then reveal.** One patient\'s brain in 3D: post-op MRI, the 15 mm margin, the model\'s committed prediction, then the actual recurrence revealed and scored.

5.  **Does more complexity help?** One forest plot orders the eight models from simplest to most complex against a zero line that is distance itself.

6.  **The truth test.** 63 patients, tournament, freeze, hash, 40 separate patients, one run. The script refuses a second run.

The 3D network is one point on the complexity axis. The project\'s strength is the fair comparison against a strong, clinically meaningful baseline.

## 2. Starting point

Two finished studies set the bar and narrow the candidate list. Both are frozen, so this project reuses their code and findings and re-fits nothing from them.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Finding                                                                                                                                        Source                                                                                                            Consequence for this project
  ---------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------
  Distance from the target predicts recurrence location; habitats, the Fisher--KPP PDE and the hybrid each add \< \~0.015 ΔPR-AUC                gbm-4d, LOPO n = 42, sealed test n = 11                                                                           Distance is the model to beat

  45.2% of recurrent enhancing tissue lies inside the target; 86% of patients recur within 2 cm                                                  gbm-4d                                                                                                            Most of the answer is already in the target, so any gain will be small

  The PDE converged to maximum diffusion and minimum proliferation (40 of 42 fits), which is a smoothed distance map                             gbm-4d RQ2                                                                                                        The PDE appears in the app as an animation, and stays out of the tournament

  Adding feature blocks to LightGBM made scores slightly worse at each step                                                                      gbm-4d RQ1                                                                                                        Candidates stay small; a simple model wins ties

  Baseline FLAIR-abnormal tissue is the largest spatial effect in both cohorts: RR 7.29 (4.73--16.44) and 3.22 (1.68--5.88)                      gbm-wm §4.4, §5                                                                                                   The main new feature. It was measured against an unmatched null on 9--13 patients, so it is a lead

  White matter did not replicate: RR 0.94 (0.54--1.34); masks reverse between cohorts                                                            gbm-wm §4.2, §5                                                                                                   White matter is excluded from every candidate

  gbm-4d\'s habitat GMM (Gaussian mixture model, an unsupervised clustering of voxel intensities) included FLAIR, and its blocks added nothing   gbm-4d D29                                                                                                        The prior against FLAIR features. The new features use the segmented FLAIR region and distance to its edge, which the GMM never encoded

  A registration threshold written for one recipe removed 21 of 72 patients when a different recipe ran                                          gbm-wm §3.6                                                                                                       Screen thresholds are fixed against the recipe that will actually run
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Expected outcome.** The most likely result is that distance remains champion, with every challenger bounded to a small margin. The plan treats that as a complete result: it would extend gbm-4d\'s result from LUMIERE to an external cohort.

## 3. Roadmap

Six phases, each closed by a gate. Phase 5 (the app) runs in parallel from week 2, because it can be built on LUMIERE cases before any RHUH-GBM result exists.

flowchart TD\
P0\[Phase 0\<br/\>Setup + ISEF forms\] \--\> P1\[Phase 1\<br/\>Tournament on LUMIERE\]\
P1 \--\> P2\[Phase 2\<br/\>Freeze + OSF\]\
P2 \--\> P3\[Phase 3\<br/\>RHUH-GBM cohort lock\]\
P3 \--\> P4\[Phase 4\<br/\>Confirmatory run, once\]\
P4 \--\> P6\[Phase 6\<br/\>Poster, report, pitches\]\
P0 \--\> P5\[Phase 5\<br/\>3D/4D explainer app\]\
P4 \--\> P5\
P5 \--\> P6

  ------------------------------------------------------------------------------------------------------------------------------------------------------------
  Phase                       Weeks                   Gate that closes it
  --------------------------- ----------------------- --------------------------------------------------------------------------------------------------------
  0\. Setup + paperwork       1                       ISEF forms signed; SRC contacted; repository scaffolded; literature search and expert outreach started

  1\. Tournament              1--2                    Every candidate scored on identical folds; challenger named by the pre-written rule

  2\. Freeze + pre-register   2                       OSF timestamp earlier than the RHUH-GBM download; model weights hashed

  3\. RHUH-GBM cohort         3                       Registration screens run; cohort locked; evaluable n recorded

  4\. Confirmatory run        4--5                    Primary computed once; champion verdict written into the pre-written sentence

  5\. App                     2--7                    Runs offline on the fair laptop; five-person usability check passed

  6\. Communication           6--8                    Poster final; report drafted; expert review recorded; pitches mock-judged; fallbacks printed
  ------------------------------------------------------------------------------------------------------------------------------------------------------------

## 4. Phase 0 --- Setup and paperwork (week 1)

ISEF requires the forms to be signed before the research starts, so this phase comes before any new model is trained or any file is downloaded.

**Step 0.1 --- ISEF forms.** Both datasets are public, de-identified and involve no interaction with people, which ISEF\'s human-participant rules list as exempt from IRB review and from the Human Participants form (4).

- [ ] Checklist for Adult Sponsor (1)

- [ ] Student Checklist (1A)

- [ ] Research Plan/Project Addendum: question, both datasets with licences, the tournament, the confirmatory test, the app

- [ ] Approval Form (1B)

- [ ] Student Support Disclosure (2A), new for 2026--27

- [ ] Continuation/Research Progression (7). Show a substantive expansion over gbm-4d and gbm-wm: a new question (which model is best, including FLAIR and 3D CNN candidates), a new external cohort, and the app. ISEF rejects a continuation that repeats the same method and question with more patients

- [ ] Email your affiliated fair\'s Scientific Review Committee (SRC): one paragraph naming LUMIERE (CC0) and RHUH-GBM open NIfTI (CC BY 4.0), and the plan to download only the open files. Keep the written confirmation of the exemption

**Step 0.2 --- Repository.** Create a new repository (working name gbm-champion) and install the gbm-4d library at its final commit (eeba777) as a pinned dependency, so the reused preprocessing, label transport and screens are the exact frozen code.

gbm-champion/\
src/gbmc/ library: features, candidates, scoring, export\
scripts/ numbered stages: 01\_ ... 40\_\
docs/ protocol.md, decisions.md, deviations.md, preregistration.md\
results/\<stage\>/ JSON results, each with run.manifest.json\
app/ the explainer app (Phase 5)\
tests/\
HANDOFF.md live status, regenerated by a pre-commit hook

**Step 0.3 --- Environment.** Reuse gbm-4d\'s lockfile as the starting point (Python 3.13.14, LightGBM 4.7.0, scikit-learn 1.9.0, SimpleITK 2.5.6, numpy 2.5.2). Add only the packages this project needs: scikit-image and trimesh for meshes, and PyTorch if the CNN gate passes. Keep the ITK thread count pinned to 1 (gbm-4d D9).

**Step 0.4 --- Hardware and starting-weights gate for the CNN candidate.** By the end of week 1, decide two things and record both in decisions.md. First, whether a CUDA GPU (a graphics card PyTorch can train on) with at least 8 GB of memory is available, on the laptop or the second machine; if none is, C5 is dropped. Second, whether C5 starts from publicly released weights of a 3D brain-tumour segmentation network with a licence that allows reuse (transfer learning: starting from a network already trained on thousands of tumour scans, which suits a 63-patient cohort better than training from scratch). If no such weights are found and verified in week 1, C5 trains from scratch.

**Step 0.5 --- Carry over the reproducibility contract** (Section 13): BLAKE2b seeds, run manifests, clean-tree checks, null constants that raise on read.

**Step 0.6 --- Literature scan, capped at 25 papers.** Judges will ask how the champion compares with published recurrence-location models. Commit a PubMed search string and an extraction form before reading, then record for each study of voxel-level or region-level recurrence-location prediction in glioblastoma:

  -------------------------------------------------------------------------------------------------
  Field                                           Why
  ----------------------------------------------- -------------------------------------------------
  Cohort size and imaging inputs                  Puts 63 + 40 patients in context

  External validation (yes/no, cohort)            The test this project runs

  Distance-to-tumour baseline reported (yes/no)   The comparator this project treats as essential

  Metric and headline number                      Whether numbers are comparable at all

  Pre-registration (yes/no)                       Design discipline
  -------------------------------------------------------------------------------------------------

The table becomes the poster\'s context panel and a report section. Report the counts exactly as found; the scan\'s purpose is to place this project among published work, whichever way the counts fall.

**Step 0.7 --- Expert outreach, sent in week 1.** Email two or three neuro-oncologists, radiation oncologists or neuroradiologists (a local hospital, a university lab, or an author found in Step 0.6) asking for one hour in week 6. They review the 3D reveal, the app\'s wording and the poster\'s clinical framing; they touch no data. A single hour of review addresses gbm-4d\'s recorded limitation V4 (no clinician in the loop), and it is the answer to a judge asking whether a doctor has seen this.

**Gate:** forms signed; SRC contacted; repository builds and its tests pass; CNN hardware and starting-weights decisions recorded; literature search string committed; expert emails sent.

## 5. Phase 1 --- Model tournament on LUMIERE (weeks 1--2)

Every candidate is scored on the same patient folds with the same endpoint. The challenger is chosen by a rule written into decisions.md before the first candidate runs.

**Step 1.1 --- Write the tournament rules first.** Commit the candidate list, the features, the folds, the scoring and the selection rule (Steps 1.2--1.6) to decisions.md. Any change after the first score is seen is a deviation, logged with its direction of effect.

**Step 1.2 --- Cohort and endpoint.** Use all 63 locked LUMIERE patients, in gbm-4d\'s baseline space, with gbm-4d\'s transported labels. Its sealed test set has already been opened, so in this project LUMIERE is development data only; record that change of role as deviation V1. The label is enhancing tumour at the first segmented RANO progression; the evaluation region is brain minus the target. Patients with no enhancing recurrence outside the target are *undefined*, never zero (gbm-4d D18).

**Step 1.3 --- Features, computed per voxel.**

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  \#                Feature                     Definition                                                                                      Source
  ----------------- --------------------------- ----------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------
  f1                Distance to target          Signed Euclidean distance (mm) to the surface of baseline enhancing tumour + resection cavity   gbm-4d Block A

  f2                FLAIR-abnormal membership   1 inside the baseline oedema/non-enhancing label, else 0                                        LUMIERE DeepBraTumIA label, using gbm-4d\'s verified label mapping (02_verify_labels.py)

  f3                Distance to FLAIR edge      Signed Euclidean distance (mm) to the surface of the f2 region                                  New

  f4                Interaction                 f1 × f2                                                                                         New

  f5                Geodesic distance           Shortest path from the target that respects the falx, ventricles and skull                      gbm-4d Block B
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Native voxel spacing, a covariate in gbm-4d, is left out: RHUH-GBM\'s open files are resampled to 1 mm, and the native spacing sits in the controlled DICOM. Record this as a decision.

**Step 1.4 --- Candidates,** listed from simplest to most complex. That order breaks ties and sets the x-axis of the complexity forest plot.

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ID             Candidate                            Inputs                                                                                                                Learner                                                                                                                                                                         Why it is in
  -------------- ------------------------------------ --------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -----------------------------------------------------------------------------------------------------------------------
  C0             Distance                             f1                                                                                                                    None: score = −f1                                                                                                                                                               The clinical margin as a ranking. Always the comparator

  C0T            Trimmed distance                     f1 with the ESTRO-EANO 2023 barrier rules: margin stops at the skull and falx, and is cut to 5 mm at the ventricles   None                                                                                                                                                                            The guideline margin written as a ranking. gbm-4d trimmed at the falx only; the full barrier set is a recorded change

  C4             FLAIR edge                           f3                                                                                                                    None: score = −f3                                                                                                                                                               Tests whether the oedema boundary alone ranks voxels better than the target boundary

  C1             Distance + FLAIR, linear             f1--f4                                                                                                                L2 logistic regression, patient-balanced weights                                                                                                                                Four coefficients; overfits least

  C3             Distance + FLAIR + anatomy, linear   f1--f5                                                                                                                L2 logistic regression                                                                                                                                                          Tests whether respecting anatomical barriers adds to C1

  C2             Distance + FLAIR, non-linear         f1--f4                                                                                                                LightGBM with gbm-4d\'s frozen hyperparameters                                                                                                                                  Tests whether a non-linear learner extracts more from the same inputs

  C5             3D CNN (gated, Step 0.4)             Baseline T1c and FLAIR (z-scored), target mask, FLAIR mask, f1 map                                                    Small 3D U-Net (a convolutional network that maps a 3D image to a same-sized 3D probability map), patch-based, starting from pretrained weights if Step 0.4 finds usable ones   Answers the judge\'s question \"why not deep learning?\" with a measured number

  C6             Rank-average ensemble                The voxel scores of C1, C3, C2, and C5 if it runs                                                                     None: mean of the members\' within-patient percentile ranks                                                                                                                     Tests whether combining the learned candidates beats the best single one
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**C5 rules, to stop it leaking or overfitting unseen:** a fixed number of epochs chosen before training (early stopping on a validation fold would leak it into the choice); augmentation limited to left--right flips and rotations up to 10°; one seed per fold; weighted loss for the rare recurrent class. With about 54 training patients per fold, the likely outcome is a CNN that fits training folds well and scores below C1 on held-out folds. The tournament records whichever happens.

**C6 rules.** An ensemble beats its members only when they rank different voxels wrongly, so their errors partly cancel. C6 tests that with nothing fitted:

1.  Membership is fixed now: C1, C3 and C2, plus C5 if the Step 0.4 gate passes. It never depends on how the members score.

2.  For each patient, each member\'s voxel scores in the evaluation region become percentile ranks from 0 to 1 (ties share the average rank). Ranks put a logistic probability, a LightGBM score and a CNN output on one scale without calibrating them.

3.  C6\'s score at a voxel is the unweighted mean of its members\' ranks.

4.  In cross-validation, C6 averages the members\' held-out predictions for each fold, so it sees no held-out labels and needs no nested cross-validation (a second, inner split used to fit weights inside each training fold).

5.  C0 is excluded as a member, because every member already includes distance f1 as an input.

Weighted stacking (a second-level model that learns how much to trust each member) is left out. It would need nested cross-validation, and with about 54 training patients per fold its weights would be noisy. Record this as a decision.

**Step 1.5 --- Folds and scoring.** One seeded, patient-grouped 7-fold split (9 patients per fold), drawn with stable_seed and shared by every candidate. For each held-out patient, compute three numbers:

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Metric                       Definition                                                                                                                                                                                                                               Why
  ---------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------
  PR-AUC (primary)             Area under the precision--recall curve, brain minus target                                                                                                                                                                               Registered endpoint, continuous with gbm-4d

  Lift over chance             PR-AUC ÷ the patient\'s recurrence prevalence (the PR-AUC of a random ranking)                                                                                                                                                           Makes a PR-AUC near 0.07 readable: \"x times better than guessing\"

  Coverage at matched volume   Share of recurrence outside the target that falls in the model\'s top-ranked voxels, when the model treats exactly the volume of that patient\'s own 15 mm guideline margin; repeated at 20 mm and at 50 ml for continuity with gbm-4d   The headline number: same dose volume, more recurrence caught?
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Report each candidate\'s mean of each metric and its Δ vs C0 with a patient-level bootstrap 95% CI (1000 resamples, BCa: bias-corrected and accelerated, as in gbm-wm).

**Step 1.6 --- Selection rule.** The challenger is the candidate from C4, C1, C3, C2, C5 and C6 with the highest mean held-out PR-AUC. C6 ranks as the most complex, so it becomes challenger only if it beats every simpler candidate by more than 0.005. If a simpler candidate is within 0.005 of it, the simpler one is chosen. The challenger goes to Phase 4 whether or not it beat C0 on LUMIERE.

**Step 1.7 --- Refit for release.** Retrain the challenger and every learned candidate on all 63 patients with the same settings. Save the weights, and record their SHA-256 hashes in the run manifest. The app shows every candidate, so all of them are refit. C6 has no weights of its own: it is rebuilt from the refit members, and its manifest lists their hashes.

**Step 1.8 --- The fourth dimension (exploratory).** LUMIERE follows most patients through several RANO progressions (median 3). Using each candidate\'s out-of-fold map from Step 1.5, score the *new* enhancing voxels at the second and third progression, the ones absent at the first, carried into baseline space by the same transport. The question: does a map trained on where the tumour came back also rank where it spreads next? Label it exploratory, report it for every candidate, and use it to drive the app\'s time slider. RHUH-GBM has one recurrence scan, so this analysis exists on LUMIERE only.

**Deliverables:** results/tournament/leaderboard.json (one row per candidate: mean PR-AUC, lift over chance, coverage at matched volume, each Δ vs C0 with its CI); results/tournament/later_progression.json; the challenger\'s name; hashed weights; the LUMIERE points for the complexity forest plot.

**Gate:** all candidates scored on identical folds; challenger named by the rule; weights hashed and committed.

## 6. Phase 2 --- Freeze and pre-register (week 2)

Everything that could be tuned toward the RHUH-GBM result is fixed and posted to OSF before the first RHUH-GBM file is downloaded. The download date is committed to the repository after the OSF timestamp.

**Step 2.1 --- Power simulation.** Simulate the minimum detectable effect (MDE: the smallest ΔPR-AUC the design detects with 80% power) from C0\'s LUMIERE scores alone, under the actual bootstrap decision rule, at 20, 25, 30 and 35 evaluable patients. gbm-4d\'s reference points were 0.05 at n = 42 and 0.08 at n = 11. The table goes into the pre-registration, and the achieved power is reported beside the result. Every effect gbm-4d measured was below 0.015, so if the simulated MDE at the expected size is above 0.05, the realistic headline is a bound on the challenger rather than a win, and the poster\'s pre-written title band for that case should lead with the bound and the matched-volume coverage.

**Step 2.2 --- Settle the RHUH-GBM definitions from its documentation.** The dataset paper gives the labels as 1 = necrosis, 2 = peritumoral signal alteration (oedema + non-enhancing tumour), 3 = enhancing tumour, and registers each timepoint separately to the SRI24 atlas. Two points it leaves open have to be pre-registered with a rule:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Open point                                                                                Pre-registered rule
  ----------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  The paper does not state how the resection cavity is labelled on the early post-op scan   On download, inspect label values inside the cavity on 5 seeded patients. If the cavity has no label, the target = post-op enhancing residual + the preoperative tumour core (labels 1 + 3), carried rigidly into post-op space and clipped to the brain mask. Record this as a deviation from gbm-4d\'s target definition

  The label mapping could differ from the published one in the files                        Re-derive the mapping from intensities, as gbm-4d\'s 02_verify_labels.py did. This check may only correct a swapped mapping, never choose between alternatives
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Step 2.3 --- The primary and the verdict rule.**

- Primary comparison: challenger vs C0, Δ mean PR-AUC on evaluable RHUH-GBM patients, patient-level bootstrap 95% BCa CI. PR-AUC stays primary because it scores the ranking at every volume, has more statistical power than a single-volume metric, and continues gbm-4d.

- Champion verdict: the challenger is champion if the CI\'s lower bound is above 0. Otherwise C0 is champion.

- Hero secondary, shown first on every deliverable: challenger vs C0 on coverage at matched volume (each patient\'s own 15 mm guideline-margin volume). At that volume C0\'s top-ranked voxels are essentially the guideline margin itself, so this reads as \"model-shaped region vs guideline margin, same volume\".

- Complexity trend: every candidate vs C0, in the Step 1.4 complexity order, reported as one forest plot (Step 4.7).

- Other secondaries, under BH-FDR (Benjamini--Hochberg false discovery rate control, q = 0.05): every other candidate vs C0; challenger vs C0T; challenger vs C0 in the escape region (\> 20 mm from the target); lift over chance; coverage at matched 20 mm volume and at 50 ml.

- Where distance fails, descriptive only: per patient, the share of recurrence beyond the 15 mm margin and beyond 20 mm, and the share of it each model catches. No test is run on it: the escape events will be few.

- Sensitivity: unscreened arm (all transported patients); gbm-4d\'s original registration recipe; alternative cavity definition; label *enhancing + necrosis*; the 11 patients scanned partly at a secondary facility removed.

- Exploratory: C1 + baseline ADC, fitted leave-one-patient-out inside RHUH-GBM, since LUMIERE has no ADC. The later-progression analysis from Step 1.8 is already on record from LUMIERE and is quoted as exploratory.

**Step 2.4 --- The result sentences, written now with blanks:**

- *Challenger wins:* \"On an independent cohort, \[challenger\] adds \[Δ\] PR-AUC (95% CI \[...\]) over distance from the treatment target, and at the volume of the 15 mm guideline margin captures \[x\]% of recurrence against \[y\]% for distance.\"

- *Distance holds:* \"On an independent cohort, the best challenger from an eight-model tournament adds less than \[upper limit\] PR-AUC over distance, and at the volume of the 15 mm guideline margin captures \[x\]% of recurrence against \[y\]% for distance; distance remains the best available predictor.\"

- *The two metrics disagree* (one CI excludes zero, the other does not): \"The verdict follows the pre-registered primary, PR-AUC, which \[does / does not\] favour \[challenger\]. At the guideline margin\'s volume alone, coverage \[does / does not\] differ (\[x\]% vs \[y\]%, 95% CI \[...\]); the difference is reported as a secondary result.\"

**Step 2.5 --- OSF checklist.**

- [ ] Tournament leaderboard and the named challenger, with the commit hash

- [ ] SHA-256 hashes of every refit model

- [ ] Feature definitions and the label mapping for each cohort

- [ ] Cavity rule and label-verification rule (Step 2.2)

- [ ] Registration recipe, screen thresholds and exclusion rules (Step 3.3)

- [ ] Power table

- [ ] Primary, verdict rule, secondaries, sensitivity and exploratory lists

- [ ] Both result sentences

- [ ] A statement of what was seen before posting: all LUMIERE results, both predecessor studies, the RHUH-GBM dataset paper, and no RHUH-GBM image

**Gate:** OSF posted; download date committed after the OSF timestamp.

## 7. Phase 3 --- RHUH-GBM cohort (week 3)

This phase turns 40 downloaded patients into a locked, screened cohort with labels in baseline space, without computing any model score.

**Step 3.1 --- Download.** Only the open files: brain-extracted NIfTI images and segmentations (2.9 GB) and the clinical CSV (CC BY 4.0). Leave the controlled DICOM alone. Commit the download date and the archive\'s SHA-256.

**Step 3.2 --- Inventory.** For each patient, confirm the early post-op scan (\< 72 h) and the recurrence scan, each with T1ce, FLAIR and a segmentation. Run the label checks from Step 2.2 (cavity labelling on 5 seeded patients; mapping re-derived from intensities) and apply the pre-registered rule. Record the 11 patients whose preoperative and follow-up scans came from a secondary facility as a covariate.

**Step 3.3 --- Label transport.** Carry the recurrence segmentation into the early post-op grid by deformable registration driven by T1ce, with the resection cavity masked out of the metric.

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Choice                           Primary                                                                                                                                                    Sensitivity
  -------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------- -----------------------------------------------------------------------
  Recipe                           gbm-wm\'s tuned SyN settings: 1.39 mm residual on synthetic deformations, against 2.13 mm for the original                                                 gbm-4d\'s original recipe, which produced the LUMIERE training labels

  Round-trip exclusion threshold   Fixed in Phase 2 against the tuned recipe\'s round-trip distribution when run on LUMIERE baseline→progression pairs, a check that uses no LUMIERE labels   The unscreened arm
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The two recipes differ because gbm-wm found the original recovered only 0.6 mm of a 2.7 mm known deformation. The mismatch with the LUMIERE training labels is recorded as a deviation, and the sensitivity row measures its effect.

**Step 3.4 --- Registration screens.** These can exclude a patient but cannot certify one, because two estimates of an unknown truth may share a bias.

- [ ] Round-trip error per patient (peritumoral median, maximum), Jacobian statistics, folded-voxel count

- [ ] Brain-surface distance against the shipped brain masks, compared with the identity transform as a null

- [ ] Multi-arbiter consensus on every patient: recipe re-driven by T2 and FLAIR, plus the shipped atlas alignment as an independent estimate

- [ ] Simulated deformation recovery on 10 seeded patients

**Step 3.5 --- Exclusions**, each logged with its reason: a missing sequence or segmentation; failed transport; screen failure; an empty target; no enhancing recurrence outside the target (evaluable = no, reported as undefined).

**Step 3.6 --- Cohort lock.** Commit a CONSORT table (the standard flow chart of how many patients were excluded at each step and why), from 40 enrolled down to the evaluable count. Compare the evaluable n with the power table and record the achieved MDE before Phase 4 starts.

**Watch for:** C5 reads image intensities, and RHUH-GBM\'s preprocessing (SynthStrip, SRI24, z-score) differs from gbm-4d\'s (N4, rigid to T1c, z-score in the brain mask). C0--C4 depend only on masks and are unaffected. Log this as a known source of shift for C5.

**Gate:** cohort locked; CONSORT table committed; achieved MDE recorded; no model score computed.

## 8. Phase 4 --- Confirmatory run and the champion verdict (weeks 4--5)

The frozen models are applied once to the locked RHUH-GBM cohort, and the verdict rule names the champion. Nothing is tuned, reselected or re-run.

**Step 4.1 --- Features on RHUH-GBM.** Compute f1--f5 in each patient\'s early post-op grid. Derive the falx and ventricle barriers for f5 the same way gbm-4d did, adapted from MNI to SRI24 space; record the adaptation as a decision before scoring.

**Step 4.2 --- Hash check.** The scoring script loads each model, recomputes its SHA-256, and stops if it differs from the value posted to OSF.

**Step 4.3 --- Primary, once.** Score every candidate on every evaluable patient. The script writes an irreversible marker when it first reads RHUH-GBM labels and refuses to run a second time (gbm-4d 30\_ and gbm-wm pattern).

**Step 4.4 --- Verdict.** Apply the Step 2.3 rule, name the champion, and fill in the matching Step 2.4 sentence. Classify the result with one of these pre-named outcomes:

  ---------------------------------------------------------------------------------------------------------------------------------------
  Outcome                 Condition                           What the poster says
  ----------------------- ----------------------------------- ---------------------------------------------------------------------------
  Challenger champion     CI lower bound \> 0                 The challenger improves on distance by \[Δ\] on an unseen cohort

  Bounded null            CI contains 0; upper bound \< MDE   Distance holds; challengers add less than \[upper bound\]

  Inconclusive            CI contains 0; upper bound ≥ MDE    Distance holds; the cohort was too small to bound the challengers tightly

  Challenger worse        CI upper bound \< 0                 Distance beats the best learned model on new patients
  ---------------------------------------------------------------------------------------------------------------------------------------

**Step 4.5 --- Secondary, sensitivity and exploratory analyses,** exactly as listed in Step 2.3, with BH-FDR applied to the secondary family only.

**Step 4.6 --- Export for the app.** For every evaluable patient and every candidate: the probability map (NIfTI, float16), the top-ranked 50 ml and 100 ml masks, per-patient PR-AUC and coverage. Save to results/confirm/maps/ with a manifest.

**Step 4.7 --- Figures.**

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Figure                                                      Content                                                                                                                                                                                                                                                          Takeaway
  ----------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------
  **Prediction vs reality** (centrepiece of poster and app)   One rule-chosen patient (the median per-patient Δ), same brain, same orientation, four panels: 15 mm guideline margin; distance prediction; champion prediction; actual recurrence. Both predictions at the margin\'s volume, coverage printed under each        What \"beat distance\" looks like in a real brain

  **Does more complexity help?**                              Forest plot on RHUH-GBM: x-axis = the eight models as ordered categories, simplest to most complex (C0, C0T, C4, C1, C3, C2, C5, C6); y-axis = Δ PR-AUC vs distance with 95% CI; the zero line is distance. LUMIERE development points drawn faint beside each   The answer to the central question, readable in five seconds

  **The truth test**                                          Protocol strip: 63 patients → tournament → freeze → hash → OSF → 40 separate patients → one run → reveal, with the OSF and download dates printed                                                                                                                Why the result can be trusted

  **Hero metric**                                             Matched-volume coverage: guideline margin, distance, champion, as one bar group with CIs                                                                                                                                                                         The number a judge remembers

  Where distance fails (descriptive)                          Per-patient share of recurrence beyond 15 mm and beyond 20 mm, and the share each model catches                                                                                                                                                                  Where a better predictor would have to win

  Coverage curve                                              Coverage against treated volume, champion vs C0                                                                                                                                                                                                                  Whether the result holds at every volume

  Per-patient strip                                           Each RHUH-GBM patient\'s Δ, ordered, with the demo patient and the failure case marked                                                                                                                                                                           Whether one patient drives the mean

  Later progression (exploratory)                             For LUMIERE patients with later progressions, how each map ranks the new tumour at each timepoint                                                                                                                                                                The fourth dimension, labelled exploratory
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Step 4.8 --- Number audit.** Write a checker, like gbm-wm\'s 21_check_manuscript.py, that reads every number quoted in the poster, report and app text and compares it with the JSON results.

**Gate:** verdict sentence filled; every figure generated from results/; audit script passes.

## 9. Phase 5 --- The 3D/4D explainer app (weeks 2--7)

The app is a static website that lets anyone rotate a real patient\'s brain in 3D, step through that patient\'s real scans in time, and see where each model predicted recurrence beside where it happened. It is educational: it runs on precomputed results from public cases and carries a fixed label on every screen.

### Step 5.1 --- Product definition

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Item                                Decision
  ----------------------------------- --------------------------------------------------------------------------------------------------------------------------------
  Audience                            Fair judges and visitors first; students and the public after the fair

  Job                                 Answer \"Can MRI-based models beat distance?\" with a real case: prediction first, then the reveal, then the score

  Cases                               Every evaluable RHUH-GBM patient (CC BY 4.0, credited on screen) and the LUMIERE patients used in gbm-4d\'s hero figures (CC0)

  Fixed label, every screen           \"Research visualization of a pre-registered method comparison on public data. Not a treatment prediction.\"

  Delivery                            Static website on GitHub Pages; an offline build for the fair laptop

  Out of scope                        Uploading scans, live inference, accounts. Those belong to the later research toolkit (Section 14)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Step 5.2 --- Screens

1.  **Story (the landing screen and the default judge path)** --- a guided walkthrough of one patient, chosen by rule: the RHUH-GBM patient with the median per-patient Δ. The model commits before the truth is shown, just as the pipeline did:

    a.  The brain after surgery: post-op MRI, brain surface, cavity and residual tumour.

    b.  The guideline margin: the 15 mm ESTRO-EANO shell around the target, trimmed at the skull, falx and ventricles.

    c.  The prediction: distance\'s and the champion\'s top-ranked regions at exactly the margin\'s volume. The recurrence stays hidden.

    d.  The reveal: the actual recurrence appears, in two colours for covered and missed.

    e.  The score: coverage for the guideline margin, distance and the champion, side by side.

    f.  The truth test: the protocol strip, from 63 patients to one run on 40.

    g.  Does more complexity help? The forest plot of all eight models against distance.

    h.  What it means, with the limits stated.

Buttons under step 5 open the failure case (the patient with the lowest Δ) and the best case (the highest), each with the same reveal.

2.  **Explore** --- pick any case; rotate, zoom and slice; toggle layers; move the time slider; switch the model shown. The reveal toggle works on every case.

3.  **Compare** --- two models side by side at the same treated volume, cameras linked, each with its coverage number.

4.  **Results** --- the complexity forest plot and the matched-volume coverage bars. Clicking a patient\'s dot in the per-patient strip opens that case in Explore.

5.  **Learn** --- how each candidate works in two or three sentences, a glossary (PR-AUC, margin, FLAIR, registration, pre-registration), the limitations, and data credits.

### Step 5.3 --- The viewer

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Element                             Behaviour
  ----------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  3D panel                            Meshes lit and rotatable; brain surface translucent; clipping plane to cut the brain open

  Slice panel                         Axial, coronal and sagittal slices of T1ce or FLAIR, with the selected model\'s probability map as a heat overlay and recurrence as an outline. Linked to the 3D panel\'s crosshair

  Layers                              Brain, target, FLAIR-abnormal region, 15 mm guideline margin (20 mm switchable), model prediction at the guideline margin\'s volume or at 50 ml, actual recurrence (covered / missed)

  Time slider                         Steps through real scans only: LUMIERE\'s post-op and each segmented progression timepoint; RHUH-GBM\'s pre-op, early post-op and recurrence. Every frame is labelled with its interval from baseline as the dataset reports it. For LUMIERE cases, each later progression is drawn over the same predicted map, so the viewer sees whether the map anticipated the spread (Step 1.8, exploratory)

  PDE mode                            Plays the Fisher--KPP simulation from day 0 to day 206, labelled \"what a diffusion model draws\", to show why it converged to a distance map

  Model selector                      C0, C0T, C4, C1, C3, C2, C5 (if run), C6; the champion badged
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Step 5.4 --- Data pipeline (scripts/40_export_case.py)

1.  Load the case\'s masks and maps in baseline space from results/.

2.  Build surfaces with marching cubes (skimage.measure.marching_cubes) at 1 mm spacing.

3.  Smooth with Taubin smoothing, which avoids the shrinkage of plain Laplacian smoothing, then decimate to a face budget: brain ≤ 60k, other layers ≤ 20k each.

4.  Convert voxel indices to millimetres in one shared coordinate frame per case.

5.  Write one GLB file (binary glTF, the standard compact 3D format browsers load) per layer per timepoint with trimesh.

6.  Write downsampled slice volumes (T1ce, FLAIR, probability maps) as compressed NIfTI for the slice panel.

7.  Write case.json (schema v1): case ID, cohort, licence and credit line, timepoints with intervals, layer files with their colour role, per-model scores, and the SHA-256 of every source file.

8.  Check each case against a size target of about 5 MB, and fail the export if it is exceeded.

The app reads every number from case.json and results.json. The page contains no typed-in numbers, so the audit script (Step 4.8) covers it.

### Step 5.5 --- Tech stack

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Layer                   Choice                                                                               Reason
  ----------------------- ------------------------------------------------------------------------------------ ----------------------------------------------------------
  Build                   Vite + TypeScript                                                                    Fast local development; static output

  3D                      three.js with GLTFLoader and OrbitControls                                           Mature, widely documented WebGL library

  Slices                  NiiVue (open-source browser viewer for NIfTI volumes)                                Reads the exported volumes directly and handles overlays

  UI                      React with React Three Fiber if you already know React; otherwise plain TypeScript   Choose once in week 2 and keep it

  Hosting                 GitHub Pages                                                                         Free static hosting from the repository

  Fair                    vite build output served locally; no network needed                                  The booth has no dependence on venue Wi-Fi
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Step 5.6 --- Design rules

- Colour roles fixed across the app, poster and figures: target, FLAIR region, C0 prediction, champion prediction, recurrence covered, recurrence missed. Choose a colour-blind-safe palette and check its contrast in light and dark themes.

- Dark background for the 3D panel, where translucent surfaces read best; light and dark themes elsewhere.

- Keyboard control for rotate, zoom, slider and layer toggles; a one-paragraph text summary of each view for screen readers.

- Target 60 frames per second on the fair laptop; show a loading state for every case. The public site also opens on a phone, so a judge who scans the poster\'s QR code can rotate a case in their hand: keep each case within the 5 MB budget and test on one iPhone and one Android phone.

- The language rules in Section 13 apply to every sentence of app text.

### Step 5.7 --- Build steps

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Week                    Build                                                                                                         Done when
  ----------------------- ------------------------------------------------------------------------------------------------------------- ------------------------------------
  2                       Stack chosen; one LUMIERE case loads as a rotatable brain + target + recurrence                               Runs from vite dev

  3                       Export pipeline for the LUMIERE hero cases; all layers; time slider; slice panel                              Three cases load under 5 MB each

  4                       Story mode with the prediction-then-reveal sequence and placeholder text for the RHUH-GBM steps; Learn page   Story runs end to end

  5                       RHUH-GBM cases and results plugged in; Compare mode                                                           Numbers match the audit script

  6                       Results page; PDE mode; accessibility pass                                                                    Keyboard-only walkthrough works

  7                       Usability check; fixes; offline build; screen recording; deploy to GitHub Pages                               Fair laptop runs it with Wi-Fi off
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Step 5.8 --- Usability check (week 7)

Five people who have not seen the project each get three tasks, with no help: find where the tumour came back in a given case; say which of two models covered more of it; explain what the fixed label means. The check passes when at least four of five complete each task. Fix whatever fails, then re-test that task with one new person.

## 10. Phase 6 --- Poster, report, pitches, booth (weeks 6--8)

Every deliverable tells the same story in the same order: the question, the tournament, the unseen-cohort test, the champion, and what that means. Every number is read from results/ and checked by the audit script.

**Step 6.1 --- Poster,** in the Z layout used for gbm-4d.

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Panel (reading order)                 Content
  ------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Title band                            Chosen after the verdict from two versions written in week 2. *Challenger champion:* \"Where does glioblastoma come back? \[Champion\] finds more of it than distance, on patients it has never seen.\" *Distance holds:* \"Where does glioblastoma come back? Eight models, one sealed test: distance still wins.\"

  The question                          \"Clinical guidelines treat a 15 mm margin. Can MRI-based models predict recurrence more precisely at the same treated volume?\" plus the guideline\'s FLAIR sentence, quoted

  Prediction vs reality (centrepiece)   The four-panel figure from Step 4.7, largest panel on the poster

  The truth test                        The protocol strip with its dates

  Hero metric                           Matched-volume coverage bars: guideline margin, distance, champion

  Does more complexity help?            The complexity forest plot, with the filled result sentence and achieved power beneath it

  Where distance fails                  The descriptive escape panel, small

  Fourth dimension                      The later-progression result and one LUMIERE patient\'s time series, labelled exploratory

  Context                               The Step 0.6 literature table, reduced to counts

  Why                                   Mechanisms from gbm-4d (recurrence is local; the PDE became a distance map) plus what this study adds

  Limits                                Registration gate not discharged; cohort size; label and preprocessing shift between cohorts

  QR codes                              The app (opens on a phone at the Story screen), OSF pre-registration, GitHub repository
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Step 6.2 --- Written report,** using TRIPOD+AI (the reporting checklist for studies that develop or validate AI prediction models) as the section template. Then run a PROBAST+AI self-audit (a tool for judging a prediction-model study\'s risk of bias) across its four domains, and include the result as an appendix.

**Step 6.3 --- Pitches.** Write each one after the verdict:

- 30 s: the 15 mm margin; the question \"Can MRI-based models beat distance?\"; eight models, one sealed test on 40 new patients; the answer and the matched-volume coverage number.

- 3 min: the 30 s pitch, then the Story screen on the laptop (prediction, reveal, score), the truth-test strip, and the complexity forest plot.

- 10 min: the 3 min pitch, then the mechanisms, the escape panel, the limits, and a live Explore session on a case the judge picks, reveal toggle included.

**Step 6.4 --- Judge Q&A, prepared answers.**

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Likely question                                          Where the answer comes from
  -------------------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  What is new here?                                        The \"first minute\" list in Section 1 and the literature table

  Why this patient for the demo?                           The rule fixed before the run: the median per-patient Δ. The failure and best cases are one click away

  Why not just use deep learning?                          C5\'s point on the complexity forest plot, its starting-weights decision, or the recorded hardware decision if C5 was dropped

  Did you try combining models?                            C6\'s point on the forest plot and the reason weighted stacking was left out

  How do you know you didn\'t tune on the test data?       OSF timestamp before the download date; the model hashes; the script that refuses to run twice

  Is 40 patients enough?                                   The power table, the achieved MDE, and why the bound is informative on its own

  Why RHUH-GBM and not a bigger dataset?                   The dataset survey: it is the open cohort with an early post-op scan, a recurrence scan and corrected labels that no earlier study here used

  Why is PR-AUC the primary if coverage is the headline?   PR-AUC scores the ranking at every volume and has more power; coverage at one volume is the clinical translation. The pre-registered sentence covers the case where they disagree

  How accurate is your registration?                       The three screens, the synthetic recovery test, and the stated limitation

  What does this mean for the 15 mm margin?                Matched-volume coverage; the result is evidence about where recurrence occurs, and margin recommendations remain the guideline\'s

  Could a doctor use this?                                 The fixed label; the expert review (Step 6.6) and what the reviewers said

  What did gbm-4d find, and why continue?                  Section 2 and Form 7\'s expansion statement

  What would you do with more time or data?                Diffusion and perfusion imaging, which see infiltration directly; the research toolkit (Section 14)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Step 6.6 --- Expert review (week 6).** Run the hour arranged in Step 0.7: walk the reviewer through the Story mode and the poster draft, and ask three questions. Is the clinical framing accurate? Is anything overstated? Which figure would a clinician look at first? Record their comments and your changes in docs/expert_review.md, and quote them on the poster only with their written permission.

**Step 6.7 --- Mock judging (week 8).** Three adults who have not seen the project each get the 3-minute pitch and ten minutes of questions, scored on your fair\'s judging rubric. Rewrite whatever loses points, then repeat once.

**Step 6.5 --- Booth kit.**

- [ ] Fair laptop with the offline app build, tested with Wi-Fi off

- [ ] Screen recording of the Story mode on loop (fallback 1)

- [ ] Printed panels of the 3D reveal and the Compare view (fallback 2)

- [ ] Form 7 displayed, as ISEF requires for continuation projects

- [ ] Printed copies of the OSF pre-registration and the CONSORT table

- [ ] Credits for LUMIERE and RHUH-GBM on the poster and in the app

**Gate:** poster file final; report drafted; expert review recorded; all three pitches rehearsed and mock-judged; booth kit packed.

## 11. Week-by-week timeline

The research track and the app track run side by side. The only hard dependency between them is in week 5, when RHUH-GBM results enter the app. The fair date is not recorded yet: the plan assumes the week of November 16, and every row moves with it.

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Week           Dates (2026)    Research track                                                                                                                     App and communication track                                                           Gate at the end of the week
  -------------- --------------- ---------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------- -----------------------------------------------------------
  1              Sep 22--28      Forms signed (first days); SRC email; repository; tournament rules committed; features f1--f5 on LUMIERE; C0, C0T, C4, C1 scored   Literature search string committed; expert emails sent; CNN starting-weights search   Forms signed; CNN decisions recorded

  2              Sep 29--Oct 5   C3, C2, C5, C6 scored; later-progression analysis; challenger named; refit and hash; power simulation; OSF posted                  Stack chosen; first rotatable LUMIERE case; both title bands written                  OSF timestamp in place

  3              Oct 6--12       RHUH-GBM download; inventory and label checks; label transport; screens; cohort lock                                               Export pipeline; layers; time slider; slice panel                                     Cohort locked, achieved MDE recorded

  4              Oct 13--19      Features on RHUH-GBM; hash check; **primary run, once**; verdict                                                                   Story mode with placeholders; Learn page; literature extraction                       Verdict sentence filled

  5              Oct 20--26      Secondaries, sensitivity, exploratory; map export; all figures; audit script                                                       RHUH-GBM cases and results in; Compare mode; literature table finished                Audit script passes

  6              Oct 27--Nov 2   Poster draft; report outline                                                                                                       Expert review; Results page; PDE mode; accessibility pass                             Poster draft reviewed by the adult sponsor and the expert

  7              Nov 3--9        Poster final; report draft; PROBAST+AI audit                                                                                       Usability check; fixes; phone test; offline build; recording; deploy                  App runs with Wi-Fi off and on a phone

  8              Nov 10--16      Pitches written and rehearsed                                                                                                      Mock judging twice; booth kit                                                         Fair-ready
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Daily rhythm that kept gbm-4d on schedule:** one numbered script per stage, run in the foreground with per-unit checkpoints; HANDOFF.md regenerated on every commit; every reportable number produced from a clean tree.

## 12. Risks, fallbacks and cut order

The two most likely problems are a small evaluable cohort and a challenger that adds nothing. The plan absorbs both: each has a pre-named outcome and a finished poster.

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Risk                                                                                                    Likelihood        Effect                                             Fallback
  ------------------------------------------------------------------------------------------------------- ----------------- -------------------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------
  Registration screens and the undefined-patient rule leave fewer than \~20 evaluable RHUH-GBM patients   High              Wide CI; outcome likely *inconclusive*             Report the achieved MDE; the unscreened arm beside the primary; the tournament and the app still stand

  No challenger beats distance                                                                            High              Distance is champion                               Pre-written *distance holds* sentence and title band; the complexity forest plot and the matched-volume coverage carry the headline

  RHUH-GBM has no cavity label                                                                            Medium            Target definition differs from LUMIERE             Pre-registered cavity rule (Step 2.2); alternative definition as a sensitivity analysis

  The FLAIR label differs between cohorts (automated in LUMIERE, hand-corrected in RHUH-GBM)              Medium            Shift that works against FLAIR-based challengers   Recorded as a known shift; a positive result survives it

  The registration recipe differs from the one that made the LUMIERE labels                               Medium            Label shift between cohorts                        Both recipes run on RHUH-GBM; the difference is reported

  No GPU for C5                                                                                           Medium            One candidate fewer                                Recorded in week 1; the Q&A answer cites the decision

  C5 overfits                                                                                             High if run       C5 ranks low                                       Reported as measured; it answers the deep-learning question

  The app slips                                                                                           Medium            Weaker booth                                       Cut order below; the screen recording and printed panels need only the week 5 figures

  WebGL fails on the fair hardware                                                                        Low               No live demo                                       Recording on loop, then printed panels

  The SRC asks for more paperwork                                                                         Low               Delay to week 1                                    Email the SRC on day 1; start once the forms are signed unless the SRC says the project needs pre-approval

  16 GB RAM limits label transport and C5                                                                 Medium            Slow runs                                          gbm-4d\'s per-unit checkpointing; foreground runs; the second machine for heavy jobs
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Cut order if time runs short,** first to last:

1.  C1 + ADC exploratory

2.  PDE mode in the app

3.  Results page in the app (the poster already carries it)

4.  C5

5.  Compare mode (Story step 5 shows the same comparison)

These are never cut: forms, the tournament on C0, C0T, C4, C1, C3, C2 and C6, OSF pre-registration, the single primary run, Story and Explore modes, the poster, the pitches.

## 13. Project rules

These rules carry over from gbm-4d and gbm-wm. Code enforces them wherever it can, so they hold without relying on memory under deadline pressure.

**Reproducibility contract**

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Rule                                          How it is enforced
  --------------------------------------------- -----------------------------------------------------------------------------------------------------------------------
  Seeds are stable across machines and runs     stable_seed from BLAKE2b, never Python\'s hash(); tested under differing PYTHONHASHSEED

  Every estimator is seeded                     Explicit random_state everywhere; PyTorch seeds and deterministic flags for C5

  Every result is traceable                     run_manifest on every file in results/: git commit, dirty flag, SHA-256 of inputs and outputs, seeds, package list

  Reportable numbers come from committed code   require_clean_tree() refuses to write them from an uncommitted tree

  Constants are set before they are used        Undecided constants are null in config.yaml and raise on read

  Every decision and departure is on record     decisions.md (D-numbers) before the number it governs is seen; deviations.md (V-numbers) with the direction of effect

  The confirmatory run happens once             Irreversible marker file; the script refuses a second run

  Models are the ones pre-registered            Hash check before scoring

  Documents quote only real numbers             The audit script checks poster, report and app text against results/

  Data stays out of git                         .gitignore covers the data folders; the resolved data path goes into every manifest
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Language rules for every deliverable**

- \"Champion\" is used only as the Step 2.3 verdict rule assigns it.

- \"Improves on distance\" appears only if the verdict names a challenger champion, and always with its Δ and CI.

- \"Predicts\" always means ranking voxels in retrospective public data. A clinical prediction for a patient is never implied.

- LUMIERE cross-validation numbers are called *development* results; only RHUH-GBM numbers are called *validation*.

- The gbm-wm FLAIR result is always tagged as an association on 9--13 patients.

- No registration screen number is quoted as a TRE (target registration error, the distance between matching anatomical landmarks after registration). TRE was never measured.

- Coverage is always stated with its denominator: all recurrence, or recurrence outside the target.

- The fixed label appears on the poster, in the app, and in the report\'s first paragraph.

- Results are stated as evidence about where recurrence occurs. Margin recommendations remain the guideline\'s, and the guideline is quoted in its own words.

- The central question is written \"Can MRI-based models beat distance?\". \"AI\" is reserved for C5 and C6; C4 is a rule and C1 and C3 are logistic regressions.

## 14. Later: the research toolkit

The toolkit turns this project\'s protocol into software that any imaging group can run on its own cohort: ingest, transport labels, screen registrations, run a tournament, freeze, confirm once, and view the results in 3D on their own machine. It starts after the fair. Three choices made now keep it cheap to build: a versioned case schema, a pipeline that accepts any labelled cohort, and a viewer with no cohort-specific code.

### 14.1 Product definition

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Item                                Decision
  ----------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Working name                        gbmc (from the fair repository\'s library)

  Users                               Imaging researchers and students studying where glioblastoma recurs; clinician-researchers working with their institution\'s de-identified data

  Job                                 Run a recurrence-location study with the discipline built in: distance baseline always included, splits frozen, one-shot confirmation, every number traceable

  What sets it apart                  A distance baseline, a sealed confirmation set and a pre-registration draft are built into every study the toolkit runs, and the commands refuse to proceed without them

  Runs where                          Entirely on the researcher\'s machine. No upload, no server, no network calls during analysis

  Fixed label                         \"For research use only. Not for clinical use.\" on every report, export and viewer screen

  Licence                             Open source (Apache-2.0 or MIT for the code); each cohort keeps its own data licence
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 14.2 Components

flowchart LR\
A\[Cohort folder\<br/\>NIfTI + labels\] \--\> B\[gbmc library\<br/\>Python\]\
B \--\> C\[gbmc CLI\<br/\>one command per stage\]\
C \--\> D\[results/ + manifests\]\
D \--\> E\[Report\<br/\>TRIPOD+AI layout\]\
D \--\> F\[Export\<br/\>GLB + case.json\]\
F \--\> G\[Local viewer\<br/\>same app, local mode\]

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Component               Contents                                                                                                                                                                                                                             Source in the fair project
  ----------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------
  Library gbmc            Cohort adapters, label transport, registration screens, features f1--f5, model interface, scoring (per-patient PR-AUC, coverage), statistics (bootstrap, BH-FDR, MDE simulation), gbm-wm\'s rotational spatial null, run manifests   src/gbmc, gbm-4d library, gbm-wm scripts

  Command-line tool       One command per protocol stage (14.3)                                                                                                                                                                                                Numbered scripts, wrapped

  Report generator        HTML and PDF: CONSORT table, leaderboard, forest plot, coverage curve, deviations list, all in the TRIPOD+AI section order                                                                                                           Phase 4 figures and audit script

  Local viewer            The explainer app with a local mode that opens exported cases from disk                                                                                                                                                              Phase 5 app
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 14.3 Workflow the command-line tool enforces

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Command                   Does                                                                                                               Refuses when
  ------------------------- ------------------------------------------------------------------------------------------------------------------ ----------------------------------------------------------------
  gbmc init                 Creates a study folder, config.yaml with every constant set to null, and empty decision and deviation registers    The folder is not a clean git repository

  gbmc ingest               Reads the cohort through an adapter; checks sequences, labels and timepoints; writes the inventory                 A label mapping has not been declared

  gbmc transport            Carries recurrence labels into baseline space                                                                      The registration recipe is unset

  gbmc screen               Round-trip, brain-surface and multi-arbiter screens; synthetic recovery test                                       Screen thresholds are unset

  gbmc lock                 Freezes the cohort, draws the development/confirmation split by seeded permutation, writes CONSORT                 The cohort has already been locked

  gbmc tournament           Scores every registered model on shared patient-grouped folds of the development set; applies the selection rule   Any confirmation-set ID reaches it

  gbmc freeze               Refits, hashes weights, writes a pre-registration draft for OSF                                                    The selection rule or result sentences are missing

  gbmc confirm              Scores challenger vs distance on the confirmation set, once                                                        It has run before, or a model hash differs from the frozen one

  gbmc report               Builds the report; checks every quoted number against results/                                                     Any number fails the check

  gbmc export / gbmc view   Writes GLB meshes and case.json; opens the local viewer on localhost                                               ---
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 14.4 Extension points

- **Cohort adapters.** BraTS layout (the label layout of the Brain Tumor Segmentation challenge) first, then LUMIERE, RHUH-GBM and MU-Glioma-Post. An adapter maps a dataset\'s files and label values to the toolkit\'s cohort schema and nothing else.

- **Models.** Any class with fit(training_cases) and predict(case), where predict returns a probability map in baseline space. Registered by name; its weights are hashed at freeze. C0--C6 ship as built-ins, and distance is always included as the comparator.

- **Features.** A registry of voxel features with their definitions, so a new feature is declared once and becomes available to every model.

### 14.5 Validation before release

- [ ] Reproduce gbm-4d\'s frozen Arm 0 result (mean PR-AUC 0.0743 on LOPO) from the same inputs, within a stated tolerance

- [ ] Reproduce gbm-wm\'s primary estimate (log odds ratio −0.065) from its frozen inputs

- [ ] Reproduce this project\'s RHUH-GBM primary from its frozen inputs

- [ ] Run end to end on three cohorts: LUMIERE, RHUH-GBM, MU-Glioma-Post

- [ ] Unit tests for every refusal in 14.3, including the confirmation script\'s second-run refusal

- [ ] Cross-machine determinism check, as in gbm-4d (identical outputs on two machines)

### 14.6 Release

1.  Package gbmc for installation with pip; pin dependencies with a lockfile.

2.  Documentation site: quick start on a public cohort, a protocol guide, the adapter and model interfaces, and a worked example that reproduces this project.

3.  GitHub release archived on Zenodo for a DOI (a permanent citable identifier).

4.  A candidate venue is a software journal such as the Journal of Open Source Software. Check its current submission criteria before writing.

### 14.7 Build order after the fair

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Stage                   Work                                                             Done when
  ----------------------- ---------------------------------------------------------------- ---------------------------------------------------------------------------
  T1                      Extract src/gbmc into an installable library; cohort schema v1   The fair analysis re-runs through the library with identical results

  T2                      BraTS adapter; MU-Glioma-Post adapter                            Ingest and transport run on MU-Glioma-Post

  T3                      Command-line tool with every refusal in 14.3                     Refusal tests pass

  T4                      Report generator; viewer local mode                              A full study on a public cohort produces a report and opens in the viewer

  T5                      Validation checklist (14.5); documentation; release              Checklist complete; DOI issued
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Privacy boundary.** Public, de-identified cohorts need nothing beyond their licences. Institutional patient data brings privacy obligations (HIPAA in the US) and an institution\'s own approvals. The toolkit\'s local-only design keeps that data on the institution\'s machines, and those approvals remain the institution\'s responsibility.

## Sources

- gbm-4d Project Write-up

- Margin-escape glioblastoma recurrence and white matter: a pre-registered external test

- gbm-4d follow-on: fresh-cohort test and 4D viewer, the earlier outline this plan expands

- [RHUH-GBM, TCIA collection page](https://www.cancerimagingarchive.net/collection/rhuh-gbm/) (doi:10.7937/4545-c905)

- [Cepeda et al. 2023, RHUH-GBM dataset paper](https://arxiv.org/pdf/2305.00005) (Data in Brief 50, 109617)

- [ISEF Human Participants rules](https://www.societyforscience.org/isef/international-rules/human-participants/)

- [ISEF Rules for All Projects](https://www.societyforscience.org/isef/international-rules/rules-for-all-projects/)

- [ISEF 2026--27 rule changes](https://sspcdn.blob.core.windows.net/files/Documents/SEP/ISEF/2027/Rules/FINAL-Rule-Modifications-for-2026-27.pdf)

- [ESTRO-EANO guideline on target delineation and radiotherapy details for glioblastoma, Radiotherapy and Oncology 2023](https://www.sciencedirect.com/science/article/pii/S0167814023002013)
