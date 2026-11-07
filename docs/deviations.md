# Deviations (V-numbers)

Every departure from the plan (`docs/plan/final_plan.md`) or from gbm-4d's protocol is recorded here with its direction of effect, before the number it affects is seen where possible.

## V1: LUMIERE is development data only

Date: 2026-09-23
Source: plan Step 1.2

Departure:
gbm-4d kept a sealed LUMIERE test set (n = 11). That set has already been opened, so in this project all 63 locked LUMIERE patients are development data.

Direction of effect:
None on the confirmatory result, which comes only from RHUH-GBM. LUMIERE numbers are reported as *development* results, never as *validation*.

## V3: Standalone code replaces gbm-4d; rigid resampling direction corrected

Date: 2026-11-03
Source: plan Steps 0.2, 1.2, 1.3, 1.8, 2.2 and 3.3; user decision

Departure:
1. The plan pins and imports gbm-4d at commit `eeba777`. That commit was never pushed and the computer holding it is no longer available, so the code is rewritten in `gbmc` (`rng`, `imaging`, `transport`, `anatomy`, `pde`, and functions in `metrics`, `candidates`, `tournament`, `power`, `lumiere`, `provenance`). The hyperparameters, constants and seed formula are unchanged. The `gbm-4d` scripts `03_`, `04_` and `23_` that `05_` and `42_` called are replaced by `gbmc` code.
2. Later-progression transport (`05_`) now runs on new code with new seeds. The one earlier unit (Patient-002 week-047) needs redoing.
3. The rigid resampling direction is corrected in new code. The registration transform maps fixed points to moving points, but gbm-4d's preprocessing and this project's `registration.transport_pair` and `22_` passed its inverse to `Resample`. On a synthetic known shift the inverse raised the error (RMSE 0.35, against 0.25 unaligned and 0.001 corrected).

Direction of effect:
- Items 1 and 2: on synthetic inputs the new code matched the old exactly (seeds, falx, ventricles, geodesic distance, N4, rigid, round trip, Jacobian, PR-AUC, logistic fit) and the PDE matched to within 3.5e-18 over the full 206-day run. LightGBM could not be run on the test Mac. Not re-checked on real scans.
- Item 3, RHUH-GBM: expected to improve label transport, which has not been run.
- Item 3, LUMIERE features: resolved on 2026-11-06. All 63 baselines were rebuilt from raw LUMIERE (`Imaging-v202211.zip`, MD5 verified) with the corrected direction and the features recomputed (`results/feature_recompute/`). The evaluation masks and labels are kept from the earlier caches and are identical for all 63, so 53 patients stay evaluable. f1 to f4 are unchanged. f5 and C0T changed in all 63: mean absolute change per patient averages 2.72 mm (f5) and 2.99 mm (C0T), up to 7.56 mm in one patient. The falx voxel count changed by a median of 10.5% (absolute), up to +115%. Matched 15 mm voxels changed by a median of 0.04%, up to 2.9%. Re-registering rebuilt T1 to T1c on the 8 most-shifted patients left at most 0.26 mm of offset. The direction of the change on any score is unknown, because nothing has been scored. Earlier caches stay in `data/interim/features_v1/` (not in git).
- Item 3, not attributable to T1: ventricle volume changed in 8 patients (up to 20,939 mm3). Ventricles use T1c and the brain mask, not T1, so this looks like a difference between computers, unproven. Those 8 patients' f5 and C0T change mixes both causes.
- Item 3, first-progression labels: whether gbm-4d's `04_` label transport had the same fault cannot be checked, because that script is lost; its recorded brain-surface screen (median 1.21 mm) suggests the labels are usable.
- Computer differences: 15 of 63 rebuilt brain masks differ from the originals, mostly near scan edges, by up to 1.68% of voxels. Keeping the original masks removes this from the tournament inputs, and reportable numbers should come from one computer.
- No score has been seen, so none of this was chosen in light of a result.

## V4: New ventricle segmentation rule for f5 and C0T

Date: 2026-11-07
Source: plan Steps 1.3 (f5 "gbm-4d Block B"), 1.4 (C0T) and 4.1 ("derive the falx and ventricle barriers the same way gbm-4d did"); user decision on 2026-11-07 ("Fix for all patients"); `DECISIONS.md` D16

Departure:
gbm-4d's ventricle segmentation (Otsu on T1c in the brain eroded by 8 mm) is replaced for every patient, on LUMIERE and RHUH-GBM alike, by rule V4 in `gbmc.anatomy.segment_ventricles`: a midpoint threshold between each scan's CSF and tissue levels, a 15 mm depth limit, a 1 mm core test and 10 mm of growth from the core (D16). The old rule stays as `segment_ventricles_otsu` for comparison only. The falx is unchanged.

Why:
The f5 cap survey (`results/f5_cap_survey/`, anatomy only) found the old masks unreliable: Patient-002's covered 52.8% of the brain (771 ml) and passed QC, and in 27 of 62 masks fewer than half the voxels passed a rough T2/FLAIR CSF screen. Overlays showed masks spreading into deep tissue.

How the rule was chosen:
Nine variants were tried on the 63 LUMIERE baselines and judged on volumes, the T2/FLAIR screen and overlays, by eye, by Claude. No recurrence label, outcome or score was read at any point. The rule favours specificity: missing some ventricle shortens a path slightly, while marking tissue as barrier distorts distances and trims the margin. It is frozen before any candidate is scored and is applied unchanged to RHUH-GBM.

Direction of effect:
- Masks: all 63 pass QC (the old rule failed Patient-074), with volumes of 6.2 to 61.7 ml (median 23.9); the old rule gave 0 to 771 ml (median 49). On the rough screen, which uses T2 and FLAIR that the rule does not, a median of 82% of mask voxels look like CSF (old 63%), and 2 masks are under half (old 27).
- Known weaknesses: thin ventricles are under-segmented (Patient-028, 6 ml), and some deep sulci remain (Patient-018). The overlay review was at thumbnail size, by one reviewer.
- On f5, C0T and the matched volume: see the recompute results below. Direction on any score: unknown, nothing scored.
- RHUH-GBM: same rule, T1c only, so it needs no sequence the plan does not already use. Untested there (no download).

Recompute (2026-11-07; `results/feature_recompute_v4/`, `results/f5_cap_survey_v4/`):
- `01_` was rerun for all 63 patients, keeping the original evaluation masks and labels (`--keep-from features_v1`). Masks, labels and f1–f4 are identical to the V3 caches. The V3 caches are kept in `data/interim/features_v3/` (not in git).
- f5: the mean absolute change per patient has a median of 3.23 mm (mean 7.55). It is largest in Patient-002 (139.2 mm), where half the brain had been barrier, then about 16 mm in Patients 008, 022 and 011. A median of 39% of voxels changed by more than 0.01 mm.
- C0T: the mean absolute change per patient has a median of 0.84 mm (mean 2.54). It is largest in Patient-002 (101.4 mm); every other patient is at or under 2.8 mm.
- Matched volume:
  - 15 mm margin: median change 0.0%, range −1.2% (Patient-072) to +3.8% (Patient-002).
  - 20 mm margin: median change 0.0%, range −1.3% to +16.2% (Patient-002).
- Capped share of evaluation voxels: median 3.5%, at most 6.0% (V3: 5.4%, up to 55%). The tissue that is truly unreached is at most 0.17 ml.
- No score has been seen, so none of this was chosen in light of a result.

## V2: No usability check with other people (plan Step 5.8 dropped)

Date: 2026-09-29
Source: plan Step 5.8; user decision

Departure:
The plan's five-person usability check is removed. ISEF exempts testing of a student-designed computer application from IRB review only when the student is the only tester; with other testers it needs IRB pre-approval (Form 4). The app is tested only by the student, against the Phase 5 engineering criteria (size budget, offline build, phone test, keyboard navigation, number audit).

Direction of effect:
None on any study result. The app's usability is unmeasured, so no claim about how well others can use it is made. The Phase 5 gate becomes "runs offline on the fair laptop; engineering criteria pass".
