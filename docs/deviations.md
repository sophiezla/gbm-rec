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
- Item 3: expected to improve RHUH-GBM label transport, which has not been run. Effect on existing data is NOT resolved: the 63 cached LUMIERE features came from gbm-4d's preprocessing, where only the T1 falx test used a moving sequence, so f5 and C0T could be affected; f1 to f4 use the native segmentation and T1c. A re-registration on 6 preprocessed Patient-002 sequences left 0.19 to 3.87 mm of offset, which does not settle it. Whether gbm-4d's `04_` label transport had the same fault cannot be checked, because that script is lost; its recorded brain-surface screen (median 1.21 mm) suggests the labels are usable.
- No score has been seen, so none of this was chosen in light of a result.

## V2: No usability check with other people (plan Step 5.8 dropped)

Date: 2026-09-29
Source: plan Step 5.8; user decision

Departure:
The plan's five-person usability check is removed. ISEF exempts testing of a student-designed computer application from IRB review only when the student is the only tester; with other testers it needs IRB pre-approval (Form 4). The app is tested only by the student, against the Phase 5 engineering criteria (size budget, offline build, phone test, keyboard navigation, number audit).

Direction of effect:
None on any study result. The app's usability is unmeasured, so no claim about how well others can use it is made. The Phase 5 gate becomes "runs offline on the fair laptop; engineering criteria pass".
