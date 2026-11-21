# Team plan: who does what, in which order (2026-11-19)

Designed for few handoffs: the Mac (user) does all data work and every reportable number (D6 item 9), and the partner works in parallel on what needs no data, plus the one GPU job. There are three handoffs. Sources: `DECISIONS.md` D6, D17–D19; `docs/preregistration.md`; `docs/future_cohorts.md`.

**Recommendation that keeps it this short:** make the H2 final model a non-GPU model (feature-based logistic regression or LightGBM, new features, ensembles). Then the MU-Glioma-Post test runs on the Mac with no extra handoff. A GPU model would need the partner for training and for MU scoring.

## Round 1: in parallel, now until the registration is posted

**User (Mac):**
1. Run `03_` (refit and hash), `04_` (power), `07_ --summary` (round-trip threshold) and `06_` (later progressions). Commit and push.
2. Approve or change D9–D15. D12 and D13 belong in the registration.
3. **Build the H2 final model on LUMIERE only** (D19). Log every change; freeze it; record its SHA-256.
4. Write MU-Glioma-Post's cohort rules into `docs/preregistration.md` section 8: eligibility, post-treatment baseline, label mapping, exclusions, power at about 44 patients.
5. Fill every remaining blank, and **post the registration** (OSF Registries; fallback AsPredicted or Zenodo).
6. After the timestamp: download RHUH-GBM and MU-Glioma-Post. Run `20_`–`23_` (RHUH-GBM) and the MU cohort lock (to build). Run `24_` and put the RHUH-GBM C5 inputs on Drive. **→ Handoff 1**

**Partner (no data needed):**
- Build the app (Node.js, which the Mac lacks): screens, wording, offline build.
- Draft the poster and the report: methods, the tournament result, and the result-sentence templates for the primary and H2.
- Review the registration draft, section 8 especially, before it is posted.
- Phase 0 leftovers: the literature search string (Step 0.6) and the expert emails (Step 0.7).
- Do not download any RHUH-GBM or MU-Glioma-Post image.

## Round 2: partner, one GPU job

7. Pull the repo and run `14_c5_score_rhuh.py` on an L4, at a commit whose `src/gbmc/c5.py` matches the folds'. Put `data/interim/rhuh/c5_maps/` on Drive. About an hour. **→ Handoff 2**

## Round 3: user, the tests, then the fair entry

8. Run `30_` once: the primary (C1 vs C0) and H2 on RHUH-GBM. Then `31_` (sensitivity).
9. Run the MU-Glioma-Post test of H2, once (script to build, with its own run-once marker).
10. Run `50_`/`51_` (figures) and `40_` (app cases with results), and put the cases on Drive. **→ Handoff 3**
11. Partner: drop the cases and numbers into the app and poster. Done.

## Code (built 2026-11-21)
- H2 scoring inside `30_`: E1 is scored next to the candidates, outside the primary's BH-FDR family.
- The MU-Glioma-Post pipeline:
  - `25_mu_eligibility` (metadata only: 44 eligible);
  - `26_mu_transport` and `27_mu_screens` (the RHUH-GBM recipe and screens);
  - `28_mu_lock`;
  - `32_mu_confirm`, the run-once H2b test.

## Guard rails
- RHUH-GBM and MU-Glioma-Post stay unopened until the registration is posted. `tests/test_config.py` enforces both.
- The H2 model is built on LUMIERE only, and its LUMIERE numbers are development numbers (V6).
- Every reported number is computed on the Mac from a clean commit.
