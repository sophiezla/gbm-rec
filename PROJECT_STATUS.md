# Project Status: gbm-rec ("Where Does Glioblastoma Come Back?")

_Last updated: 2026-09-23_

## Current Objective
Carry out the final plan (`docs/plan/final_plan.md`, dated 2026-09-22) as written. The current phase is **Phase 0: Setup and paperwork (week 1, Sep 22–28)**.

## Phase Gates (plan §3)
| Phase | Weeks | Gate | State |
|---|---|---|---|
| 0 Setup + paperwork | 1 | ISEF forms signed; SRC contacted; repo scaffolded, builds, tests pass; CNN hardware + starting-weights decisions recorded; literature search string committed; expert emails sent | **Open.** Repo, build and tests done |
| 1 Tournament on LUMIERE | 1–2 | All candidates scored on identical folds; challenger named by rule; weights hashed | Not started |
| 2 Freeze + OSF | 2 | OSF timestamp precedes the RHUH-GBM download date | Not started |
| 3 RHUH-GBM cohort lock | 3 | Cohort locked; CONSORT committed; achieved MDE recorded; no model score computed | Not started |
| 4 Confirmatory run, once | 4–5 | Verdict sentence filled; figures from results/; audit passes | Not started |
| 5 App | 2–7 | Runs offline; five-person usability check passed | Not started |
| 6 Communication | 6–8 | Poster final; report drafted; expert review; mock judging; booth kit | Not started |

## Phase 0 Checklist
- [ ] 0.1 ISEF forms 1, 1A, Research Plan/Addendum, 1B, 2A, 7; SRC email sent
- [x] 0.2 Repository scaffolded; gbm4d pinned at `eeba777` (see Blockers)
- [x] 0.3 Environment: Python 3.13.14, gbm-4d lock + scikit-image + trimesh; ITK threads pinned to 1
- [ ] 0.4 CNN gates: hardware and starting weights (facts so far below)
- [x] 0.5 Reproducibility contract carried over (`gbmc.rng`, `gbmc.config`, `gbmc.provenance`, tests)
- [ ] 0.6 PubMed search string and extraction form committed
- [ ] 0.7 Expert outreach emails sent

### Step 0.4 facts (no decision yet)
- This laptop: Intel Arc Graphics, about 2 GB adapter memory, no CUDA. It does not meet the ≥ 8 GB CUDA requirement.
- Second machine: not checked yet. Its GPU and memory decide whether C5 runs.
- Pretrained 3D brain-tumour weights with a reuse licence: not searched yet.

## Completed
- 2026-09-23: Documentation scaffold; the plan is stored in `docs/plan/`.
- 2026-09-23: Steps 0.2, 0.3 and 0.5. `pytest`: 18 passed. `ruff`: clean.

## Next Steps
1. Step 0.1: sign the forms and email the SRC. **No model training and no data download before this.**
2. Step 0.4: check the second machine's GPU; search for pretrained weights; record both in DECISIONS.md by Sep 28.
3. Step 0.6: commit the PubMed search string and extraction form before reading.
4. Step 0.7: send the expert emails.

## Known Issues
- gbm-4d commit `eeba777` is not on GitHub (local gbm-4d is 48 commits ahead of `origin/main`). A fresh clone of this repo cannot install gbm4d until gbm-4d is pushed.

## Blockers
- ISEF forms not signed (blocks Phase 1).

## Items Requiring Approval
- Pushing gbm-4d, so the `eeba777` pin resolves from GitHub.
- The fair date. The plan assumes the week of Nov 16 but says the date is not recorded yet.

## Recent Changes
See `CHANGELOG.md`.
