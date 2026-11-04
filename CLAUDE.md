# CLAUDE.md: gbm-rec ("Where Does Glioblastoma Come Back? Can MRI-based models beat distance?")

These are standing instructions for every Claude session. The user gave them on 2026-09-23.

## The plan is the law
`docs/plan/final_plan.docx` is **the ultimate plan to stick to** (the user's words). `docs/plan/final_plan.md` is a readable conversion; if the two disagree, the .docx wins. Anything that departs from the plan is a deviation: log it with its direction of effect and ask the user first when it touches methodology.

### Hard rules from the plan (never break them)
- **No model training and no data downloads before the ISEF forms are signed** (Step 0.1).
- **Do not download RHUH-GBM before the OSF pre-registration is posted.** The download date is committed after the OSF timestamp.
- **The confirmatory run happens once.** An irreversible marker file blocks any rerun. Nothing is tuned, reselected or rerun on RHUH-GBM.
- **Write tournament rules into DECISIONS.md before the first candidate is scored** (Step 1.1). Any change after a score has been seen is a deviation.
- **This project is standalone** (deviation V3). It does not import or install gbm-4d or gbm-wm; their write-ups are only sources for constants, findings and the first-progression labels already in the feature caches. Re-fit nothing from them.
- **LUMIERE is development data only** (deviation V1). Only RHUH-GBM numbers may be called *validation*.
- Patients with no enhancing recurrence outside the target are *undefined*, never zero.
- Reproducibility contract (plan §13): BLAKE2b `stable_seed`; explicit seeds everywhere; a `run_manifest` on every result; `require_clean_tree()` before any reportable number; undecided constants are null and raise when read; model hash check before scoring; number-audit script; data never goes into git.
- Language rules (plan §13):
  - "Champion" only as the verdict rule assigns it.
  - "Predicts" means ranking voxels in retrospective public data.
  - Coverage is always stated with its denominator.
  - "AI" only for C5/C6.
  - No registration number is ever called a TRE.
  - The gbm-wm FLAIR result is always tagged as an association on 9–13 patients.
  - Fixed label: "Research visualization of a pre-registered method comparison on public data. Not a treatment prediction."

## Session start
1. `git pull`, then read `HANDOFF.md`, then `PROJECT_STATUS.md`. Two people share this repo, so never start from stale work.
2. Read the plan section for the current phase, and `DECISIONS.md` before any methodological change.
3. Read the relevant source files. Continue from the existing state.

## Session end (after substantial work)
Verify and run checks; update `CHANGELOG.md`, `PROJECT_STATUS.md`, `HANDOFF.md`, and `DECISIONS.md` / `docs/deviations.md` as needed; name the remaining issues and the single next step.

## General rules
- **Never invent project facts**: datasets, results, metrics, API behavior, test outcomes, requirements, decisions, sources, or commands that were not actually run. Write `TODO: verify` or `UNKNOWN` instead. Keep confirmed facts, assumptions, proposals and unverified information visibly separate.
- **Never claim code works without validation.** Report `Validation performed / Result: PASS|FAIL / Notes`, or `Not tested: [reason]`.
- Prefer small, testable changes. Add no dependencies beyond those the plan names without approval. Modify existing work rather than replacing it.
- Documentation must match the code in the same session.
- Never silently change observations, labels, splits, outliers, preprocessing, or evaluation criteria.
- If a request conflicts with the plan or a recorded decision, flag the conflict; do not overwrite silently.

## Ask the user before
Changing the question, cohorts, candidates, features, metrics, the selection or verdict rules, or the timeline; adding major dependencies; removing functionality; discarding results; large refactors; **pushing to the public remote**.

## Priority when requirements conflict
1. The user's explicit current request → 2. the final plan and confirmed requirements → 3. recorded decisions → 4. correctness and reproducibility → 5. maintainability → 6. convenience.

## Reporting format
`## Completed` / `## Changed` / `## Validation` / `## Remaining` / `## Recommended Next Step`. Keep it concise and concrete.

## Golden rule
The repository is the long-term source of truth. It must stay understandable if the conversation history disappears.
