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
