# Changelog: gbm-rec

## [2026-09-23]

### Added
- Project documentation: `README.md`, `PROJECT_STATUS.md`, `HANDOFF.md`, `DECISIONS.md` (D0–D4), `CHANGELOG.md`.
- `docs/plan/final_plan.md`: the project plan (D1).
- `docs/protocol.md`, `docs/deviations.md` (V1), `docs/preregistration.md`.
- Package `gbmc` (`src/gbmc/`: `rng`, `config`, `provenance`) carrying over gbm-4d's reproducibility contract (D3).
- `pyproject.toml`, `requirements.lock` (Python 3.13.14; gbm-4d lock + scikit-image + trimesh, D4), `config.yaml`.
- Test suite (18 tests) and `.gitignore` rules keeping data and images out of git.
