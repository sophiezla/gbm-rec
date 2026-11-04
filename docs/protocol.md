# Protocol

The analysis protocol is the plan, `docs/plan/final_plan.md`. This file will hold the operational details the plan leaves to implementation (file layouts, script order, exact commands) as each phase is built.

## Phase 1: tournament on LUMIERE

Rules: `DECISIONS.md` D6 (draft until committed).

| Script | Does | Writes |
|---|---|---|
| `scripts/01_lumiere_features.py` | f1-f5 and the C0T distance for the 63 patients; ~20 s per patient; resumable | `data/interim/features/<patient>.{npz,json}` (not in git) |
| `scripts/02_tournament.py` | every candidate on the shared 7 folds; metrics; Δ vs C0 with BCa CIs; challenger by rule. Refuses a dirty tree or a null master seed | `results/tournament/`, `data/interim/oof/` |
| `scripts/03_refit_release.py` | Step 1.7: refit C1, C3, C2 on all 63; save and SHA-256 the weights | `results/release/` |
| `scripts/04_power.py` | Step 2.1: MDE at n = 20–35 from C0's LUMIERE scores, BCa decision rule | `results/power/` |
| `scripts/05_later_progressions.py` | Step 1.8 inputs: preprocess and transport progressions 2 and 3 with gbm-4d's frozen scripts (D9). Long | `data/interim/{preprocessed_later,label_field_later}/` |
| `scripts/06_later_progression_scores.py` | Step 1.8: exploratory scores on new enhancing tumour | `results/tournament/later_progression/` |
| `scripts/07_roundtrip_threshold.py` | Step 3.3: tuned-recipe round-trip distribution on 63 LUMIERE pairs, registration only (D12). Long | `data/interim/roundtrip_tuned/`, `--summary` -> `results/registration_threshold/` |
| `scripts/08_pack_bundle.py` / `09_import_bundle.py` | run `05_` and `07_` on another PC and merge the results back | `../gbmrec_bundle(.zip)` |
| `scripts/20_rhuh_inventory.py` | Steps 3.1-3.2: download record, inventory, mapping check, cavity inspection | `results/rhuh_inventory/` |
| `scripts/21_rhuh_transport.py` | Step 3.3: transport, tuned (primary) or `--recipe gbm4d` (sensitivity) | `data/interim/rhuh/label_field_*/` |
| `scripts/22_rhuh_screens.py` | Step 3.4: surface, arbiters, simulate, verdict | `results/rhuh_screens/` |
| `scripts/23_rhuh_lock.py` | Steps 3.5-3.6: CONSORT, evaluable n, achieved MDE | `results/rhuh_lock/` |
| `scripts/30_confirm.py` | Steps 4.2-4.6: hash check, run-once marker, primary, verdict, secondaries (BH-FDR), maps | `results/confirm/` |
| `scripts/31_sensitivity.py` | Step 4.5: sensitivity arms, once, after 30_ | `results/confirm_sensitivity/` |
| `scripts/42_pde_frames.py` | PDE mode: gbm-4d's Fisher-KPP re-run in frames (D15) | `data/interim/pde_frames/` |
| `scripts/50_figures.py` / `51_reveal_figure.py` | Step 4.7 figures from results/ | `results/figures*/` |
| `scripts/update_handoff.py` / `install_hooks.py` | regenerate HANDOFF.md's status block; install it as a pre-commit hook (not installed) | `HANDOFF.md` |
| `scripts/40_export_case.py` | Step 5.4: GLB layers, slice volumes, case.json per LUMIERE hero case | `app/public/cases/` (not in git, D11) |
| `scripts/41_audit_numbers.py` | Step 4.8: every quoted number must exist in `results/` | exit code |

Every script that writes to `results/` refuses a dirty tree. `02_`–`04_` and `06_` also need `seeds.master`.

Still to build: C5 (Step 0.4 gate) and the exploratory C1 + ADC arm. Every RHUH-GBM script refuses until the OSF record and the verified file layout are in `config.yaml`.

## App (Phase 5)

`app/`: Vite + TypeScript + three.js + NiiVue (D11). `npm install`, then `npx vite` in `app/` for development or `npx vite build` for the static build. Screens: Story (eight steps, prediction before reveal), Explore (layers, models, time slider, slices with recurrence overlay, PDE mode), Compare (two models, linked cameras), Results (verdict, forest plot, coverage bars, from `results.json`), Learn (`app/src/content/learn.html`). App prose lives in `app/src/content/` so `41_` audits it.
