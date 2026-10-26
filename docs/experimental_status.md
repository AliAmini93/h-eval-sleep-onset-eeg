# Experimental status of every result family

Status words follow the manuscript.
- **Prespecified:** analysis plan, model and roster were fixed before the evaluation; executed once.
- **Retrospective / post-hoc:** formulated after earlier held-out outcomes were examined; descriptive.
- **Exploratory (TRAIN only):** development data only; no held-out data involved.

| Result family | Cohort(s) | Status | Code | Aggregate snapshot(s) | Paper |
|---|---|---|---|---|---|
| SpecLR-18 vs constant (primary) | VAL | prespecified; met criteria | `gate_e7b/gate_e7b_v1_2.py` | `gate_e7b/execution_v1_2/*` | Table 1 |
| SpecLR-18 vs constant (primary); RawCNN-46 vs SpecLR-18 (sequential); RawCNN-46 vs SpecLR-46 (descriptive) | TEST | prespecified; primary and sequential criteria met | `gate_t1/gate_t1.py` | `gate_t1/execution_v1/*` | Table 1, Table 4 |
| Same comparisons | EXT | prespecified; primary **inconclusive**; CNN comparison not formally tested | `gate_x1/gate_x1.py` | `gate_x1/execution_v1/*` | Table 1, Fig. 5 |
| Stage attribution of original score differences | VAL, TEST, EXT (+ TRAIN CV) | retrospective, descriptive | `repair_r2/`, `repair_r1/` | `repair_r2/r2_stage_results.json`, `repair_r1/r1_stage_construct.json` | Fig. 2, Table 6 |
| Stage-only / Stage+time references and incremental EEG skill | VAL, TEST, EXT | retrospective, descriptive; no multiplicity adjustment | `repair_r3b/r3b_audit.py` | `repair_r3b/r3b_results_v1.json` | Figs 3–4, Tables 2, 7 |
| TRAIN state references vs out-of-fold EEG | TRAIN | retrospective, development only | `repair_r3a/` | (not exported; values quoted in App. J) | App. J |
| Wake-to-first-N1 programme (E1–E3, recalibration, personalisation) | TRAIN | exploratory | `discovery_w2s/` | `discovery_w2s/*` | Fig. 6, Tables 8–9 |
| Architecture ablations | TRAIN CV | development | `cth_v1/cth_experiment.py` | `cth_v1/cth_cv_results_v1.json` | Fig. 7 |
| Representation / perturbation | TRAIN CV | post-hoc, development | `interp_v1/interp_v1.py` | `interp_v1/*` | Fig. 8 |

## What must not be inferred
- The Stage-only and Stage+time references use retrospectively scored stages: they are neither deployable forecasters nor performance ceilings.
- Stage-attribution shares above 100% reflect negative wake contributions; they are not causal fractions.
- No result shows that the CNN internally detects N1 or sleep stage.
- The positive within-state contrasts (VAL wake for SpecLR-18; TEST Stage 1 for RawCNN-46) did not replicate in the other held-out cohorts.
- SpecLR-18 vs RawCNN-46 confounds architecture with training population; SpecLR-46 vs RawCNN-46 holds the population fixed.
