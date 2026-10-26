# Provenance

This repository was assembled on 2026-10-26 from the authors' development repository by an allowlist script (`scripts/release/assemble_paper_repo_v8r5.py` in the development repository). It was not mirrored, and no commit history was copied. The aggregate snapshots were produced there by `scripts/release/export_paper_aggregates_v8r5.py`.

## Copied files (`docs/file_manifest.csv`)
- 77 files were copied: 70 verbatim and 7 with explicit, asserted text patches.
- Each row records the source path, source SHA-256, release SHA-256 and the patch description.

| Patched file | Change |
|---|---|
| `scripts/decision/h_eval/gate_e4/v3/gate_e4.py` | remove unused personal-path constant (REPO is not referenced anywhere in the module) |
| `scripts/decision/h_eval/gate_e7a/gate_e7a.py` | redact per-person results and identifiers in an erratum text record (documentation only; no computation) |
| `scripts/paper/make_figures_v8r5.py` | docstring: release paths; frozen-input check: snapshot manifest instead of full-file hashes; output path; provenance path; read aggregate-only snapshots |
| `scripts/paper/make_tables_paper_v7.py` | read aggregate-only snapshots; unused standalone output -> build/ |
| `scripts/paper/make_tables_paper_v8r1.py` | intermediate output -> build/; read aggregate-only snapshots |
| `scripts/paper/regenerate_tables_v8r2.py` | compare against the manuscript table file |
| `scripts/paper/verify_final_claims_v8r5.py` | docstring; portable interpreter check (no personal path); read aggregate-only snapshots |

No patch changes a computation:
- The paper scripts only change input/output paths and frozen-input checks.
- The two method-code patches:
  - remove one unused constant that held a local workstation path;
  - redact person identifiers and per-person results from a documentation-only erratum text record.

## Aggregate snapshots (`results/aggregates/MANIFEST.json`)
- 20 snapshots, keeping exactly the fields read by the table, figure and claim generators, with the same key paths as the frozen source files.
- Re-running all generators on the snapshots reproduces:
  - the tables byte-identically (Git blob `f4ac863b0130ef5c8c9253d14bbc2e375b550cd9`);
  - all 16 figure files byte-identically;
  - the claim checks identically (29 checks, status PASS).
- The snapshots contain no person or recording identifiers.

| Snapshot | Source SHA-256 (frozen result file, not redistributed) | Snapshot SHA-256 | Leaf values |
|---|---|---|---|
| `cth_v1/cth_cv_results_v1.json` | `a8e9b368f5f1677806759e22614dcb7c9bb0769736aecf826a2d7d8f743f9a96` | `51dffb27f882fa31e1d3a47c709ef40db11b20ec06bf547da6ec0fd8c72a8dd5` | 16 |
| `discovery_w2s/w2s_classical_results_v1.json` | `f8006848f42903c46ff361cdcca81fd8f5329a9aa2b3a374966a9bf440ed583a` | `e03adc254db408c2c8b402c7bcc76a6515eb919dafdd63df7a238f0272993451` | 22 |
| `discovery_w2s/w2s_neural_results_v1.json` | `96d91fef9a6d31ace42fabcd31c758e8ae7c43f7158999872a737e54c0eb6bbd` | `bb9761a3ba3551ede1c6d452b127bfc6e987b29cb4d4ea30a3ede6793ad3e2aa` | 32 |
| `discovery_w2s/w2s_p0a_diagnostic_v1.json` | `48ecb36ee2641ec3d20877157da5b4d9279fdb8cf5e9fa0e40c739e35ba6939d` | `8663172fdb5fb1460436463c89b4901cf8e4a24365475f812c13d17d6bb090cd` | 16 |
| `discovery_w2s/w2s_p0b_calibration_v1.json` | `e222fb4cc874e54af70873c178ee43a3acfb9b05bdab1d1a3f9a06e4c465339c` | `052e931e486d2cce0f9b07fd6af452c3dd4885c8ba9e94d64a0dcdc659b79f6e` | 30 |
| `discovery_w2s/w2s_p1_results_v1.json` | `e8655d793578223f79356eccc5ce597b9f7e5d4b4beb0996574ba17c42d15d92` | `f2cda8276da9b0c1713d686c3d5f84acc1460f6568b9e84337c0486a6a3398d0` | 100 |
| `discovery_w2s/w2s_trial_log.json` | `90a602349df17790c64afd644d3b72e4c75d954dc70e7974f3480b26fd2f6350` | `637a1a0516e0826e3d99d3f670b3591857559b3d5168651c420baeade0015314` | 3 |
| `discovery_w2s/w2s_v7r1_descriptive_ci.json` | `978911d15558210676c9ea288389bb4aae52365a0b3052fae0187bfdb3e3c502` | `fe57831bfc87d5c07d2e9d499fd929a9564542ed2761a3755d8cb8fe3f9502b9` | 12 |
| `gate_e7b/execution_v1_2/gate_e7b_primary_result_v1_2.json` | `83ac5b7060222d33b1b5c99b5c7bdf5392aa9407ff4841bc1dced309dc2997be` | `1abf1d5148fea3d4b6129fbb9ee4a8dcd97b74bdedeeebf46bdf0d5b7f96d042` | 8 |
| `gate_e7b/execution_v1_2/gate_e7b_sensitivity_result_v1_2.json` | `2580bd7c89d349e502ee5fa19615fd094e3879408c3e11fc186a8c8c91f8c960` | `fccbbe9b797654dfa1d7a4aebaa1d46c7446879687061662c1fa89f659e31c50` | 2 |
| `gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json` | `72e0a8ad334020a87531da508bd8aca39e138a4168dfd990064aa3f7dffc652f` | `87132d5fd4f098142e93ce516b813701e31a8d79e8f1a6becc4cbab1ff6ada32` | 37 |
| `gate_t1/execution_v1/gate_t1_sensitivity_result_v1.json` | `137ae79ad8daf0e9fcd5d088cd26304d67be99aa1288008a0f997a6058e770c3` | `0f2539c2af326354c91ea3d80751c16d108136454bf5048889ad25ddc90b4931` | 6 |
| `gate_x1/execution_v1/gate_x1_result_v1.json` | `088119fe1817c8127f8d629a1cdfa10bfda5e4a01bc7376b4a0ee1ee3492df18` | `075598c412d8b5c0aa72d4f42eb21e9d9344dcb7305dbb04b78ca3aa833b8462` | 48 |
| `interp_v1/interp_followup_F1_F2_v1.json` | `882c1c8aaa328e04bf4a1f7435cb2e9ef17030084757e2b0222590134a33db74` | `f5bfe79678ebca40be7ffc828c902bce48a67a8c04d51f33ff554c8d54234050` | 6 |
| `interp_v1/interp_partA_perturbation_v1.json` | `e0ce6e45f84d1feff47901b61d5f8faf3de0310e75174769d30fa29f52dae55d` | `8dc06772d11f9e1a1b21f2a1014e02d86d8b58f2e09367566388ddb62fe77091` | 54 |
| `interp_v1/interp_partB_cv_results_v1.json` | `33ef9b0c98254671ed8626fdabb9f46e4019fbfe8bf68f802e65bf24bc829fbf` | `4147878befc4fa629cf9a59e7222759531f971ae57d1eef4e087fa99cea78ddd` | 17 |
| `repair_r1/r1_cohort_flow.json` | `676415ea3623b3dbe208b3395caab3552cf0f814773eb7f79f277483a4121dd2` | `3e558ecd5acf60e0dab689905972185f018c74f8e467daac3353dc16ccd5eb2d` | 25 |
| `repair_r1/r1_stage_construct.json` | `eb3855d8a9bf3a16e65f47b6e528046b51b438ff7724431e4c07d00eff5f4340` | `c8498547f98b1555a67fd0541cad262e2829300591f40b619bb7caa7c6835cf5` | 6 |
| `repair_r2/r2_stage_results.json` | `42983c0c3621a1600b29f58ab94613fff0d8fc16c6d7ea7dbba74835f3a17800` | `94016a88392c147fd581b11a80c9db960f233573ee2b34d3f7b2baf1d9a0f8a2` | 197 |
| `repair_r3b/r3b_results_v1.json` | `ec0910011c0d4aa59e025d7f682986fb254855ccfebbedf542947d9a5a9f4771` | `afaf56edf9eb774ce4885079291a656a03754a380953db687be3d553d392024b` | 661 |

## Figures and manuscript
- `paper/figures/` holds the eight author-approved figures (PDF + PNG).
  - SHA-256 values are in `paper/figures/FIGURE_SHA256.txt`.
  - They are byte-identical to the approved figure files in the development repository.
  - The per-value ledger is in `paper/figures/provenance/`.
- `paper/latex/` holds the anonymous manuscript sources:
  - `tables_generated.tex` is the frozen table file (Git blob `f4ac863b…`);
  - `neurips_2026.sty` is the conference style file, included only to build the PDF.

## Privacy decisions
- **Public record codes.** Sleep-EDF record and subject codes (e.g. `SC4001`) are public, pseudonymous dataset identifiers.
- **Split procedure.** The person-level split is a deterministic salted-hash function of these codes, implemented in `scripts/decision/h_eval/gate_e3/gate_e3.py` (method code required to define the cohorts). Split membership is therefore reproducible from public data by design.
- **Not included:**
  - roster files;
  - per-person or per-window results;
  - eligibility exclusion lists with identifiers.
- **Where public record codes remain:**
  - unit-test fixtures (synthetic or boundary-guard tests);
  - one public file-format note in `gate_e/gate_e.py`;
  - none in `results/aggregates/`.

## Exclusion manifest (development-repository candidates not copied)
The release plan listed 116 candidate paths. Reconciliation:
- excluded: 41
- included: 55
- replaced by aggregate-only snapshot: 20

| Candidate path | Decision |
|---|---|
| `results/decision/h_eval/cth_v1/cth_cv_results_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/cth_v1/cth_vs_m0_table_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/cth_v1/frozen_m0_46_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/cth_v2/m2_epoch_schedule_v2.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/discovery_w2s/w2s_classical_results_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/discovery_w2s/w2s_e1_diagnostic_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/discovery_w2s/w2s_e1_timematched_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/discovery_w2s/w2s_neural_results_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/discovery_w2s/w2s_p0a_diagnostic_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/discovery_w2s/w2s_p0b_calibration_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/discovery_w2s/w2s_p0p1_trial_ledger.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/discovery_w2s/w2s_p1_results_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/discovery_w2s/w2s_trial_log.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/discovery_w2s/w2s_v7r1_descriptive_ci.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/gate_b/hmc_gate_b_summary.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_c/gate_c_preflight_summary.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_d/gate_d_preflight.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e3/gate_e3_sc_group_split.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e3/gate_e3_summary.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e5/gate_e5_summary.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e6/gate_e6_cluster_bootstrap.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e6/gate_e6_summary.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e7a/gate_e6_erratum_01.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e7a/gate_e7a_summary.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e7a/gate_e7a_time_matched_discrimination.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e7a/gate_e7a_verification.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e7b/execution_v1_2/gate_e7b_primary_result_v1_2.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/gate_e7b/execution_v1_2/gate_e7b_sensitivity_result_v1_2.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/gate_e7b/frozen_models_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_e7b/gate_e7b_prepare_report_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/gate_t1/execution_v1/gate_t1_sensitivity_result_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/gate_x1/execution_v1/gate_x1_result_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/hmc_readiness/hmc_partition_spec_draft_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/interp_v1/interp_followup_F1_F2_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/interp_v1/interp_partA_explanations_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/interp_v1/interp_partA_perturbation_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/interp_v1/interp_partB_cv_results_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/repair_r1/r1_cohort_flow.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/repair_r1/r1_small_sample_coverage.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r1/r1_stage_construct.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/repair_r1/r1_time_baseline_results.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r2/r2_integrity.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r2/r2_stage_results.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/repair_r2/r2_synthetic_tests.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r3a/r3a_state_reference_results_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r3b/r3b_execution_log.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r3b/r3b_integrity.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r3b/r3b_references_v1.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `results/decision/h_eval/repair_r3b/r3b_results_v1.json` | replaced by aggregate-only snapshot |
| `results/decision/h_eval/repair_r3b/r3b_synthetic_tests.json` | excluded: not read by any paper table/figure/claim generator; full result files may contain identifiers or per-person data |
| `scripts/decision/h_eval/discovery_w2s/w2s_diag_e1.py` | excluded: not in the dependency closure of the paper analyses (earlier or unused diagnostics) |
| `scripts/decision/h_eval/discovery_w2s/w2s_diag_e1_timematched.py` | excluded: not in the dependency closure of the paper analyses (earlier or unused diagnostics) |
| `scripts/decision/h_eval/repair_r1/r1_small_sample.py` | excluded: not in the dependency closure of the paper analyses (earlier or unused diagnostics) |
| `scripts/paper/make_figures_interp_v1.py` | excluded: superseded by the v8r5 generators/verifier |
| `scripts/paper/make_figures_paper_interp_v8r4.py` | excluded: superseded by the v8r5 generators/verifier |
| `scripts/paper/make_figures_paper_main_v8r4.py` | excluded: superseded by the v8r5 generators/verifier |
| `scripts/paper/make_figures_paper_v8r4.py` | excluded: superseded by the v8r5 generators/verifier |
| `scripts/paper/make_figures_v4.py` | excluded: superseded by the v8r5 generators/verifier |
| `scripts/paper/make_figures_w2s_v8r4.py` | excluded: superseded by the v8r5 generators/verifier |
| `scripts/paper/verify_claims_v8r1.py` | excluded: superseded by the v8r5 generators/verifier |

**Never included** (whether or not they were listed as candidates):
- raw EEG/EDF and hypnograms;
- PhysioNet subject spreadsheets and the synthetic mock spreadsheets used by two `gate_e2` tests;
- split rosters and participant/recording identifier lists;
- per-window predictions and per-window stage joins;
- cached datasets (`.local/`);
- frozen model parameter files and checkpoints;
- protocol approval records;
- the authors' manuscript history and reviews;
- literature PDFs;
- code from earlier gates on other datasets (`gate_b`, `gate_c`, `gate_d`, `hmc_readiness`).
