# Reproducibility: what this package reproduces, and what it does not

All results below were obtained on 2026-10-26 on one Linux workstation with an existing Python environment:
- Python 3.11.15;
- numpy 1.26.4, scipy 1.16.0, scikit-learn 1.6.1, torch 2.9.1, matplotlib 3.10.7, Pillow 11.3.0, pytest 9.1.1;
- TeX Live pdfTeX 1.40.25.

**No clean-room reproduction on a second machine has been performed yet.**

## Tier 1 — aggregate-only reproduction (verified; no EEG data)
```bash
export PYTHON=/path/to/existing/python3
"$PYTHON" -m pytest                                   # tests/test_paper_artifacts.py
"$PYTHON" scripts/paper/regenerate_tables_v8r2.py     # tables vs paper/latex/tables_generated.tex
"$PYTHON" scripts/paper/make_figures_v8r5.py          # rewrites paper/figures (+ provenance); compare with FIGURE_SHA256.txt
"$PYTHON" scripts/paper/verify_final_claims_v8r5.py   # 29 numerical manuscript claims
PYTHON="$PYTHON" bash scripts/paper/build_paper.sh    # all of the above + LaTeX -> build/h_eval_paper_anonymous.pdf
```
| Check | Result |
|---|---|
| `pytest` (5 tests) | **5 passed** |
| Tables: generator `make_tables_paper_v8r1.py` (+ `make_tables_paper_v7.py`) and 51 recorded presentation edits | output identical to `paper/latex/tables_generated.tex`, Git blob `f4ac863b0130ef5c8c9253d14bbc2e375b550cd9` |
| Figures: `make_figures_v8r5.py` from `results/aggregates` | all 16 files (8 PDF, 8 PNG) byte-identical to the author-approved figures (`paper/figures/FIGURE_SHA256.txt`) |
| Numerical claims | 29/29 pass (`status: PASS`) |
| PDF build (anonymous mode) | compiles; cross-references stable |

**Scope:**
- **Covered:** these checks reproduce the *paper artefacts from the frozen aggregate results*.
- **Not covered:** they do not re-estimate any model, prediction or interval.

The figure PNG/PDF bytes depend on the matplotlib version and on the Liberation Sans font; another environment may produce visually identical figures with different bytes.

## Tier 2 — method unit tests (synthetic inputs; partly runnable)
```bash
"$PYTHON" -m pytest scripts/decision/h_eval/<module>/test_<name>.py
```
Totals over 18 test files: **196 passed, 1 skipped, 9 failed, 3 errors.** Every non-passing test needs material that this package excludes on purpose.

| Test file | Result here | Cause of failures / errors | Same tests in the development repository |
|---|---|---|---|
| `cth_v1/test_cth.py` | 11 passed | — | — |
| `gate_e/test_gate_e.py` | 11 passed | — | — |
| `gate_e2/test_gate_e2.py` | 13 passed, 2 failed | synthetic `fixtures/SC-mock.xls`, `ST-mock.xls` not included (binary spreadsheets are excluded by the release policy) | pass |
| `gate_e2/test_full_gate_e2.py` | 1 failed | same mock workbooks | pass |
| `gate_e3/test_gate_e3.py` | 15 passed | — | — |
| `gate_e4/v3/test_gate_e4.py`, `test_resume_v2.py`, `test_resume_v3.py` | 9 + 10 + 7 passed | — | — |
| `gate_e5/test_gate_e5.py` | 13 passed | — | — |
| `gate_e6/test_gate_e6.py` | 18 passed, 1 failed | integration test needs private per-window predictions (`.local/…/gate_e6_window_predictions.csv`) | fails (needs `.local/` private derived data) |
| `gate_e7a/test_gate_e7a.py` | 11 passed, 1 failed | integration test needs development-repository protocol records and private per-window predictions | fails (needs `.local/` private derived data) |
| `gate_e7b/test_gate_e7b.py` | 10 passed, 1 skipped, 3 errors | roster tests derive the real split rosters from the locally downloaded Sleep-EDF metadata (`data/raw/…/SHA256SUMS.txt`) | errors (needs the PhysioNet download) |
| `gate_e7b/test_gate_e7b_v1_2.py` | 17 passed, 1 failed | needs the frozen model manifest `frozen_models_v1.json` (excluded: model parameters/rosters without distribution approval) | pass |
| `gate_t1/test_gate_t1.py` | 6 passed, 1 failed | real-roster test: needs the locally downloaded Sleep-EDF metadata | fails (needs the PhysioNet download) |
| `gate_x1/test_gate_x1.py` | 6 passed, 1 failed | real-roster test: needs the locally downloaded Sleep-EDF metadata | fails (needs the PhysioNet download) |
| `interp_v1/test_interp_v1.py` | 7 passed, 1 failed | needs the cached development dataset (`.local/…/dataset_v1.npz`) and protocol records | fails (needs `.local/` private derived data) |
| `repair_r2/test_r2_stage_audit.py` | 20 passed | — | — |
| `repair_r3b/test_r3b_audit.py` | 12 passed | — | — |

"Development repository" means the authors' version-controlled working tree (without local data):
- 4 of the 12 non-passing tests pass there.
- The other 8 also fail there, because they need either the PhysioNet download (`data/raw/`) or private derived data (`.local/`), neither of which is ever committed.

## Tier 3 — end-to-end replay from raw EEG: **not supported by this package**
The method code is included so that every analysis can be inspected. A full replay is not possible from this package for these reasons:
1. **Raw data.** Sleep-EDF Expanded must be downloaded from PhysioNet (not redistributed).
2. **Development-repository paths and preconditions.** The data-preparation and evaluation gates read and hash-check upstream artefacts at development-repository paths (`results/decision/h_eval/…`, `.local/…`, `docs/decision/…`). These include identifier-bearing manifests and rosters, per-window predictions, approval records and, for `gate_e/gate_e.py`, a header preflight from an earlier gate on a different dataset (not part of this paper). These artefacts are deliberately excluded.
3. **One-shot status.** The held-out VAL/TEST/EXT evaluations were one-shot prespecified tests. Re-executing them would be an independent replication, not a reproduction of the original tests.
4. **Model artefacts.** Frozen model parameters/checkpoints are not distributed (no distribution approval).

What would be needed for an independent end-to-end replication:
- the PhysioNet download;
- the split procedure in `gate_e3` (salted hash);
- the feature, model and evaluation code under `scripts/decision/h_eval/`;
- adaptation of the input/output paths and preconditions noted above.

Such a replication has **not** been performed.

## Known open items
- Clean-room run of Tier 1 on a second machine.
- Licence selection and citation metadata.
- An anonymised reviewer snapshot for double-blind review.
- Exhaustive per-experiment compute accounting (the paper reports only partial runtimes).
