# EEG Sleep-Onset Forecasting — State-Conditioned Evaluation

Private pre-submission research code for *Where Does EEG Sleep-Onset Forecasting Skill Arise? A State-Conditioned Evaluation*.

## Repair branch status (2026-10-26)

Known layout defects in the stripped-down src release were repaired on branch fix/reproduction-imports-and-preflight-20261026:

- Restored 53 exact historical Git blobs (49 Python modules/tests) from annotated tag v0.1.0-paper-review to scripts/decision/h_eval, preserving original code and test implementations.
- Corrected relocated src module repository roots, fixed misplaced Stage Reference import search path, and restored W2S import-name shims, XLS reader and diagnostic helper.
- Added a fail-closed source/syntax and private-input preflight and five independent synthetic gate tests.
- The repair branch has **not** been independently trained, scored against frozen predictions or replayed end-to-end. Its preflight checks source layout only, not scientific numerical parity.

## Safe check on the owner Linux workstation

Create a separate checkout under /media/armin/External/Ali/ rather than switching an active Claude/Codex worktree.

    cd /media/armin/External/Ali
    git clone --branch fix/reproduction-imports-and-preflight-20261026 git@github.com:AliAmini93/h-eval-sleep-onset-eeg.git h-eval-sleep-onset-eeg-repair
    cd h-eval-sleep-onset-eeg-repair
    bash scripts/run_release_checks.sh

The runner uses only the exact Python /mnt/HDD/AliWorks/HCI Tagging Database/HCI/bin/python3. It installs nothing, does not open EEG or TEST data, does not train a model, and reports missing private input paths rather than manufacturing artifacts. An exit code 2 after successful source checks denotes the expected missing-private-inputs gate.

## Code map

- src/data: Sleep-EDF eligibility, split and EEG windows.
- src/models: original spectral/CNN models.
- src/evaluation: internal/external and retrospective state/cluster analysis.
- src/analysis: wake-to-N1 and representation perturbations.
- scripts/decision/h_eval: unchanged historical source bodies, kept as compatibility dependencies and for frozen code review.
- scripts/release_preflight.py: safe source and dependency inventory; it cannot validate numerical parity.

## Scientific and licensing limits

Data source: https://physionet.org/content/sleep-edfx/1.0.0/ . Sleep-EDF must be downloaded separately and official checksums verified. Do not commit raw signals, participant lists, individual predictions, private checkpoints, authorization markers or hidden protocol inputs. Internal train/held-out prerequisites are catalogued in docs/reproducibility.md.

Retrospective sleep-stage reference results are not online-deployable predictors. Post-hoc subgroup comparisons are not prespecified held-out tests. The historical v0.1.0-paper-review tag contains the already documented aggregate-only reproduction, not an authorized raw-signal replication. Reuse terms and citation metadata remain subject to author approval.

See docs/REPAIR_REPORT_20261026.md and docs/experimental_status.md. Sanitized local execution logs and metrics must be committed, pushed and verified against remote Git SHA for collaborator review.