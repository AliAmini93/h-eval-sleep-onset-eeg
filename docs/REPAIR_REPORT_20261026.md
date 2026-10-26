# Source-Release Repair (2026-10-26)

Branch: fix/reproduction-imports-and-preflight-20261026

- Recovered 53 unchanged Git blobs (49 Python modules/tests) by hash from the annotated v0.1.0-paper-review tag under the historically expected scripts/decision/h_eval/ paths; no paper PDF, private recordings, prediction arrays or weights copied.
- Fixed module-relative repo-root resolution in 11 relocated research sources; repaired the misplaced stage-reference import search path.
- Restored the XLS reader, W2S diagnostic and old import-name shims. This preserves existing scientific calculations rather than replacing them.
- Added scripts/release_preflight.py (AST/source and private-input gate), five stdlib synthetic tests and scripts/run_release_checks.sh (requires the exact existing HCI Python; installs nothing).

**Evidence:** The new gate logic passed five synthetic unit tests in the assistant's temporary CPU environment. However GitHub/private source checkout was not available in that runtime: branch-wide compilation, imports, model runs and end-to-end replay on the owner's Ubuntu were **not** independently verified. Source gate is only syntactic, not semantic. The historical tag documents an already-validated aggregate-only artifact tier, separate from raw EEG replication.

**Unresolved external prerequisites:** official PhysioNet Sleep-EDF Expanded recordings/checksums, private source-derived eligibility/split manifests, frozen model weights, recorded protocol approvals, and author-only internal output files. The preflight reports missing paths without reading study data or opening held-out cohorts. Do not synthesize those files or treat confirmation tests as repeated model selection.

**Owner execution (new checkout, no changes to active worktrees):**
  cd /media/armin/External/Ali
  git clone --branch fix/reproduction-imports-and-preflight-20261026 git@github.com:AliAmini93/h-eval-sleep-onset-eeg.git h-eval-sleep-onset-eeg-repair
  cd h-eval-sleep-onset-eeg-repair
  bash scripts/run_release_checks.sh

All future source fixes and sanitized test logs must be committed/pushed and remote commit verified before claiming GitHub synchronization. Protected, post-hoc or retrospective stage analyses remain explicitly non-deployable/non-prespecified.
