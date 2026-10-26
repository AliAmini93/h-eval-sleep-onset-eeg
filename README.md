# EEG Sleep-Onset Forecasting — State-Conditioned Evaluation

Source code accompanying **Where Does EEG Sleep-Onset Forecasting Skill Arise? A State-Conditioned Evaluation**.

This repository contains **only research code directly relevant to the reported EEG models, cohort construction, held-out comparisons, and state-conditioned / exploratory analyses**. It does not contain the manuscript, LaTeX, figures, result snapshots, development logs, or unrelated project work.

## Code

- `src/data/` — Sleep-EDF eligibility, person split, EEG windows.
- `src/models/` — spectral EEG features and raw-waveform CNN implementation.
- `src/evaluation/` — original internal/external evaluations, cluster uncertainty, cohort and state-conditioned comparisons.
- `src/analysis/` — exploratory wake-to-first-N1 models, calibration, personalization, and representation perturbations.

## Data

Source dataset: [Sleep-EDF Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet). Download it separately and comply with its terms. Raw signals, person-level rosters, held-out predictions, fitted weights, and nonpublic evaluation artifacts are not distributed here.

## Scope and execution status

These are **original research implementations, reorganized for code review**; original source bodies are retained. This compact version is **not yet a fully runnable or independently validated end-to-end reproduction package**: some legacy internal imports, artifact paths, and private input dependencies still require adaptation. In particular, moving files to descriptive paths has **not** been validated by running all scripts. The historical source snapshot remains available to authorized collaborators at Git tag `v0.1.0-paper-review`.

The paper reports the outcomes and describes their statistical and experimental limitations. Do not treat post-hoc subgroup results as prespecified held-out evidence or retrospectively scored sleep-stage references as online deployable predictors.

**Status:** private pre-submission source release. Software reuse terms and citation metadata remain subject to author approval.
