# H-EVAL: state-conditioned evaluation of EEG sleep-onset forecasting

Source code for the paper *Where Does EEG Sleep-Onset Forecasting Skill Arise? A State-Conditioned Evaluation*.
Every 30 s after lights-off, models predict whether the first scored Stage 2 epoch (N2) begins within 300 s from the
preceding 30 s of two-channel EEG (Fpz-Cz, Pz-Oz; Sleep-EDF Expanded). Brier-score skill over a constant is split
exactly into contributions from inputs scored as wake (W) and as Stage 1, and EEG forecasts are compared with
Stage-only and Stage+time references, which use the scored stage retrospectively.
The code covers cohort construction, the SpecLR and RawCNN models, the person-cluster bootstrap, the prespecified
held-out decision rules, the post-hoc stage analyses and the TRAIN-only wake-to-first-N1 analysis.

```text
src/heval/
├── data/        Sleep-EDF EDF/EDF+ and lights-off spreadsheet readers, eligibility, salted person split, decision windows
├── models/      SpecLR (spectral.py), RawCNN (raw_cnn.py, 114,817 parameters), Stage-only / Stage+time references,
│                wake-to-N1 temporal CNNs
├── evaluation/  Brier / log loss / calibration / AUROC variants, person-cluster bootstrap, signed W/N1 attribution,
│                scored-stage reference analysis, prespecified TEST/EXT decision rules
└── analysis/    TRAIN-only wake-to-first-N1 estimands, references, nested recalibration, warning metrics
tests/           deterministic synthetic tests (no data needed)
```

## Quick start (no data needed)
Use an existing Python 3.11 environment with the packages in `requirements.txt`; nothing is installed by this repository.
```bash
PY=/path/to/python3
PYTHONPATH=src "$PY" -m pytest -q tests              # 59 synthetic tests
PYTHONPATH=src "$PY" -m heval info                   # environment + RawCNN parameter check (114,817)
PYTHONPATH=src "$PY" -m heval demo --out demo.json   # synthetic end-to-end run of the whole pipeline
PYTHONPATH=src "$PY" -m heval --help                 # all commands; each has --help
```
Tested on the authors' workstation (Python 3.11.15, numpy 1.26.4, scipy 1.16.0, scikit-learn 1.6.1, torch 2.9.1, CPU):
`PYTHONPATH=src python -m pytest -q tests` passed 59 of 59 tests. No run on a second machine has been performed yet.

## Running on Sleep-EDF
1. Download Sleep-EDF Expanded v1.0.0 from https://physionet.org/content/sleep-edfx/1.0.0/ (ODC-By 1.0 licence).
   Keep the PhysioNet layout (`metadata/`, `sleep-cassette/`, `sleep-telemetry/`); official SHA-256 sums are checked.
2. Build the eligibility table and the person split (lights-off from the subject spreadsheets, no EEG samples read):
   `PYTHONPATH=src "$PY" -m heval cohort --data-root DIR --out-dir out` (or set `HEVAL_SLEEP_EDF_ROOT`).
3. Extract decision windows and spectral features per eligible recording:
   `... -m heval windows --data-root DIR --cohort-csv out/cohort.csv --recording <code> --out <code>.npz`.
4. Fit and apply the models on TRAIN window files: `fit-speclr`, `fit-rawcnn` (seeds 0, 1, 2; 4 epochs), `predict`.
5. Score any reference/candidate pair: `... -m heval score --predictions preds.csv --ref <col> --cand <col> --stage-col stage`.

Input schemas: window files hold `raw` (N, 2, 3000) microvolts, `spectral` (N, 10), `elapsed_s`, `time_to_n2_s`, `y`,
`anchor`, `person`, `recording`; prediction CSVs need `person`, `y` (0/1), probability columns and optionally the scored
input stage (`W`/`1`). `fit-references` takes TRAIN windows with `person`, `elapsed_s`, `stage`, `y`.
Rosters (TRAIN all nights; VAL/TEST lowest night per person; EXT all telemetry nights) and the 18 SpecLR-18 training
nights follow `heval.data.cohort.roster` and `speclr18_training_nights`.

## Methods in the code
| Paper element | Module | Notes |
|---|---|---|
| Risk set, labels, scored input stage | `data/cohort.py`, `data/windows.py` | t < T2 on the 30 s grid; Y = 1{T2 - t <= 300 s} |
| Person-disjoint split | `data/cohort.py: person_split` | salted SHA-256 order of subject codes; 20% TEST, 20% VAL, rest TRAIN; telemetry = EXT |
| SpecLR-18 / SpecLR-46 | `models/spectral.py` | 10 log band-power/RMS features; z-score; 350-step L2 logistic GD |
| RawCNN-46 | `models/raw_cnn.py` | multiscale 1D CNN, AdamW, 4 epochs, mean of seeds 0-2 |
| Stage-only / Stage+time | `models/references.py` | Laplace rates; B-spline logistic selected by inner CV (5 knots, C = 10) |
| Brier skill, CIs | `evaluation/bootstrap.py` | 95% person-cluster percentile bootstrap, 10,000 replicates, no CI below 8 persons |
| Signed W/N1 contributions | `evaluation/stage_analysis.py` | delta_WW = C_W + C_1 with the pooled denominator |
| Prespecified TEST/EXT rules | `evaluation/heldout.py` | H1 GO/NO-GO with donor and permutation controls; H2 sequential; H3 descriptive |
| Wake-to-first-N1 analysis | `analysis/wake_to_n1.py`, `models/temporal_cnn.py` | TRAIN only, exploratory |

**Stage references are oracle diagnostics.** They read the stage scored retrospectively from the hypnogram, which is not
available when a forecast is made. They are neither deployable forecasters nor performance ceilings.

**Lineage.** Each module names the research stage its functions come from: E2 and E3 for the cohort, E5–E7B for SpecLR,
CTH v1/v2 for RawCNN, T1/X1 for the held-out rules, R2/R3B for the stage analyses and W2S for wake-to-N1. Those stages
are preserved in the tagged archive `v0.1.0-paper-review`. The extracted arithmetic was compared with the original
functions on synthetic inputs and on TRAIN-only development files: features, model fits, RawCNN outputs from the frozen
weights, references, bootstrap intervals and recalibration results matched exactly. Golden values from that comparison
are fixed in `tests/`.

## What this repository does not contain
- No EEG, hypnograms or subject spreadsheets: download them from PhysioNet.
- No split rosters, per-window predictions, stage joins or model weights.
- No held-out results, and no manuscript or figure sources.
- The VAL, TEST and EXT evaluations were one-shot tests. Their reported numbers depend on frozen inputs and model
  weights that are withheld, so they cannot be recreated from this source alone. Rerunning the pipeline on public data
  refits the models; it does not reproduce those tests.
- Not included: approval, manifest and marker machinery of the original one-shot runs, the architecture-ablation study
  and the representation/perturbation analysis (both development-only appendices).

## Licence and citation
The repository is private. A software licence and citation metadata are pending the authors' decision; no reuse rights
are granted yet.
