# Dataset: Sleep-EDF Expanded v1.0.0

## Source and licence
- **Dataset:** Sleep-EDF Expanded **v1.0.0**, PhysioNet: <https://physionet.org/content/sleep-edfx/1.0.0/>, DOI 10.13026/C2X676.
- **Licence:** Open Data Commons Attribution License v1.0. Cite the dataset and PhysioNet as the dataset page requests.
- **Studies used:**
  - **Sleep Cassette (SC):** home recordings of healthy adults; development and internal held-out cohorts.
  - **Sleep Telemetry (ST):** hospital recordings on placebo and temazepam nights; external cohort.
- **Scoring:** hypnograms scored with Rechtschaffen & Kales rules (30 s epochs).
- **Channels used:** Fpz-Cz and Pz-Oz EEG at 100 Hz.

**This repository does not redistribute any recording, hypnogram, derived per-window feature, prediction, or participant/recording roster.** To run data-dependent code, download the dataset from PhysioNet yourself and verify the official `SHA256SUMS`.

## Eligibility (applied to scored annotations; metadata only)
A night is eligible if:
1. lights-off lies within the signal;
2. no scoring gap or overlap precedes the first N2;
3. N2 is not active at lights-off; and
4. the scored stage at lights-off is W.

Lights-off times come from the official subject spreadsheets.

| | Sleep Cassette (SC) | Sleep Telemetry (ST) |
|---|---|---|
| Source recordings (persons) | 153 (78) | 44 (22) |
| Lights-off outside signal | −3 | −3 |
| Scoring gap/overlap before first N2 | −1 | −13 |
| N2 active at lights-off | −5 | 0 |
| Not awake (W) at lights-off | −3 | 0 |
| **Analysable** | **141 (76)** | **28 (17)** |
| TRAIN (development) | 84 (46) | — |
| VAL (primary + sensitivity nights) | 15 + 12 (15) | — |
| TEST (primary + sensitivity nights) | 15 + 15 (15) | — |
| EXT (all eligible nights) | — | 28 (17) |

The values come from `results/aggregates/repair_r1/r1_cohort_flow.json` (counts only) and match paper Table 3.

## Code that touches the data (not runnable from this package alone)
- `scripts/decision/h_eval/gate_e/`: metadata-only eligibility audit (record lists, checksums, hypnograms).
- `scripts/decision/h_eval/gate_e2/`: lights-off and recording-origin audit from the official subject spreadsheets.
- `scripts/decision/h_eval/gate_e3/`: person-level split of SC by salted hash.
- `scripts/decision/h_eval/gate_e4/v3/`: SHA-verified signal QC and window construction.
- `scripts/decision/h_eval/gate_e5/`: spectral feature extraction (SpecLR inputs).

These modules expect the dataset and upstream provenance files in development-repository paths; see `docs/reproducibility.md`.
