# Protocol summary

This is a summary for orientation. The manuscript (`paper/latex/main.tex`, Sections 3–4 and Appendices B–C) is authoritative.

## Task and estimand
- **Decision points:** for a night with lights-off at time 0, decision points are t ∈ {30, 60, …} s.
- **Input:** each decision uses only the EEG window [t − 30 s, t).
- **Target:** T₂ is the onset of the first scored Stage 2 epoch.
- **Risk set:** 𝒜₀ = {t < T₂}.
- **Label:** Y_t = 1{T₂ ≤ t + 300 s}.
- **Scored input stage:** S(t) ∈ {W, 1}. It is **never** given to the models; it only defines post-hoc strata.

## Scores
- **Window-weighted Brier improvement:** Δ_WW = mean over 𝒜 of [(p_ref − Y)² − (p_cand − Y)²].
- **Brier skill:** R = Δ_WW / B̄_ref.
- **Stage attribution:** C_s restricts the sum to decisions with S(t) = s while keeping the pooled denominator. Hence Δ_WW = C_W + C_1 exactly. This is an accounting identity, not a causal decomposition.
- **Intervals:** 95% person-cluster percentile bootstrap. Each prespecified evaluation used its own original bootstrap; the post-hoc analysis re-bootstrapped every contrast (10,000 replicates).

## Splits and cohorts
- **SC split:** Sleep Cassette persons were split once, by salted hash (`gate_e3`), into:
  - TRAIN (46 persons, 84 nights);
  - VAL (15 persons);
  - TEST (15 persons).
- **EXT:** the analysable Sleep Telemetry nights (17 persons, 28 nights).
- **Night selection:** VAL and TEST primary analyses used one predeclared night per person; all-night analyses were predeclared sensitivity analyses.
- **Not published here:** the split rosters themselves.

## Models
- **Constant:** the training prevalence.
- **SpecLR-18:**
  - Logistic regression on ten log band-power/amplitude features from both channels (`gate_e5`, `gate_e7b`).
  - Developed on one night from each of 18 TRAIN persons. Six of these persons were selected by observed latency, so this training set is outcome-dependent.
  - Frozen before VAL and carried unchanged into TEST and EXT.
- **SpecLR-46:** the same recipe trained on all 46 TRAIN persons.
- **RawCNN-46:**
  - Multiscale 1D CNN on the raw window (114,817 parameters; mean of three seeds).
  - Code: `cth_v1/cth_experiment.py` (`RawBranch`, `CTHNet`, `M2_raw_cnn_binary`) and `cth_v2/cth_final_v2.py`.
  - It replaced a predeclared task-specific network after development cross-validation; this is a documented deviation (paper App. H).

## Held-out procedure
- **Fixed in advance:** model parameters, rosters, comparison rules and analysis code were fixed before each designated held-out evaluation:
  - VAL: `gate_e7b/gate_e7b_v1_2.py`;
  - TEST: `gate_t1/gate_t1.py`;
  - EXT: `gate_x1/gate_x1.py`.
- **Comparisons:**
  - The primary comparison was SpecLR-18 against the constant.
  - The CNN comparison against SpecLR-18 was sequential, run only if the primary criteria were met.
  - The equal-training-population contrast (RawCNN-46 vs SpecLR-46) was descriptive.
- **One-shot status:** each held-out evaluation was executed once. These scripts are archived for inspection, not for re-execution as new tests.

## Retrospective (post-hoc) analyses
- **Development of the questions:** the state-conditioned questions were developed **after** earlier held-out outcomes had been examined.
- **R1 (`repair_r1`):** TRAIN-only stage construction and time-only references.
- **R2 (`repair_r2`):** attribution of the original recorded held-out score differences to the scored input state.
- **R3B (`repair_r3b`):**
  - TRAIN-fitted Stage-only and Stage+time references (and 18-person refits), applied without refitting to the held-out predictions;
  - contrasts re-bootstrapped.
- **R3A (`repair_r3a`):** TRAIN-only state references compared with out-of-fold EEG predictions.
- **Status:** all of these are retrospective and descriptive.

## Leakage safeguards (as implemented in the method code)
- Person-disjoint splits.
- TRAIN-only fitting of all references and recalibrators.
- Nested, person-disjoint cross-fitting for exploratory recalibration.
- Hash-checked frozen inputs.
- Boundary guards in the held-out gates that refuse to read other cohorts.
