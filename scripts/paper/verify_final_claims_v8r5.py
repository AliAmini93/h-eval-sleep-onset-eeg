#!/usr/bin/env python3
"""Source-only scientific claim verification for the H-EVAL v8r5 manuscript.

Reads the aggregate-only snapshots in results/aggregates. No EEG data, model loading, inference, bootstrap,
sampling or changes to any result file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIN_PYTHON = (3, 11)


def check_environment():
    if sys.version_info[:2] < MIN_PYTHON:
        raise SystemExit(f"STOP: Python >= 3.11 required; running {sys.version.split()[0]} at {sys.executable}")


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def percentage(value):
    return round(100.0 * value, 1)


def check(label, actual, expected, results, precision=1):
    observed = round(actual, precision)
    ok = observed == expected
    results.append({"claim": label, "observed": observed, "expected": expected, "pass": ok})
    if not ok:
        raise AssertionError(f"{label}: got {observed}, expected {expected}")


def main():
    check_environment()
    t1 = load("results/aggregates/gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json")["content"]["results"]
    x1 = load("results/aggregates/gate_x1/execution_v1/gate_x1_result_v1.json")["content"]["results"]
    r2 = load("results/aggregates/repair_r2/r2_stage_results.json")["populations"]
    r3 = load("results/aggregates/repair_r3b/r3b_results_v1.json")
    w2s = load("results/aggregates/discovery_w2s/w2s_trial_log.json")
    checks = []

    # Q1: original held-out comparisons, as percentages.
    for title, metric, expected in [
        ("TEST prespecified primary Brier skill", t1["H1"]["scores"]["primary"]["R"], 18.3),
        ("TEST CNN-46 vs SpecLR-18", t1["H2"]["R"], 23.8),
        ("EXT prespecified primary Brier skill", x1["H1"]["scores"]["primary"]["R"], 3.4),
    ]:
        check(title, percentage(metric), expected, checks)
    assert x1["H1"]["decision"]["outcome"] == "INCONCLUSIVE"

    # Exact reported headline intervals. Percentage point values (1 decimal).
    for title, values, expected in [
        ("TEST primary CI", t1["H1"]["scores"]["primary"]["R_ci95"], (8.8, 28.8)),
        ("TEST CNN comparison CI", t1["H2"]["R_ci95"], (4.7, 41.2)),
        ("EXT primary CI", x1["H1"]["scores"]["primary"]["R_ci95"], (-0.9, 7.6)),
    ]:
        for suffix, value, exp in zip(("lower", "upper"), values, expected):
            check(title + " " + suffix, percentage(value), exp, checks)

    # Q2/Q3: retrospective R3B reference-conditional comparisons.
    assert r3["status"] == "POST_HOC_EXPLORATORY_DESCRIPTIVE"
    for cohort, stratum, contrast, expected in [
        ("TEST", "ALL", "1:CONSTANT->A", 26.3),
        ("TEST", "ALL", "2:CONSTANT->B", 30.3),
        ("EXT", "ALL", "1:CONSTANT->A", 29.5),
        ("EXT", "ALL", "2:CONSTANT->B", 32.2),
        ("TEST", "STAGE_1", "8:B->RawCNN-46", 28.5),
        ("EXT", "STAGE_1", "8:B->RawCNN-46", -15.5),
        ("VAL", "W", "4:B->SpecLR-18", 14.6),
    ]:
        value = r3["populations"][cohort]["strata"][stratum]["contrasts"][contrast]["R"]
        check(f"{cohort} {stratum} {contrast}", percentage(value), expected, checks)
    assert "RawCNN-46" not in r3["populations"]["VAL"]["strata"]["ALL"]["models"]
    assert "SpecLR-46" not in r3["populations"]["VAL"]["strata"]["ALL"]["models"]

    # Q2: stage-conditional composition and additive, signed score attribution.
    for cohort, expected in [("VAL", 62), ("TEST", 71), ("EXT", 70)]:
        composition = r2[cohort]["composition"]
        positive_one = composition["1"]["positive"]
        positive_w = composition["W"]["positive"]
        fraction = 100.0 * positive_one / (positive_w + positive_one)
        check(cohort + " positive windows Stage1 share", fraction, expected, checks, 0)

    for cohort, model, stage, expected in [
        ("VAL", "SpecLR-18", "W", 37),
        ("TEST", "SpecLR-18", "1", 99),
        ("TEST", "SpecLR-46", "1", 99),
        ("TEST", "RawCNN-46", "1", 102),
        ("EXT", "SpecLR-18", "1", 35),
        ("EXT", "SpecLR-46", "1", 63),
        ("EXT", "RawCNN-46", "1", 104),
    ]:
        candidates = [(k, v) for k, v in r2[cohort]["stage_contribution"].items()
                      if k.startswith(f"CONSTANT->{model} [")]
        assert len(candidates) == 1, (cohort, model, len(candidates))
        contribution = candidates[0][1]
        check(f"{cohort} {model} share {stage}", 100.0 * contribution["share"][stage],
              expected, checks, 0)
        # Verify the exact additive accounting identity in the source.
        total = contribution["absolute"]["1"] + contribution["absolute"]["W"]
        assert abs(total - contribution["delta_ww_all"]) < 1e-12

    for title, val, target in [
        ("TRAIN classical fold fits", w2s["n_classical_fold_fits"], 30),
        ("TRAIN neural fits", w2s["n_neural_fits"], 60),
        ("TRAIN classical fit seconds", w2s["classical_runtime_s"], 653.3),
    ]:
        check(title, float(val), target, checks)

    print(json.dumps({
        "status": "PASS",
        "checks": len(checks),
        "scope": "Frozen JSON values/identities only; no model replay, no comprehensive paper audit",
        "details": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
