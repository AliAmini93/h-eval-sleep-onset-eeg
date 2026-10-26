"""Stage-conditioned analysis of fixed predictions: signed W / Stage-1 attribution and scored-stage references.

Signed attribution: C_s = sum over windows with scored input stage s of [(p_ref - y)^2 - (p_cand - y)^2] divided by
the number of ALL windows, so delta_WW = C_W + C_1 exactly (an accounting identity, not a causal decomposition).
Shares C_s / delta_WW can be negative or exceed 100% when the other stage contributes negatively.

Lineage (research archive, tag v0.1.0-paper-review): contributions and stage_attribution from the post-hoc stage
audit R2; state_reference_analysis is the statistics phase of the reference audit R3B (same strata, contrasts,
seed and order of random draws). Input verification against the frozen files is not part of this package.
"""
import numpy as np

from .bootstrap import (BOOT_N, MIN_CI_PERSONS, SEED_REFERENCE_AUDIT, boot_index, reference_contrast,
                        stratum_contrast)
from .metrics import auroc, calibration_in_stratum, matched_auroc, within_recording_auroc
from ..models.references import predict_A, predict_B

CONTRASTS = [('1', 'CONSTANT', 'A'), ('2', 'CONSTANT', 'B'), ('3', 'A', 'SpecLR-18'), ('4', 'B', 'SpecLR-18'),
             ('3s', 'A18', 'SpecLR-18'), ('4s', 'B18', 'SpecLR-18'), ('5', 'A', 'SpecLR-46'), ('6', 'B', 'SpecLR-46'),
             ('7', 'A', 'RawCNN-46'), ('8', 'B', 'RawCNN-46'), ('9', 'SpecLR-46', 'RawCNN-46'),
             ('0a', 'CONSTANT', 'SpecLR-18'), ('0b', 'CONSTANT', 'SpecLR-46'), ('0c', 'CONSTANT', 'RawCNN-46')]


def contributions(b_ref, b_cand, cats):
    """Per-window Brier losses of reference and candidate + category per window -> signed contributions."""
    b_ref, b_cand, cats = np.asarray(b_ref, float), np.asarray(b_cand, float), np.asarray(cats)
    if not (b_ref.shape == b_cand.shape == cats.shape) or b_ref.ndim != 1 or len(b_ref) == 0:
        raise ValueError('b_ref, b_cand and cats must be non-empty 1-D arrays of equal length')
    total = float((b_ref - b_cand).mean())
    c = {s: float((b_ref - b_cand)[cats == s].sum() / len(cats)) for s in sorted(set(cats))}
    if abs(sum(c.values()) - total) > 1e-12:
        raise ValueError('contribution decomposition does not sum to delta_ww')
    return dict(delta_ww_all=total, absolute=c,
                share=({s: v / total for s, v in c.items()} if abs(total) > 1e-9 else 'undefined_delta_near_0'))


def stage_attribution(y, person, recording, stage, pre_first_sleep, P, pairs, seed=None, condition=None):
    """Stage-stratified contrasts and signed contributions for one cohort.

    P: {model: probabilities}; pairs: {name: (reference model, candidate model)}; stage: 'W', '1' or MIXED labels;
    pre_first_sleep: bool per window; condition: optional per-window condition labels (EXT placebo / temazepam)."""
    y = np.asarray(y, float)
    person, rec, stage = np.asarray(person), np.asarray(recording), np.asarray(stage)
    pre = np.asarray(pre_first_sleep, bool)
    subsets = {'ALL': np.ones(len(y), bool)}
    for s in sorted(set(stage)):
        subsets['STAGE_' + s] = stage == s
    subsets['PRE_FIRST_SLEEP'] = pre
    subsets['POST_FIRST_SLEEP'] = ~pre
    subsets['W_POST_FIRST_SLEEP'] = (stage == 'W') & ~pre
    if condition is not None:
        cond = np.asarray(condition)
        for c in sorted(set(cond)):
            subsets[f'{c}:ALL'] = cond == c
            subsets[f'{c}:STAGE_W'] = (cond == c) & (stage == 'W')
            subsets[f'{c}:STAGE_1'] = (cond == c) & (stage == '1')
            subsets[f'{c}:PRE_FIRST_SLEEP'] = (cond == c) & pre
    out = {}
    for sname, m in subsets.items():
        info = dict(n_windows=int(m.sum()), n_positive=int(y[m].sum()), n_negative=int((1 - y[m]).sum()),
                    n_persons=len(set(person[m])), n_recordings=len(set(rec[m])))
        if m.sum() == 0:
            out[sname] = dict(**info, contrasts='empty')
            continue
        pu = sorted(set(person[m]))
        kw = {} if seed is None else dict(seed=seed)
        I = boot_index(len(pu), **kw) if len(pu) >= MIN_CI_PERSONS else None
        info['contrasts'] = {k: stratum_contrast(P[a][m], P[b][m], y[m], person[m], I) for k, (a, b) in pairs.items()}
        out[sname] = info
    contrib = {k: contributions((P[a] - y) ** 2, (P[b] - y) ** 2, stage) for k, (a, b) in pairs.items()}
    contrib_pre = {k: contributions((P[a] - y) ** 2, (P[b] - y) ** 2, np.where(pre, 'PRE_FIRST_SLEEP', 'POST_FIRST_SLEEP'))
                   for k, (a, b) in pairs.items()}
    w = stage == 'W'
    wa = {m: within_recording_auroc(P[m][w], y[w], rec[w]) for m in {b for _, b in pairs.values()}}
    comp = {s: dict(negative=int(((stage == s) & (y == 0)).sum()), positive=int(((stage == s) & (y == 1)).sum()),
                    persons=len(set(person[stage == s])), recordings=len(set(rec[stage == s]))) for s in sorted(set(stage))}
    return dict(composition=comp, subsets=out, stage_contribution=contrib, pre_post_contribution=contrib_pre,
                within_recording_auroc_w_only=wa)


def state_reference_analysis(populations, refs, seed=SEED_REFERENCE_AUDIT, n_boot=BOOT_N):
    """Scored-stage reference analysis for held-out cohorts processed in the given order.

    populations: {name: dict(y, person, recording, elapsed, s1, pre, P)} where P maps the available model names
    ('CONSTANT', 'SpecLR-18', 'SpecLR-46', 'RawCNN-46') to probabilities. refs: TRAIN-fitted references
    {'A', 'B', 'A18', 'B18'} (fit_references output for all 46 persons and for the SpecLR-18 persons); they are only
    applied, never refitted. Returns per-population strata with model metrics and the CONTRASTS table."""
    rng = np.random.default_rng(seed)
    results = {}
    lo, hi = refs['B']['logt_lo'], refs['B']['logt_hi']
    for pop, D in populations.items():
        y = np.asarray(D['y'], float)
        person, recs = np.asarray(D['person']), np.asarray(D['recording'])
        t, s1, pre = np.asarray(D['elapsed'], float), np.asarray(D['s1'], float), np.asarray(D['pre'], bool)
        P = {k: np.asarray(v, float) for k, v in D['P'].items()}
        P['A'], P['B'] = predict_A(refs['A'], s1), predict_B(refs['B'], t, s1)
        P['A18'], P['B18'] = predict_A(refs['A18'], s1), predict_B(refs['B18'], t, s1)
        kidx = np.floor(t / 30 + 1e-9).astype(int)
        strata = {'ALL': np.ones(len(y), bool), 'W': s1 == 0, 'STAGE_1': s1 == 1, 'PRE_FIRST_SLEEP': pre,
                  'RETURNING_W': (s1 == 0) & ~pre}
        R = dict(out_of_support_windows_B=int(((np.log(t) < lo) | (np.log(t) > hi)).sum()), strata={})
        for sn, m in strata.items():
            S = dict(n_windows=int(m.sum()), n_positive=int(y[m].sum()), n_persons=len(set(person[m])),
                     n_recordings=len(set(recs[m])), prevalence=float(y[m].mean()) if m.any() else None, models={}, contrasts={})
            if not m.any():
                R['strata'][sn] = S
                continue
            pu = sorted(set(person[m]))
            I = rng.integers(0, len(pu), size=(n_boot, len(pu))) if len(pu) >= MIN_CI_PERSONS else None
            for name, p in P.items():
                pm, ym, perm = p[m], y[m], person[m]
                cil, slope, why = calibration_in_stratum(pm, ym)
                b = (pm - ym) ** 2
                pooled = auroc(pm, ym)
                S['models'][name] = dict(brier_ww=float(b.mean()), brier_se=float(np.mean([b[perm == q].mean() for q in pu])),
                                         logloss_ww=float(np.mean(-(ym * np.log(np.clip(pm, 1e-6, 1)) + (1 - ym) * np.log(np.clip(1 - pm, 1e-6, 1))))),
                                         cal_in_large=cil, cal_slope=slope, cal_slope_undefined_reason=why,
                                         pooled_auroc=pooled, matched_time_auroc=matched_auroc(pm, ym, perm, kidx[m], rng))
            for cid, a, b in CONTRASTS:
                key = f'{cid}:{a}->{b}'
                S['contrasts'][key] = 'not_available' if a not in P or b not in P else reference_contrast(P[a][m], P[b][m], y[m], person[m], I)
            R['strata'][sn] = S
        results[pop] = R
    return results
