"""Prespecified held-out comparison logic (VAL, TEST, EXT): scorer, negative controls and decision rules.

H1 (primary): SpecLR-18 vs the TRAIN-prevalence constant. GO requires all of: delta_WW 95% CI lower bound > 0;
R >= 0.05; delta_SE > 0; within-person (EXT: within-recording) AUROC CI lower bound > 0.5; the elapsed-time-matched
donor control (another person's EEG at the same elapsed time, 5 draws) does NOT reproduce the gain (CI lower <= 0).
NO-GO if the R CI upper bound < 0.05 or both point deltas <= 0; otherwise INCONCLUSIVE; INVALID if the permutation
null (NC1: mean within-person AUROC under label permutation in [0.48, 0.52]) or the frozen-input check (NC4) fails.
H2 (sequential): RawCNN-46 vs SpecLR-18, SUPPORTED if R >= 0.05 and delta_WW CI lower > 0, tested only after H1 GO.
H3 (descriptive): RawCNN-46 vs SpecLR-46 (same 46 TRAIN persons).
Seeds: 20261024 for H1, 20261026 for H2/H3; 10,000 person-cluster bootstrap replicates.

Lineage (research archive, tag v0.1.0-paper-review): Scorer, donor_plan, decide and score_primary from E7B (the
VAL evaluation, reused unchanged by T1 and X1); h2_outcome, loo_R and contrast_block from T1; condition_paired from
X1. Approval records, manifests, markers and atomic publication of the one-shot runs are not part of this package.
"""
import hashlib
from collections import defaultdict

import numpy as np

from .metrics import auroc, level_within, pair_matrices, tie_ranks, weighted_auc

SEED_H1, SEED_H2 = 20261024, 20261026
N_BOOT, N_PERM, N_TM_BOOT, DRAWS = 10_000, 2_000, 2_000, 5
DONOR_SALT, STRATUM_S, PRACTICAL, NC1_BAND = 'H-EVAL-E7B-donor-v1', 300., 0.05, (0.48, 0.52)


def donor_plan(person, elapsed_s, draw):
    """Elapsed-time-matched donor window for every recipient window (another person, same 300 s stratum or nearest).
    Uses only person keys and elapsed time, never labels or predictions."""
    stratum = [int(e // STRATUM_S) for e in elapsed_s]
    by = defaultdict(list)
    for i, (p, k) in enumerate(zip(person, stratum)):
        by[(p, k)].append(i)
    strata_people = defaultdict(set)
    for p, k in by:
        strata_people[k].add(p)
    all_k = sorted(strata_people)
    plan, fallback_n, fallback_maxd = [], 0, 0
    for i in range(len(person)):
        v, k = person[i], stratum[i]
        ks = sorted(all_k, key=lambda kk: (abs(kk - k), kk))
        kk = next((x for x in ks if strata_people[x] - {v}), None)
        if kk is None:
            raise ValueError('No donor available in any stratum (cohort has a single person)')
        if kk != k:
            fallback_n += 1
            fallback_maxd = max(fallback_maxd, abs(kk - k))
        cands = sorted(strata_people[kk] - {v}, key=lambda d: hashlib.sha256(f'{DONOR_SALT}|{draw}|{v}|{k}|{d}'.encode()).hexdigest())
        d = cands[0]
        j = min(by[(d, kk)], key=lambda j: (abs(elapsed_s[j] - elapsed_s[i]), elapsed_s[j]))
        plan.append(j)
    return plan, dict(fallback_windows=fallback_n, max_stratum_distance=fallback_maxd)


class Scorer:
    """Person-cluster scorer for one cohort. rows: dicts with 'person', 'y', 'elapsed_s'; P: {model: probabilities}."""

    def __init__(self, rows, P, seed=SEED_H1, n_boot=N_BOOT):
        if not rows:
            raise ValueError('empty cohort')
        self.rows, self.P = rows, P
        self.persons = sorted({r['person'] for r in rows})
        self.pidx = np.array([self.persons.index(r['person']) for r in rows])
        self.y = np.array([r['y'] for r in rows], float)
        if not set(np.unique(self.y)) <= {0., 1.}:
            raise ValueError('y must be 0/1')
        for m, p in P.items():
            p = np.asarray(p, float)
            if p.shape != self.y.shape or not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
                raise ValueError(f'predictions of {m} must be finite, in (0, 1) and aligned with rows')
        self.n = np.bincount(self.pidx, minlength=len(self.persons)).astype(float)
        self.I = np.random.default_rng(seed).integers(0, len(self.persons), size=(n_boot, len(self.persons)))
        self.seed = seed

    def _sums(self, v):
        return np.bincount(self.pidx, weights=v, minlength=len(self.persons))

    def brier(self, p):
        return (p - self.y) ** 2

    def logloss(self, p):
        return -(self.y * np.log(p) + (1 - self.y) * np.log1p(-p))

    def contrast(self, loss_ref, loss_cand, ratio=True):
        sr, sc = self._sums(loss_ref), self._sums(loss_cand)
        I, n = self.I, self.n
        Br, Bc = sr[I].sum(1) / n[I].sum(1), sc[I].sum(1) / n[I].sum(1)
        d_ww = Br - Bc
        pr, pc = sr / n, sc / n
        d_se = (pr - pc)[I].mean(1)
        point_ww = sr.sum() / n.sum() - sc.sum() / n.sum()
        out = dict(ref_ww=float(sr.sum() / n.sum()), cand_ww=float(sc.sum() / n.sum()),
                   delta_ww=float(point_ww), delta_ww_ci95=[float(x) for x in np.percentile(d_ww, [2.5, 97.5])],
                   delta_se=float((pr - pc).mean()), delta_se_ci95=[float(x) for x in np.percentile(d_se, [2.5, 97.5])],
                   per_person_delta={p: float(v) for p, v in zip(self.persons, pr - pc)},
                   persons_favouring_candidate=int(((pr - pc) > 0).sum()))
        if ratio:
            out['R'] = float(point_ww / (sr.sum() / n.sum()))
            out['R_ci95'] = [float(x) for x in np.percentile(d_ww / Br, [2.5, 97.5])]
        return out

    def level_within(self, ref, cand):
        L = {m: np.zeros(len(self.persons)) for m in (ref, cand)}
        W = {m: np.zeros(len(self.persons)) for m in (ref, cand)}
        for i in range(len(self.persons)):
            k = self.pidx == i
            for m in (ref, cand):
                L[m][i], W[m][i] = level_within(self.P[m][k], self.y[k])
        out = {}
        for comp, D in (('level', L), ('within', W)):
            d = D[ref] - D[cand]
            se, ww = d[self.I].mean(1), (d * self.n)[self.I].sum(1) / self.n[self.I].sum(1)
            out[comp] = dict(subject_equal=float(d.mean()), subject_equal_ci95=[float(x) for x in np.percentile(se, [2.5, 97.5])],
                             window_weighted=float((d * self.n).sum() / self.n.sum()),
                             window_weighted_ci95=[float(x) for x in np.percentile(ww, [2.5, 97.5])])
        return out

    def within_auroc(self, m):
        vals = [(p, auroc(self.P[m][self.pidx == i], self.y[self.pidx == i])) for i, p in enumerate(self.persons)]
        arr = np.array([a for _, a in vals if a is not None])
        if not len(arr):
            raise ValueError('no person contributes both classes')
        J = np.random.default_rng(self.seed).integers(0, len(arr), size=(N_BOOT, len(arr)))
        return dict(n_defined=len(arr), n_undefined=sum(a is None for _, a in vals), mean=float(arr.mean()),
                    ci95=[float(x) for x in np.percentile(arr[J].mean(1), [2.5, 97.5])])

    def permutation_null(self, m):
        rng = np.random.default_rng(self.seed)
        items = []
        for i in range(len(self.persons)):
            k = self.pidx == i
            yy = self.y[k].astype(int)
            if 0 < yy.sum() < len(yy):
                items.append((tie_ranks(self.P[m][k]), int(yy.sum()), len(yy)))
        null = np.array([np.mean([(rk[rng.permutation(nn)[:npos]].sum() - npos * (npos + 1) / 2) / (npos * (nn - npos))
                                  for rk, npos, nn in items]) for _ in range(N_PERM)])
        return dict(null_mean=float(null.mean()), null_95th=float(np.percentile(null, 95)),
                    passes=bool(NC1_BAND[0] <= null.mean() <= NC1_BAND[1]))

    def time_matched(self, m):
        person = np.array([r['person'] for r in self.rows])
        stratum = np.floor(np.array([r['elapsed_s'] for r in self.rows]) / STRATUM_S).astype(int)
        win, pairs, used = pair_matrices(self.P[m], self.y.astype(int), person, stratum, self.persons)
        Ib = np.random.default_rng(self.seed).integers(0, len(self.persons), size=(N_TM_BOOT, len(self.persons)))
        C = np.stack([np.bincount(r, minlength=len(self.persons)) for r in Ib]).astype(float)
        boots = np.array([weighted_auc(win, pairs, c) for c in C])
        return dict(auroc=weighted_auc(win, pairs, np.ones(len(self.persons))),
                    ci95=[float(x) for x in np.nanpercentile(boots, [2.5, 97.5])],
                    n_cross_person_pairs=int(pairs.sum()), n_strata_contributing=int(used))


def decide(prim, se_delta, s4, nc1_pass, nc2, nc4_pass):
    """H1 decision from the primary contrast, delta_SE, the AUROC block s4 and the negative controls."""
    flags = dict(STATISTICALLY_POSITIVE=bool(prim['delta_ww_ci95'][0] > 0),
                 PRACTICALLY_MEANINGFUL_POINT=bool(prim['R'] >= PRACTICAL),
                 PRACTICALLY_ESTABLISHED=bool(prim['R_ci95'][0] >= PRACTICAL))
    if not (nc1_pass and nc4_pass):
        return dict(outcome='INVALID', flags=flags, failed=[n for n, ok in (('NC1', nc1_pass), ('NC4', nc4_pass)) if not ok])
    go_cond = dict(statistically_positive=flags['STATISTICALLY_POSITIVE'],
                   practically_meaningful_point=flags['PRACTICALLY_MEANINGFUL_POINT'],
                   subject_equal_delta_gt_0=bool(se_delta > 0),
                   within_auroc_ci_lower_gt_half=bool(s4['ci95'][0] > 0.5),
                   nc2_donor_does_not_reproduce_gain=bool(nc2['delta_ww_ci95'][0] <= 0))
    go = all(go_cond.values())
    nogo = bool(prim['R_ci95'][1] < PRACTICAL or (prim['delta_ww'] <= 0 and se_delta <= 0))
    failed = [k for k, ok in go_cond.items() if not ok]
    if go and nogo:
        return dict(outcome='INCONCLUSIVE', flags=flags, go_conditions=go_cond, failed=['anomaly: GO and NO-GO both hold'])
    if go:
        return dict(outcome='GO', flags=flags, go_conditions=go_cond, failed=[])
    if nogo:
        return dict(outcome='NO-GO', flags=flags, go_conditions=go_cond, failed=failed)
    return dict(outcome='INCONCLUSIVE', flags=flags, go_conditions=go_cond, failed=failed)


def score_primary(rows, P, donor_predict, nc4_pass=True, s4=None):
    """H1 for one cohort. P must contain 'CONSTANT' and 'EEG_LOGIT'; donor_predict(plan) returns the candidate's
    probabilities when every window's EEG is replaced by its donor window (time features kept). s4 overrides the
    within-person AUROC block (EXT uses recording_auroc)."""
    S = Scorer(rows, P)
    b = lambda m: S.brier(P[m])  # noqa: E731
    prim = S.contrast(b('CONSTANT'), b('EEG_LOGIT'))
    person = [r['person'] for r in rows]
    elapsed = [r['elapsed_s'] for r in rows]
    donor = [S.brier(np.asarray(donor_predict(donor_plan(person, elapsed, d)[0]), float)) for d in range(DRAWS)]
    nc2 = S.contrast(b('CONSTANT'), np.mean(donor, axis=0))
    nc1 = S.permutation_null('EEG_LOGIT')
    s4 = S.within_auroc('EEG_LOGIT') if s4 is None else s4
    decision = decide(prim, prim['delta_se'], s4, nc1['passes'], nc2, nc4_pass)
    return dict(outcome=decision['outcome'], decision=decision, primary=prim, NC1_permutation=nc1, NC2_donor=nc2,
                S2_logloss=S.contrast(S.logloss(P['CONSTANT']), S.logloss(P['EEG_LOGIT']), ratio=False),
                S3_level_within=S.level_within('CONSTANT', 'EEG_LOGIT'), S4_within_auroc=s4,
                S5_time_matched=S.time_matched('EEG_LOGIT'))


def loo_R(S, ref, cand):
    """Smallest Brier skill after leaving out one person."""
    br, bc = S._sums(S.brier(S.P[ref])), S._sums(S.brier(S.P[cand]))
    return float(min(((br.sum() - br[i]) - (bc.sum() - bc[i])) / (br.sum() - br[i]) for i in range(len(br))))


def contrast_block(S, ref, cand):
    c = S.contrast(S.brier(S.P[ref]), S.brier(S.P[cand]))
    return dict(c, loo_R_min=loo_R(S, ref, cand), persons_worse=int(sum(v < 0 for v in c['per_person_delta'].values())))


def h2_outcome(h1_outcome, h2, integrity_ok):
    if not integrity_ok:
        return 'INVALID'
    supported = bool(h2['R'] >= PRACTICAL and h2['delta_ww_ci95'][0] > 0)
    if h1_outcome != 'GO':
        return 'NOT_TESTED' + ('(would_be_SUPPORTED)' if supported else '(would_be_NOT_SUPPORTED)')
    return 'SUPPORTED' if supported else 'NOT_SUPPORTED'


def score_sequential(rows, Q, h1_outcome, integrity_ok=True):
    """H2 (SpecLR-18 -> RawCNN-46) and H3 (SpecLR-46 -> RawCNN-46). Q keys: 'E7B_EEG', 'M0_46', 'M2_46'."""
    S = Scorer(rows, Q, seed=SEED_H2)
    h2 = contrast_block(S, 'E7B_EEG', 'M2_46')
    h3 = contrast_block(S, 'M0_46', 'M2_46')
    return dict(H2=dict(outcome=h2_outcome(h1_outcome, h2, integrity_ok), **h2), H3=dict(status='DESCRIPTIVE_ONLY', **h3))


def condition_paired(rows, Q, ref, cand, seed=SEED_H2, conditions=('placebo', 'temazepam')):
    """Persons observed in both conditions: per-person (placebo - temazepam) Brier gain of cand over ref. Descriptive."""
    y = np.array([r['y'] for r in rows], float)
    lr, lc = (Q[ref] - y) ** 2, (Q[cand] - y) ** 2
    g = defaultdict(dict)
    for cond in conditions:
        for p in sorted({r['person'] for r in rows}):
            k = np.array([r['person'] == p and r['condition'] == cond for r in rows])
            if k.any():
                g[p][cond] = float(lr[k].mean() - lc[k].mean())
    both = sorted(p for p, v in g.items() if len(v) == 2)
    if not both:
        raise ValueError('no person observed in both conditions')
    d = np.array([g[p][conditions[0]] - g[p][conditions[1]] for p in both])
    J = np.random.default_rng(seed).integers(0, len(d), size=(N_BOOT, len(d)))
    return dict(n_persons=len(both), placebo_minus_temazepam_mean=float(d.mean()),
                ci95=[float(x) for x in np.percentile(d[J].mean(1), [2.5, 97.5])])
