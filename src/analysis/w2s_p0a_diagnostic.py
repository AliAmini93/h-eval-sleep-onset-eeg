#!/usr/bin/env python3
"""H-EVAL W2S P0a: audit of the E1 time-matched cross-person AUROC (design w2s_p0p1_design_v1.md, P0a).
Existing partition-A OOF only. Variants V1-V4 (V4 primary), permutation null for V4, residual-variation report."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402

LOC = ev.REPO / '.local/h_eval/discovery_w2s'
OUT = ev.REPO / 'results/decision/h_eval/discovery_w2s/w2s_p0a_diagnostic_v1.json'
B_BOOT, N_PERM = 2000, 1000


def load_preds(d):
    C = np.load(LOC / 'w2s_classical_oof_A.npz')
    P = {'TIME_pop': C['E1__TIME_pop'], 'CONST_pop': C['E1__CONST_pop'], 'C1_logistic': C['E1__C1_logistic']}
    for k in ('N2_tcn5min', 'N3_tcn5min_state'):
        P[k] = np.load(LOC / f'w2s_neural_oof_{k}.npz')['p_n1']
    return P


def pair_index(strata, fold, pid, y, within_fold):
    """All cross-person (pos, neg) pairs in the same stratum (and fold if within_fold). Returns arrays of indices."""
    pi, ni = [], []
    key = strata if not within_fold else strata * 10 + fold
    for s in np.unique(key):
        idx = np.where(key == s)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if not len(pos) or not len(neg):
            continue
        P, N = np.meshgrid(pos, neg, indexing='ij')
        keep = pid[P] != pid[N]
        pi.append(P[keep])
        ni.append(N[keep])
    return np.concatenate(pi), np.concatenate(ni)


def auc_pairs(p, pi, ni):
    return float(((p[pi] > p[ni]) + 0.5 * (p[pi] == p[ni])).mean())


def main():
    d = ev.load()
    m = d['e1'].astype(bool)
    pf = ev.folds(d['person'], ev.FOLD_SALTS['A'])[m]
    y = d['y1'][m].astype(int)
    el = d['elapsed'][m]
    per = d['person'][m]
    pu = sorted(set(per))
    pid = np.searchsorted(pu, per)
    P = {k: v[m] for k, v in load_preds(d).items()}
    variants = {'V1_300s_pooled': ((el // 300).astype(int), False), 'V2_300s_within_fold': ((el // 300).astype(int), True),
                'V3_30s_exact_pooled': ((el // 30).astype(int), False),
                'V4_30s_exact_within_fold_PRIMARY': ((el // 30).astype(int), True)}
    rng = np.random.default_rng(ev.BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, len(pu), len(pu)), minlength=len(pu)) for _ in range(B_BOOT)]).astype(float)
    res = {}
    for vn, (strata, wf) in variants.items():
        pi, ni = pair_index(strata, pf, pid, y, wf)
        res[vn] = dict(n_pairs=int(len(pi)), n_pos_windows=int(len(set(pi))), n_neg_windows=int(len(set(ni))), models={})
        for k, p in P.items():
            c = (p[pi] > p[ni]) + 0.5 * (p[pi] == p[ni])
            est = float(c.mean())
            bs = [float((c * w[pid[pi]] * w[pid[ni]]).sum() / (w[pid[pi]] * w[pid[ni]]).sum()) for w in W]
            res[vn]['models'][k] = dict(auroc=est, ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])])
    # permutation null for V4: permute predictions within (fold, elapsed_s) cells
    strata, _ = variants['V4_30s_exact_within_fold_PRIMARY']
    cell = strata * 10 + pf
    pi, ni = pair_index(strata, pf, pid, y, True)
    groups = [np.where(cell == c)[0] for c in np.unique(cell)]
    prng = np.random.default_rng(20261025)
    perm = {}
    for k, p in P.items():
        obs = res['V4_30s_exact_within_fold_PRIMARY']['models'][k]['auroc']
        null = []
        for _ in range(N_PERM):
            q = p.copy()
            for g in groups:
                if len(g) > 1:
                    q[g] = p[prng.permutation(g)]
            null.append(auc_pairs(q, pi, ni))
        null = np.array(null)
        perm[k] = dict(observed=obs, null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)),
                       p_value=float((1 + (null >= obs).sum()) / (N_PERM + 1)))
    # residual variation
    folds = {}
    for f in range(5):
        i = pf == f
        folds[str(f)] = dict(n=int(i.sum()), persons=len(set(per[i])), event_rate=float(y[i].mean()),
                             **{f'mean_{k}': float(P[k][i].mean()) for k in P})
    bins = (el // 300).astype(int)
    within_bin_sd = float(np.mean([P['TIME_pop'][(bins == b) & (pf == f)].std()
                                   for b in np.unique(bins) for f in range(5) if ((bins == b) & (pf == f)).sum() > 1]))
    corr = {k: float(np.corrcoef([folds[str(f)][f'mean_{k}'] for f in range(5)],
                                 [folds[str(f)]['event_rate'] for f in range(5)])[0, 1]) for k in P}
    out = dict(version='H-EVAL-W2S-P0a-v1-2026-10-24', status='TRAIN_ONLY_EXPLORATORY_DIAGNOSTIC', estimand='E1 (oracle at-risk set)',
               design_sha256=hashlib.sha256((ev.REPO / 'docs/decision/h_eval/discovery_w2s/w2s_p0p1_design_v1.md').read_bytes()).hexdigest(),
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               variants=res, permutation_null_V4=perm, fold_levels=folds,
               fold_mean_vs_event_rate_correlation=corr, time_pop_mean_within_300s_bin_sd_within_fold=within_bin_sd)
    with OUT.open('x') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    for vn, r in res.items():
        print(vn, r['n_pairs'], {k: f"{v['auroc']:.3f} ({v['ci95'][0]:.3f},{v['ci95'][1]:.3f})" for k, v in r['models'].items()})
    print(json.dumps(perm, indent=0))
    print(json.dumps(folds, indent=0))
    print('fold mean vs event rate corr', {k: round(v, 2) for k, v in corr.items()}, 'TIME within-bin sd', within_bin_sd)


if __name__ == '__main__':
    main()
