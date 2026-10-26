#!/usr/bin/env python3
"""H-EVAL R3A: TRAIN-only oracle-state references A (state-only) and B (state + elapsed time), secondary C (+ time since
first scored sleep) and D (B + 10 spectral features), compared with existing EEG OOF predictions on identical windows.
Design: docs/decision/h_eval/repair_r3a/r3a_state_reference_design_v1.md (hash-checked). No neural network is trained;
no VAL/TEST/EXT data are read. Writes results/decision/h_eval/repair_r3a/r3a_state_reference_results_v1.json."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'discovery_w2s'))
import w2s_eval as ev  # noqa: E402
import w2s_p0a_diagnostic as p0a  # noqa: E402

REPO = ev.REPO
DESIGN = REPO / 'docs/decision/h_eval/repair_r3a/r3a_state_reference_design_v1.md'
FREEZE = REPO / 'results/decision/h_eval/repair_r3a/r3a_design_freeze.json'
OUT = REPO / 'results/decision/h_eval/repair_r3a/r3a_state_reference_results_v1.json'
INNER_SALT, N_INNER, MIN_P = 'H-EVAL-R3A-inner', 4, 8
B_GRID = [(k, C, inter) for k in (3, 5) for C in (0.1, 1.0, 10.0) for inter in (False, True)]
C_GRID = (0.1, 1.0, 10.0)


def inner_folds(persons):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{INNER_SALT}|{p}'.encode()).hexdigest())
    f = {p: i % N_INNER for i, p in enumerate(ranked)}
    return np.array([f[p] for p in persons])


def design_matrix(t, s1, extra, st, inter):
    S = st.transform(np.log(t)[:, None])
    X = [s1[:, None], S]
    if inter:
        X.append(S * s1[:, None])
    if extra is not None:
        X.append(extra)
    return np.hstack(X)


def fit_predict_B(tr, te, cfg, d, extra=None, C_override=None):
    k, C, inter = cfg
    t, s1, y = d['t'], d['s1'], d['y']
    st = SplineTransformer(n_knots=k, degree=3, extrapolation='constant').fit(np.log(t[tr])[:, None])
    ex_tr = ex_te = None
    if extra is not None:
        mu, sd = extra[tr].mean(0), extra[tr].std(0) + 1e-9
        ex_tr, ex_te = (extra[tr] - mu) / sd, (extra[te] - mu) / sd
    clf = LogisticRegression(C=C_override or C, max_iter=5000).fit(design_matrix(t[tr], s1[tr], ex_tr, st, inter), y[tr])
    return np.clip(clf.predict_proba(design_matrix(t[te], s1[te], ex_te, st, inter))[:, 1], 1e-6, 1 - 1e-6)


def select(fitfn, grid, tr, d):
    fi = inner_folds(d['person'][tr])
    best, best_s = None, np.inf
    for g in grid:
        s = sum(((fitfn(tr[fi != h], tr[fi == h], g) - d['y'][tr[fi == h]]) ** 2).sum() for h in range(N_INNER))
        if s < best_s - 1e-12:
            best, best_s = g, s
    return best


def auc_exact(p, d, m, pf, rng, nboot=2000):
    per = d['person'][m]
    pu = sorted(set(per))
    pid = np.searchsorted(pu, per)
    pi, ni = p0a.pair_index((d['t'][m] // 30).astype(int), pf[m], pid, d['y'][m].astype(int), True)
    if len(pi) == 0:
        return dict(auroc=None, n_pairs=0)
    pm = p[m]
    c = (pm[pi] > pm[ni]) + 0.5 * (pm[pi] == pm[ni])
    out = dict(auroc=float(c.mean()), n_pairs=int(len(pi)), n_persons_pos=len(set(pid[pi])), n_persons_neg=len(set(pid[ni])))
    if out['n_persons_pos'] >= MIN_P and out['n_persons_neg'] >= MIN_P:
        bs = []
        for _ in range(nboot):
            w = np.bincount(rng.integers(0, len(pu), len(pu)), minlength=len(pu)).astype(float)
            ww = w[pid[pi]] * w[pid[ni]]
            bs.append((c * ww).sum() / ww.sum() if ww.sum() else np.nan)
        out['ci95'] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
    else:
        out['ci95'] = 'sparse_not_reported'
    return out


def pooled_auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    fr = json.loads(FREEZE.read_text())
    if hashlib.sha256(DESIGN.read_bytes()).hexdigest() != fr['sha256']:
        raise SystemExit('design differs from frozen hash')
    w = ev.load()
    z = np.load(REPO / '.local/h_eval/cth_v1/dataset_v1.npz')
    o = np.load(REPO / '.local/h_eval/cth_v1/cth_oof_predictions_v1.npz')
    assert np.array_equal(o['person'].astype(str), w['person']) and np.array_equal(o['y'], w['y'])
    assert np.array_equal(z['person'].astype(str), w['person']) and np.allclose(z['elapsed'], w['elapsed'])
    stage = w['stage']
    assert set(stage) == {'W', '1'}, set(stage)
    d = dict(t=w['elapsed'].astype(float), s1=(stage == '1').astype(float), y=w['y'].astype(float), person=w['person'],
             since_sleep=np.log1p(np.maximum(w['elapsed'] - w['first_sleep_s'], 0) / 60)[:, None])
    e5 = z['e5'].astype(float)
    pf = ev.folds(w['person'], ev.FOLD_SALTS['A'])
    n = len(d['y'])
    P = {k: np.full(n, np.nan) for k in ('A_state', 'B_state_time', 'C_state_time_since_sleep', 'D_B_plus_spectral')}
    sel = []
    for f in range(ev.N_FOLDS):
        tr, te = np.where(pf != f)[0], np.where(pf == f)[0]
        for s in (0.0, 1.0):
            a = tr[d['s1'][tr] == s]
            b = te[d['s1'][te] == s]
            P['A_state'][b] = (d['y'][a].sum() + 1) / (len(a) + 2)
        cfg = select(lambda a, b, g: fit_predict_B(a, b, g, d), B_GRID, tr, d)
        P['B_state_time'][te] = fit_predict_B(tr, te, cfg, d)
        cC = select(lambda a, b, g: fit_predict_B(a, b, cfg, d, d['since_sleep'], g), C_GRID, tr, d)
        P['C_state_time_since_sleep'][te] = fit_predict_B(tr, te, cfg, d, d['since_sleep'], cC)
        cD = select(lambda a, b, g: fit_predict_B(a, b, cfg, d, e5, g), C_GRID, tr, d)
        P['D_B_plus_spectral'][te] = fit_predict_B(tr, te, cfg, d, e5, cD)
        sel.append(dict(fold=f, B_cfg=dict(knots=cfg[0], C=cfg[1], interaction=cfg[2]), C_C=cC, D_C=cD))
    assert all(np.isfinite(v).all() for v in P.values())
    P.update(T0_constant=o['T0_constant'], M0_spectral_oof=o['M0_e5_logistic'], M2_rawcnn_oof=o['M2_raw_cnn_binary'])
    strata = {'ALL': np.ones(n, bool), 'W': d['s1'] == 0, 'STAGE_1': d['s1'] == 1}
    pairs = [('T0_constant', 'A_state'), ('T0_constant', 'B_state_time'), ('A_state', 'M0_spectral_oof'),
             ('A_state', 'M2_rawcnn_oof'), ('B_state_time', 'M0_spectral_oof'), ('B_state_time', 'M2_rawcnn_oof'),
             ('B_state_time', 'C_state_time_since_sleep'), ('B_state_time', 'D_B_plus_spectral'),
             ('C_state_time_since_sleep', 'M0_spectral_oof'), ('C_state_time_since_sleep', 'M2_rawcnn_oof'),
             ('T0_constant', 'M0_spectral_oof'), ('T0_constant', 'M2_rawcnn_oof')]
    rng = np.random.default_rng(ev.BOOT_SEED)
    res = dict(version='H-EVAL-R3A-state-ref-v1-2026-10-25', status='POST_HOC_EXPLORATORY_TRAIN_ONLY_ORACLE_STATE_DIAGNOSTIC',
               design_sha256=fr['sha256'], code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               selection=sel, strata={})
    for sn, m in strata.items():
        y, per = d['y'][m], d['person'][m]
        S = dict(n_windows=int(m.sum()), n_positive=int(y.sum()), n_persons=len(set(per)), n_recordings=len(set(w['recording'][m])),
                 prevalence=float(y.mean()), models={}, contrasts={})
        for k, p in P.items():
            mm = ev.model_metrics(p[m], y, per)
            mm['pooled_auroc'] = pooled_auc(p[m], y)
            mm['exact_time_within_fold_auroc'] = auc_exact(p, d, m, pf, rng)
            S['models'][k] = mm
        for a, b in pairs:
            c = ev.paired(P[a][m], P[b][m], y, per)
            c['ci_reported'] = c['n_persons'] >= MIN_P
            S['contrasts'][f'{a}->{b}'] = c
        res['strata'][sn] = S
    with OUT.open('x') as fh:
        fh.write(json.dumps(res, indent=1) + '\n')
    for sn, S in res['strata'].items():
        print(f"== {sn}: windows {S['n_windows']} pos {S['n_positive']} persons {S['n_persons']} prev {S['prevalence']:.3f}")
        for k, m in S['models'].items():
            e = m['exact_time_within_fold_auroc']
            ci = e.get('ci95')
            print(f"   {k:26s} BrierWW {m['brier_ww']:.4f} SE {m['brier_se']:.4f} CIL {m['cal_in_large']:+.3f} slope {m['cal_slope'] if m['cal_slope'] is None else round(m['cal_slope'], 2)} "
                  f"AUROC {m['pooled_auroc'] if m['pooled_auroc'] is None else round(m['pooled_auroc'], 3)} exactAUROC {e['auroc'] if e['auroc'] is None else round(e['auroc'], 3)} {ci if isinstance(ci, str) or ci is None else [round(x, 3) for x in ci]} pairs {e['n_pairs']}")
        for k, c in S['contrasts'].items():
            print(f"   {k:52s} dWW {c['delta_ww']:+.4f} R {100 * c['R']:6.1f} ({100 * c['R_ci95'][0]:.1f}, {100 * c['R_ci95'][1]:.1f}) dSE {c['delta_se']:+.4f} ({c['delta_se_ci95'][0]:+.4f}, {c['delta_se_ci95'][1]:+.4f}) fav {c['persons_favouring']}/{c['n_persons']}")
    print(json.dumps(sel))


if __name__ == '__main__':
    main()
