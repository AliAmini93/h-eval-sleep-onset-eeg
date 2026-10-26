#!/usr/bin/env python3
"""H-EVAL W2S: references (CONST_pop, TIME_pop, ORACLE_state) and classical families C1/C2 on TRAIN.

Design: docs/decision/h_eval/discovery_w2s/w2s_design_v1.md. TRAIN-only, person-disjoint, two fold partitions.
Writes OOF predictions to .local/h_eval/discovery_w2s/ and aggregate results to results/decision/h_eval/discovery_w2s/.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402

REPO = ev.REPO
RES = REPO / 'results/decision/h_eval/discovery_w2s'
LOC = REPO / '.local/h_eval/discovery_w2s'
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TIME_GRID = [(k, c) for k in (3, 5) for c in (0.1, 1.0, 10.0)]


def features(d):
    return np.hstack([d['feat_cur'], d['feat_mean_prev'], d['feat_slope'], ev.time_feats(d['elapsed'])])


def run_partition(d, part):
    pf = ev.folds(d['person'], ev.FOLD_SALTS[part])
    X = features(d)
    t = d['elapsed']
    post = (t >= d['first_sleep_s']).astype(float)
    oracle_extra = np.column_stack([post, np.log1p(np.maximum(t - d['first_sleep_s'], 0) / 60)])
    out, log = {}, []
    for E, (mk, yk) in ev.ESTIMANDS.items():
        pop, y = d[mk].astype(bool), d[yk].astype(float)
        P = {k: np.full(len(y), np.nan) for k in ('CONST_pop', 'TIME_pop', 'ORACLE_state', 'C1_logistic', 'C2_hgb')}
        for f in range(ev.N_FOLDS):
            tr, te = np.where(pop & (pf != f))[0], np.where(pop & (pf == f))[0]
            P['CONST_pop'][te] = ev.fit_const(y[tr])(len(te))
            cfg = ev.select_inner(lambda a, b, c: ev.spline_logit(t[a], y[a], t[b], *c), TIME_GRID, tr, y, d['person'])
            P['TIME_pop'][te] = ev.spline_logit(t[tr], y[tr], t[te], *cfg)
            P['ORACLE_state'][te] = ev.spline_logit(t[tr], y[tr], t[te], 3, 1.0, oracle_extra[tr], oracle_extra[te])

            def c1(a, b, C):
                mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
                clf = LogisticRegression(C=C, max_iter=5000).fit((X[a] - mu) / sd, y[a])
                return np.clip(clf.predict_proba((X[b] - mu) / sd)[:, 1], 1e-6, 1 - 1e-6)
            C = ev.select_inner(c1, C_GRID, tr, y, d['person'])
            P['C1_logistic'][te] = c1(tr, te, C)
            hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                 min_samples_leaf=40, early_stopping=False, random_state=0)
            P['C2_hgb'][te] = np.clip(hgb.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            log.append(dict(partition=part, estimand=E, fold=f, time_cfg=list(cfg), c1_C=C, n_train=int(len(tr)),
                            n_test=int(len(te))))
        out[E] = P
    return out, log


def summarise(d, oof, oof_existing=None):
    res = {}
    for E, (mk, yk) in ev.ESTIMANDS.items():
        y = d[yk].astype(float)
        P = oof[E]
        theta = 2 * P['CONST_pop']
        tte = (d['first_sleep_s'] - d['elapsed']) if E == 'E1' else d['tau']
        refs = {k: P[k] for k in ('CONST_pop', 'TIME_pop')}
        preds = {k: P[k] for k in ('C1_logistic', 'C2_hgb')}
        if E != 'E1':
            preds['ORACLE_state(diagnostic)'] = P['ORACLE_state']
            if oof_existing is not None:
                preds['M0_existing_allstage(context)'] = oof_existing['M0_e5_logistic']
                preds['M2_existing_allstage(context)'] = oof_existing['M2_raw_cnn_binary']
        res[E] = [ev.evaluate(s, preds, refs, d, y, d[s].astype(bool), theta, tte) for s in ev.EVAL_SUBSETS[E]]
    return res


def main():
    t0 = time.time()
    d = ev.load()
    o = np.load(REPO / '.local/h_eval/cth_v1/cth_oof_predictions_v1.npz')
    assert np.array_equal(o['person'].astype(str), d['person']) and np.array_equal(o['y'], d['y'])
    results, logs = {}, []
    for part in ('A', 'B'):
        oof, log = run_partition(d, part)
        logs += log
        np.savez(LOC / f'w2s_classical_oof_{part}.npz', **{f'{E}__{k}': v for E, P in oof.items() for k, v in P.items()})
        results[part] = summarise(d, oof, o if part == 'A' else None)
    out = dict(version='H-EVAL-W2S-screen-v1-2026-10-24', status='TRAIN_ONLY_EXPLORATORY',
               design_sha256=hashlib.sha256((REPO / 'docs/decision/h_eval/discovery_w2s/w2s_design_v1.md').read_bytes()).hexdigest(),
               table_sha256=hashlib.sha256(ev.TABLE.read_bytes()).hexdigest(),
               code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (Path(__file__), Path(ev.__file__))},
               note='Existing M0/M2 OOF are fold-A only (they were produced on the CTH partition).',
               results=results, fold_log=logs, runtime_s=round(time.time() - t0, 1))
    RES.mkdir(parents=True, exist_ok=True)
    with (RES / 'w2s_classical_results_v1.json').open('x') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    for part in ('A', 'B'):
        for E, subs in results[part].items():
            for s in subs:
                print(part, E, s['subset'], s['n_windows'], s['n_positive'], s['n_persons'])
                for k, v in s['vs_time'].items():
                    print(f"   {k:32s} R_vs_time {v['R']*100:6.1f} ({v['R_ci95'][0]*100:.1f},{v['R_ci95'][1]*100:.1f}) "
                          f"dSE {v['delta_se']:+.4f} ({v['delta_se_ci95'][0]:+.4f},{v['delta_se_ci95'][1]:+.4f}) "
                          f"| R_vs_const {s['vs_const'][k]['R']*100:6.1f} fav {v['persons_favouring']}/{v['n_persons']}")


if __name__ == '__main__':
    main()
