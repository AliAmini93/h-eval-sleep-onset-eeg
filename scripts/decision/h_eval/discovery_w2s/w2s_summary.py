#!/usr/bin/env python3
"""H-EVAL W2S: evaluate neural OOF (partition A) against the same in-population references as the classical screen;
seed-level stability; fusion eligibility check (design §4 F6, §6 rule 1). Aggregates only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402
from w2s_neural import MODELS, SEEDS  # noqa: E402

REPO = ev.REPO
LOC = REPO / '.local/h_eval/discovery_w2s'
RES = REPO / 'results/decision/h_eval/discovery_w2s'


def main():
    d = ev.load()
    C = np.load(LOC / 'w2s_classical_oof_A.npz')
    N = {m: np.load(LOC / f'w2s_neural_oof_{m}.npz') for m in MODELS}
    res, seeds, fusion = {}, {}, {}
    for E, (mk, yk) in ev.ESTIMANDS.items():
        y = d[yk].astype(float)
        head = 'p_n1' if E == 'E1' else 'p_n2'
        refs = {k: C[f'{E}__{k}'] for k in ('CONST_pop', 'TIME_pop')}
        preds = {m: N[m][head] for m in MODELS}
        preds['C1_logistic'] = C[f'{E}__C1_logistic']
        theta = 2 * refs['CONST_pop']
        tte = (d['first_sleep_s'] - d['elapsed']) if E == 'E1' else d['tau']
        res[E] = [ev.evaluate(s, preds, refs, d, y, d[s].astype(bool), theta, tte) for s in ev.EVAL_SUBSETS[E]]
        seeds[E] = {}
        for s in ev.EVAL_SUBSETS[E]:
            m = d[s].astype(bool)
            seeds[E][s] = {mod: [ev.paired(refs['TIME_pop'][m], N[mod][f'{head}_seed{k}'][m], y[m], d['person'][m])['R']
                                 for k in SEEDS] for mod in MODELS}
        # fusion eligibility: both members must beat TIME_pop (CI lower > 0) on the primary subset
        m = d[ev.EVAL_SUBSETS[E][0]].astype(bool)
        vt = res[E][0]['vs_time']
        best_n = max(MODELS, key=lambda k: vt[k]['R'])
        elig = vt[best_n]['R_ci95'][0] > 0 and vt['C1_logistic']['R_ci95'][0] > 0
        r = np.corrcoef(preds[best_n][m] - y[m], preds['C1_logistic'][m] - y[m])[0, 1]
        fusion[E] = dict(best_neural=best_n, residual_corr_with_C1=float(r), eligible=bool(elig))
        if elig:
            fused = (preds[best_n] + preds['C1_logistic']) / 2
            fusion[E]['fused_vs_time'] = ev.paired(refs['TIME_pop'][m], fused[m], y[m], d['person'][m])
    out = dict(version='H-EVAL-W2S-screen-v1-2026-10-24', status='TRAIN_ONLY_EXPLORATORY', partition='A',
               logs={m: json.loads((LOC / f'w2s_neural_log_{m}.json').read_text()) for m in MODELS},
               code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            Path(__file__).resolve().parent.glob('w2s_*.py')},
               results=res, seed_R_vs_time=seeds, fusion=fusion)
    with (RES / 'w2s_neural_results_v1.json').open('x') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    for E, subs in res.items():
        for s in subs:
            print(E, s['subset'], s['n_windows'], s['n_positive'])
            for k, v in s['vs_time'].items():
                mm = s['models'][k]
                w = mm.get('warning', {})
                print(f"   {k:20s} R_vs_time {v['R']*100:6.1f} ({v['R_ci95'][0]*100:6.1f},{v['R_ci95'][1]*100:5.1f}) "
                      f"dSE {v['delta_se']:+.4f} ({v['delta_se_ci95'][0]:+.4f},{v['delta_se_ci95'][1]:+.4f}) fav {v['persons_favouring']}/46 "
                      f"CIL {mm['cal_in_large']:+.3f} slope {mm['cal_slope'] if mm['cal_slope'] is None else round(mm['cal_slope'], 2)} "
                      f"sens {w.get('sensitivity')} FA/h {None if w.get('false_alarms_per_hour') is None else round(w['false_alarms_per_hour'], 1)}"
                      + (f" seeds {[round(x*100,1) for x in seeds[E][s['subset']][k]]}" if k in seeds[E][s['subset']] else ''))
    print(json.dumps(fusion, indent=1))


if __name__ == '__main__':
    main()
