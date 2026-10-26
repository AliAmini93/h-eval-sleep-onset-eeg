#!/usr/bin/env python3
"""H-EVAL W2S P1 evaluation (design w2s_p0p1_design_v1.md, P1): variants A-E on partitions A and B, against the
within-population E1 references of the same partition; seeds; donor-history control; nested-calibrated A and E with the
calibrator selected in P0b. Aggregates only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402
import w2s_p0a_diagnostic as p0a  # noqa: E402
import w2s_p0b_calibration as p0b  # noqa: E402

LOC = ev.REPO / '.local/h_eval/discovery_w2s'
P1 = LOC / 'p1'
RES = ev.REPO / 'results/decision/h_eval/discovery_w2s'
SEEDS = (0, 1, 2)


def nested_cal(kind, outer, inner, m, pf, y, u_all):
    out = np.full(len(y), np.nan)
    for f in range(ev.N_FOLDS):
        cal, te = m & (pf != f), m & (pf == f)
        pin = inner[f'outer{f}']
        assert np.isfinite(pin[cal]).all()
        uc = u_all[cal]
        mu, sd = uc.mean(), uc.std() + 1e-9
        out[te], _ = p0b.calibrate_fold(kind, p0b.logit(pin[cal]), (uc - mu) / sd, y[cal], pf[cal],
                                        p0b.logit(outer[te]), (u_all[te] - mu) / sd)
    return out


def extras(p, d, m, pf, y):
    per, el = d['person'], d['elapsed']
    pid = np.searchsorted(sorted(set(per[m])), per[m])
    pi, ni = p0a.pair_index((el[m] // 30).astype(int), pf[m], pid, y[m].astype(int), True)
    pm = p[m]
    lev = wit = 0.0
    for q in sorted(set(per[m])):
        i = per[m] == q
        pb, yb = pm[i].mean(), y[m][i].mean()
        lev += i.sum() * (pb - yb) ** 2
        wit += (((pm[i] - pb) - (y[m][i] - yb)) ** 2).sum()
    return dict(V4_auroc=p0a.auc_pairs(pm, pi, ni), level=lev / m.sum(), within=wit / m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrator', required=True, choices=('K1', 'K2', 'K3'), help='selected in P0b by its primary endpoint')
    a = ap.parse_args()
    d = ev.load()
    m = d['e1'].astype(bool)
    y = d['y1'].astype(float)
    u_all = np.log1p(d['elapsed'] / 60)
    out = {}
    for part in ('A', 'B'):
        pf = ev.folds(d['person'], ev.FOLD_SALTS[part])
        C = np.load(LOC / f'w2s_classical_oof_{part}.npz')
        refs = {'CONST_pop': C['E1__CONST_pop'], 'TIME_pop': C['E1__TIME_pop']}
        preds, seeds, donor = {}, {}, {}
        for vn in 'ABCDE':
            z = np.load(P1 / f'p1_outer_{part}_{vn}.npz')
            preds[vn] = z['p1']
            seeds[vn] = [ev.paired(refs['TIME_pop'][m], z[f'p1_seed{s}'][m], y[m], d['person'][m])['R'] for s in SEEDS]
            if vn in 'BCE':
                preds[f'{vn}|donor_history'] = z['p1_donor']
                donor[vn] = dict(own_minus_donor=ev.paired(z['p1_donor'][m], z['p1'][m], y[m], d['person'][m]))
        for vn in 'AE':
            f = P1 / f'p1_inner_{part}_{vn}.npz'
            if f.exists():
                preds[f'{vn}|{a.calibrator}'] = nested_cal(a.calibrator, preds[vn], np.load(f), m, pf, y, u_all)
        res = ev.evaluate('e1', preds, refs, d, y, m)
        theta = 2 * refs['CONST_pop']
        ext = {k: dict(**extras(p, d, m, pf, y),
                       warnings=p0b.warnings(p, theta, y, d['recording'], d['elapsed'], m)) for k, p in {**refs, **preds}.items()}
        out[part] = dict(results=res, seed_R_vs_time=seeds, donor_control=donor, extra=ext)
        for k in preds:
            v, mm, e = res['vs_time'][k], res['models'][k], ext[k]
            print(f"{part} {k:18s} R_vs_time {v['R']*100:6.1f} ({v['R_ci95'][0]*100:6.1f},{v['R_ci95'][1]*100:5.1f}) "
                  f"dSE {v['delta_se']:+.4f} ({v['delta_se_ci95'][0]:+.4f},{v['delta_se_ci95'][1]:+.4f}) LL {mm['logloss_ww']:.4f} "
                  f"CIL {mm['cal_in_large']:+.3f} slope {mm['cal_slope'] and round(mm['cal_slope'], 2)} V4 {e['V4_auroc']:.3f} "
                  f"lvl {e['level']:.4f} win {e['within']:.4f} FAep/h {e['warnings']['false_alarm_episodes_per_hour']:.2f} "
                  f"sens {e['warnings']['event_sensitivity']:.2f}" + (f" seeds {[round(x*100, 1) for x in seeds[k]]}" if k in seeds else ''))
        for vn, dc in donor.items():
            o = dc['own_minus_donor']
            print(f"{part} donor control {vn}: own vs donor-history R {o['R']*100:.1f} ({o['R_ci95'][0]*100:.1f},{o['R_ci95'][1]*100:.1f})")
    doc = dict(version='H-EVAL-W2S-P1-v1-2026-10-24', status='TRAIN_ONLY_EXPLORATORY', calibrator=a.calibrator,
               design_sha256=hashlib.sha256((ev.REPO / 'docs/decision/h_eval/discovery_w2s/w2s_p0p1_design_v1.md').read_bytes()).hexdigest(),
               code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (Path(__file__), Path(__file__).parent / 'w2s_p1_personal.py')},
               note='Partitions A and B are re-splits of the same 46 TRAIN persons; not independent validation.', partitions=out)
    with (RES / 'w2s_p1_results_v1.json').open('x') as f:
        f.write(json.dumps(doc, indent=1) + '\n')


if __name__ == '__main__':
    main()
