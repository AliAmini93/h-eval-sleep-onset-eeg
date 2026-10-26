#!/usr/bin/env python3
"""H-EVAL v7r1: DESCRIPTIVE person-cluster uncertainty for two point estimates shown without CIs in manuscript v7.
Uses ONLY existing frozen TRAIN out-of-fold predictions; no model is trained, tuned or recalibrated; no held-out data.

(1) TRAIN Stage-1-input R (SpecLR recipe M0 and RawCNN recipe M2 vs Constant T0), CTH partition-A OOF
    (.local/h_eval/cth_v1/cth_oof_predictions_v1.npz), stage from the W2S TRAIN table. Paired person-cluster
    percentile bootstrap, 10,000 replicates, seed 20261024 (same estimator as R1/R2).
(2) E1 lead-time AUROC bins (0,60], (60,120], (120,180], (180,300] s before first N1, positives vs far negatives
    (time to N1 > 600 s), within-fold exact-landmark cross-person pairs (P0b definition), state-aware TCN (N3) OOF.
    Person-cluster bootstrap (2,000 replicates, seed 20261024; pair weight = product of person multiplicities).
Rule fixed before computing: an interval is reported only if >= 8 persons contribute positive windows (and >= 8
contribute negatives); otherwise it is marked 'sparse_not_reported'. Point estimates must reproduce the v7 values."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402
import w2s_p0a_diagnostic as p0a  # noqa: E402

REPO = ev.REPO
OUT = REPO / 'results/decision/h_eval/discovery_w2s/w2s_v7r1_descriptive_ci.json'
MIN_P = 8


def main():
    d = ev.load()
    o = np.load(REPO / '.local/h_eval/cth_v1/cth_oof_predictions_v1.npz')
    assert np.array_equal(o['person'].astype(str), d['person']) and np.array_equal(o['y'], d['y'])
    y = d['y'].astype(float)
    s1 = d['e3'].astype(bool)
    res = dict(status='DESCRIPTIVE_FOLLOW_UP_TRAIN_OOF_ONLY_NO_NEW_MODEL', min_persons_for_ci=MIN_P, stage1={}, lead_time={})
    for name, key in (('SpecLR (M0 recipe)', 'M0_e5_logistic'), ('RawCNN (M2 recipe)', 'M2_raw_cnn_binary')):
        c = ev.paired(o['T0_constant'][s1], o[key][s1], y[s1], d['person'][s1])
        c['ci_reported'] = c['n_persons'] >= MIN_P
        res['stage1'][name] = {k: v for k, v in c.items()}
    # lead-time bins
    m = d['e1'].astype(bool)
    pf = ev.folds(d['person'], ev.FOLD_SALTS['A'])
    tte = d['first_sleep_s'] - d['elapsed']
    p = np.load(REPO / '.local/h_eval/discovery_w2s/w2s_neural_oof_N3_tcn5min_state.npz')['p_n1']
    far = m & (tte > 600)
    rng = np.random.default_rng(ev.BOOT_SEED)
    for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 300)):
        pos = m & (tte > lo) & (tte <= hi)
        sub = pos | far
        per = d['person'][sub]
        pu = sorted(set(per))
        pid = np.searchsorted(pu, per)
        pi, ni = p0a.pair_index((d['elapsed'][sub] // 30).astype(int), pf[sub], pid, pos[sub].astype(int), True)
        ps = p[sub]
        c = (ps[pi] > ps[ni]) + 0.5 * (ps[pi] == ps[ni])
        est = float(c.mean())
        n_pos_p, n_neg_p = len(set(pid[pi])), len(set(pid[ni]))
        row = dict(n_pairs=int(len(pi)), n_persons_with_positive_pairs=n_pos_p, n_persons_with_negative_pairs=n_neg_p,
                   n_positive_windows=int(pos.sum()), auroc=est)
        if n_pos_p >= MIN_P and n_neg_p >= MIN_P:
            bs = []
            for _ in range(2000):
                w = np.bincount(rng.integers(0, len(pu), len(pu)), minlength=len(pu)).astype(float)
                ww = w[pid[pi]] * w[pid[ni]]
                bs.append(float((c * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan)
            bs = np.array(bs)
            row.update(ci95=[float(x) for x in np.nanpercentile(bs, [2.5, 97.5])], bootstrap_undefined=int(np.isnan(bs).sum()))
        else:
            row['ci95'] = 'sparse_not_reported'
        res['lead_time'][f'({lo},{hi}]'] = row
    # consistency with v7 point estimates
    p0b = json.loads((REPO / 'results/decision/h_eval/discovery_w2s/w2s_p0b_calibration_v1.json').read_text())
    v7 = p0b['extra']['N3_tcn5min_state|K0']['lead_time']
    r1 = json.loads((REPO / 'results/decision/h_eval/repair_r1/r1_stage_construct.json').read_text())['stage_stratified_brier_ww']['1']
    res['consistency'] = dict(
        lead_time_point_estimates_equal_v7=all(abs(res['lead_time'][k]['auroc'] - v7[k]['auroc_vs_far_V4']) < 1e-12 for k in v7),
        stage1_point_estimates_equal_v7={'M0': abs(res['stage1']['SpecLR (M0 recipe)']['R'] - (r1['T0'] - r1['M0']) / r1['T0']) < 1e-12,
                                         'M2': abs(res['stage1']['RawCNN (M2 recipe)']['R'] - (r1['T0'] - r1['M2']) / r1['T0']) < 1e-12})
    res['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with OUT.open('x') as f:
        f.write(json.dumps(res, indent=1) + '\n')
    for k, v in res['stage1'].items():
        print(k, round(100 * v['R'], 1), [round(100 * x, 1) for x in v['R_ci95']], 'persons', v['n_persons'])
    for k, v in res['lead_time'].items():
        print(k, round(v['auroc'], 3), v['ci95'] if isinstance(v['ci95'], str) else [round(x, 3) for x in v['ci95']],
              'pairs', v['n_pairs'], 'persons+', v['n_persons_with_positive_pairs'])
    print(res['consistency'])


if __name__ == '__main__':
    main()
