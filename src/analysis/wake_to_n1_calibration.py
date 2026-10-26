#!/usr/bin/env python3
"""H-EVAL W2S P0b: leakage-safe calibration of existing E1 predictions by nested person-disjoint cross-fitting
(design w2s_p0p1_design_v1.md, P0b). Partition A.

  inner   train inner models (same arch/seed; 3 folds; fixed epochs = outer model's best epoch) -> inner-OOF on outer-train
  eval    fit calibrators K1-K3 on inner-OOF of outer-train E1 windows; apply to the EXISTING outer predictions of fold f
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402
import w2s_neural as wn  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

LOC = ev.REPO / '.local/h_eval/discovery_w2s'
RES = ev.REPO / 'results/decision/h_eval/discovery_w2s'
ARCHS = ('N2_tcn5min', 'N3_tcn5min_state')
LAMBDAS = (0.1, 1.0, 10.0, 100.0)


# ------------------------------------------------------------------------ inner models (fixed epochs, no early stopping)
def train_fixed(d, cfg, tr, te, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = wn.Data(d, tr, cfg)
    net = wn.Net(cfg['enc'], cfg['state']).to(wn.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=wn.LR, weight_decay=wn.WD)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), wn.BATCH):
            b = torch.tensor(order[i:i + wn.BATCH], device=wn.DEV)
            (l1, m1), (l2, m2), s = net(D.batch(b))
            loss = wn.hazard_nll(l2, m2, D.K2[b], torch.ones(len(b), device=wn.DEV)) + wn.hazard_nll(l1, m1, D.K1[b], D.e1[b])
            if cfg['state']:
                loss = loss + wn.STAGE_W * F.binary_cross_entropy_with_logits(s, D.st[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            (l1, m1), _, _ = net(D.batch(torch.tensor(te[i:i + 256], device=wn.DEV)))
            out.append((1 - torch.exp(m1.sum(1))).cpu().numpy())
    return np.concatenate(out)


def cmd_inner(archs):
    d = ev.load()
    pf = ev.folds(d['person'], ev.FOLD_SALTS['A'])
    ledger = []
    for arch in archs:
        path = LOC / f'w2s_p0b_inner_{arch}.npz'
        if path.exists():
            raise SystemExit(f'{path} exists (write-once)')
        log = json.loads((LOC / f'w2s_neural_log_{arch}.json').read_text())['log']
        best_ep = {(r['fold'], r['seed']): r['epochs'] - wn.PATIENCE for r in log}
        assert all(r['epochs'] < wn.MAX_EPOCHS for r in log)
        inner = {f: np.full(len(d['y']), np.nan) for f in range(ev.N_FOLDS)}
        for f in range(ev.N_FOLDS):
            for h in [g for g in range(ev.N_FOLDS) if g != f]:
                te = np.where(pf == h)[0]
                tr = np.where((pf != f) & (pf != h))[0]
                ps = []
                for s in wn.SEEDS:
                    t0 = time.time()
                    ps.append(train_fixed(d, wn.MODELS[arch], tr, te, s, best_ep[(f, s)]))
                    ledger.append(dict(arch=arch, outer_fold=f, inner_fold=h, seed=s, epochs=best_ep[(f, s)],
                                       n_train=int(len(tr)), seconds=round(time.time() - t0, 1), status='ok'))
                    print(f'[P0b] {arch} f{f} h{h} s{s} ep{best_ep[(f, s)]}', flush=True)
                inner[f][te] = np.mean(ps, 0)
        np.savez(path, **{f'outer{f}': v for f, v in inner.items()})
        (LOC / f'w2s_p0b_inner_ledger_{arch}.json').write_text(json.dumps(ledger, indent=1))


# ------------------------------------------------------------------------ calibrators
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_cal(kind, s, u, y, lam):
    """Penalised log-loss fit. Returns params (a, b, c)."""
    if kind == 'K1':
        f = lambda th: -np.mean(y * -np.logaddexp(0, -(s + th[0])) + (1 - y) * -np.logaddexp(0, s + th[0]))  # noqa: E731
        th = minimize(f, [0.0], method='BFGS').x
        return th[0], 1.0, 0.0

    def obj(th):
        a, b = th[0], th[1]
        c = th[2] if kind == 'K3' else 0.0
        z = a + b * s + c * u
        nll = np.mean(np.logaddexp(0, z) - y * z)
        return nll + lam * ((b - 1) ** 2 + c ** 2) / len(y)          # = (sum NLL + lam * penalty) / n
    th = minimize(obj, [0.0, 1.0, 0.0] if kind == 'K3' else [0.0, 1.0], method='BFGS').x
    return th[0], th[1], (th[2] if kind == 'K3' else 0.0)


def apply_cal(params, s, u):
    a, b, c = params
    return np.clip(1 / (1 + np.exp(-(a + b * s + c * u))), 1e-6, 1 - 1e-6)


def calibrate_fold(kind, s_cal, u_cal, y_cal, g_cal, s_te, u_te):
    """Choose lambda by leave-one-inner-fold-out (window-weighted Brier) on the calibration set; refit; apply."""
    lam = None
    if kind in ('K2', 'K3'):
        best = np.inf
        for L in LAMBDAS:
            sc = 0.0
            for g in np.unique(g_cal):
                a, b = g_cal != g, g_cal == g
                sc += ((apply_cal(fit_cal(kind, s_cal[a], u_cal[a], y_cal[a], L), s_cal[b], u_cal[b]) - y_cal[b]) ** 2).sum()
            if sc < best - 1e-12:
                best, lam = sc, L
    params = fit_cal(kind, s_cal, u_cal, y_cal, lam or 0.0)
    return apply_cal(params, s_te, u_te), dict(lambda_=lam, a=float(params[0]), b=float(params[1]), c=float(params[2]))


# ------------------------------------------------------------------------ warnings (corrected) and lead-time
def warnings(p, theta, y, rec, el, mask):
    alarm = (p >= theta) & mask
    hours = mask.sum() * 30 / 3600
    ep_fa, det, nev = 0, 0, 0
    for r in np.unique(rec[mask]):
        idx = np.where(mask & (rec == r))[0]
        idx = idx[np.argsort(el[idx])]
        a = alarm[idx]
        run = []
        for j, flag in enumerate(a):
            if flag:
                run.append(idx[j])
            if (not flag or j == len(a) - 1) and run:
                if y[run].sum() == 0:
                    ep_fa += 1
                run = []
        if y[idx].sum():
            nev += 1
            det += int(alarm[idx][y[idx] == 1].any())
    return dict(alarm_positive_windows_per_hour=float((alarm & (y == 0)).sum() / hours),
                false_alarm_episodes_per_hour=float(ep_fa / hours), event_sensitivity=det / nev if nev else None,
                n_events=nev)


def cmd_eval():
    d = ev.load()
    pf = ev.folds(d['person'], ev.FOLD_SALTS['A'])
    m = d['e1'].astype(bool)
    y = d['y1'].astype(float)
    el, rec, per = d['elapsed'], d['recording'], d['person']
    tte = d['first_sleep_s'] - el
    C = np.load(LOC / 'w2s_classical_oof_A.npz')
    refs = {'CONST_pop': C['E1__CONST_pop'], 'TIME_pop': C['E1__TIME_pop']}
    u_all = np.log1p(el / 60)
    preds, params = {}, {}
    for arch in ARCHS:
        outer = np.load(LOC / f'w2s_neural_oof_{arch}.npz')['p_n1']
        inner = np.load(LOC / f'w2s_p0b_inner_{arch}.npz')
        preds[f'{arch}|K0'] = outer.copy()
        for kind in ('K1', 'K2', 'K3'):
            out = np.full(len(y), np.nan)
            for f in range(ev.N_FOLDS):
                cal = m & (pf != f)
                te = m & (pf == f)
                pin = inner[f'outer{f}']
                assert np.isfinite(pin[cal]).all() and np.isnan(pin[pf == f]).all()
                uc = u_all[cal]
                mu, sd = uc.mean(), uc.std() + 1e-9
                out[te], info = calibrate_fold(kind, logit(pin[cal]), (uc - mu) / sd, y[cal], pf[cal],
                                               logit(outer[te]), (u_all[te] - mu) / sd)
                params[f'{arch}|{kind}|fold{f}'] = info
            preds[f'{arch}|{kind}'] = out
    res = ev.evaluate('e1', preds, refs, d, y, m)
    # V4 time-matched AUROC (within fold, exact landmark), level/within, warnings, lead-time bins
    import w2s_p0a_diagnostic as p0a
    pid = np.searchsorted(sorted(set(per[m])), per[m])
    pi, ni = p0a.pair_index((el[m] // 30).astype(int), pf[m], pid, y[m].astype(int), True)
    extra = {}
    for k, p in {**refs, **preds}.items():
        pm = p[m]
        lev = wit = 0.0
        for q in sorted(set(per[m])):
            i = per[m] == q
            pb, yb = pm[i].mean(), y[m][i].mean()
            lev += i.sum() * (pb - yb) ** 2
            wit += (((pm[i] - pb) - (y[m][i] - yb)) ** 2).sum()
        lead = {}
        far = m & (tte > 600)
        for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 300)):
            pos = m & (tte > lo) & (tte <= hi)
            sub = pos | far
            yy = pos[sub].astype(int)
            pid2 = np.searchsorted(sorted(set(per[sub])), per[sub])
            a, b = p0a.pair_index((el[sub] // 30).astype(int), pf[sub], pid2, yy, True)
            lead[f'({lo},{hi}]'] = dict(n_windows=int(pos.sum()), n_persons=len(set(per[pos])), mean_pred=float(p[pos].mean()),
                                        auroc_vs_far_V4=(p0a.auc_pairs(p[sub], a, b) if len(a) else None), n_pairs=int(len(a)))
        extra[k] = dict(V4_auroc=p0a.auc_pairs(pm, pi, ni), level=lev / m.sum(), within=wit / m.sum(),
                        warnings=warnings(p, 2 * refs['CONST_pop'], y, rec, el, m), lead_time=lead)
    out = dict(version='H-EVAL-W2S-P0b-v1-2026-10-24', status='TRAIN_ONLY_EXPLORATORY_NESTED_CROSSFIT', partition='A',
               design_sha256=hashlib.sha256((ev.REPO / 'docs/decision/h_eval/discovery_w2s/w2s_p0p1_design_v1.md').read_bytes()).hexdigest(),
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               deployable_note=dict(all_pre_n2_windows=int(len(y)), e1_oracle_at_risk_windows=int(m.sum()),
                                    windows_after_first_sleep=int((~m).sum())),
               results=res, extra=extra, calibrator_params=params)
    with (RES / 'w2s_p0b_calibration_v1.json').open('x') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    for k in preds:
        v, mm, e = res['vs_time'][k], res['models'][k], extra[k]
        print(f"{k:26s} R_vs_time {v['R']*100:6.1f} ({v['R_ci95'][0]*100:6.1f},{v['R_ci95'][1]*100:5.1f}) dSE {v['delta_se']:+.4f} "
              f"({v['delta_se_ci95'][0]:+.4f},{v['delta_se_ci95'][1]:+.4f}) LL {mm['logloss_ww']:.4f} CIL {mm['cal_in_large']:+.3f} "
              f"slope {mm['cal_slope'] and round(mm['cal_slope'], 2)} V4 {e['V4_auroc']:.3f} level {e['level']:.4f} within {e['within']:.4f} "
              f"| sens {e['warnings']['event_sensitivity']:.2f} alarmwin/h {e['warnings']['alarm_positive_windows_per_hour']:.1f} "
              f"FAep/h {e['warnings']['false_alarm_episodes_per_hour']:.2f}")
    for k in ('TIME_pop', 'CONST_pop'):
        mm, e = res['models'][k], extra[k]
        print(f"{k:26s} brier {mm['brier_ww']:.4f} LL {mm['logloss_ww']:.4f} level {e['level']:.4f} within {e['within']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('inner', 'eval'))
    ap.add_argument('--archs', nargs='+', default=list(ARCHS))
    a = ap.parse_args()
    cmd_inner(a.archs) if a.cmd == 'inner' else cmd_eval()


if __name__ == '__main__':
    main()
