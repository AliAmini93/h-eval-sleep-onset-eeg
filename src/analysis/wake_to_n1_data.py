#!/usr/bin/env python3
"""H-EVAL W2S: TRAIN-only estimand table and causal nonlinear-dynamical features (design w2s_design_v1.md §1-4).

Reads only the CTH v1 TRAIN dataset and the 84 SHA-checked TRAIN hypnograms (via the R1 attribution code).
Writes .local/h_eval/discovery_w2s/w2s_table_v1.npz (git-ignored). Never touches VAL/TEST/EXT/ST/Muse/HMC.
"""
import hashlib
import math
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[4]
LOC = REPO / '.local/h_eval/discovery_w2s'
OUT = LOC / 'w2s_table_v1.npz'
FS = 100.0


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------ per-window features (one window, one channel)
def perm_entropy(x, order=5, delay=1):
    n = len(x) - (order - 1) * delay
    emb = np.stack([x[i * delay:i * delay + n] for i in range(order)], 1)
    ranks = np.argsort(emb, 1)
    codes = (ranks * (order ** np.arange(order))).sum(1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def dfa(x, scales=(16, 25, 40, 63, 100, 160, 250, 400)):
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        n = len(y) // s
        seg = y[:n * s].reshape(n, s)
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        res = seg.T - A @ coef
        F.append(np.sqrt((res ** 2).mean()))
    return float(np.polyfit(np.log(scales), np.log(np.array(F) + 1e-12), 1)[0])


def hjorth(x):
    d1, d2 = np.diff(x), np.diff(x, 2)
    v0, v1, v2 = x.var() + 1e-12, d1.var() + 1e-12, d2.var() + 1e-12
    mob = np.sqrt(v1 / v0)
    return float(mob), float(np.sqrt(v2 / v1) / mob)


def channel_features(x):
    x = x - x.mean()
    f, P = welch(x, fs=FS, nperseg=200, noverlap=100)
    band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum()   # noqa: E731
    tot = band(0.5, 30) + 1e-12
    fit = (f >= 2) & (f <= 30) & ~((f >= 7) & (f <= 14))
    lf, lp = np.log10(f[fit]), np.log10(P[fit] + 1e-12)
    slope, off = np.polyfit(lf, lp, 1)
    pred = off + slope * lf
    r2 = 1 - ((lp - pred) ** 2).sum() / (((lp - lp.mean()) ** 2).sum() + 1e-12)
    ap = lambda m: off + slope * np.log10(f[m])           # noqa: E731
    th, al = (f >= 4) & (f < 8), (f >= 8) & (f < 12)
    th_res = float(np.max(np.log10(P[th] + 1e-12) - ap(th)))
    al_res = float(np.max(np.log10(P[al] + 1e-12) - ap(al)))
    m = (f >= 0.5) & (f <= 30)
    pn = P[m] / (P[m].sum() + 1e-12)
    sent = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    cum = np.cumsum(P[m]) / (P[m].sum() + 1e-12)
    sef95 = float(f[m][np.searchsorted(cum, 0.95)])
    mob, comp = hjorth(x)
    return [-slope, off, r2, th_res, al_res, band(0.5, 4) / tot, band(4, 8) / tot, band(8, 12) / tot,
            band(12, 15) / tot, band(15, 30) / tot, np.log(tot), sent, sef95, mob, comp,
            perm_entropy(x, 5, 1), perm_entropy(x, 5, 4), dfa(x)]


FEAT_NAMES = ['aperiodic_exponent', 'aperiodic_offset', 'aperiodic_r2', 'theta_periodic_resid', 'alpha_periodic_resid',
              'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta', 'log_total_power', 'spectral_entropy',
              'sef95', 'hjorth_mobility', 'hjorth_complexity', 'perm_entropy_d1', 'perm_entropy_d4', 'dfa_alpha']
SOS = {b: butter(4, lims, btype='band', fs=FS, output='sos') for b, lims in (('theta', (4, 8)), ('alpha', (8, 12)))}


def plv(x1, x2, band):
    # zero-phase filtering is applied WITHIN the 30 s window only (no samples outside [t-30, t))
    a, b = sosfiltfilt(SOS[band], x1), sosfiltfilt(SOS[band], x2)
    return float(np.abs(np.exp(1j * (np.angle(hilbert(a)) - np.angle(hilbert(b)))).mean()))


def window_features(raw):
    out = np.zeros((len(raw), 2 * len(FEAT_NAMES) + 2))
    for i, w in enumerate(raw):
        x0, x1 = w[0].astype(float), w[1].astype(float)
        out[i] = channel_features(x0) + channel_features(x1) + [plv(x0, x1, 'theta'), plv(x0, x1, 'alpha')]
    return out


def history_features(F, hist):
    """Causal: mean of previous <=4 windows and OLS slope over <=10 windows (current included); 0 if unavailable."""
    n, d = F.shape
    mean_prev, slope = np.zeros((n, d)), np.zeros((n, d))
    for i in range(n):
        idx = hist[i][hist[i] >= 0]                 # chronological, last = current
        prev = idx[:-1][-4:]
        if len(prev):
            mean_prev[i] = F[prev].mean(0)
        if len(idx) >= 3:
            t = np.arange(len(idx), dtype=float)
            t -= t.mean()
            slope[i] = (t[:, None] * (F[idx] - F[idx].mean(0))).sum(0) / (t ** 2).sum()
    return mean_prev, slope


def main():
    r1 = _imp(REPO / 'scripts/decision/h_eval/repair_r1/r1_analysis.py', 'r1_for_w2s')
    d, _ = r1.load()
    z = np.load(r1.DATA)
    raw = z['raw']
    stage, recinfo = r1.train_hypnogram_stages(d)
    fs = np.array([recinfo[r]['first_any_sleep_s'] for r in d['recording']], float)
    first_s1 = np.array([recinfo[r]['first_s1_s'] for r in d['recording']], float)
    assert np.allclose(fs, first_s1), 'first sleep is not Stage 1 in some TRAIN night'
    el = d['elapsed']
    e1 = el < fs - 1e-6
    y1 = (e1 & (fs - el <= 300 + 1e-8)).astype(int)
    k1 = np.where(e1 & (fs - el <= 300 + 1e-8), np.clip(np.ceil((fs - el) / 30 - 1e-9), 1, 10), 11).astype(int)
    print('[W2S] window features ...', flush=True)
    F = window_features(raw)
    mp, sl = history_features(F, d['hist'])
    arrays = dict(stage=np.array([s for s in stage]).astype(str), first_sleep_s=fs, e1=e1, y1=y1, k1=k1,
                  e2=(stage == 'W'), e2pre=(stage == 'W') & e1, e3=(stage == '1'), y=d['y'], tau=d['tau'],
                  elapsed=el, person=d['person'], recording=d['recording'], night=d['night'], hist=d['hist'],
                  feat_cur=F, feat_mean_prev=mp, feat_slope=sl)
    LOC.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **arrays)
    summ = dict(n=int(len(el)), E1=dict(n=int(e1.sum()), pos=int(y1.sum())),
                E2=dict(n=int((stage == 'W').sum()), pos=int(d['y'][stage == 'W'].sum())),
                E2pre=dict(n=int(((stage == 'W') & e1).sum()), pos=int(d['y'][(stage == 'W') & e1].sum())),
                E3=dict(n=int((stage == '1').sum()), pos=int(d['y'][stage == '1'].sum())),
                n_features=int(F.shape[1]), table_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
