#!/usr/bin/env python3
"""H-EVAL Repair Gate R1: TRAIN-only flexible time baseline, stage-construct analysis and cohort flow.

Implements docs/decision/h_eval/repair_r1/r1_time_baseline_design.md. POST-HOC, EXPLORATORY, TRAIN ONLY.
Reads only: the CTH v1 TRAIN dataset and its OOF predictions (.local/h_eval/cth_v1/), official TRAIN hypnograms
(SHA-256 checked), and existing metadata/roster JSON/CSV. Never reads VAL/TEST/EXT signals or hypnograms, never
loads or scores a frozen model, and writes only new files (refuses to overwrite).

Usage:  r1_analysis.py time    flexible time baselines (TS, TK) + comparisons with existing OOF predictions
        r1_analysis.py stage   stage-construct analysis on TRAIN hypnograms
        r1_analysis.py cohort  cohort flow from E2/E3 inventories and approved rosters (metadata only)
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / 'docs/decision/h_eval/repair_r1/r1_time_baseline_design.md'
DATA = REPO / '.local/h_eval/cth_v1/dataset_v1.npz'
OOF = REPO / '.local/h_eval/cth_v1/cth_oof_predictions_v1.npz'
RES = REPO / 'results/decision/h_eval/repair_r1'
LOC = REPO / '.local/h_eval/repair_r1'
VERSION = 'H-EVAL-R1-time-baseline-v1-2026-10-24'
H = 10
INNER_SALT = 'H-EVAL-R1-inner'
N_INNER = 4
KNOTS = (2, 3, 4, 5, 7, 9)
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
MS = (0, 1, 2, 5, 10, 20, 50, 100)
BOOT_N, BOOT_SEED = 10_000, 20261023
EXISTING = ('T0_constant', 'T1_time_cont', 'TH_time_hazard', 'M0_e5_logistic', 'M2_raw_cnn_binary')


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:                      # never overwrite
        f.write(data)


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


cx = _imp(REPO / 'scripts/decision/h_eval/cth_v1/cth_experiment.py', 'cth_for_r1')


def load():
    z = np.load(DATA, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != 'raw'}         # raw EEG is not needed by R1
    d['person'] = d['person'].astype(str)
    d['recording'] = d['recording'].astype(str)
    o = np.load(OOF, allow_pickle=False)
    oof = {k: o[k] for k in EXISTING}
    if not (np.array_equal(o['y'], d['y']) and np.array_equal(o['person'].astype(str), d['person'])):
        raise RuntimeError('OOF predictions are not aligned with the TRAIN dataset')
    lat = d['elapsed'] + d['tau']
    if not np.allclose(lat % 30, 0) or not np.allclose(d['elapsed'] % 30, 0):
        raise RuntimeError('latencies are not on the 30 s lattice')
    d['J'] = np.rint(lat / 30).astype(int)
    d['k'] = np.rint(d['elapsed'] / 30).astype(int)
    return d, oof


# ------------------------------------------------------------------------ time-only models
def rec_latencies(d, idx):
    """One (J, weight) per training recording; idx = training window indices."""
    recs = {}
    for i in idx:
        recs[d['recording'][i]] = d['J'][i]
    return np.array(sorted(recs.values()))


def fit_ts(Js, n_knots, C):
    rows_j, rows_e = [], []
    for J in Js:
        j = np.arange(2, J + 1)
        rows_j.append(j)
        rows_e.append((j == J).astype(int))
    j = np.concatenate(rows_j)
    e = np.concatenate(rows_e)
    x = np.log(30.0 * j)[:, None]
    st = SplineTransformer(n_knots=n_knots, degree=3, extrapolation='constant', include_bias=False).fit(x)
    clf = LogisticRegression(C=C, max_iter=5000).fit(st.transform(x), e)

    def hazard(jq):
        return clf.predict_proba(st.transform(np.log(30.0 * np.asarray(jq, float))[:, None]))[:, 1]

    def pred(k):
        k = np.asarray(k)
        jj = k[:, None] + np.arange(1, H + 1)[None, :]
        hz = hazard(jj.ravel()).reshape(jj.shape)
        return np.clip(1 - np.prod(1 - hz, 1), 1e-6, 1 - 1e-6)
    return pred


def fit_tk(Js, m, p0):
    Js = np.asarray(Js)

    def pred(k):
        k = np.asarray(k)
        at = (Js[None, :] > k[:, None]).sum(1)
        ev = ((Js[None, :] > k[:, None]) & (Js[None, :] <= k[:, None] + H)).sum(1)
        return np.clip((ev + m * p0) / np.maximum(at + m, 1e-12), 1e-6, 1 - 1e-6)
    return pred


def inner_folds(persons):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{INNER_SALT}|{p}'.encode()).hexdigest())
    return {p: i % N_INNER for i, p in enumerate(ranked)}


def nested_time(d, tr, te):
    """Select TS and TK flexibility by inner person-disjoint CV on tr; refit on tr; predict te."""
    f = inner_folds(d['person'][tr])
    fi = np.array([f[p] for p in d['person'][tr]])
    y = d['y'].astype(float)
    ts_scores, tk_scores = defaultdict(float), defaultdict(float)
    for g in range(N_INNER):
        itr, iva = tr[fi != g], tr[fi == g]
        Js = rec_latencies(d, itr)
        p0 = (y[itr].sum() + 1) / (len(itr) + 2)
        for nk in KNOTS:
            for C in CS:
                ts_scores[(nk, C)] += ((fit_ts(Js, nk, C)(d['k'][iva]) - y[iva]) ** 2).sum()
        for m in MS:
            tk_scores[m] += ((fit_tk(Js, m, p0)(d['k'][iva]) - y[iva]) ** 2).sum()
    best_ts = min(ts_scores, key=lambda c: (round(ts_scores[c] / len(tr), 9), c[0], c[1]))
    best_tk = min(tk_scores, key=lambda m: (round(tk_scores[m] / len(tr), 9), m))
    Js = rec_latencies(d, tr)
    p0 = (y[tr].sum() + 1) / (len(tr) + 2)
    return (fit_ts(Js, *best_ts)(d['k'][te]), fit_tk(Js, best_tk, p0)(d['k'][te]),
            dict(ts_knots=best_ts[0], ts_C=best_ts[1], tk_m=best_tk,
                 inner_brier_ts={f'{a}|{b}': v / len(tr) for (a, b), v in ts_scores.items()},
                 inner_brier_tk={str(m): v / len(tr) for m, v in tk_scores.items()}))


# ------------------------------------------------------------------------ metrics
def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def model_metrics(p, d):
    m = cx.metrics(p, d)
    return {k: m[k] for k in ('brier_ww', 'brier_se', 'logloss_ww', 'logloss_se', 'cal_in_large', 'cal_slope',
                              'ece10', 'within_recording_auroc_mean')}


def paired(p_ref, p_cand, d, mask=None):
    y = d['y'].astype(float)
    persons = d['person']
    mask = np.ones(len(y), bool) if mask is None else mask
    pu = sorted(set(persons[mask]))
    idx = np.searchsorted(pu, persons[mask])
    sr = np.bincount(idx, weights=brier(p_ref[mask], y[mask]), minlength=len(pu))
    sc = np.bincount(idx, weights=brier(p_cand[mask], y[mask]), minlength=len(pu))
    lr = np.bincount(idx, weights=logloss(p_ref[mask], y[mask]), minlength=len(pu))
    lc = np.bincount(idx, weights=logloss(p_cand[mask], y[mask]), minlength=len(pu))
    n = np.bincount(idx, minlength=len(pu)).astype(float)
    I = np.random.default_rng(BOOT_SEED).integers(0, len(pu), size=(BOOT_N, len(pu)))
    Br, Bc = sr[I].sum(1) / n[I].sum(1), sc[I].sum(1) / n[I].sum(1)
    se_b = ((sr - sc) / n)[I].mean(1)
    ll_b = (lr[I].sum(1) - lc[I].sum(1)) / n[I].sum(1)
    per = (sr - sc) / n
    loo = [((sr.sum() - sr[i]) - (sc.sum() - sc[i])) / (sr.sum() - sr[i]) for i in range(len(pu))]
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]   # noqa: E731
    return dict(n_persons=len(pu), n_windows=int(n.sum()),
                brier_ref_ww=float(sr.sum() / n.sum()), brier_cand_ww=float(sc.sum() / n.sum()),
                delta_ww=float((sr.sum() - sc.sum()) / n.sum()), delta_ww_ci95=ci(Br - Bc),
                R=float((sr.sum() - sc.sum()) / sr.sum()), R_ci95=ci((Br - Bc) / Br),
                delta_se=float(per.mean()), delta_se_ci95=ci(se_b),
                delta_logloss_ww=float((lr.sum() - lc.sum()) / n.sum()), delta_logloss_ww_ci95=ci(ll_b),
                loo_R_min=float(min(loo)), persons_favouring_candidate=int((per > 0).sum()),
                per_person_delta={p: float(v) for p, v in zip(pu, per)})


# ------------------------------------------------------------------------ TIME
def run_time():
    d, oof = load()
    folds = cx.person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    ts, tk = np.full(len(d['y']), np.nan), np.full(len(d['y']), np.nan)
    sel = []
    for f in range(cx.N_FOLDS):
        te, tr = np.where(pf == f)[0], np.where(pf != f)[0]
        a, b, info = nested_time(d, tr, te)
        ts[te], tk[te] = a, b
        sel.append(dict(fold=f, n_train_persons=len(set(d['person'][tr])), n_test_persons=len(set(d['person'][te])),
                        **{k: v for k, v in info.items() if not k.startswith('inner')},
                        inner_brier_ts_best=min(info['inner_brier_ts'].values()),
                        inner_brier_ts_worst=max(info['inner_brier_ts'].values())))
    assert np.isfinite(ts).all() and np.isfinite(tk).all()
    P = dict(oof, TS_spline_hazard=ts, TK_empirical_shrunk=tk)
    # optimistic in-sample ceiling (all 46 persons, evaluated in-sample)
    allidx = np.arange(len(d['y']))
    _, _, info_all = nested_time(d, allidx, allidx[:1])
    Js = rec_latencies(d, allidx)
    p0 = (d['y'].sum() + 1) / (len(d['y']) + 2)
    ceil_ts = fit_ts(Js, info_all['ts_knots'], info_all['ts_C'])(d['k'])
    ceil_tk = fit_tk(Js, info_all['tk_m'], p0)(d['k'])
    ceil_tk0 = fit_tk(Js, 0, p0)(d['k'])
    ceil_const = np.full(len(d['y']), p0)
    pairs = [('T0_constant', 'TS_spline_hazard'), ('T0_constant', 'TK_empirical_shrunk'),
             ('T0_constant', 'T1_time_cont'), ('T0_constant', 'TH_time_hazard'),
             ('TS_spline_hazard', 'M0_e5_logistic'), ('TS_spline_hazard', 'M2_raw_cnn_binary'),
             ('TK_empirical_shrunk', 'M0_e5_logistic'), ('TK_empirical_shrunk', 'M2_raw_cnn_binary'),
             ('T0_constant', 'M0_e5_logistic'), ('T0_constant', 'M2_raw_cnn_binary'),
             ('TS_spline_hazard', 'TH_time_hazard'), ('TS_spline_hazard', 'T1_time_cont')]
    comps = {f'{a}__vs__{b}': paired(P[a], P[b], d) for a, b in pairs}
    first = np.array([n == min(d['night'][d['person'] == s]) for n, s in zip(d['night'], d['person'])])
    comps_first_night = {f'{a}__vs__{b}': {k: v for k, v in paired(P[a], P[b], d, first).items() if k != 'per_person_delta'}
                         for a, b in pairs[:2] + pairs[4:6]}
    ceiling = dict(note='In-sample: fitted and evaluated on all 46 TRAIN persons. Optimistic; not an out-of-sample estimate.',
                   selected=dict(ts_knots=info_all['ts_knots'], ts_C=info_all['ts_C'], tk_m=info_all['tk_m']),
                   R_TS_vs_const=paired(ceil_const, ceil_ts, d)['R'], R_TK_vs_const=paired(ceil_const, ceil_tk, d)['R'],
                   R_unshrunk_empirical_vs_const=paired(ceil_const, ceil_tk0, d)['R'],
                   brier_ww=dict(constant=float(brier(ceil_const, d['y']).mean()), TS=float(brier(ceil_ts, d['y']).mean()),
                                 TK=float(brier(ceil_tk, d['y']).mean()), empirical_m0=float(brier(ceil_tk0, d['y']).mean())))
    # curve of the in-sample TS and TK forecast vs elapsed time (for the report)
    kk = np.arange(1, 241)
    curve = dict(elapsed_s=(30 * kk).tolist(),
                 TS=fit_ts(Js, info_all['ts_knots'], info_all['ts_C'])(kk).round(5).tolist(),
                 TK=fit_tk(Js, info_all['tk_m'], p0)(kk).round(5).tolist(),
                 n_at_risk=[int((Js > k).sum()) for k in kk])
    res = dict(version=VERSION, status='POST_HOC_EXPLORATORY_TRAIN_ONLY_NOT_PREREGISTERED_NOT_CONFIRMATORY',
               design_sha256=digest(DESIGN), dataset_sha256=digest(DATA), oof_sha256=digest(OOF),
               code_sha256=digest(Path(__file__)),
               data=dict(n_windows=int(len(d['y'])), n_positive=int(d['y'].sum()), n_persons=len(set(d['person'])),
                         n_recordings=len(set(d['recording']))),
               selection_per_fold=sel,
               models={m: model_metrics(p, d) for m, p in P.items()},
               comparisons={k: {a: b for a, b in v.items() if a != 'per_person_delta'} for k, v in comps.items()},
               comparisons_first_night_only=comps_first_night,
               in_sample_ceiling=ceiling, forecast_curve_in_sample=curve,
               boundary=dict(val_test_ext_accessed=False, frozen_models_loaded=False, eeg_signal_read=False))
    pp = []
    for p in sorted(set(d['person'])):
        row = dict(person=p, n_windows=int((d['person'] == p).sum()), n_recordings=len(set(d['recording'][d['person'] == p])))
        for k in ('T0_constant__vs__TS_spline_hazard', 'TS_spline_hazard__vs__M0_e5_logistic',
                  'TS_spline_hazard__vs__M2_raw_cnn_binary', 'T0_constant__vs__M0_e5_logistic'):
            row['delta_' + k] = round(comps[k]['per_person_delta'][p], 6)
        pp.append(row)
    write_new(RES / 'r1_time_baseline_results.json', (json.dumps(res, indent=2) + '\n').encode())
    write_new(RES / 'r1_time_baseline_per_person.csv', to_csv(pp))
    LOC.mkdir(parents=True, exist_ok=True)
    np.savez(LOC / 'r1_time_oof.npz', TS_spline_hazard=ts, TK_empirical_shrunk=tk)
    print(json.dumps(dict(selection=sel, ceiling=ceiling,
                          key={k: {a: v[a] for a in ('R', 'R_ci95', 'delta_ww', 'delta_ww_ci95', 'delta_se', 'delta_se_ci95',
                                                     'delta_logloss_ww', 'persons_favouring_candidate')}
                               for k, v in comps.items()}), indent=1))


# ------------------------------------------------------------------------ STAGE
def train_hypnogram_stages(d):
    """Return per-window input-epoch stage and per-recording first-sleep times (TRAIN only, SHA-checked)."""
    ge = _imp(REPO / 'scripts/decision/h_eval/gate_e/gate_e.py', 'gate_e_for_r1')
    e3 = _imp(REPO / 'scripts/decision/h_eval/gate_e3/gate_e3.py', 'gate_e3_for_r1')
    split = json.loads((REPO / 'results/decision/h_eval/gate_e3/gate_e3_sc_group_split.json').read_text())['by_subject']
    inv = {r['recording']: r for r in csv.DictReader((REPO / 'results/decision/h_eval/gate_e2/sleep_edf_origin_inventory.csv').open())}
    sums = ge.sums_parse((REPO / 'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt').read_bytes())
    stage = np.empty(len(d['y']), dtype=object)
    recinfo = {}
    for rec in sorted(set(d['recording'])):
        row = inv[rec]
        if split.get(row['subject_key']) != 'SC_TRAIN':
            raise RuntimeError('PROTECTED SPLIT VIOLATION: ' + rec)
        hp = REPO / 'data/raw/sleep_edfx_v1_0_0' / row['hypnogram_path']
        raw = hp.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sums[row['hypnogram_path']]:
            raise RuntimeError('hypnogram checksum mismatch ' + rec)
        ev, _ = ge.decode_edfplus_annotations(raw)
        off, n2 = float(row['lights_off_rel_s']), float(row['first_n2_after_off_s'])
        idx = np.where(d['recording'] == rec)[0]
        for i in idx:
            t = off + d['elapsed'][i]
            a, b = e3.stage_at(ev, t - 30), e3.stage_at(ev, t - 0.5)
            if a != b:
                raise RuntimeError(f'input window spans two stages {rec} {t}')
            stage[i] = a
        # first epoch of any sleep (1,2,3,4,R) and of Stage 1 at/after LightsOff
        tt = off
        first_sleep = first_s1 = None
        while tt < n2 + 1e-6:
            s = e3.stage_at(ev, tt)
            if s in ('1', '2', '3', '4', 'R') and first_sleep is None:
                first_sleep = tt - off
            if s == '1' and first_s1 is None:
                first_s1 = tt - off
            tt += 30.0
        if abs(off + d['J'][idx[0]] * 30 - n2) > 1e-6 or e3.stage_at(ev, n2) != '2':
            raise RuntimeError('N2 latency mismatch ' + rec)
        recinfo[rec] = dict(person=row['subject_key'], latency_n2_s=n2 - off, first_any_sleep_s=first_sleep,
                            first_s1_s=first_s1)
    return stage, recinfo


def auroc(p, y):
    return cx.auroc(p, y)


def run_stage():
    d, oof = load()
    ts_oof = np.load(LOC / 'r1_time_oof.npz')
    P = dict(T0=oof['T0_constant'], TS=ts_oof['TS_spline_hazard'], M0=oof['M0_e5_logistic'], M2=oof['M2_raw_cnn_binary'])
    stage, recinfo = train_hypnogram_stages(d)
    y = d['y'].astype(float)
    # stage-recognition diagnostic SR (uses the scorer's label; not prospectively available)
    folds = cx.person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    s1 = (stage == '1').astype(float)
    X = np.column_stack([s1, cx.time_feats(d['elapsed'])])
    sr = np.full(len(y), np.nan)
    for f in range(cx.N_FOLDS):
        te, tr = np.where(pf == f)[0], np.where(pf != f)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        sr[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    P['SR_stage_recognition'] = sr
    comp = Counter(zip(stage, d['y']))
    composition = {s: dict(negative=comp.get((s, 0), 0), positive=comp.get((s, 1), 0))
                   for s in sorted(set(stage))}
    # stage-stratified Brier and contribution of each stage to the total WW gain vs T0
    strat, contrib = {}, {}
    for s in sorted(set(stage)):
        m = stage == s
        strat[s] = {k: float(brier(v[m], y[m]).mean()) for k, v in P.items()}
        strat[s]['n'] = int(m.sum())
        strat[s]['prevalence'] = float(y[m].mean())
    for k in ('TS', 'M0', 'M2', 'SR_stage_recognition'):
        tot = (brier(P['T0'], y) - brier(P[k], y)).sum()
        contrib[k] = {s: float((brier(P['T0'][stage == s], y[stage == s]) - brier(P[k][stage == s], y[stage == s])).sum() / tot)
                      for s in sorted(set(stage))}
        contrib[k]['total_delta_ww'] = float(tot / len(y))
    # W-only paired comparisons (person-cluster CI)
    w = stage == 'W'
    w_only = {f'T0_vs_{k}': {a: b for a, b in paired(P['T0'], P[k], d, w).items() if a != 'per_person_delta'}
              for k in ('TS', 'M0', 'M2')}
    w_only.update({f'TS_vs_{k}': {a: b for a, b in paired(P['TS'], P[k], d, w).items() if a != 'per_person_delta'}
                   for k in ('M0', 'M2')})
    w_only['SR_vs_M0'] = {a: b for a, b in paired(P['SR_stage_recognition'], P['M0'], d).items() if a != 'per_person_delta'}
    w_only['SR_vs_M2'] = {a: b for a, b in paired(P['SR_stage_recognition'], P['M2'], d).items() if a != 'per_person_delta'}
    w_only['T0_vs_SR'] = {a: b for a, b in paired(P['T0'], P['SR_stage_recognition'], d).items() if a != 'per_person_delta'}
    # within-recording AUROC on W windows only
    wa = {}
    for k in ('TS', 'M0', 'M2'):
        vals = [auroc(P[k][(d['rec_idx'] == r) & w], d['y'][(d['rec_idx'] == r) & w]) for r in np.unique(d['rec_idx'])]
        vals = [v for v in vals if v is not None]
        wa[k] = dict(mean=float(np.mean(vals)), median=float(np.median(vals)), n_recordings=len(vals))
    pooled_w = {k: auroc(P[k][w], d['y'][w]) for k in ('TS', 'M0', 'M2')}
    pooled_s1 = {k: auroc(P[k][stage == '1'], d['y'][stage == '1']) for k in ('TS', 'M0', 'M2')}
    # latency strata: tau bins (time to N2) and recording-latency tertiles
    tau = d['tau']
    tb = np.digitize(tau, [300.001, 600.001, 1200.001])
    tau_strata = {}
    for b, lab in enumerate(['tau<=300 (positive)', '300<tau<=600', '600<tau<=1200', 'tau>1200']):
        m = tb == b
        tau_strata[lab] = dict(n=int(m.sum()), share_stage1=float((stage[m] == '1').mean()),
                               **{k: float(brier(v[m], y[m]).mean()) for k, v in P.items()})
    lat_rec = np.array([recinfo[r]['latency_n2_s'] for r in d['recording']])
    q = np.quantile([v['latency_n2_s'] for v in recinfo.values()], [1 / 3, 2 / 3])
    lt = np.digitize(lat_rec, q)
    lat_strata = {}
    for b, lab in enumerate([f'short (<= {q[0]:.0f} s)', f'middle', f'long (> {q[1]:.0f} s)']):
        m = lt == b
        lat_strata[lab] = dict(n_windows=int(m.sum()), n_recordings=len(set(d['recording'][m])),
                               prevalence=float(y[m].mean()), share_stage1=float((stage[m] == '1').mean()),
                               **{k: float(brier(v[m], y[m]).mean()) for k, v in P.items()})
    # first-N2 vs first-any-sleep target
    fs = np.array([recinfo[r]['first_any_sleep_s'] for r in d['recording']], dtype=float)
    after_sleep = d['elapsed'] - 30 >= fs - 1e-6           # input epoch at/after the first sleep epoch
    alt_at_risk = d['elapsed'] <= fs + 1e-6                 # input epoch entirely before first sleep epoch
    alt_y = (fs - d['elapsed'] <= 300 + 1e-8) & alt_at_risk
    rec_rows = list(recinfo.values())
    gap = np.array([r['latency_n2_s'] - r['first_any_sleep_s'] for r in rec_rows])
    targets = dict(
        n_recordings=len(rec_rows),
        recordings_with_sleep_before_n2=int((gap > 0).sum()),
        first_sleep_stage_is_s1=int(sum(r['first_s1_s'] is not None and abs(r['first_s1_s'] - r['first_any_sleep_s']) < 1e-6
                                        for r in rec_rows)),
        latency_n2_median_s=float(np.median([r['latency_n2_s'] for r in rec_rows])),
        latency_first_sleep_median_s=float(np.median([r['first_any_sleep_s'] for r in rec_rows])),
        gap_n2_minus_first_sleep_s=dict(median=float(np.median(gap)), q25=float(np.quantile(gap, .25)),
                                        q75=float(np.quantile(gap, .75)), max=float(gap.max()),
                                        share_gap_le_300=float((gap <= 300).mean())),
        windows_after_first_sleep=dict(n=int(after_sleep.sum()), share=float(after_sleep.mean()),
                                       positives=int(d['y'][after_sleep].sum()),
                                       share_of_all_positives=float(d['y'][after_sleep].sum() / d['y'].sum())),
        first_any_sleep_target=dict(at_risk_windows=int(alt_at_risk.sum()), positives=int(alt_y.sum()),
                                    prevalence=float(alt_y.sum() / alt_at_risk.sum())),
        brier_on_windows_before_first_sleep={k: float(brier(v[~after_sleep], y[~after_sleep]).mean()) for k, v in P.items()},
        paired_before_first_sleep={f'{a}_vs_{b}': {x: z for x, z in paired(P[a], P[b], d, ~after_sleep).items()
                                                   if x != 'per_person_delta'}
                                   for a, b in (('T0', 'M0'), ('T0', 'M2'), ('TS', 'M0'), ('TS', 'M2'), ('T0', 'TS'))})
    res = dict(version=VERSION, status='POST_HOC_DESCRIPTIVE_TRAIN_ONLY', code_sha256=digest(Path(__file__)),
               note=('Stage is the R&K label of the input epoch [t-30, t). SR uses that label, which a prospective '
                     'forecaster does not have; it is a diagnostic of stage-recognition skill, not a usable baseline.'),
               composition=composition, stage_stratified_brier_ww=strat, share_of_total_gain_vs_T0_by_stage=contrib,
               w_only_and_sr_comparisons=w_only, within_recording_auroc_w_only=wa,
               pooled_auroc_w_only=pooled_w, pooled_auroc_stage1_only=pooled_s1,
               models_overall={k: model_metrics(v, d) for k, v in P.items() if k == 'SR_stage_recognition'},
               tau_strata=tau_strata, recording_latency_tertiles=lat_strata, first_n2_vs_first_sleep=targets,
               boundary=dict(hypnograms_read='SC_TRAIN only (84 recordings, SHA-256 checked)',
                             val_test_ext_accessed=False))
    write_new(RES / 'r1_stage_construct.json', (json.dumps(res, indent=2) + '\n').encode())
    print(json.dumps({k: res[k] for k in ('composition', 'share_of_total_gain_vs_T0_by_stage', 'within_recording_auroc_w_only',
                                          'pooled_auroc_w_only', 'pooled_auroc_stage1_only', 'tau_strata',
                                          'recording_latency_tertiles', 'first_n2_vs_first_sleep')}, indent=1))
    print(json.dumps({k: {a: v[a] for a in ('R', 'R_ci95', 'delta_ww', 'delta_ww_ci95', 'n_windows')}
                      for k, v in w_only.items()}, indent=1))


# ------------------------------------------------------------------------ COHORT
def run_cohort():
    ev = REPO / 'results/decision/h_eval'
    inv = list(csv.DictReader((ev / 'gate_e2/sleep_edf_origin_inventory.csv').open()))
    stage = {r['recording']: r for r in csv.DictReader((ev / 'gate_e3/gate_e3_stage_diagnostics.csv').open())}
    split = json.loads((ev / 'gate_e3/gate_e3_sc_group_split.json').read_text())['by_subject']
    cth = json.loads((ev / 'cth_v1/cth_data_manifest_v1.json').read_text())
    e6 = json.loads((ev / 'gate_e6/gate_e6_summary.json').read_text())
    val = json.loads((ev / 'gate_e7b/gate_e7b_validation_roster_v1.json').read_text())
    t1 = json.loads((ev / 'gate_t1/gate_t1_roster_v1.json').read_text())
    x1 = json.loads((ev / 'gate_x1/gate_x1_roster_v1.json').read_text())
    role = {}
    for r in cth['recordings']:
        role[r['recording']] = 'TRAIN (CTH/SpecLR-46/RawCNN-46 development, all eligible nights)'
    e6p = set(e6['persons_by_origin']['E5_RETAINED'] + e6['persons_by_origin']['E6_HASH'])
    for name, ro in (('VAL', val), ('TEST', t1), ('EXT', x1)):
        for r in ro['recordings']:
            role[r['recording']] = f"{name} {r['role']}"
    rows = []
    for r in sorted(inv, key=lambda r: r['recording']):
        st = stage[r['recording']]['stage_at_lights_off']
        sp = split.get(r['subject_key'], 'NOT_IN_SPLIT')
        if r['status'] != 'ELIGIBLE_ORIGIN_INDEPENDENT':
            excl = 'E2: ' + r['status']
        elif st != 'W':
            excl = f'E3/E6 rule: stage at LightsOff = {st} (not W)'
        else:
            excl = ''
        rows.append(dict(recording=r['recording'], study=r['study'], person=r['subject_key'], night=r['night'],
                         condition=r['condition'], split=sp, e2_status=r['status'], stage_at_lights_off=st,
                         exclusion=excl, final_role=role.get(r['recording'], ''),
                         in_e6_18_person_pilot=r['subject_key'] in e6p))
    # integrity: every analysed recording is eligible and W-start; every eligible W-start recording has a role
    for x in rows:
        if x['final_role'] and x['exclusion']:
            raise RuntimeError('analysed recording has an exclusion: ' + x['recording'])
        if not x['exclusion'] and not x['final_role']:
            raise RuntimeError('eligible W-start recording without a role: ' + x['recording'])
    def tally(study, field=None):
        sub = [x for x in rows if x['study'] == study]
        return dict(persons=len({x['person'] for x in sub}), recordings=len(sub))
    flow = {}
    for study, s in (('sleep-cassette', 'SC'), ('sleep-telemetry', 'ST')):
        sub = [x for x in rows if x['study'] == study]
        f = dict(source=tally(study))
        f['excluded_by_reason'] = {k: dict(recordings=v, persons=sorted({x['person'] for x in sub if x['exclusion'] == k}))
                                   for k, v in Counter(x['exclusion'] for x in sub if x['exclusion']).items()}
        f['analysable_recordings'] = sum(not x['exclusion'] for x in sub)
        f['analysable_persons'] = len({x['person'] for x in sub if not x['exclusion']})
        f['persons_with_no_analysable_night'] = sorted({x['person'] for x in sub} - {x['person'] for x in sub if not x['exclusion']})
        bysplit = defaultdict(lambda: dict(persons=set(), recordings=0, primary=0, sensitivity=0, persons_in_split=set()))
        for x in sub:
            b = bysplit[x['split']]
            b['persons_in_split'].add(x['person'])
            if x['final_role']:
                b['persons'].add(x['person'])
                b['recordings'] += 1
                b['primary'] += 'PRIMARY' in x['final_role']
                b['sensitivity'] += 'SENSITIVITY' in x['final_role']
        f['by_split'] = {k: dict(persons_in_split=len(v['persons_in_split']), analysed_persons=len(v['persons']),
                                 analysed_recordings=v['recordings'], primary_recordings=v['primary'],
                                 sensitivity_recordings=v['sensitivity'],
                                 split_persons_not_analysed=sorted(v['persons_in_split'] - v['persons']))
                         for k, v in sorted(bysplit.items())}
        flow[s] = f
    out = dict(version=VERSION, status='METADATA_ONLY_RECONSTRUCTION',
               primary_night_rule=dict(
                   TRAIN='All eligible W-start nights of each TRAIN person (84 recordings of 46 persons) for CTH, SpecLR-46 and '
                         'RawCNN-46; the 18-person E6 pilot and SpecLR-18 used one night per person (E5 retained 6 + 12 by hash).',
                   VAL='Primary = lowest-numbered eligible W-start night per person; sensitivity = remaining eligible nights.',
                   TEST='Primary = lowest-numbered eligible W-start night per person; sensitivity = remaining eligible nights.',
                   EXT='All eligible W-start nights, both conditions; person is the bootstrap cluster.'),
               flow=flow,
               inputs_sha256={p: digest(ev / p) for p in ('gate_e2/sleep_edf_origin_inventory.csv',
                                                          'gate_e3/gate_e3_stage_diagnostics.csv',
                                                          'gate_e3/gate_e3_sc_group_split.json',
                                                          'cth_v1/cth_data_manifest_v1.json',
                                                          'gate_e7b/gate_e7b_validation_roster_v1.json',
                                                          'gate_t1/gate_t1_roster_v1.json', 'gate_x1/gate_x1_roster_v1.json')},
               code_sha256=digest(Path(__file__)))
    write_new(RES / 'r1_cohort_flow.json', (json.dumps(out, indent=2) + '\n').encode())
    write_new(RES / 'r1_cohort_recordings.csv', to_csv(rows))
    print(json.dumps(flow, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('time', 'stage', 'cohort'))
    a = ap.parse_args(argv)
    {'time': run_time, 'stage': run_stage, 'cohort': run_cohort}[a.cmd]()


if __name__ == '__main__':
    main()
