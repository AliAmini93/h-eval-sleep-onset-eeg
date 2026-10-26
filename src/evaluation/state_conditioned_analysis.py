#!/usr/bin/env python3
"""H-EVAL R3B: frozen held-out state-aware reference audit (protocol docs/decision/h_eval/repair_r3b/r3b_protocol_v1.md).
POST-HOC, EXPLORATORY. Commands:
  fit      fit final TRAIN-only references A, B (and A18, B18) -> results/.../r3b_references_v1.json   (TRAIN only)
  freeze   run synthetic tests; write the write-once freeze record (protocol, code, tests, references)
  execute  verify held-out inputs (R2 routines + join checks); compute aggregates; write results (write-once)
Never reads raw EEG or hypnograms; never runs or recalibrates an EEG model; never fits on held-out data."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROTOCOL = REPO / 'docs/decision/h_eval/repair_r3b/r3b_protocol_v1.md'
TESTS = HERE / 'test_r3b_audit.py'
RES = REPO / 'results/decision/h_eval/repair_r3b'
LOC = REPO / '.local/h_eval/repair_r3b'
REFS = RES / 'r3b_references_v1.json'
FREEZE = RES / 'r3b_freeze_record.json'
TABLE = REPO / '.local/h_eval/discovery_w2s/w2s_table_v1.npz'
E7B_INPUTS = REPO / 'results/decision/h_eval/gate_e7b/gate_e7b_train_reference_inputs_v1.csv'
GRID = [(k, C, inter) for k in (3, 5) for C in (0.1, 1.0, 10.0) for inter in (False, True)]
SEL_SALT, N_SEL, MIN_P = 'H-EVAL-R3B-select', 4, 8
BOOT_N, AUC_BOOT_N, SEED = 10_000, 2_000, 20261026
JOIN_SHA = {'VAL': '132a012d3a879c7f583c701994e5e998bdf6f2a369f24cf9a11897ab2c580ce1',
            'TEST': 'a7527cb6afb2c7356b4ca526b577e8a02e64b4fd9b8fbb10a5d45aa2ead3af3f',
            'EXT': '57231b3ca8d796385d6cd361a96b90d8781c1a79708aacdae85168445aa11937'}
EXPECTED_N = {'VAL': (905, 15), 'TEST': (809, 15), 'EXT': (1438, 17)}
MODELS = {'VAL': {'CONSTANT': 'CONSTANT', 'SpecLR-18': 'EEG_LOGIT'},
          'TEST': {'CONSTANT': 'E7B_CONSTANT', 'SpecLR-18': 'E7B_EEG_LOGIT', 'SpecLR-46': 'M0_46', 'RawCNN-46': 'M2_46'},
          'EXT': {'CONSTANT': 'E7B_CONSTANT', 'SpecLR-18': 'E7B_EEG_LOGIT', 'SpecLR-46': 'M0_46', 'RawCNN-46': 'M2_46'}}
CONTRASTS = [('1', 'CONSTANT', 'A'), ('2', 'CONSTANT', 'B'), ('3', 'A', 'SpecLR-18'), ('4', 'B', 'SpecLR-18'),
             ('3s', 'A18', 'SpecLR-18'), ('4s', 'B18', 'SpecLR-18'), ('5', 'A', 'SpecLR-46'), ('6', 'B', 'SpecLR-46'),
             ('7', 'A', 'RawCNN-46'), ('8', 'B', 'RawCNN-46'), ('9', 'SpecLR-46', 'RawCNN-46'),
             ('0a', 'CONSTANT', 'SpecLR-18'), ('0b', 'CONSTANT', 'SpecLR-46'), ('0c', 'CONSTANT', 'RawCNN-46')]


class Abort(RuntimeError):
    pass


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------ references (pure functions of stored parameters)
def spline_basis(logt, lo, hi, k):
    """Cubic B-spline basis, uniform knots on [lo, hi], constant extrapolation (x clipped to [lo, hi])."""
    step = (hi - lo) / (k - 1)
    knots = np.r_[lo - 3 * step + np.arange(3) * step, np.linspace(lo, hi, k), hi + step * np.arange(1, 4)]
    x = np.clip(np.asarray(logt, float), lo, hi)
    return BSpline.design_matrix(x, knots, 3).toarray()


def design(t, s1, lo, hi, k, inter):
    S = spline_basis(np.log(t), lo, hi, k)
    X = [np.asarray(s1, float)[:, None], S]
    if inter:
        X.append(S * np.asarray(s1, float)[:, None])
    return np.hstack(X)


def predict_A(ref, s1):
    s1 = np.asarray(s1)
    return np.where(s1 == 1, ref['p_stage1'], ref['p_wake'])


def predict_B(ref, t, s1):
    X = design(t, s1, ref['logt_lo'], ref['logt_hi'], ref['knots'], ref['interaction'])
    z = ref['intercept'] + X @ np.asarray(ref['coef'])
    return np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)


def fit_A(y, s1):
    y, s1 = np.asarray(y, float), np.asarray(s1)
    return dict(p_wake=float((y[s1 == 0].sum() + 1) / ((s1 == 0).sum() + 2)),
                p_stage1=float((y[s1 == 1].sum() + 1) / ((s1 == 1).sum() + 2)))


def fit_B(t, s1, y, cfg, lo=None, hi=None):
    k, C, inter = cfg
    lt = np.log(t)
    lo = float(lt.min()) if lo is None else lo
    hi = float(lt.max()) if hi is None else hi
    clf = LogisticRegression(C=C, max_iter=5000).fit(design(t, s1, lo, hi, k, inter), y)
    return dict(knots=k, C=C, interaction=bool(inter), logt_lo=lo, logt_hi=hi, intercept=float(clf.intercept_[0]),
                coef=[float(c) for c in clf.coef_[0]])


def select_B(t, s1, y, person, salt=SEL_SALT):
    ranked = sorted(set(person), key=lambda p: hashlib.sha256(f'{salt}|{p}'.encode()).hexdigest())
    fold = {p: i % N_SEL for i, p in enumerate(ranked)}
    fi = np.array([fold[p] for p in person])
    scores = {}
    for cfg in GRID:
        s = 0.0
        for h in range(N_SEL):
            a, b = fi != h, fi == h
            ref = fit_B(t[a], s1[a], y[a], cfg)
            s += float(((predict_B(ref, t[b], s1[b]) - y[b]) ** 2).sum())
        scores[cfg] = s
    best = None
    for cfg in GRID:                                   # ties -> earlier grid entry
        if best is None or scores[cfg] < scores[best] - 1e-12:
            best = cfg
    return best, {f'k{c[0]}_C{c[1]}_int{int(c[2])}': v for c, v in scores.items()}


# ------------------------------------------------------------------------ statistics
def paired(p_ref, p_c, y, person, I):
    pu = sorted(set(person))
    k = np.searchsorted(pu, person)
    n = np.bincount(k, minlength=len(pu)).astype(float)
    sr, sc = np.bincount(k, (p_ref - y) ** 2, len(pu)), np.bincount(k, (p_c - y) ** 2, len(pu))
    lr = -(y * np.log(np.clip(p_ref, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p_ref, 1e-6, 1)))
    lc = -(y * np.log(np.clip(p_c, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p_c, 1e-6, 1)))
    per = (sr - sc) / n
    out = dict(n_persons=len(pu), n_windows=int(n.sum()), brier_ref_ww=float(sr.sum() / n.sum()),
               brier_cand_ww=float(sc.sum() / n.sum()), delta_ww=float((sr.sum() - sc.sum()) / n.sum()),
               R=float((sr.sum() - sc.sum()) / sr.sum()) if sr.sum() > 0 else None, delta_se=float(per.mean()),
               delta_logloss_ww=float((lr.sum() - lc.sum()) / n.sum()), persons_favouring_candidate=int((per > 0).sum()),
               interval_label='R3B re-bootstrap')
    if len(pu) < MIN_P or I is None:
        out['ci'] = 'not_reported_lt_8_persons'
        return out
    Br, Bc, N = sr[I].sum(1), sc[I].sum(1), n[I].sum(1)
    q = lambda a: [float(x) for x in np.nanpercentile(a, [2.5, 97.5])]   # noqa: E731
    with np.errstate(divide='ignore', invalid='ignore'):
        out.update(delta_ww_ci95=q((Br - Bc) / N), R_ci95=q(np.where(Br > 0, (Br - Bc) / Br, np.nan)),
                   delta_se_ci95=q(per[I].mean(1)))
    return out


def calib(p, y):
    cil = float(p.mean() - y.mean())
    if y.min() == y.max():
        return cil, None, 'undefined: outcome constant in stratum'
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if lp.std() < 1e-9:
        return cil, None, 'undefined: prediction constant in stratum'
    return cil, float(LogisticRegression(C=1e6, max_iter=2000).fit(lp[:, None], y).coef_[0, 0]), None


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def matched_pairs(kidx, pid, y):
    pi, ni = [], []
    for kk in np.unique(kidx):
        idx = np.where(kidx == kk)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(pos) and len(neg):
            P, N = np.meshgrid(pos, neg, indexing='ij')
            keep = pid[P] != pid[N]
            pi.append(P[keep])
            ni.append(N[keep])
    if not pi:
        return np.array([], int), np.array([], int)
    return np.concatenate(pi), np.concatenate(ni)


def matched_auroc(p, y, person, kidx, rng):
    pu = sorted(set(person))
    pid = np.searchsorted(pu, person)
    pi, ni = matched_pairs(kidx, pid, y.astype(int))
    if not len(pi):
        return dict(auroc=None, n_pairs=0, reason='no matched cross-person pairs')
    c = (p[pi] > p[ni]) + 0.5 * (p[pi] == p[ni])
    out = dict(auroc=float(c.mean()), n_pairs=int(len(pi)), n_persons_pos=len(set(pid[pi])), n_persons_neg=len(set(pid[ni])))
    if out['n_persons_pos'] >= MIN_P and out['n_persons_neg'] >= MIN_P:
        bs = []
        for _ in range(AUC_BOOT_N):
            w = np.bincount(rng.integers(0, len(pu), len(pu)), minlength=len(pu)).astype(float)
            ww = w[pid[pi]] * w[pid[ni]]
            bs.append((c * ww).sum() / ww.sum() if ww.sum() else np.nan)
        out['ci95'] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
    else:
        out['ci95'] = 'not_reported_lt_8_persons'
    return out


# ------------------------------------------------------------------------ commands
def cmd_fit():
    if REFS.exists():
        raise SystemExit('references exist (write-once)')
    w2sc = json.loads((REPO / 'results/decision/h_eval/discovery_w2s/w2s_classical_results_v1.json').read_text())
    if digest(TABLE) != w2sc['table_sha256']:
        raise Abort('TRAIN table hash mismatch')
    z = np.load(TABLE, allow_pickle=False)
    stage, t, y = z['stage'].astype(str), z['elapsed'].astype(float), z['y'].astype(float)
    person, rec = z['person'].astype(str), z['recording'].astype(str)
    if set(stage) != {'W', '1'} or len(y) != 4792 or len(set(person)) != 46:
        raise Abort('TRAIN table content unexpected')
    s1 = (stage == '1').astype(float)
    A = fit_A(y, s1)
    cfg, scores = select_B(t, s1, y, person)
    B = fit_B(t, s1, y, cfg)
    # matched 18-person SpecLR-18 roster
    rows = list(csv.DictReader(E7B_INPUTS.open()))
    key = {(r, float(e)): i for i, (r, e) in enumerate(zip(rec, t))}
    idx = np.array([key[(r['recording'], float(r['elapsed_s']))] for r in rows])
    if len(idx) != 1271 or len(set(person[idx])) != 18 or not all(int(r['y']) == y[i] for r, i in zip(rows, idx)):
        raise Abort('SpecLR-18 roster does not reproduce')
    A18 = fit_A(y[idx], s1[idx])
    cfg18, scores18 = select_B(t[idx], s1[idx], y[idx], person[idx])
    B18 = fit_B(t[idx], s1[idx], y[idx], cfg18)
    pins = json.loads((REPO / 'results/decision/h_eval/gate_e7b/gate_e7b_freeze_manifest_v1_2.json').read_text())['files']
    pin = pins.get('results/decision/h_eval/gate_e7b/gate_e7b_train_reference_inputs_v1.csv')
    for name, ref, pb in (('B', B, predict_B(B, t, s1)), ('B18', B18, predict_B(B18, t[idx], s1[idx]))):
        if not (np.all(pb > 0) and np.all(pb < 1)):
            raise Abort('probability bounds ' + name)
    out = dict(version='H-EVAL-R3B-references-v1', status='TRAIN_ONLY_FROZEN_ORACLE_STAGE_REFERENCES',
               note='Oracle diagnostics using retrospective scorer stage; not deployable stagers or forecasters.',
               train_table_sha256=digest(TABLE), train_persons=sorted(set(person)), n_train_windows=int(len(y)),
               A=A, B=B, B_selection=dict(selected=dict(knots=cfg[0], C=cfg[1], interaction=cfg[2]), inner_brier_sum=scores),
               speclr18_roster=dict(file=str(E7B_INPUTS.relative_to(REPO)), sha256=digest(E7B_INPUTS), pinned_in_e7b_manifest_v1_2=pin,
                                    pin_matches=(pin == digest(E7B_INPUTS)) if pin else None,
                                    persons=sorted(set(person[idx])), n_windows=int(len(idx))),
               A18=A18, B18=B18, B18_selection=dict(selected=dict(knots=cfg18[0], C=cfg18[1], interaction=cfg18[2]), inner_brier_sum=scores18),
               train_in_sample=dict(brier_A=float(((predict_A(A, s1) - y) ** 2).mean()), brier_B=float(((predict_B(B, t, s1) - y) ** 2).mean()),
                                    elapsed_support_s=[float(t.min()), float(t.max())]),
               code_sha256=digest(Path(__file__)))
    write_new(REFS, (json.dumps(out, indent=1) + '\n').encode())
    print(json.dumps({k: out[k] for k in ('A', 'B_selection', 'A18', 'B18_selection', 'speclr18_roster', 'train_in_sample')}, indent=1)[:3000])


def cmd_freeze():
    if FREEZE.exists():
        raise SystemExit('freeze record exists')
    p = subprocess.run([sys.executable, '-m', 'unittest', '-v', TESTS.name], cwd=HERE, capture_output=True, text=True)
    log = p.stdout + p.stderr
    if p.returncode != 0:
        print(log)
        raise SystemExit('synthetic tests failed')
    tests = [l.split(' ... ')[0] for l in log.splitlines() if l.rstrip().endswith('... ok')]
    write_new(RES / 'r3b_synthetic_tests.json', (json.dumps(dict(passed=True, n_tests=len(tests), tests=tests,
                                                                 log_sha256=hashlib.sha256(log.encode()).hexdigest(),
                                                                 held_out_data_accessed=False), indent=1) + '\n').encode())
    rec = dict(version='H-EVAL-R3B-v1-2026-10-25', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               protocol_sha256=digest(PROTOCOL), code_sha256=digest(Path(__file__)), tests_sha256=digest(TESTS),
               synthetic_tests_sha256=digest(RES / 'r3b_synthetic_tests.json'), references_sha256=digest(REFS),
               held_out_files_parsed_before_freeze=False,
               note='Prospectively frozen computational procedure for a retrospective, post-hoc question; not a preregistered confirmatory test.')
    write_new(FREEZE, (json.dumps(rec, indent=1) + '\n').encode())
    print(json.dumps(rec, indent=1))


def load_join(pop):
    with (LOC.parent / 'repair_r2' / f'r2_window_stage_join_{pop}.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def cmd_execute():
    fr = json.loads(FREEZE.read_text())
    now = dict(protocol_sha256=digest(PROTOCOL), code_sha256=digest(Path(__file__)), tests_sha256=digest(TESTS), references_sha256=digest(REFS))
    if any(fr[k] != v for k, v in now.items()):
        raise SystemExit('frozen hashes differ; refusing to run')
    for p in (RES / 'r3b_integrity.json', RES / 'r3b_results_v1.json'):
        if p.exists():
            raise SystemExit(f'{p} exists (write-once)')
    t0 = time.time()
    refs = json.loads(REFS.read_text())
    r2 = _imp(REPO / 'scripts/decision/h_eval/repair_r2/r2_stage_audit.py', 'r2_for_r3b')
    e5 = _imp(REPO / 'scripts/decision/h_eval/gate_e5/gate_e5.py', 'e5_for_r3b')
    inv = {r['recording']: r for r in csv.DictReader((REPO / 'results/decision/h_eval/gate_e2/sleep_edf_origin_inventory.csv').open(newline=''))}
    r2agg = json.loads((REPO / 'results/decision/h_eval/repair_r2/r2_stage_results.json').read_text())['populations']
    integ, data = dict(version='H-EVAL-R3B-v1', freeze_record_sha256=digest(FREEZE)), {}
    # ---- phase 2: verification (all populations before any statistic)
    for pop in ('VAL', 'TEST', 'EXT'):
        cfg = r2.POPS[pop]
        content, recs, rep = r2.verify_population(pop, cfg, inv, e5)                        # A1-A6, A10
        yv, Pv, per = [], {m: [] for m in [cfg['ref']] + list(cfg['cands'])}, []
        truth = {}
        for rc in recs:
            el = np.array([float(r['elapsed_s']) for r in rc['rows']])
            yy, _ = r2.labels(rc['off'], rc['n2'], el)
            for r, lab in zip(rc['rows'], yy):
                truth[(r['recording'], float(r['elapsed_s']))] = (int(lab), r)
                yv.append(lab)
                per.append(r['person'])
                for m in Pv:
                    Pv[m].append(float(r['p_' + m]))
        chk = r2.reproduce(pop, cfg, content, np.array(yv, float), {m: np.array(v) for m, v in Pv.items()}, np.array(per))
        jpath = LOC.parent / 'repair_r2' / f'r2_window_stage_join_{pop}.csv'
        if digest(jpath) != JOIN_SHA[pop]:
            raise Abort(f'{pop}: join hash mismatch')
        J = load_join(pop)
        keys = [(r['recording'], float(r['elapsed_s'])) for r in J]
        if len(set(keys)) != len(keys) or set(keys) != set(truth):
            raise Abort(f'{pop}: join rows differ from verified predictions')
        for r in J:
            lab, orig = truth[(r['recording'], float(r['elapsed_s']))]
            if int(r['y']) != lab or r['person'] != orig['person'] or r.get('condition', '') != orig.get('condition', ''):
                raise Abort(f'{pop}: join label/person mismatch')
            for m in Pv:
                if abs(float(r['p_' + m]) - float(orig['p_' + m])) > 1e-12:
                    raise Abort(f'{pop}: join prediction mismatch')
            if r['stage'] not in ('W', '1'):
                raise Abort(f'{pop}: unexpected stage {r["stage"]}')
        n_exp, p_exp = EXPECTED_N[pop]
        if len(J) != n_exp or len({r['person'] for r in J}) != p_exp or (pop == 'EXT' and len({r['recording'] for r in J}) != 28):
            raise Abort(f'{pop}: counts')
        comp = r2agg[pop]['composition']
        for s in ('W', '1'):
            pos = sum(1 for r in J if r['stage'] == s and r['y'] == '1')
            neg = sum(1 for r in J if r['stage'] == s and r['y'] == '0')
            if (pos, neg) != (comp[s]['positive'], comp[s]['negative']):
                raise Abort(f'{pop}: stage composition differs from R2')
        data[pop] = J
        integ[pop] = dict(envelope_reproduction=chk, n_windows=len(J), n_persons=len({r['person'] for r in J}),
                          n_recordings=len({r['recording'] for r in J}), join_sha256=JOIN_SHA[pop], join_complete_and_consistent=True,
                          stage_composition_equals_R2=True, verify_population=rep)
    if sum(len(v) for v in data.values()) != 3152:
        raise Abort('total windows != 3152')
    # ---- phase 3/4: statistics
    rng = np.random.default_rng(SEED)
    results, private = {}, {}
    lo, hi = refs['B']['logt_lo'], refs['B']['logt_hi']
    for pop, J in data.items():
        y = np.array([int(r['y']) for r in J], float)
        person = np.array([r['person'] for r in J])
        recs = np.array([r['recording'] for r in J])
        t = np.array([float(r['elapsed_s']) for r in J])
        s1 = np.array([1.0 if r['stage'] == '1' else 0.0 for r in J])
        pre = np.array([r['pre_first_sleep'] == 'True' for r in J])
        P = {name: np.array([float(r['p_' + col]) for r in J]) for name, col in MODELS[pop].items()}
        P['A'], P['B'] = predict_A(refs['A'], s1), predict_B(refs['B'], t, s1)
        P['A18'], P['B18'] = predict_A(refs['A18'], s1), predict_B(refs['B18'], t, s1)
        for k in ('A', 'B', 'A18', 'B18'):
            assert np.all(P[k] > 0) and np.all(P[k] < 1)
        out_of_support = int(((np.log(t) < lo) | (np.log(t) > hi)).sum())
        kidx = np.floor(t / 30 + 1e-9).astype(int)
        strata = {'ALL': np.ones(len(y), bool), 'W': s1 == 0, 'STAGE_1': s1 == 1, 'PRE_FIRST_SLEEP': pre, 'RETURNING_W': (s1 == 0) & ~pre}
        R = dict(available_models=sorted(MODELS[pop]), not_available=sorted(set(['SpecLR-46', 'RawCNN-46']) - set(MODELS[pop])),
                 out_of_support_windows_B=out_of_support, strata={})
        for sn, m in strata.items():
            S = dict(n_windows=int(m.sum()), n_positive=int(y[m].sum()), n_persons=len(set(person[m])),
                     n_recordings=len(set(recs[m])), prevalence=float(y[m].mean()) if m.any() else None, models={}, contrasts={})
            if not m.any():
                R['strata'][sn] = S
                continue
            pu = sorted(set(person[m]))
            I = rng.integers(0, len(pu), size=(BOOT_N, len(pu))) if len(pu) >= MIN_P else None
            for name, p in P.items():
                pm, ym, perm = p[m], y[m], person[m]
                cil, slope, why = calib(pm, ym)
                b = (pm - ym) ** 2
                S['models'][name] = dict(brier_ww=float(b.mean()), brier_se=float(np.mean([b[perm == q].mean() for q in pu])),
                                         logloss_ww=float(np.mean(-(ym * np.log(np.clip(pm, 1e-6, 1)) + (1 - ym) * np.log(np.clip(1 - pm, 1e-6, 1))))),
                                         cal_in_large=cil, cal_slope=slope, cal_slope_undefined_reason=why,
                                         pooled_auroc=auroc(pm, ym),
                                         pooled_auroc_undefined_reason=None if auroc(pm, ym) is not None else 'one class absent',
                                         matched_time_auroc=matched_auroc(pm, ym, perm, kidx[m], rng))
            for cid, a, b in CONTRASTS:
                if a not in P or b not in P:
                    S['contrasts'][f'{cid}:{a}->{b}'] = 'not_available'
                    continue
                S['contrasts'][f'{cid}:{a}->{b}'] = paired(P[a][m], P[b][m], y[m], person[m], I)
            R['strata'][sn] = S
        results[pop] = R
        private[pop] = dict(y=y, person=person, recording=recs, elapsed=t, s1=s1, pre=pre, **{f'p_{k}': v for k, v in P.items()})
    # reconciliation with R2 ALL-subset point estimates
    recon = {}
    keymap = {'VAL': {'SpecLR-18': 'CONSTANT->SpecLR-18 [EEG_LOGIT]'},
              'TEST': {'SpecLR-18': 'CONSTANT->SpecLR-18 [E7B_EEG_LOGIT]', 'SpecLR-46': 'CONSTANT->SpecLR-46 [M0_46]', 'RawCNN-46': 'CONSTANT->RawCNN-46 [M2_46]'},
              'EXT': {'SpecLR-18': 'CONSTANT->SpecLR-18 [E7B_EEG_LOGIT]', 'SpecLR-46': 'CONSTANT->SpecLR-46 [M0_46]', 'RawCNN-46': 'CONSTANT->RawCNN-46 [M2_46]'}}
    cid = {'SpecLR-18': '0a', 'SpecLR-46': '0b', 'RawCNN-46': '0c'}
    for pop, km in keymap.items():
        for mdl, k2 in km.items():
            for sn, s2 in (('ALL', 'ALL'), ('W', 'STAGE_W'), ('STAGE_1', 'STAGE_1'), ('PRE_FIRST_SLEEP', 'PRE_FIRST_SLEEP')):
                a = results[pop]['strata'][sn]['contrasts'][f'{cid[mdl]}:CONSTANT->{mdl}']
                b = r2agg[pop]['subsets'][s2]['contrasts'][k2]
                ok = abs(a['delta_ww'] - b['delta_ww']) < 1e-12 and abs(a['R'] - b['R']) < 1e-12
                recon[f'{pop}|{mdl}|{sn}'] = ok
                if not ok:
                    raise Abort(f'R2 reconciliation failed {pop} {mdl} {sn}')
    integ['r2_point_estimate_reconciliation'] = dict(n_checked=len(recon), all_equal=all(recon.values()))
    integ['execution'] = dict(raw_eeg_read=False, hypnograms_read=False, eeg_models_run=False, recalibration=False,
                              runtime_s=round(time.time() - t0, 1))
    LOC.mkdir(parents=True, exist_ok=True)
    for pop, arr in private.items():
        buf = io.BytesIO()
        np.savez(buf, **arr)
        write_new(LOC / f'r3b_heldout_reference_join_{pop}.npz', buf.getvalue())
    write_new(RES / 'r3b_integrity.json', (json.dumps(integ, indent=1, default=str) + '\n').encode())
    write_new(RES / 'r3b_results_v1.json', (json.dumps(dict(version='H-EVAL-R3B-v1', status='POST_HOC_EXPLORATORY_DESCRIPTIVE',
                                                              bootstrap=dict(n=BOOT_N, auc_n=AUC_BOOT_N, seed=SEED, cluster='person', min_persons=MIN_P),
                                                              populations=results), indent=1) + '\n').encode())
    print('integrity OK; results written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('fit', 'freeze', 'execute'))
    a = ap.parse_args()
    try:
        {'fit': cmd_fit, 'freeze': cmd_freeze, 'execute': cmd_execute}[a.cmd]()
    except Abort as exc:
        raise SystemExit('R3B ABORT (no results written): ' + str(exc))


if __name__ == '__main__':
    main()
