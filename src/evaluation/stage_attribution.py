#!/usr/bin/env python3
"""H-EVAL Repair Gate R2: post-hoc descriptive stage stratification of FROZEN VAL/TEST/EXT predictions.

Implements docs/decision/h_eval/repair_r2/r2_protocol.md. POST-HOC, EXPLORATORY, DESCRIPTIVE.
Reads only frozen prediction CSVs, result envelopes, freeze manifests, execution markers, rosters, E2/E3 metadata and
official hypnograms of roster recordings. Never opens a PSG/EEG file, never loads a model, never runs any gate command.

Usage:  r2_stage_audit.py freeze    run synthetic tests, then write the freeze record (before any held-out hypnogram)
        r2_stage_audit.py execute   verify inputs, align, stratify; write aggregate results (write-once)
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VERSION = 'H-EVAL-R2-stage-audit-v1-2026-10-24'
PROTOCOL = REPO / 'docs/decision/h_eval/repair_r2/r2_protocol.md'
TESTS = REPO / 'scripts/decision/h_eval/repair_r2/test_r2_stage_audit.py'
RES = REPO / 'results/decision/h_eval/repair_r2'
LOC = REPO / '.local/h_eval/repair_r2'
FREEZE = RES / 'r2_freeze_record.json'
RAW = REPO / 'data/raw/sleep_edfx_v1_0_0'
EV = REPO / 'results/decision/h_eval'
BOOT_N, BOOT_SEED, MIN_CI_PERSONS = 10_000, 20261025, 8
EPS = 1e-6
SLEEP = {'1', '2', '3', '4', 'R'}
UNEXPECTED = {'2', '3', '4'}

POPS = {
    'VAL': dict(pred='.local/h_eval/gate_e7b/validation_primary_predictions_v1_2.csv',
                env='results/decision/h_eval/gate_e7b/execution_v1_2/gate_e7b_primary_result_v1_2.json',
                manifest='results/decision/h_eval/gate_e7b/gate_e7b_freeze_manifest_v1_2.json',
                marker='.local/h_eval/gate_e7b/E7B_V1_2_EXECUTION_MARKER.json',
                roster='results/decision/h_eval/gate_e7b/gate_e7b_validation_roster_v1.json', roles=('PRIMARY',),
                ref='CONSTANT', n_persons=15,
                cands={'EEG_LOGIT': 'SpecLR-18', 'TIME_CONT_EEG_LOGIT': 'SpecLR-18+time',
                       'EEG_RESID_LOGIT': 'SpecLR-18 drift-residual', 'TIME_CONT': 'time-only (context)'},
                m2m={}),
    'TEST': dict(pred='.local/h_eval/gate_t1/protected_primary_predictions_v1.csv',
                 env='results/decision/h_eval/gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json',
                 manifest='results/decision/h_eval/gate_t1/gate_t1_freeze_manifest_v1.json',
                 marker='.local/h_eval/gate_t1/T1_EXECUTION_MARKER.json',
                 roster='results/decision/h_eval/gate_t1/gate_t1_roster_v1.json', roles=('PRIMARY',),
                 ref='E7B_CONSTANT', n_persons=15,
                 cands={'E7B_EEG_LOGIT': 'SpecLR-18', 'M0_46': 'SpecLR-46', 'M2_46': 'RawCNN-46',
                        'E7B_TIME_CONT_EEG_LOGIT': 'SpecLR-18+time', 'E7B_EEG_RESID_LOGIT': 'SpecLR-18 drift-residual',
                        'E7B_TIME_CONT': 'time-only (context)'},
                 m2m={'SpecLR-18->RawCNN-46': ('E7B_EEG_LOGIT', 'M2_46'), 'SpecLR-46->RawCNN-46': ('M0_46', 'M2_46')}),
    'EXT': dict(pred='.local/h_eval/gate_x1/st_predictions_v1.csv',
                env='results/decision/h_eval/gate_x1/execution_v1/gate_x1_result_v1.json',
                manifest='results/decision/h_eval/gate_x1/gate_x1_freeze_manifest_v1.json',
                marker='.local/h_eval/gate_x1/X1_EXECUTION_MARKER.json',
                roster='results/decision/h_eval/gate_x1/gate_x1_roster_v1.json', roles=('PRIMARY',),
                ref='E7B_CONSTANT', n_persons=17,
                cands={'E7B_EEG_LOGIT': 'SpecLR-18', 'M0_46': 'SpecLR-46', 'M2_46': 'RawCNN-46',
                       'E7B_TIME_CONT_EEG_LOGIT': 'SpecLR-18+time', 'E7B_EEG_RESID_LOGIT': 'SpecLR-18 drift-residual',
                       'E7B_TIME_CONT': 'time-only (context)'},
                m2m={'SpecLR-18->RawCNN-46': ('E7B_EEG_LOGIT', 'M2_46'), 'SpecLR-46->RawCNN-46': ('M0_46', 'M2_46')}),
}
MANIFEST_PINS = ('results/decision/h_eval/gate_e2/sleep_edf_origin_inventory.csv',
                 'results/decision/h_eval/gate_e3/gate_e3_stage_diagnostics.csv',
                 'scripts/decision/h_eval/gate_e5/gate_e5.py')


class Abort(RuntimeError):
    """Protocol §2 abort: nothing is written."""


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_new(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


# ------------------------------------------------------------------------ pure functions (synthetic-tested)
def stage_events(events):
    """(onset, end, label) for 'Sleep stage X' annotations; label = X upper-cased."""
    out = []
    for onset, dur, lab in events:
        if lab.lower().startswith('sleep stage '):
            out.append((float(onset), float(onset) + float(dur), lab[len('sleep stage '):].strip().upper()))
    return sorted(out)


def attribute(stages, a, b):
    """Stage category of [a, b) per protocol §3."""
    ov = [(max(s, a), min(e, b), lab) for s, e, lab in stages if min(e, b) - max(s, a) > EPS]
    if not ov:
        return 'UNSCORED'
    ov.sort()
    for i in range(len(ov)):
        for j in range(i + 1, len(ov)):
            if ov[j][0] < ov[i][1] - EPS and ov[i][2] != ov[j][2]:
                return 'AMBIGUOUS'
    cur = a
    for s, e, _ in ov:                       # coverage check (same-label overlaps allowed)
        if s > cur + EPS:
            return 'UNSCORED'
        cur = max(cur, e)
    if cur < b - EPS:
        return 'UNSCORED'
    labs = []
    for _, _, lab in ov:
        if lab not in labs:
            labs.append(lab)
    return labs[0] if len(labs) == 1 else 'MIXED:' + '|'.join(labs)


def first_sleep(stages, off):
    c = [max(s, off) for s, e, lab in stages if lab in SLEEP and e > off + EPS]
    return min(c) if c else None


def check_alignment(stages, off, n2):
    """A8: a stage-2 annotation starts exactly at n2 and no stage-2 annotation overlaps [off, n2)."""
    if not any(lab == '2' and abs(s - n2) <= EPS for s, e, lab in stages):
        raise Abort(f'A8: no stage-2 annotation starts at n2={n2}')
    for s, e, lab in stages:
        if lab == '2' and min(e, n2) - max(s, off) > EPS:
            raise Abort(f'A8: stage 2 inside [off, n2) at {s}')


def labels(off, n2, elapsed):
    t = off + np.asarray(elapsed, float)
    return ((n2 - t) <= 300 + 1e-8).astype(int), t


def boot_index(n_persons, seed=BOOT_SEED, n=BOOT_N):
    return np.random.default_rng(seed).integers(0, n_persons, size=(n, n_persons))


def contrast(p_ref, p_cand, y, person, I=None):
    """Aggregate paired statistics on one subset (arrays already masked). I: bootstrap index over sorted persons."""
    pu = sorted(set(person))
    k = np.searchsorted(pu, person)
    br, bc = (p_ref - y) ** 2, (p_cand - y) ** 2
    pr, pc = np.clip(p_ref, 1e-6, 1 - 1e-6), np.clip(p_cand, 1e-6, 1 - 1e-6)
    lr = -(y * np.log(pr) + (1 - y) * np.log1p(-pr))
    lc = -(y * np.log(pc) + (1 - y) * np.log1p(-pc))
    n = np.bincount(k, minlength=len(pu)).astype(float)
    sr, sc = np.bincount(k, br, len(pu)), np.bincount(k, bc, len(pu))
    per = (sr - sc) / n
    ref_ww = sr.sum() / n.sum()
    out = dict(n_windows=int(n.sum()), n_persons=len(pu), brier_ref_ww=float(ref_ww), brier_cand_ww=float(sc.sum() / n.sum()),
               delta_ww=float((sr.sum() - sc.sum()) / n.sum()),
               R=(float((sr.sum() - sc.sum()) / sr.sum()) if ref_ww > 1e-12 else None),
               delta_se=float(per.mean()), delta_logloss_ww=float((lr.sum() - lc.sum()) / n.sum()),
               persons_favouring_candidate=int((per > 0).sum()))
    if len(pu) < MIN_CI_PERSONS or I is None:
        out['ci'] = 'not_reported_lt_8_persons' if len(pu) < MIN_CI_PERSONS else 'not_computed'
        return out
    Br, Bc, N = sr[I].sum(1), sc[I].sum(1), n[I].sum(1)
    ok = Br / N > 1e-12
    q = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]   # noqa: E731
    out.update(delta_ww_ci95=q((Br - Bc) / N), R_ci95=q(((Br - Bc) / Br)[ok]), R_boot_undefined=int((~ok).sum()),
               delta_se_ci95=q(per[I].mean(1)))
    return out


def contributions(b_ref, b_cand, cats):
    total = float((b_ref - b_cand).mean())
    c = {s: float((b_ref - b_cand)[cats == s].sum() / len(cats)) for s in sorted(set(cats))}
    if abs(sum(c.values()) - total) > 1e-12:
        raise Abort('contribution decomposition does not sum to delta_ww')
    return dict(delta_ww_all=total, absolute=c,
                share=({s: v / total for s, v in c.items()} if abs(total) > 1e-9 else 'undefined_delta_near_0'))


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


# ------------------------------------------------------------------------ input verification
def canon_sha(content):
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=float).encode()).hexdigest()


def verify_population(name, cfg, inv, e5):
    rep = dict(population=name)
    env = json.loads((REPO / cfg['env']).read_text())
    if canon_sha(env['content']) != env['content_sha256']:
        raise Abort(f'A1 {name}: envelope content hash')
    c = env['content']
    if digest(REPO / cfg['pred']) != c['predictions_sha256']:
        raise Abort(f'A1 {name}: predictions hash')
    if digest(REPO / cfg['manifest']) != c['manifest_sha256']:
        raise Abort(f'A1 {name}: manifest hash')
    marker = json.loads((REPO / cfg['marker']).read_text())
    if marker['manifest_sha256'] != c['manifest_sha256'] or marker['approval_ref'] != c['approval_ref']:
        raise Abort(f'A1 {name}: marker disagrees with envelope')
    pins = json.loads((REPO / cfg['manifest']).read_text())['files']
    for p in MANIFEST_PINS + (cfg['roster'],):
        if p not in pins or digest(REPO / p) != pins[p]:
            raise Abort(f'A1 {name}: manifest pin mismatch {p}')
    rep.update(envelope_content_sha256=env['content_sha256'], predictions_sha256=c['predictions_sha256'],
               manifest_sha256=c['manifest_sha256'], approval_ref=c['approval_ref'], roster_sha256=pins[cfg['roster']])
    roster = [r for r in json.loads((REPO / cfg['roster']).read_text())['recordings'] if r['role'] in cfg['roles']]
    rmap = {r['recording']: r for r in roster}
    with (REPO / cfg['pred']).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if {r['recording'] for r in rows} != set(rmap):
        raise Abort(f'A2 {name}: prediction recordings differ from roster')
    keys = [(r['recording'], float(r['elapsed_s'])) for r in rows]
    if len(set(keys)) != len(keys):
        raise Abort(f'A3 {name}: duplicate landmarks')
    by = defaultdict(list)
    for r in rows:
        by[r['recording']].append(r)
        ro = rmap[r['recording']]
        if r['person'] != ro['subject_key'] or ('condition' in r and r['condition'] != ro.get('condition')):
            raise Abort(f'A5 {name}: person/condition mismatch {r["recording"]}')
    cols = [cfg['ref']] + list(cfg['cands'])
    for r in rows:
        for m in cols:
            v = float(r['p_' + m])
            if not (math.isfinite(v) and 0 < v < 1):
                raise Abort(f'A6 {name}: invalid prediction')
    recs = []
    for rec, rr in sorted(by.items()):
        row = inv[rec]
        off, n2 = float(row['lights_off_rel_s']), float(row['first_n2_after_off_s'])
        lm = e5.make_landmarks(off, n2, row)
        rr.sort(key=lambda r: float(r['elapsed_s']))
        t = [off + float(r['elapsed_s']) for r in rr]
        if len(t) != len(lm) or any(abs(a - b) > 1e-6 for a, b in zip(t, lm)):
            raise Abort(f'A4 {name}: landmark set mismatch {rec}')
        recs.append(dict(recording=rec, row=row, roster=rmap[rec], off=off, n2=n2, rows=rr))
    persons = {r['subject_key'] for r in roster}
    if len(persons) != cfg['n_persons']:
        raise Abort(f'A10 {name}: person count {len(persons)}')
    if name == 'EXT':
        cc = Counter(r['condition'] for r in roster)
        if cc != Counter(placebo=15, temazepam=13):
            raise Abort('A10 EXT: condition counts ' + str(cc))
    rep.update(n_roster_recordings=len(roster), n_persons=len(persons), n_prediction_rows=len(rows),
               rows_match_make_landmarks=True)
    return env['content'], recs, rep


def reproduce(name, cfg, content, y, P, person):
    """Protocol §6: recompute published values on ALL windows from original predictions + reconstructed labels."""
    tol = 1e-10
    chk = []

    def eq(label, a, b):
        ok = abs(float(a) - float(b)) <= tol
        chk.append(dict(value=label, published=float(b), recomputed=float(a), ok=bool(ok)))

    def ww(m):
        return float(((P[m] - y) ** 2).mean())

    def se(m):
        return float(np.mean([((P[m][person == p] - y[person == p]) ** 2).mean() for p in sorted(set(person))]))
    if name == 'VAL':
        s = content['scores']
        eq('n_windows', len(y), s['n_windows'])
        eq('n_positive', y.sum(), s['n_positive'])
        pr = s['primary']
        eq('primary.ref_ww', ww('CONSTANT'), pr['ref_ww'])
        eq('primary.cand_ww', ww('EEG_LOGIT'), pr['cand_ww'])
        eq('primary.delta_ww', ww('CONSTANT') - ww('EEG_LOGIT'), pr['delta_ww'])
        eq('primary.R', (ww('CONSTANT') - ww('EEG_LOGIT')) / ww('CONSTANT'), pr['R'])
        eq('primary.delta_se', se('CONSTANT') - se('EEG_LOGIT'), pr['delta_se'])
    else:
        r = content['results']
        eq('n_windows', len(y), r['n_windows'])
        eq('n_positive', y.sum(), r['n_positive'])
        for pub, col in (('E7B_CONSTANT', 'E7B_CONSTANT'), ('E7B_EEG', 'E7B_EEG_LOGIT'), ('M0_46', 'M0_46'), ('M2_46', 'M2_46')):
            eq(f'models.{pub}.brier_ww', ww(col), r['models'][pub]['brier_ww'])
            eq(f'models.{pub}.brier_se', se(col), r['models'][pub]['brier_se'])
        for h, (a, b) in (('H2', ('E7B_EEG_LOGIT', 'M2_46')), ('H3', ('M0_46', 'M2_46'))):
            eq(f'{h}.delta_ww', ww(a) - ww(b), r[h]['delta_ww'])
            eq(f'{h}.R', (ww(a) - ww(b)) / ww(a), r[h]['R'])
    if not all(c['ok'] for c in chk):
        raise Abort(f'A7 {name}: envelope reproduction failed ' + json.dumps([c for c in chk if not c['ok']]))
    return chk


# ------------------------------------------------------------------------ population analysis
def analyse(name, cfg, recs, ge, sums):
    win = []
    hyp = []
    for rc in recs:
        row = rc['row']
        hp = row['hypnogram_path']
        if name == 'EXT' and rc['roster'].get('hypnogram_path') != hp:
            raise Abort('A1 EXT: roster/inventory hypnogram path differ ' + rc['recording'])
        raw = (RAW / hp).read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        if h != sums.get(hp) or (name == 'EXT' and h != rc['roster']['hypnogram_official_sha256']):
            raise Abort('A1: hypnogram hash ' + hp)
        hyp.append(dict(recording=rc['recording'], hypnogram_sha256=h))
        st = stage_events(ge.decode_edfplus_annotations(raw)[0])
        check_alignment(st, rc['off'], rc['n2'])
        fs = first_sleep(st, rc['off'])
        el = np.array([float(r['elapsed_s']) for r in rc['rows']])
        y, t = labels(rc['off'], rc['n2'], el)
        for r, yy, tt in zip(rc['rows'], y, t):
            if tt - 30 < -EPS:
                raise Abort('A8: window before signal start')
            cat = attribute(st, tt - 30, tt)
            if cat in UNEXPECTED or any(x in UNEXPECTED for x in cat.split(':')[-1].split('|')):
                raise Abort(f'A9 {name}: unexpected stage {cat} in {rc["recording"]}')
            win.append(dict(recording=rc['recording'], person=r['person'], condition=r.get('condition', ''),
                            elapsed_s=float(r['elapsed_s']), y=int(yy), stage=cat,
                            pre_first_sleep=bool(fs is not None and tt <= fs + EPS),
                            first_sleep_after_off_s=(None if fs is None else fs - rc['off']),
                            **{'p_' + m: float(r['p_' + m]) for m in [cfg['ref']] + list(cfg['cands'])}))
    return win, hyp


def summarise(name, cfg, win):
    y = np.array([w['y'] for w in win], float)
    person = np.array([w['person'] for w in win])
    rec = np.array([w['recording'] for w in win])
    stage = np.array([w['stage'] for w in win])
    pre = np.array([w['pre_first_sleep'] for w in win])
    cond = np.array([w['condition'] for w in win])
    P = {m: np.array([w['p_' + m] for w in win]) for m in [cfg['ref']] + list(cfg['cands'])}
    subsets = {'ALL': np.ones(len(y), bool)}
    for s in sorted(set(stage)):
        subsets['STAGE_' + s] = stage == s
    subsets['PRE_FIRST_SLEEP'] = pre
    subsets['POST_FIRST_SLEEP'] = ~pre
    subsets['W_POST_FIRST_SLEEP'] = (stage == 'W') & ~pre
    if name == 'EXT':
        for c in ('placebo', 'temazepam'):
            subsets[f'{c}:ALL'] = cond == c
            subsets[f'{c}:STAGE_W'] = (cond == c) & (stage == 'W')
            subsets[f'{c}:STAGE_1'] = (cond == c) & (stage == '1')
            subsets[f'{c}:PRE_FIRST_SLEEP'] = (cond == c) & pre
    pairs = {f'CONSTANT->{lab} [{m}]': (cfg['ref'], m) for m, lab in cfg['cands'].items()}
    pairs.update({f'{k} [{a}->{b}]': (a, b) for k, (a, b) in cfg['m2m'].items()})
    out = {}
    for sname, m in subsets.items():
        info = dict(n_windows=int(m.sum()), n_positive=int(y[m].sum()), n_negative=int((1 - y[m]).sum()),
                    n_persons=len(set(person[m])), n_recordings=len(set(rec[m])))
        if m.sum() == 0:
            out[sname] = dict(**info, contrasts='empty')
            continue
        pu = sorted(set(person[m]))
        I = boot_index(len(pu)) if len(pu) >= MIN_CI_PERSONS else None
        info['contrasts'] = {k: contrast(P[a][m], P[b][m], y[m], person[m], I) for k, (a, b) in pairs.items()}
        out[sname] = info
    contrib = {k: contributions((P[a] - y) ** 2, (P[b] - y) ** 2, stage) for k, (a, b) in pairs.items()}
    contrib_pre = {k: contributions((P[a] - y) ** 2, (P[b] - y) ** 2,
                                    np.where(pre, 'PRE_FIRST_SLEEP', 'POST_FIRST_SLEEP')) for k, (a, b) in pairs.items()}
    wa = {}
    w = stage == 'W'
    for mname in cfg['cands']:
        vals = [auroc(P[mname][(rec == r) & w], y[(rec == r) & w]) for r in sorted(set(rec))]
        d = [v for v in vals if v is not None]
        wa[mname] = dict(mean=(float(np.mean(d)) if d else None), median=(float(np.median(d)) if d else None),
                         n_defined=len(d), n_undefined=len(vals) - len(d))
    comp = {s: dict(negative=int(((stage == s) & (y == 0)).sum()), positive=int(((stage == s) & (y == 1)).sum()),
                    persons=len(set(person[stage == s])), recordings=len(set(rec[stage == s])))
            for s in sorted(set(stage))}
    fsl = {}
    for r in sorted(set(rec)):
        v = [x for x in win if x['recording'] == r][0]['first_sleep_after_off_s']
        fsl[r] = v
    return dict(composition=comp, subsets=out, stage_contribution=contrib, pre_post_contribution=contrib_pre,
                within_recording_auroc_w_only=wa,
                first_sleep=dict(n_recordings=len(fsl), n_with_sleep_before_n2=sum(v is not None for v in fsl.values()),
                                 median_first_sleep_after_off_s=float(np.median([v for v in fsl.values() if v is not None]))))


# ------------------------------------------------------------------------ commands
def code_hashes():
    return dict(protocol_sha256=digest(PROTOCOL), code_sha256=digest(Path(__file__)), tests_sha256=digest(TESTS))


def cmd_freeze():
    if FREEZE.exists():
        raise SystemExit('freeze record exists; refusing to overwrite')
    t0 = time.time()
    p = subprocess.run([sys.executable, '-m', 'unittest', '-v', str(TESTS.name)], cwd=TESTS.parent,
                       capture_output=True, text=True)
    log = p.stdout + p.stderr
    if p.returncode != 0:
        print(log)
        raise SystemExit('synthetic tests failed; not frozen')
    n = sum(1 for line in log.splitlines() if line.rstrip().endswith('... ok'))
    write_new(RES / 'r2_synthetic_tests.json', (json.dumps(dict(
        version=VERSION, passed=True, n_tests=n, runtime_s=round(time.time() - t0, 2),
        log_sha256=hashlib.sha256(log.encode()).hexdigest(), held_out_data_accessed=False,
        tests=[l.split(' ... ')[0] for l in log.splitlines() if l.rstrip().endswith('... ok')]), indent=2) + '\n').encode())
    rec = dict(version=VERSION, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **code_hashes(),
               synthetic_tests_sha256=digest(RES / 'r2_synthetic_tests.json'), held_out_hypnograms_read=False,
               note='Frozen before any held-out hypnogram was read. Analysis remains POST-HOC and EXPLORATORY.')
    write_new(FREEZE, (json.dumps(rec, indent=2) + '\n').encode())
    print(json.dumps(rec, indent=2))


def cmd_execute():
    fr = json.loads(FREEZE.read_text())
    now = code_hashes()
    if any(fr[k] != now[k] for k in now):
        raise SystemExit('protocol/code/tests differ from the freeze record; refusing to run')
    for p in (RES / 'r2_integrity.json', RES / 'r2_stage_results.json'):
        if p.exists():
            raise SystemExit(f'{p} exists; R2 executes once (write-once outputs)')
    t0 = time.time()
    ge = _imp(REPO / 'scripts/decision/h_eval/gate_e/gate_e.py', 'gate_e_r2')
    e5 = _imp(REPO / 'scripts/decision/h_eval/gate_e5/gate_e5.py', 'gate_e5_r2')
    sums = ge.sums_parse((RAW / 'metadata/SHA256SUMS.txt').read_bytes())
    inv = {r['recording']: r for r in csv.DictReader((EV / 'gate_e2/sleep_edf_origin_inventory.csv').open(newline=''))}
    verified = {}
    for name, cfg in POPS.items():                      # phase 2: ALL hashes / rosters / rows before any hypnogram
        verified[name] = verify_population(name, cfg, inv, e5)
    integ, results, private = dict(version=VERSION, freeze_record_sha256=digest(FREEZE)), {}, {}
    for name, cfg in POPS.items():
        content, recs, rep = verified[name]
        win, hyp = analyse(name, cfg, recs, ge, sums)
        y = np.array([w['y'] for w in win], float)
        person = np.array([w['person'] for w in win])
        P = {m: np.array([w['p_' + m] for w in win]) for m in [cfg['ref']] + list(cfg['cands'])}
        rep['envelope_reproduction'] = reproduce(name, cfg, content, y, P, person)
        rep['hypnograms'] = dict(n=len(hyp), all_sha256_match_official=True)
        rep['stage_categories'] = dict(Counter(w['stage'] for w in win))
        rep['n_windows_joined'] = len(win)
        integ[name] = rep
        results[name] = dict(population=name, original_decisions=decisions(name, content), **summarise(name, cfg, win))
        private[name] = win
    integ['execution_audit'] = dict(psg_files_opened=0, models_loaded=0, gate_commands_run=0,
                                    sensitivity_nights_read=False, runtime_s=round(time.time() - t0, 2))
    out = dict(version=VERSION, status='POST_HOC_EXPLORATORY_DESCRIPTIVE', protocol_sha256=fr['protocol_sha256'],
               bootstrap=dict(n=BOOT_N, seed=BOOT_SEED, min_persons_for_ci=MIN_CI_PERSONS), populations=results)
    LOC.mkdir(parents=True, exist_ok=True)
    for name, win in private.items():
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(win[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(win)
        write_new(LOC / f'r2_window_stage_join_{name}.csv', buf.getvalue().encode())
    write_new(RES / 'r2_integrity.json', (json.dumps(integ, indent=2, default=str) + '\n').encode())
    write_new(RES / 'r2_stage_results.json', (json.dumps(out, indent=2) + '\n').encode())
    print(json.dumps({k: v['composition'] for k, v in results.items()}, indent=1))


def decisions(name, content):
    if name == 'VAL':
        return dict(H1=content['decision']['outcome'])
    r = content['results']
    return dict(H1=r['H1']['outcome'], H2=r['H2']['outcome'], H3=r['H3'].get('status'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('freeze', 'execute'))
    a = ap.parse_args(argv)
    try:
        {'freeze': cmd_freeze, 'execute': cmd_execute}[a.cmd]()
    except Abort as exc:
        raise SystemExit('R2 ABORT (no results written): ' + str(exc))


if __name__ == '__main__':
    main()
