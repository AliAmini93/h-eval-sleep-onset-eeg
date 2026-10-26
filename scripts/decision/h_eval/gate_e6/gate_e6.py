#!/usr/bin/env python3
"""H-EVAL Gate E6: expanded SC_TRAIN-only leave-one-person-out EEG evaluation pilot.

Implements docs/decision/h_eval/gate_e6/gate_e6_protocol.md (frozen; SHA-256 checked at run time).
EXPLORATORY: SC_TRAIN only. Never reads SC_VALIDATION, SC_PROTECTED_MODEL_TEST, ST or Muse signals.
Reuses (read-only, hashed) the E4 v3 EDF reader / resumable downloader and the E5 landmark and
spectral-feature code. Logistic estimator re-implemented in NumPy and checked against E5 at run time.
No GPU, no package installation, no git operations, no overwrites.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION = 'H-EVAL-E6-expanded-TRAIN-LOPO-v1-2026-10-23'
PROTOCOL_REL = 'docs/decision/h_eval/gate_e6/gate_e6_protocol.md'
PROTOCOL_SHA256 = 'f6de7f0bcb50bb234f47f81d7d47358f124bc3585301c56cc4084a130f7f51c0'
SALT = 'H-EVAL-E6-SC-TRAIN-expansion-20261023-v1'
N_NEW = 12
BUDGET_BYTES = 850_000_000
EACH_CAP = 70_000_000
BOOT_N = 10_000
BOOT_SEED = 20261023
BINS = (300., 600., 1200., 1800., 3600., 7200.)
HORIZON = 300.
LOOKBACK = 1200.
E5_MODELS = ('CONSTANT', 'TIME_BIN', 'TIME_LOGIT', 'EEG_LOGIT', 'TIME_EEG_LOGIT')
MODELS = ('CONSTANT', 'TIME_BIN', 'TIME_LOGIT', 'TIME_CONT', 'EEG_LOGIT', 'TIME_EEG_LOGIT', 'TIME_CONT_EEG_LOGIT')
BASELINES = ('CONSTANT', 'TIME_BIN', 'TIME_LOGIT', 'TIME_CONT')
EEG_MODELS = ('TIME_CONT_EEG_LOGIT', 'EEG_LOGIT')
SUPPORTS = ('FULL', 'ANCHOR')
SPLIT_COUNTS = {'SC_TRAIN': 47, 'SC_VALIDATION': 15, 'SC_PROTECTED_MODEL_TEST': 15, 'ST_EXTERNAL_RESERVED': 17}
BASE_URL = 'https://physionet.org/files/sleep-edfx/1.0.0/'


class CriticalStop(RuntimeError):
    """Scientific or data-integrity blocker: the affected experiment must not run."""


# ------------------------------------------------------------------------------ utilities
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def import_py(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('Could not import ' + str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def immut(path, data: bytes):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise CriticalStop('Existing output differs; refusing overwrite: ' + str(path))
        return 'identical_existing'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)
    return 'written'


def to_csv(rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


def subject_rank_key(subject_key):
    return hashlib.sha256((SALT + '|' + subject_key).encode()).hexdigest()


# ------------------------------------------------------------------------ inputs and guards
def load_inputs(repo: Path):
    ev = repo / 'results/decision/h_eval'
    files = {
        'protocol': repo / PROTOCOL_REL,
        'e2_inventory': ev / 'gate_e2/sleep_edf_origin_inventory.csv',
        'e3_split': ev / 'gate_e3/gate_e3_sc_group_split.json',
        'e3_stage_diagnostics': ev / 'gate_e3/gate_e3_stage_diagnostics.csv',
        'e4_qc': ev / 'gate_e4/gate_e4_signal_qc.json',
        'e5_summary': ev / 'gate_e5/gate_e5_summary.json',
        'e4_code': repo / 'scripts/decision/h_eval/gate_e4/v3/gate_e4.py',
        'e5_code': repo / 'scripts/decision/h_eval/gate_e5/gate_e5.py',
        'official_sha256sums': repo / 'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt',
        'e6_code': Path(__file__).resolve(),
    }
    for k, p in files.items():
        if not p.is_file():
            raise CriticalStop(f'Missing upstream input {k}: {p}')
    hashes = {k: digest(p) for k, p in files.items()}
    if hashes['protocol'] != PROTOCOL_SHA256:
        raise CriticalStop('Protocol file differs from the frozen protocol hash')
    split_doc = json.loads(files['e3_split'].read_text())
    split = split_doc['by_subject']
    counts = defaultdict(int)
    for v in split.values():
        counts[v] += 1
    if dict(counts) != SPLIT_COUNTS:
        raise CriticalStop('E3 split counts changed: ' + str(dict(counts)))
    if split_doc['input_gate_e2_inventory_sha256'] != hashes['e2_inventory']:
        raise CriticalStop('E3 split no longer matches the E2 inventory')
    e4qc = json.loads(files['e4_qc'].read_text())
    if e4qc['provenance']['gate_e3_split_sha256'] != hashes['e3_split']:
        raise CriticalStop('E4 provenance does not match the frozen E3 split')
    with files['e2_inventory'].open(newline='') as f:
        inventory = {r['recording']: r for r in csv.DictReader(f)}
    with files['e3_stage_diagnostics'].open(newline='') as f:
        stage = {r['recording']: r['stage_at_lights_off'] for r in csv.DictReader(f)}
    if len(inventory) != 197:
        raise CriticalStop('E2 inventory size changed')
    sig = import_py(files['e4_code'], 'heval_e4_v3_readonly_for_e6')
    e5 = import_py(files['e5_code'], 'heval_e5_readonly_for_e6')
    sums = sig.manifest_hashes(files['official_sha256sums'])
    e5_summary = json.loads(files['e5_summary'].read_text())
    return dict(files=files, hashes=hashes, split=split, inventory=inventory, stage=stage,
                e4qc=e4qc, sig=sig, e5=e5, sums=sums, e5_summary=e5_summary)


def assert_train_signal(subject_key, psg_path, split):
    """Hard guard before any signal byte is downloaded or read."""
    if split.get(subject_key) != 'SC_TRAIN' or not psg_path.startswith('sleep-cassette/SC'):
        raise CriticalStop(f'PROTECTED SPLIT VIOLATION: refusing signal access {subject_key} {psg_path}')
    if psg_path[len('sleep-cassette/SC'):len('sleep-cassette/SC') + 3] != subject_key[2:5]:
        raise CriticalStop(f'PSG path does not belong to {subject_key}: {psg_path}')


def eligible(row, split, stage):
    return (split.get(row['subject_key']) == 'SC_TRAIN' and row['status'] == 'ELIGIBLE_ORIGIN_INDEPENDENT'
            and row['study'] == 'sleep-cassette' and stage.get(row['recording']) == 'W')


def select_subjects(inventory, split, stage, sums, retained_recordings):
    """Protocol §3. Uses only split, eligibility status, stage at LightsOff, night number, file size and the
    subject key hash: no latency, landmark count or label enters the selection."""
    retained = []
    for rid in retained_recordings:
        row = inventory[rid]
        if not eligible(row, split, stage):
            raise CriticalStop('Retained E5 recording no longer eligible: ' + rid)
        retained.append(row)
    retained_subjects = {r['subject_key'] for r in retained}
    if len(retained_subjects) != len(retained):
        raise CriticalStop('Retained recordings must be distinct subjects')
    by_subject = defaultdict(list)
    for row in inventory.values():
        if row['subject_key'] in retained_subjects or not eligible(row, split, stage):
            continue
        by_subject[row['subject_key']].append(row)
    candidates = []
    for s, rows in by_subject.items():
        first = sorted(rows, key=lambda r: (int(r['night']), r['recording']))[0]
        candidates.append(dict(subject_key=s, recording=first['recording'], night=int(first['night']),
                               psg_path=first['psg_path'], bytes=int(first['psg_size_bytes']),
                               hash_key=subject_rank_key(s)))
    candidates.sort(key=lambda c: c['hash_key'])
    selected, total, stop_reason = [], 0, 'reached_n_new'
    for rank, c in enumerate(candidates, start=1):
        c['hash_rank'] = rank
        if len(selected) == N_NEW:
            continue
        if stop_reason != 'reached_n_new':
            continue
        if c['bytes'] > EACH_CAP or total + c['bytes'] > BUDGET_BYTES:
            stop_reason = 'budget_prefix_stop_at_rank_%d' % rank
            continue
        selected.append(c)
        total += c['bytes']
    for c in selected:
        assert_train_signal(c['subject_key'], c['psg_path'], split)
        if c['psg_path'] not in sums:
            raise CriticalStop('Selected PSG absent from official checksum manifest: ' + c['psg_path'])
        c['source_sha256'] = sums[c['psg_path']]
    return retained, selected, candidates, total, stop_reason


# ------------------------------------------------------------------------ download / verify
def acquire(repo, records, sig, offline, log):
    """Reuse or download each PSG, verify official SHA-256. Network failure stops further downloads
    (verified prefix kept); integrity failure is a critical stop."""
    root = repo / 'data/raw/sleep_edfx_v1_0_0'
    out, network_stop = [], None
    for rec in records:
        target = root / rec['psg_path']
        if network_stop is not None:
            rec.update(status='NOT_ATTEMPTED_AFTER_NETWORK_STOP')
            out.append(rec)
            continue
        existed = target.exists()
        try:
            if offline:
                if not existed:
                    raise RuntimeError('offline mode and file absent')
                state = 'reused_existing'
            else:
                state = sig.resume_download(BASE_URL + rec['psg_path'], target, rec['source_sha256'], rec['bytes'])
        except RuntimeError as exc:                     # bounded network failure; partial preserved
            network_stop = repr(exc)[:400]
            rec.update(status='NETWORK_FAILURE_PARTIAL_PRESERVED', error=network_stop)
            log(f'[E6] network stop at {rec["recording"]}: {network_stop}')
            out.append(rec)
            continue
        except ValueError as exc:                       # checksum / layout / range violation
            raise CriticalStop(f'Integrity failure for {rec["psg_path"]}: {exc}') from exc
        size, h = target.stat().st_size, digest(target)
        if size != rec['bytes'] or h != rec['source_sha256']:
            raise CriticalStop('Post-acquisition SHA/size verification failed: ' + rec['psg_path'])
        rec.update(status='VERIFIED', acquisition=('reused_verified' if existed else state),
                   computed_sha256=h, verified_bytes=size)
        out.append(rec)
    return out, network_stop


# ------------------------------------------------------------------------ landmarks and features
def extract(repo, records, inventory, split, sig, e5, log):
    rows, diags = [], []
    root = repo / 'data/raw/sleep_edfx_v1_0_0'
    for rec in records:
        assert_train_signal(rec['subject_key'], rec['psg_path'], split)
        inv = inventory[rec['recording']]
        off, n2 = float(inv['lights_off_rel_s']), float(inv['first_n2_after_off_s'])
        try:
            landmarks = e5.make_landmarks(off, n2, inv)       # checks counts against E2 exactly
        except ValueError as exc:
            raise CriticalStop('Landmark reconstruction mismatch: ' + str(exc)) from exc
        path = root / rec['psg_path']
        bad, maxclip = 0, 0.0
        with path.open('rb') as f:
            try:
                h = sig.edf_header(f, path.stat().st_size)
            except ValueError as exc:
                raise CriticalStop(f'EDF layout/channel failure {rec["recording"]}: {exc}') from exc
            if n2 > h['duration_s'] or min(landmarks) - 30 < 0:
                raise CriticalStop('Landmark outside EEG signal: ' + rec['recording'])
            for t in landmarks:
                x = []
                for ch in h['eeg_channels']:
                    samples, digital = sig.read_window(f, h, ch, t - 30, t)   # strictly [t-30, t)
                    clip = sum(v in (ch[1]['dig_min'], ch[1]['dig_max']) for v in digital) / len(digital)
                    maxclip = max(maxclip, clip)
                    bad += clip > .05
                    x.extend(e5.channel_features(samples))
                rows.append(dict(recording=rec['recording'], subject_key=rec['subject_key'], origin=rec['origin'],
                                 elapsed_s=t - off, landmark_end_s=t, time_to_n2_s=n2 - t,
                                 anchor=bool(n2 - t <= LOOKBACK + 1e-8), y=int(n2 - t <= HORIZON + 1e-8),
                                 features=x))
        diags.append(dict(recording=rec['recording'], subject_key=rec['subject_key'], origin=rec['origin'],
                          full_windows=len(landmarks), anchor_windows=sum(n2 - t <= LOOKBACK + 1e-8 for t in landmarks),
                          positive_windows=sum(n2 - t <= HORIZON + 1e-8 for t in landmarks),
                          high_clip_channel_windows=int(bad), largest_clip_fraction=round(maxclip, 6)))
        log(f'[E6] features {rec["recording"]}: {len(landmarks)} landmarks')
    return rows, diags


# ------------------------------------------------------------------------ models (NumPy, E5-equivalent)
def bin_idx(t):
    return int(sum(t >= b for b in BINS))


def time_onehot(t):
    v = np.zeros(7)
    v[bin_idx(t)] = 1.
    return v


def time_cont_raw(t):
    return np.array([math.log1p(t / 60.), t / 3600.])


def zfit(X):
    mu = X.mean(axis=0)
    sd = np.sqrt(((X - mu) ** 2).mean(axis=0))
    return mu, np.maximum(sd, 1e-6)


def zapply(X, mu, sd):
    return np.clip((X - mu) / sd, -5., 5.)


def sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1. / (1. + np.exp(-np.minimum(60., x[pos])))
    z = np.exp(np.maximum(-60., x[~pos]))
    out[~pos] = z / (1. + z)
    return out


def logistic_fit_np(X, y, steps=350, penalty=.025, lr=.3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) == 0 or len(X) != len(y):
        raise ValueError('Logistic feature matrix invalid')
    n, d = X.shape
    p = (y.sum() + 1) / (n + 2)
    w = np.zeros(d + 1)
    w[0] = math.log(p / (1 - p))
    for step in range(steps):
        err = sigmoid_np(w[0] + X @ w[1:]) - y
        g0, g = err.sum(), X.T @ err
        eta = lr / (1. + step / 200.)
        w[0] -= eta * g0 / n
        w[1:] -= eta * (g / n + penalty * w[1:])
    return w


def logistic_predict_np(w, X):
    return np.clip(sigmoid_np(w[0] + np.asarray(X, dtype=float) @ w[1:]), 1e-6, 1 - 1e-6)


def design(kind, rows, ztrain, ctrain):
    """Feature matrix for one logistic model; EEG (z) and continuous time (c) already standardised."""
    oh = np.array([time_onehot(r['elapsed_s']) for r in rows])
    return {'TIME_LOGIT': oh, 'TIME_CONT': ctrain, 'EEG_LOGIT': ztrain,
            'TIME_EEG_LOGIT': np.hstack([oh, ztrain]),
            'TIME_CONT_EEG_LOGIT': np.hstack([ctrain, ztrain])}[kind]


def lopo_predictions(rows, models=MODELS):
    """Out-of-person predictions for every window, per training support and model.
    pred[train_support][model] is aligned to rows. Also returns the training-person sets for audit."""
    persons = sorted({r['subject_key'] for r in rows})
    if len(persons) < 3:
        raise ValueError('Too few persons for LOPO')
    pred = {ts: {m: np.full(len(rows), np.nan) for m in models} for ts in SUPPORTS}
    fold_audit = []
    feats = np.array([r['features'] for r in rows], dtype=float)
    traw = np.array([time_cont_raw(r['elapsed_s']) for r in rows])
    yall = np.array([r['y'] for r in rows], dtype=float)
    subj = np.array([r['subject_key'] for r in rows])
    anchor = np.array([r['anchor'] for r in rows])
    for held in persons:
        hold = np.where(subj == held)[0]
        for ts in SUPPORTS:
            tr = np.where((subj != held) & ((ts == 'FULL') | anchor))[0]
            tr_persons = sorted(set(subj[tr]))
            if tr_persons != [p for p in persons if p != held]:
                raise ValueError('Incomplete LOPO training persons')
            fold_audit.append(dict(held=held, train_support=ts, train_persons=tr_persons, n_train=int(len(tr))))
            ytr = yall[tr]
            prior = (ytr.sum() + 1) / (len(tr) + 2)
            buckets = defaultdict(lambda: [0., 0.])
            for i in tr:
                b = buckets[bin_idx(rows[i]['elapsed_s'])]
                b[0] += yall[i]
                b[1] += 1
            tsmooth = {k: (b[0] + 5 * prior) / (b[1] + 5) for k, b in buckets.items()}
            mu, sd = zfit(feats[tr])
            cmu, csd = zfit(traw[tr])
            ztr, zho = zapply(feats[tr], mu, sd), zapply(feats[hold], mu, sd)
            ctr, cho = zapply(traw[tr], cmu, csd), zapply(traw[hold], cmu, csd)
            rtr = [rows[i] for i in tr]
            rho = [rows[i] for i in hold]
            for m in models:
                if m == 'CONSTANT':
                    p = np.full(len(hold), prior)
                elif m == 'TIME_BIN':
                    p = np.array([tsmooth.get(bin_idx(r['elapsed_s']), prior) for r in rho])
                else:
                    w = logistic_fit_np(design(m, rtr, ztr, ctr), ytr)
                    p = logistic_predict_np(w, design(m, rho, zho, cho))
                pred[ts][m][hold] = p
    return pred, fold_audit


def window_scores(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log1p(-p))


def person_table(rows, pred, models=MODELS):
    """Per person x cell x model scores (cell = train support x evaluation support)."""
    y = np.array([r['y'] for r in rows], dtype=float)
    subj = np.array([r['subject_key'] for r in rows])
    anchor = np.array([r['anchor'] for r in rows])
    origin = {r['subject_key']: r['origin'] for r in rows}
    recording = {r['subject_key']: r['recording'] for r in rows}
    out = []
    for ts in SUPPORTS:
        for es in SUPPORTS:
            mask = np.ones(len(rows), bool) if es == 'FULL' else anchor
            for m in models:
                b, ll = window_scores(pred[ts][m], y)
                for s in sorted(set(subj)):
                    k = mask & (subj == s)
                    if not k.any():
                        raise ValueError(f'Empty evaluation support for {s} in {es}')
                    out.append(dict(person=s, origin=origin[s], recording=recording[s], train_support=ts,
                                    eval_support=es, model=m, n_windows=int(k.sum()), n_positive=int(y[k].sum()),
                                    brier=float(b[k].mean()), logloss=float(ll[k].mean()),
                                    brier_sum=float(b[k].sum()), logloss_sum=float(ll[k].sum())))
    return out


def aggregate(ptab, models=MODELS):
    res = []
    for ts in SUPPORTS:
        for es in SUPPORTS:
            for m in models:
                rr = [r for r in ptab if r['train_support'] == ts and r['eval_support'] == es and r['model'] == m]
                nw = sum(r['n_windows'] for r in rr)
                res.append(dict(train_support=ts, eval_support=es, model=m, n_subjects=len(rr), total_eval_windows=nw,
                                brier_subject_equal=float(np.mean([r['brier'] for r in rr])),
                                logloss_subject_equal=float(np.mean([r['logloss'] for r in rr])),
                                brier_window_weighted=sum(r['brier_sum'] for r in rr) / nw,
                                logloss_window_weighted=sum(r['logloss_sum'] for r in rr) / nw))
    return res


def reproduce_e5(rows, e5_summary, tol=1e-9):
    """Integration check: the E6 estimator, on the E5 six recordings with the E5 model set, must reproduce
    every E5 aggregate metric."""
    six = [r for r in rows if r['origin'] == 'E5_RETAINED']
    pred, _ = lopo_predictions(six, models=E5_MODELS)
    agg = aggregate(person_table(six, pred, models=E5_MODELS), models=E5_MODELS)
    mine = {(a['train_support'], a['eval_support'], a['model']): a for a in agg}
    worst = 0.0
    checked = 0
    for ref in e5_summary['metrics']:
        a = mine[(ref['train_support'], ref['eval_support'], ref['model'])]
        for k in ('brier_subject_equal', 'logloss_subject_equal', 'brier_window_weighted'):
            worst = max(worst, abs(a[k] - ref[k]))
            checked += 1
    return dict(passed=bool(worst <= tol and checked == 60), n_values_checked=checked, max_abs_diff=worst, tolerance=tol,
                n_landmarks=len(six), e5_n_landmarks=e5_summary['n_landmarks'])


# ------------------------------------------------------------------------ bootstrap and decisions
class Bootstrap:
    """Subject-cluster bootstrap; one draw matrix per person set, shared by all contrasts on that set."""

    def __init__(self, persons, n=BOOT_N, seed=BOOT_SEED):
        self.persons = sorted(persons)
        self.I = np.random.default_rng(seed).integers(0, len(self.persons), size=(n, len(self.persons)))

    def contrast(self, ptab, ts, es, ref, cand, metric='brier'):
        def get(m):
            d = {r['person']: r for r in ptab if r['train_support'] == ts and r['eval_support'] == es and r['model'] == m}
            return [d[p] for p in self.persons]
        a, b = get(ref), get(cand)
        dmean = np.array([x[metric] - y[metric] for x, y in zip(a, b)])
        dsum = np.array([x[metric + '_sum'] - y[metric + '_sum'] for x, y in zip(a, b)])
        nwin = np.array([x['n_windows'] for x in a], dtype=float)
        se = dmean[self.I].mean(axis=1)
        ww = dsum[self.I].sum(axis=1) / nwin[self.I].sum(axis=1)
        loo = [float(np.delete(dmean, i).mean()) for i in range(len(dmean))]
        point = float(dmean.mean())
        return dict(train_support=ts, eval_support=es, reference=ref, candidate=cand, metric=metric,
                    n_persons=len(self.persons),
                    subject_equal=dict(delta=point, ci95=[float(x) for x in np.percentile(se, [2.5, 97.5])]),
                    window_weighted=dict(delta=float(dsum.sum() / nwin.sum()),
                                         ci95=[float(x) for x in np.percentile(ww, [2.5, 97.5])]),
                    persons_favouring_candidate=int((dmean > 0).sum()), persons_favouring_reference=int((dmean < 0).sum()),
                    leave_one_person_out=dict(min=min(loo), max=max(loo),
                                              sign_flips=int(sum(np.sign(x) != np.sign(point) for x in loo))),
                    per_person_delta={p: float(v) for p, v in zip(self.persons, dmean)})


def classify(primary, primary_ll, vs_baselines, hash12_delta):
    """Protocol §9 for one cell. primary/vs_baselines: Brier contrasts (subject-equal)."""
    lo, hi = primary['subject_equal']['ci95']
    beats_all = all(c['subject_equal']['ci95'][0] > 0 for c in vs_baselines)
    worse_than_some = any(c['subject_equal']['ci95'][1] < 0 for c in vs_baselines)
    conditions = dict(primary_ci_lower_gt_0=bool(lo > 0), beats_all_no_eeg_baselines_ci=bool(beats_all),
                      logloss_primary_point_gt_0=bool(primary_ll['subject_equal']['delta'] > 0),
                      hash12_primary_point_gt_0=bool(hash12_delta is not None and hash12_delta > 0),
                      no_loo_sign_flip=bool(primary['leave_one_person_out']['sign_flips'] == 0))
    if all(conditions.values()):
        label = 'EEG_INCREMENT_SUPPORTED_EXPLORATORY'
    elif hi < 0 or worse_than_some:
        label = 'EEG_INCREMENT_NOT_SUPPORTED'
    else:
        label = 'INCONCLUSIVE'
    return dict(label=label, conditions=conditions, worse_than_some_baseline_ci=bool(worse_than_some))


def kendall_tau(order_a, order_b):
    pos_b = {m: i for i, m in enumerate(order_b)}
    seq = [pos_b[m] for m in order_a]
    n = len(seq)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            conc += seq[i] < seq[j]
            disc += seq[i] > seq[j]
    return (conc - disc) / (n * (n - 1) / 2)


def analyse(rows, ptab, agg):
    persons = sorted({r['subject_key'] for r in rows})
    hash12 = sorted({r['subject_key'] for r in rows if r['origin'] == 'E6_HASH'})
    six = sorted({r['subject_key'] for r in rows if r['origin'] == 'E5_RETAINED'})
    B = Bootstrap(persons)
    contrasts, cells = [], {}
    for ts in SUPPORTS:
        for es in SUPPORTS:
            cell = f'train_{ts}__eval_{es}'
            prim = B.contrast(ptab, ts, es, 'TIME_CONT', 'TIME_CONT_EEG_LOGIT', 'brier')
            prim_ll = B.contrast(ptab, ts, es, 'TIME_CONT', 'TIME_CONT_EEG_LOGIT', 'logloss')
            base = {cand: [B.contrast(ptab, ts, es, b, cand, met) for b in BASELINES]
                    for cand in EEG_MODELS for met in ('brier',)}
            base_ll = {cand: [B.contrast(ptab, ts, es, b, cand, 'logloss') for b in BASELINES] for cand in EEG_MODELS}
            sub = {}
            for name, ps in (('hash12', hash12), ('e5_six', six)):
                if len(ps) >= 3:
                    sub[name] = Bootstrap(ps).contrast([r for r in ptab if r['person'] in ps], ts, es,
                                                       'TIME_CONT', 'TIME_CONT_EEG_LOGIT', 'brier')
            ranking = sorted([a for a in agg if a['train_support'] == ts and a['eval_support'] == es],
                             key=lambda a: a['brier_subject_equal'])
            cls = classify(prim, prim_ll, base['TIME_CONT_EEG_LOGIT'],
                           sub['hash12']['subject_equal']['delta'] if 'hash12' in sub else None)
            cells[cell] = dict(primary_brier=prim, primary_logloss=prim_ll, subsets=sub,
                               ranking_subject_equal_brier=[a['model'] for a in ranking],
                               best_model=ranking[0]['model'],
                               strongest_no_eeg_baseline=next(a['model'] for a in ranking if a['model'] in BASELINES),
                               q1_classification=cls)
            contrasts += [prim, prim_ll] + base['TIME_CONT_EEG_LOGIT'] + base['EEG_LOGIT'] + \
                base_ll['TIME_CONT_EEG_LOGIT'] + base_ll['EEG_LOGIT']
    q2 = {}
    sensitive = False
    for es in SUPPORTS:
        a, b = cells[f'train_FULL__eval_{es}'], cells[f'train_ANCHOR__eval_{es}']
        diff_cls = a['q1_classification']['label'] != b['q1_classification']['label']
        diff_best = a['best_model'] != b['best_model']
        sensitive |= diff_cls or diff_best
        q2[f'eval_{es}'] = dict(q1_class_train_FULL=a['q1_classification']['label'],
                                q1_class_train_ANCHOR=b['q1_classification']['label'],
                                best_train_FULL=a['best_model'], best_train_ANCHOR=b['best_model'],
                                kendall_tau_rankings=kendall_tau(a['ranking_subject_equal_brier'], b['ranking_subject_equal_brier']),
                                classification_differs=bool(diff_cls), best_model_differs=bool(diff_best))
    return cells, contrasts, q2, ('SENSITIVE_TO_TRAINING_SUPPORT' if sensitive else 'NOT_SENSITIVE'), B


def prevalence(rows, pred):
    y = np.array([r['y'] for r in rows], dtype=float)
    anchor = np.array([r['anchor'] for r in rows])
    return dict(eval_prevalence=dict(FULL=float(y.mean()), ANCHOR=float(y[anchor].mean())),
                pooled_training_prevalence=dict(FULL=float(y.mean()), ANCHOR=float(y[anchor].mean())),
                constant_prediction_mean=dict(FULL=float(np.mean(pred['FULL']['CONSTANT'])),
                                              ANCHOR=float(np.mean(pred['ANCHOR']['CONSTANT']))),
                note=('Anchoring keeps only the last 1200 s before Stage 2, so the positive rate rises; training on ANCHOR '
                      'shifts every model base rate upward. Brier values are not comparable across EVALUATION supports.'))


# ------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--plan-only', action='store_true')
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    if not (repo / '.git').is_dir():
        raise CriticalStop('Must run inside the existing repository')
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    X = load_inputs(repo)
    retained_ids = [r['recording'] for r in X['e4qc']['sampled_train_recordings']]
    retained, selected, candidates, planned_bytes, stop_reason = select_subjects(
        X['inventory'], X['split'], X['stage'], X['sums'], retained_ids)
    plan = dict(salt=SALT, n_candidates=len(candidates), n_selected=len(selected), planned_new_bytes=planned_bytes,
                budget_bytes=BUDGET_BYTES, stop_reason=stop_reason,
                candidates=[{k: c[k] for k in ('hash_rank', 'subject_key', 'recording', 'night', 'bytes', 'hash_key')}
                            | {'selected': c in selected} for c in candidates])
    log(json.dumps({'selected': [(c['hash_rank'], c['subject_key'], c['recording'], c['bytes']) for c in selected],
                    'planned_new_bytes': planned_bytes, 'stop_reason': stop_reason}))
    if a.plan_only:
        return plan

    records = [dict(subject_key=r['subject_key'], recording=r['recording'], night=int(r['night']), psg_path=r['psg_path'],
                    bytes=int(r['psg_size_bytes']), source_sha256=X['sums'][r['psg_path']], origin='E5_RETAINED')
               for r in retained]
    records += [dict(subject_key=c['subject_key'], recording=c['recording'], night=c['night'], psg_path=c['psg_path'],
                     bytes=c['bytes'], source_sha256=c['source_sha256'], origin='E6_HASH', hash_rank=c['hash_rank'])
                for c in selected]
    for r in records:
        assert_train_signal(r['subject_key'], r['psg_path'], X['split'])
    acquired, network_stop = acquire(repo, records, X['sig'], a.offline, log)
    usable = [r for r in acquired if r['status'] == 'VERIFIED']
    deviations = []
    if network_stop:
        deviations.append('Network failure: analysis restricted to retained six plus verified hash-order prefix (protocol §4)')
    rows, diags = extract(repo, usable, X['inventory'], X['split'], X['sig'], X['e5'], log)

    repro = reproduce_e5(rows, X['e5_summary'])
    log('[E6] E5 reproduction: ' + json.dumps(repro))
    if not repro['passed']:
        raise CriticalStop('E5 reproduction check failed: ' + json.dumps(repro))

    pred, fold_audit = lopo_predictions(rows)
    ptab = person_table(rows, pred)
    agg = aggregate(ptab)
    cells, contrasts, q2, q2_label, B = analyse(rows, ptab, agg)
    prim_cell = cells['train_FULL__eval_FULL']
    persons = sorted({r['subject_key'] for r in rows})

    summary = dict(
        version=VERSION, status='SC_TRAIN_ONLY_EXPLORATORY_NOT_CONFIRMATORY', protocol=PROTOCOL_REL,
        protocol_sha256=PROTOCOL_SHA256,
        n_persons=len(persons), n_recordings=len({r['recording'] for r in rows}), n_landmarks=len(rows),
        n_anchored_landmarks=int(sum(r['anchor'] for r in rows)), n_positive_h300=int(sum(r['y'] for r in rows)),
        persons_by_origin=dict(E5_RETAINED=sorted({r['subject_key'] for r in rows if r['origin'] == 'E5_RETAINED'}),
                               E6_HASH=sorted({r['subject_key'] for r in rows if r['origin'] == 'E6_HASH'})),
        e5_reproduction_check=repro,
        q1=dict(question='EEG information beyond elapsed time (primary cell train FULL / eval FULL, subject-equal Brier)',
                primary_contrast='TIME_CONT minus TIME_CONT_EEG_LOGIT (positive favours EEG)',
                primary_delta=prim_cell['primary_brier']['subject_equal'],
                primary_delta_window_weighted=prim_cell['primary_brier']['window_weighted'],
                primary_delta_logloss=prim_cell['primary_logloss']['subject_equal'],
                strongest_no_eeg_baseline=prim_cell['strongest_no_eeg_baseline'],
                classification=prim_cell['q1_classification']),
        q2=dict(question='Sensitivity of the Q1 conclusion and model ranking to TRAINING support (evaluation support fixed)',
                classification=q2_label, by_evaluation_support=q2),
        cells={k: dict(q1_classification=v['q1_classification'], best_model=v['best_model'],
                       strongest_no_eeg_baseline=v['strongest_no_eeg_baseline'],
                       ranking_subject_equal_brier=v['ranking_subject_equal_brier'],
                       primary_delta_subject_equal=v['primary_brier']['subject_equal'],
                       primary_delta_logloss=v['primary_logloss']['subject_equal'],
                       subsets={n: s['subject_equal'] for n, s in v['subsets'].items()},
                       leave_one_person_out=v['primary_brier']['leave_one_person_out']) for k, v in cells.items()},
        metrics=agg, prevalence=prevalence(rows, pred), qc=diags, deviations=deviations,
        runtime_s=round(time.time() - t0, 1),
        limitations=[
            'SC_TRAIN development data only; leave-one-person-out, not a validation or protected test',
            'Six retained E5 persons were selected by observed sleep latency (outcome-dependent); see hash12 subset',
            'n persons is small; cluster-bootstrap CIs over persons are approximate',
            'Retrospective all-event cohort: every night reaches Stage 2; no censoring/no-event nights',
            'First Stage 2 after metadata LightsOff is an R&K stage-label target, not clinical sleep onset',
            'Fixed, untuned models and features; results are not an upper bound on EEG information',
            'Evaluation-support comparisons (FULL vs ANCHOR) are different estimands with different prevalence',
            'No SC_VALIDATION, SC_PROTECTED_MODEL_TEST, ST or Muse data used'])
    boot = dict(version=VERSION, method='subject-cluster percentile bootstrap', n_boot=BOOT_N, seed=BOOT_SEED,
                persons=persons, draw_matrix_sha256=hashlib.sha256(B.I.tobytes()).hexdigest(),
                delta_definition='Brier/logloss(reference) - Brier/logloss(candidate); positive favours candidate',
                contrasts=contrasts,
                subsets={c: {n: s for n, s in v['subsets'].items()} for c, v in cells.items()})
    a.out.mkdir(parents=True, exist_ok=True)
    blobs = {'gate_e6_summary.json': (json.dumps(summary, indent=2) + '\n').encode(),
             'gate_e6_person_scores.csv': to_csv([{k: v for k, v in r.items()} for r in ptab]),
             'gate_e6_cluster_bootstrap.json': (json.dumps(boot, indent=2) + '\n').encode()}
    stage = repo / '.local/h_eval/gate_e6'
    win_rows = [dict(recording=r['recording'], person=r['subject_key'], origin=r['origin'], elapsed_s=r['elapsed_s'],
                     time_to_n2_s=r['time_to_n2_s'], anchor=int(r['anchor']), y=r['y'],
                     **{f'p_{ts}_{m}': float(pred[ts][m][i]) for ts in SUPPORTS for m in MODELS})
                for i, r in enumerate(rows)]
    local_blobs = {'gate_e6_window_predictions.csv': to_csv(win_rows),
                   'gate_e6_run_log.txt': ('\n'.join(log_lines) + '\n').encode(),
                   'gate_e6_fold_audit.json': (json.dumps(fold_audit, indent=1) + '\n').encode()}
    for n, b in blobs.items():
        f = a.out / n
        if f.exists() and f.read_bytes() != b:
            raise CriticalStop('Output conflict, refusing overwrite: ' + str(f))
    written = {n: immut(a.out / n, b) for n, b in blobs.items()}
    for n, b in local_blobs.items():
        immut(stage / n, b)
    prov = dict(version=VERSION, protocol_sha256=PROTOCOL_SHA256, input_sha256=X['hashes'], selection=plan,
                recordings=[{k: v for k, v in r.items()} for r in acquired], network_stop=network_stop,
                outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
                local_outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in local_blobs.items()},
                raw_data_location='data/raw/sleep_edfx_v1_0_0/ (git-ignored)', guards=dict(
                    signals_read_only_from='SC_TRAIN', hypnograms_read=False, gpu=False, git_operations=False,
                    package_installation=False))
    immut(a.out / 'gate_e6_data_provenance.json', (json.dumps(prov, indent=2) + '\n').encode())
    print(json.dumps(dict(status=summary['status'], n_persons=len(persons), n_landmarks=len(rows),
                          q1=summary['q1']['classification']['label'], q2=q2_label,
                          primary_delta=summary['q1']['primary_delta'], outputs=written, runtime_s=summary['runtime_s']),
                     indent=2), flush=True)
    return summary


if __name__ == '__main__':
    main()
