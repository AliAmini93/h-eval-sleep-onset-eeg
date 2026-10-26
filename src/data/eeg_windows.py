#!/usr/bin/env python3
"""CTH v1 data builder (SC_TRAIN only): cohort selection, SHA-verified acquisition, causal feature extraction.

Writes .local/h_eval/cth_v1/dataset_v1.npz (git-ignored; EEG-derived arrays) and
results/decision/h_eval/cth_v1/cth_data_manifest_v1.json (counts, hashes). Never touches SC_VALIDATION,
SC_PROTECTED_MODEL_TEST, ST or Muse signals.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MIRROR = 'https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/'
FALLBACK = 'https://physionet.org/files/sleep-edfx/1.0.0/'
FS = 100
WIN = 3000
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 15), 'beta': (15, 30)}
RICH_NAMES = [f'{ch}_{n}' for ch in ('fpz', 'pz') for n in
              ('logtot', 'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta', 'log_alpha_theta',
               'log_delta_theta', 'log_rms')]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def select_cohort(X):
    """All eligible (E2 eligible + W at LightsOff) nights of all SC_TRAIN persons. Outcome-independent."""
    by = defaultdict(list)
    for r in X['inventory'].values():
        if (X['split'].get(r['subject_key']) == 'SC_TRAIN' and r['status'] == 'ELIGIBLE_ORIGIN_INDEPENDENT'
                and r['study'] == 'sleep-cassette' and X['stage'].get(r['recording']) == 'W'):
            by[r['subject_key']].append(r)
    recs = []
    for s in sorted(by):
        for r in sorted(by[s], key=lambda r: (int(r['night']), r['recording'])):
            recs.append(dict(subject_key=s, recording=r['recording'], night=int(r['night']), psg_path=r['psg_path'],
                             bytes=int(r['psg_size_bytes']), official_sha256=X['sums'][r['psg_path']]))
    return recs


def guard(rec, split):
    if split.get(rec['subject_key']) != 'SC_TRAIN' or rec['psg_path'][17:20] != rec['subject_key'][2:5]:
        raise RuntimeError('DATA BOUNDARY: refusing non-SC_TRAIN signal ' + rec['psg_path'])


def acquire(recs, X, root):
    log = []
    for r in recs:
        guard(r, X['split'])
        target = root / r['psg_path']
        st = None
        for b in (MIRROR, FALLBACK):
            try:
                st = X['sig'].resume_download(b + r['psg_path'], target, r['official_sha256'], r['bytes'])
                break
            except RuntimeError as exc:
                st = 'NETWORK_INCOMPLETE ' + repr(exc)[:150]
        if target.is_file() is False or digest(target) != r['official_sha256']:
            raise RuntimeError(f'Acquisition incomplete or invalid for {r["psg_path"]}: {st}')
        log.append(dict(recording=r['recording'], state=st))
    return log


def rich_features(x):
    """x: (WIN,) µV for one channel, the 30 s window ending at the decision time. 9 features."""
    w = np.hanning(400)  # symmetric Hann, as E5 (0.5-0.5cos(2pi i/399))
    freqs = np.fft.rfftfreq(512, 1 / FS)
    P = np.zeros(len(freqs))
    for s in (0, 600, 1200, 1800, 2400):
        fr = x[s:s + 400]
        fr = (fr - fr.mean()) * w
        P += np.abs(np.fft.rfft(fr, 512)) ** 2
    bp = {k: P[(freqs >= lo) & (freqs < hi)].sum() for k, (lo, hi) in BANDS.items()}
    tot = sum(bp.values()) + 1e-12
    rms = np.sqrt(((x - x.mean()) ** 2).mean())
    return [np.log(tot), *(bp[k] / tot for k in BANDS), np.log((bp['alpha'] + 1e-9) / (bp['theta'] + 1e-9)),
            np.log((bp['delta'] + 1e-9) / (bp['theta'] + 1e-9)), np.log1p(rms)]


def extract(recs, X, root, e5):
    out = defaultdict(list)
    for ri, r in enumerate(recs):
        guard(r, X['split'])
        inv = X['inventory'][r['recording']]
        off, n2 = float(inv['lights_off_rel_s']), float(inv['first_n2_after_off_s'])
        landmarks = e5.make_landmarks(off, n2, inv)         # verifies E2 counts exactly
        path = root / r['psg_path']
        with path.open('rb') as f:
            h = X['sig'].edf_header(f, path.stat().st_size)
            if n2 > h['duration_s']:
                raise RuntimeError('Annotation beyond signal ' + r['recording'])
            for t in landmarks:
                chans = [np.array(X['sig'].read_window(f, h, ch, t - 30, t)[0]) for ch in h['eeg_channels']]
                if any(len(c) != WIN or not np.isfinite(c).all() for c in chans):
                    raise RuntimeError('Bad window ' + r['recording'])
                out['raw'].append(np.stack(chans).astype(np.float32))
                out['e5'].append([v for c in chans for v in e5.channel_features(c.tolist())])
                out['rich'].append([v for c in chans for v in rich_features(c)])
                tau = n2 - t
                out['tau'].append(tau)
                out['y'].append(int(tau <= 300 + 1e-8))
                out['elapsed'].append(t - off)
                out['rec_idx'].append(ri)
                out['person'].append(r['subject_key'])
                out['recording'].append(r['recording'])
                out['night'].append(r['night'])
        print(f'[CTH] {r["recording"]}: {len(landmarks)} landmarks', flush=True)
    arr = dict(raw=np.stack(out['raw']), e5=np.array(out['e5']), rich=np.array(out['rich']), tau=np.array(out['tau']),
               y=np.array(out['y']), elapsed=np.array(out['elapsed']), rec_idx=np.array(out['rec_idx']),
               person=np.array(out['person']), recording=np.array(out['recording']), night=np.array(out['night']))
    # history index: previous landmarks of the same recording (consecutive 30 s grid), -1 = masked
    hist = np.full((len(arr['y']), 10), -1, dtype=np.int64)
    for i in range(len(arr['y'])):
        for k in range(10):
            j = i - k
            if j >= 0 and arr['rec_idx'][j] == arr['rec_idx'][i] and abs(arr['elapsed'][i] - arr['elapsed'][j] - 30 * k) < 1e-6:
                hist[i, 9 - k] = j          # chronological: column 9 = current window
            else:
                break
    arr['hist'] = hist
    return arr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-download', action='store_true')
    a = ap.parse_args(argv)
    e6 = _imp(REPO / 'scripts/decision/h_eval/gate_e6/gate_e6.py', 'e6_for_cth')
    X = e6.load_inputs(REPO)
    root = REPO / 'data/raw/sleep_edfx_v1_0_0'
    recs = select_cohort(X)
    t0 = time.time()
    log = [] if a.no_download else acquire(recs, X, root)
    for r in recs:
        if digest(root / r['psg_path']) != r['official_sha256']:
            raise RuntimeError('SHA mismatch ' + r['psg_path'])
    arr = extract(recs, X, root, X['e5'])
    loc = REPO / '.local/h_eval/cth_v1'
    loc.mkdir(parents=True, exist_ok=True)
    out = loc / 'dataset_v1.npz'
    if out.exists():
        raise RuntimeError('dataset exists; refusing overwrite')
    np.savez(out, **arr)
    man = dict(version='H-EVAL-CTH-v1-data', protocol_sha256=digest(REPO / 'docs/decision/h_eval/cth_v1/cth_v1_protocol.md'),
               n_persons=len({r['subject_key'] for r in recs}), n_recordings=len(recs), n_windows=int(len(arr['y'])),
               n_positive=int(arr['y'].sum()), n_windows_first_night=int(sum(n == min(rr['night'] for rr in recs if rr['subject_key'] == p)
                                                                       for n, p in zip(arr['night'], arr['person']))),
               recordings=[dict(r, sha256_verified=True) for r in recs], acquisition_log=log,
               rich_feature_names=RICH_NAMES, dataset_file='.local/h_eval/cth_v1/dataset_v1.npz',
               dataset_sha256=digest(out), e5_code_sha256=digest(REPO / 'scripts/decision/h_eval/gate_e5/gate_e5.py'),
               e4_code_sha256=digest(REPO / 'scripts/decision/h_eval/gate_e4/v3/gate_e4.py'),
               builder_sha256=digest(Path(__file__)), runtime_s=round(time.time() - t0, 1),
               boundary=dict(splits_read=['SC_TRAIN'], validation_test_st_muse_signals=False))
    res = REPO / 'results/decision/h_eval/cth_v1'
    res.mkdir(parents=True, exist_ok=True)
    p = res / 'cth_data_manifest_v1.json'
    if p.exists():
        raise RuntimeError('manifest exists; refusing overwrite')
    p.write_text(json.dumps(man, indent=2) + '\n')
    print(json.dumps({k: man[k] for k in ('n_persons', 'n_recordings', 'n_windows', 'n_positive', 'dataset_sha256', 'runtime_s')}, indent=2))


if __name__ == '__main__':
    main()
