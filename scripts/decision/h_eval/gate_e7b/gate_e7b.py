#!/usr/bin/env python3
"""H-EVAL Gate E7B: frozen-model SC_VALIDATION evaluation (docs/decision/h_eval/gate_e7b/gate_e7b_protocol_v1.md).

Subcommands
  prepare    SC_TRAIN-only: freeze roster (metadata only), extract TRAIN features, fit the 5 frozen models once,
             serialise, write TRAIN reference inputs/predictions and the freeze manifest. No validation access.
  verify     Reload frozen models, require byte-identical TRAIN reference predictions, check manifest hashes.
  dry-run    In-sample pipeline exercise of the full scoring/decision code on TRAIN as a stand-in cohort
             (NOT a result). No validation access.
  acquire    [approval required; not run during preparation] download + SHA-verify the 27 roster EDFs.
  execute    [approval required; one-shot; not run during preparation] the frozen validation evaluation.
No GPU, no package installation, no git operations, no overwrites.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VERSION = 'H-EVAL-E7B-frozen-validation-v1-2026-10-23'
PROTOCOL_REL = 'docs/decision/h_eval/gate_e7b/gate_e7b_protocol_v1.md'
OUT_REL = 'results/decision/h_eval/gate_e7b'
LOCAL_REL = '.local/h_eval/gate_e7b'
MANIFEST_NAME = 'gate_e7b_freeze_manifest_v1_1.json'
AMENDMENT_REL = 'docs/decision/h_eval/gate_e7b/gate_e7b_protocol_v1_1_amendment.md'
SEED = 20261024
N_BOOT = 10_000
N_PERM = 2_000
N_TM_BOOT = 2_000
DRAWS = 5
DONOR_SALT = 'H-EVAL-E7B-donor-v1'
STRATUM_S = 300.
PRACTICAL = 0.05
NC1_BAND = (0.48, 0.52)
MODELS = ('CONSTANT', 'TIME_CONT', 'EEG_LOGIT', 'TIME_CONT_EEG_LOGIT', 'EEG_RESID_LOGIT')
MIRROR = 'https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/'
FALLBACK = 'https://physionet.org/files/sleep-edfx/1.0.0/'
CONFIRM_FLAG = '--confirm-one-shot-e7b'
N_PRIMARY, N_SENS = 15, 12


class E7BStop(RuntimeError):
    """Critical stop: integrity, boundary or reproducibility failure."""


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def modules(repo):
    e6 = _imp(repo / 'scripts/decision/h_eval/gate_e6/gate_e6.py', 'e6_for_e7b')
    e7a = _imp(repo / 'scripts/decision/h_eval/gate_e7a/gate_e7a.py', 'e7a_for_e7b')
    am = _imp(repo / 'scripts/decision/h_eval/gate_e7a/gate_e7a_amendment01.py', 'e7a_am_for_e7b')
    return e6, e7a, am


def write_once(path, data: bytes):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise E7BStop('Refusing to overwrite differing file: ' + str(path))
        return 'identical_existing'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return 'written'


def to_csv(rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


# ------------------------------------------------------------------------ roster and boundary
def build_roster(X):
    """Metadata only: split, eligibility, stage at LightsOff, night, size, official SHA. No outcome fields."""
    by = defaultdict(list)
    for r in X['inventory'].values():
        if (X['split'].get(r['subject_key']) == 'SC_VALIDATION' and r['status'] == 'ELIGIBLE_ORIGIN_INDEPENDENT'
                and r['study'] == 'sleep-cassette' and X['stage'].get(r['recording']) == 'W'):
            by[r['subject_key']].append(r)
    entries = []
    for s in sorted(by):
        for i, r in enumerate(sorted(by[s], key=lambda r: (int(r['night']), r['recording']))):
            entries.append(dict(subject_key=s, recording=r['recording'], night=int(r['night']), psg_path=r['psg_path'],
                                bytes=int(r['psg_size_bytes']), official_sha256=X['sums'][r['psg_path']],
                                role='PRIMARY' if i == 0 else 'SENSITIVITY'))
    prim = [e for e in entries if e['role'] == 'PRIMARY']
    sens = [e for e in entries if e['role'] == 'SENSITIVITY']
    if len(prim) != N_PRIMARY or len(sens) != N_SENS or len({e['subject_key'] for e in prim}) != N_PRIMARY:
        raise E7BStop(f'Roster size differs from protocol: {len(prim)} primary, {len(sens)} sensitivity')
    return dict(version=VERSION, split='SC_VALIDATION', selection='E2 eligible, E3 stage W at LightsOff; primary = lowest '
                'night per person; sensitivity = remaining eligible nights', n_primary=len(prim), n_sensitivity=len(sens),
                primary_bytes=sum(e['bytes'] for e in prim), sensitivity_bytes=sum(e['bytes'] for e in sens),
                outcome_fields_included=False, recordings=entries)


def assert_signal_allowed(subject, psg_path, split, allowed, roster_paths=None):
    if split.get(subject) != allowed:
        raise E7BStop(f'DATA BOUNDARY: {subject} is {split.get(subject)}, not {allowed}')
    if not psg_path.startswith('sleep-cassette/SC') or psg_path[17:20] != subject[2:5]:
        raise E7BStop(f'DATA BOUNDARY: {psg_path} does not belong to {subject}')
    if roster_paths is not None and psg_path not in roster_paths:
        raise E7BStop(f'DATA BOUNDARY: {psg_path} not in frozen roster')


# ------------------------------------------------------------------------ extraction
def extract_cohort(repo, records, X, allowed, roster_paths=None):
    root = repo / 'data/raw/sleep_edfx_v1_0_0'
    rows = []
    for rec in records:
        assert_signal_allowed(rec['subject_key'], rec['psg_path'], X['split'], allowed, roster_paths)
        inv = X['inventory'][rec['recording']]
        off, n2 = float(inv['lights_off_rel_s']), float(inv['first_n2_after_off_s'])
        try:
            landmarks = X['e5'].make_landmarks(off, n2, inv)
        except ValueError as exc:
            raise E7BStop('Landmark reconstruction mismatch: ' + str(exc)) from exc
        path = root / rec['psg_path']
        with path.open('rb') as f:
            h = X['sig'].edf_header(f, path.stat().st_size)
            if n2 > h['duration_s'] or min(landmarks) - 30 < 0:
                raise E7BStop('Landmark outside signal ' + rec['recording'])
            for t in landmarks:
                feats = []
                for ch in h['eeg_channels']:
                    samples, _ = X['sig'].read_window(f, h, ch, t - 30, t)
                    feats.extend(X['e5'].channel_features(samples))
                rows.append(dict(recording=rec['recording'], person=rec['subject_key'], elapsed_s=t - off,
                                 time_to_n2_s=n2 - t, anchor=int(n2 - t <= 1200 + 1e-8), y=int(n2 - t <= 300 + 1e-8),
                                 features=feats))
    return rows


def rows_matrices(rows, e6):
    F = np.array([r['features'] for r in rows], float)
    T = np.array([e6.time_cont_raw(r['elapsed_s']) for r in rows])
    y = np.array([r['y'] for r in rows], float)
    return F, T, y


# ------------------------------------------------------------------------ frozen models
def fit_frozen(rows, e6):
    F, T, y = rows_matrices(rows, e6)
    prior = (y.sum() + 1) / (len(y) + 2)
    cmu, csd = e6.zfit(T)
    mu, sd = e6.zfit(F)
    C, Z = e6.zapply(T, cmu, csd), e6.zapply(F, mu, sd)
    D = np.column_stack([np.ones(len(T)), T])
    beta = np.linalg.lstsq(D, F, rcond=None)[0]
    R = F - D @ beta
    rmu, rsd = e6.zfit(R)
    RZ = e6.zapply(R, rmu, rsd)
    w = {'TIME_CONT': e6.logistic_fit_np(C, y), 'EEG_LOGIT': e6.logistic_fit_np(Z, y),
         'TIME_CONT_EEG_LOGIT': e6.logistic_fit_np(np.hstack([C, Z]), y),
         'EEG_RESID_LOGIT': e6.logistic_fit_np(RZ, y)}
    L = lambda a: [float(x) for x in np.ravel(a)]  # noqa: E731
    return dict(version=VERSION, training=dict(persons=sorted({r['person'] for r in rows}), n_windows=len(rows),
                                               n_positive=int(y.sum()), support='FULL'),
                constant_prior=float(prior),
                time_features=dict(definition=['log1p(elapsed_s/60)', 'elapsed_s/3600'], mu=L(cmu), sd=L(csd)),
                eeg_features=dict(definition='E5 channel_features on [t-30s,t) of EEG Fpz-Cz then EEG Pz-Oz: log1p power '
                                             'delta/theta/alpha/beta + log1p AC RMS', mu=L(mu), sd=L(sd)),
                drift=dict(definition='OLS of each EEG feature on [1, log1p(t/60), t/3600] on TRAIN windows',
                           beta=[L(beta[i]) for i in range(beta.shape[0])], resid_mu=L(rmu), resid_sd=L(rsd)),
                weights={k: L(v) for k, v in w.items()},
                estimator='E6 logistic_fit_np: 350 steps, lr 0.3/(1+step/200), L2 0.025, clip [1e-6,1-1e-6]; '
                          'z-score population SD floor 1e-6 clipped +-5')


def predict_frozen(M, F, T, e6):
    a = lambda v: np.array(v, float)  # noqa: E731
    C = e6.zapply(T, a(M['time_features']['mu']), a(M['time_features']['sd']))
    Z = e6.zapply(F, a(M['eeg_features']['mu']), a(M['eeg_features']['sd']))
    beta = a(M['drift']['beta'])
    R = F - np.column_stack([np.ones(len(T)), T]) @ beta
    RZ = e6.zapply(R, a(M['drift']['resid_mu']), a(M['drift']['resid_sd']))
    W = {k: a(v) for k, v in M['weights'].items()}
    return {'CONSTANT': np.full(len(F), M['constant_prior']),
            'TIME_CONT': e6.logistic_predict_np(W['TIME_CONT'], C),
            'EEG_LOGIT': e6.logistic_predict_np(W['EEG_LOGIT'], Z),
            'TIME_CONT_EEG_LOGIT': e6.logistic_predict_np(W['TIME_CONT_EEG_LOGIT'], np.hstack([C, Z])),
            'EEG_RESID_LOGIT': e6.logistic_predict_np(W['EEG_RESID_LOGIT'], RZ)}


def reference_inputs_csv(rows):
    return to_csv([dict(recording=r['recording'], person=r['person'], elapsed_s=r['elapsed_s'], time_to_n2_s=r['time_to_n2_s'],
                        anchor=r['anchor'], y=r['y'], **{f'f{j}': float(v) for j, v in enumerate(r['features'])})
                   for r in rows])


def read_reference_inputs(path):
    with Path(path).open(newline='') as f:
        rr = list(csv.DictReader(f))
    return [dict(recording=r['recording'], person=r['person'], elapsed_s=float(r['elapsed_s']),
                 time_to_n2_s=float(r['time_to_n2_s']), anchor=int(r['anchor']), y=int(r['y']),
                 features=[float(r[f'f{j}']) for j in range(10)]) for r in rr]


def predictions_csv(rows, P):
    return to_csv([dict(recording=r['recording'], person=r['person'], elapsed_s=r['elapsed_s'],
                        **{f'p_{m}': float(P[m][i]) for m in MODELS}) for i, r in enumerate(rows)])


# ------------------------------------------------------------------------ negative-control donors
def donor_plan(rows, draw):
    """Deterministic elapsed-time-matched donor window index for every recipient window (protocol §5 NC2).
    Uses only person keys and elapsed time - never labels, scores or predictions."""
    person = [r['person'] for r in rows]
    stratum = [int(r['elapsed_s'] // STRATUM_S) for r in rows]
    by = defaultdict(list)
    for i, (p, k) in enumerate(zip(person, stratum)):
        by[(p, k)].append(i)
    strata_people = defaultdict(set)
    for p, k in by:
        strata_people[k].add(p)
    all_k = sorted(strata_people)
    plan, fallback_n, fallback_maxd = [], 0, 0
    for i, r in enumerate(rows):
        v, k = person[i], stratum[i]
        ks = sorted(all_k, key=lambda kk: (abs(kk - k), kk))
        kk = next((x for x in ks if strata_people[x] - {v}), None)
        if kk is None:
            raise E7BStop('No donor available in any stratum (cohort has a single person)')
        if kk != k:
            fallback_n += 1
            fallback_maxd = max(fallback_maxd, abs(kk - k))
        cands = sorted(strata_people[kk] - {v},
                       key=lambda d: hashlib.sha256(f'{DONOR_SALT}|{draw}|{v}|{k}|{d}'.encode()).hexdigest())
        d = cands[0]
        j = min(by[(d, kk)], key=lambda j: (abs(rows[j]['elapsed_s'] - r['elapsed_s']), rows[j]['elapsed_s']))
        plan.append(j)
    return plan, dict(fallback_windows=fallback_n, max_stratum_distance=fallback_maxd)


# ------------------------------------------------------------------------ scoring
class Scorer:
    def __init__(self, rows, P, e7a, am, seed=SEED, n_boot=N_BOOT):
        self.rows, self.P, self.e7a, self.am = rows, P, e7a, am
        self.persons = sorted({r['person'] for r in rows})
        self.pidx = np.array([self.persons.index(r['person']) for r in rows])
        self.y = np.array([r['y'] for r in rows], float)
        self.n = np.bincount(self.pidx, minlength=len(self.persons)).astype(float)
        self.I = np.random.default_rng(seed).integers(0, len(self.persons), size=(n_boot, len(self.persons)))
        self.seed = seed

    def _sums(self, v):
        return np.bincount(self.pidx, weights=v, minlength=len(self.persons))

    def brier(self, p):
        return (p - self.y) ** 2

    def logloss(self, p):
        return -(self.y * np.log(p) + (1 - self.y) * np.log1p(-p))

    def contrast(self, loss_ref, loss_cand, ratio=True):
        sr, sc = self._sums(loss_ref), self._sums(loss_cand)
        I, n = self.I, self.n
        Br, Bc = sr[I].sum(1) / n[I].sum(1), sc[I].sum(1) / n[I].sum(1)
        d_ww = Br - Bc
        pr, pc = sr / n, sc / n
        d_se = (pr - pc)[I].mean(1)
        point_ww = sr.sum() / n.sum() - sc.sum() / n.sum()
        out = dict(ref_ww=float(sr.sum() / n.sum()), cand_ww=float(sc.sum() / n.sum()),
                   delta_ww=float(point_ww), delta_ww_ci95=[float(x) for x in np.percentile(d_ww, [2.5, 97.5])],
                   delta_se=float((pr - pc).mean()), delta_se_ci95=[float(x) for x in np.percentile(d_se, [2.5, 97.5])],
                   per_person_delta={p: float(v) for p, v in zip(self.persons, pr - pc)},
                   persons_favouring_candidate=int(((pr - pc) > 0).sum()))
        if ratio:
            out['R'] = float(point_ww / (sr.sum() / n.sum()))
            out['R_ci95'] = [float(x) for x in np.percentile(d_ww / Br, [2.5, 97.5])]
        return out

    def level_within(self, ref, cand):
        L = {m: np.zeros(len(self.persons)) for m in (ref, cand)}
        W = {m: np.zeros(len(self.persons)) for m in (ref, cand)}
        for i in range(len(self.persons)):
            k = self.pidx == i
            for m in (ref, cand):
                L[m][i], W[m][i] = self.e7a.level_within(self.P[m][k], self.y[k])
        out = {}
        for comp, D in (('level', L), ('within', W)):
            d = D[ref] - D[cand]
            se, ww = d[self.I].mean(1), (d * self.n)[self.I].sum(1) / self.n[self.I].sum(1)
            out[comp] = dict(subject_equal=float(d.mean()), subject_equal_ci95=[float(x) for x in np.percentile(se, [2.5, 97.5])],
                             window_weighted=float((d * self.n).sum() / self.n.sum()),
                             window_weighted_ci95=[float(x) for x in np.percentile(ww, [2.5, 97.5])])
        return out

    def within_auroc(self, m):
        vals = []
        for i, p in enumerate(self.persons):
            k = self.pidx == i
            a = self.e7a.auroc(self.P[m][k], self.y[k])
            vals.append((p, a))
        defined = [(p, a) for p, a in vals if a is not None]
        arr = np.array([a for _, a in defined])
        J = np.random.default_rng(self.seed).integers(0, len(arr), size=(N_BOOT, len(arr)))
        return dict(per_person={p: a for p, a in vals}, n_defined=len(arr), undefined=[p for p, a in vals if a is None],
                    mean=float(arr.mean()), ci95=[float(x) for x in np.percentile(arr[J].mean(1), [2.5, 97.5])])

    def permutation_null(self, m):
        rng = np.random.default_rng(self.seed)
        items = []
        for i in range(len(self.persons)):
            k = self.pidx == i
            yy = self.y[k].astype(int)
            if 0 < yy.sum() < len(yy):
                items.append((self.e7a.tie_ranks(self.P[m][k]), int(yy.sum()), len(yy)))
        null = np.array([np.mean([(rk[rng.permutation(nn)[:npos]].sum() - npos * (npos + 1) / 2) / (npos * (nn - npos))
                                  for rk, npos, nn in items]) for _ in range(N_PERM)])
        return dict(null_mean=float(null.mean()), null_95th=float(np.percentile(null, 95)),
                    passes=bool(NC1_BAND[0] <= null.mean() <= NC1_BAND[1]), null=null)

    def time_matched(self, m):
        person = np.array([r['person'] for r in self.rows])
        stratum = np.floor(np.array([r['elapsed_s'] for r in self.rows]) / STRATUM_S).astype(int)
        win, pairs, used = self.am.pair_matrices(self.P[m], self.y.astype(int), person, stratum, self.persons)
        Ib = np.random.default_rng(self.seed).integers(0, len(self.persons), size=(N_TM_BOOT, len(self.persons)))
        C = np.stack([np.bincount(r, minlength=len(self.persons)) for r in Ib]).astype(float)
        boots = np.array([self.am.weighted_auc(win, pairs, c) for c in C])
        return dict(auroc=self.am.weighted_auc(win, pairs, np.ones(len(self.persons))),
                    ci95=[float(x) for x in np.nanpercentile(boots, [2.5, 97.5])],
                    n_cross_person_pairs=int(pairs.sum()), n_strata_contributing=int(used))

    def tertiles(self, ref, cand):
        order = sorted(range(len(self.persons)), key=lambda i: (self.n[i], self.persons[i]))
        k = len(order) // 3
        groups = {'short': order[:k], 'middle': order[k:2 * k], 'long': order[2 * k:]}
        br, bc = self._sums(self.brier(self.P[ref])), self._sums(self.brier(self.P[cand]))
        out = {}
        for g, idx in groups.items():
            idx = np.array(idx)
            out[g] = dict(persons=[self.persons[i] for i in idx], n_windows=[int(self.n[i]) for i in idx],
                          delta_ww=float((br[idx].sum() - bc[idx].sum()) / self.n[idx].sum()),
                          delta_se=float((br[idx] / self.n[idx] - bc[idx] / self.n[idx]).mean()))
        return out


def decide(prim, se_delta, s4, nc1_pass, nc2, nc4_pass):
    flags = dict(STATISTICALLY_POSITIVE=bool(prim['delta_ww_ci95'][0] > 0),
                 PRACTICALLY_MEANINGFUL_POINT=bool(prim['R'] >= PRACTICAL),
                 PRACTICALLY_ESTABLISHED=bool(prim['R_ci95'][0] >= PRACTICAL))
    if not (nc1_pass and nc4_pass):
        return dict(outcome='INVALID', flags=flags, failed=[n for n, ok in (('NC1', nc1_pass), ('NC4', nc4_pass)) if not ok])
    go_cond = dict(statistically_positive=flags['STATISTICALLY_POSITIVE'],
                   practically_meaningful_point=flags['PRACTICALLY_MEANINGFUL_POINT'],
                   subject_equal_delta_gt_0=bool(se_delta > 0),
                   within_auroc_ci_lower_gt_half=bool(s4['ci95'][0] > 0.5),
                   nc2_donor_does_not_reproduce_gain=bool(nc2['delta_ww_ci95'][0] <= 0))
    go = all(go_cond.values())
    nogo = bool(prim['R_ci95'][1] < PRACTICAL or (prim['delta_ww'] <= 0 and se_delta <= 0))
    failed = [k for k, ok in go_cond.items() if not ok]
    if go and nogo:
        return dict(outcome='INCONCLUSIVE', flags=flags, go_conditions=go_cond, failed=['anomaly: GO and NO-GO both hold'])
    if go:
        return dict(outcome='GO', flags=flags, go_conditions=go_cond, failed=[])
    if nogo:
        return dict(outcome='NO-GO', flags=flags, go_conditions=go_cond, failed=failed)
    return dict(outcome='INCONCLUSIVE', flags=flags, go_conditions=go_cond, failed=failed)


def score_cohort(rows, P, M, e6, e7a, am):
    S = Scorer(rows, P, e7a, am)
    b = lambda m: S.brier(P[m])  # noqa: E731
    prim = S.contrast(b('CONSTANT'), b('EEG_LOGIT'))
    F, T, _ = rows_matrices(rows, e6)
    donor_losses, donor_info = [], []
    for d in range(DRAWS):
        plan, info = donor_plan(rows, d)
        Pd = predict_frozen(M, F[plan], T, e6)
        donor_losses.append(S.brier(Pd['EEG_LOGIT']))
        donor_info.append(info)
    nc2 = S.contrast(b('CONSTANT'), np.mean(donor_losses, axis=0))
    nc1 = S.permutation_null('EEG_LOGIT')
    s4 = S.within_auroc('EEG_LOGIT')
    res = dict(
        n_persons=len(S.persons), n_windows=len(rows), n_positive=int(S.y.sum()),
        primary=prim,
        S1_subject_equal=dict(delta=prim['delta_se'], ci95=prim['delta_se_ci95']),
        S2_logloss=S.contrast(S.logloss(P['CONSTANT']), S.logloss(P['EEG_LOGIT']), ratio=False),
        S3_level_within=S.level_within('CONSTANT', 'EEG_LOGIT'),
        S4_within_auroc=s4,
        S5_time_matched=S.time_matched('EEG_LOGIT'),
        S6_time_cont_to_time_cont_eeg=S.contrast(b('TIME_CONT'), b('TIME_CONT_EEG_LOGIT'), ratio=False),
        S7_latency_tertiles=S.tertiles('CONSTANT', 'EEG_LOGIT'),
        NC1_permutation={k: v for k, v in nc1.items() if k != 'null'},
        NC2_donor=dict(nc2, draws=DRAWS, salt=DONOR_SALT, donor_fallback=donor_info),
        NC3_time_cont_vs_constant=S.contrast(b('CONSTANT'), b('TIME_CONT'), ratio=False),
        drift_sensitivity=dict(vs_constant=S.contrast(b('CONSTANT'), b('EEG_RESID_LOGIT')),
                               within_auroc=S.within_auroc('EEG_RESID_LOGIT'),
                               time_matched=S.time_matched('EEG_RESID_LOGIT')))
    return res, nc1['passes'], nc2, s4


# ------------------------------------------------------------------------ inputs
def load_inputs(repo):
    e6, e7a, am = modules(repo)
    X = e6.load_inputs(repo)     # verifies E6 protocol hash, E3 split counts, E2/E3/E4 provenance
    return X, e6, e7a, am


def train_records(repo):
    prov = json.loads((repo / 'results/decision/h_eval/gate_e6/gate_e6_data_provenance.json').read_text())
    recs = [r for r in prov['recordings'] if r['status'] == 'VERIFIED']
    if len(recs) != 18:
        raise E7BStop('Expected the 18 verified E6 training recordings')
    return recs


def check_train_rows_against_e6(repo, rows):
    with (repo / '.local/h_eval/gate_e6/gate_e6_window_predictions.csv').open(newline='') as f:
        e6w = list(csv.DictReader(f))
    if len(e6w) != len(rows):
        raise E7BStop('TRAIN landmark count differs from E6')
    for a, b in zip(rows, e6w):
        if (a['recording'], a['person'], a['elapsed_s'], a['time_to_n2_s'], a['anchor'], a['y']) != (
                b['recording'], b['person'], float(b['elapsed_s']), float(b['time_to_n2_s']), int(b['anchor']), int(b['y'])):
            raise E7BStop('TRAIN landmark table differs from E6 window table')
    return True


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=metadata.version('scipy'))


# ------------------------------------------------------------------------ prepare / verify / dry-run
FROZEN_FILES = ('gate_e7b_validation_roster_v1.json', 'frozen_models_v1.json', 'gate_e7b_train_reference_inputs_v1.csv',
                'gate_e7b_train_reference_predictions_v1.csv', 'gate_e7b_prepare_report_v1.json')


def prepare(repo):
    X, e6, e7a, am = load_inputs(repo)
    out = repo / OUT_REL
    roster = build_roster(X)
    recs = train_records(repo)
    for r in recs:
        p = repo / 'data/raw/sleep_edfx_v1_0_0' / r['psg_path']
        if digest(p) != X['sums'][r['psg_path']]:
            raise E7BStop('TRAIN EDF hash mismatch ' + r['psg_path'])
    rows = extract_cohort(repo, recs, X, 'SC_TRAIN')
    check_train_rows_against_e6(repo, rows)
    M = fit_frozen(rows, e6)
    M2 = fit_frozen(read_reference_inputs_from_bytes(reference_inputs_csv(rows)), e6)
    if json.dumps(M, sort_keys=True) != json.dumps(M2, sort_keys=True):
        raise E7BStop('Second independent fit from serialised inputs differs')
    F, T, _ = rows_matrices(rows, e6)
    P = predict_frozen(M, F, T, e6)
    blobs = {'gate_e7b_validation_roster_v1.json': (json.dumps(roster, indent=2) + '\n').encode(),
             'frozen_models_v1.json': (json.dumps(M, indent=2) + '\n').encode(),
             'gate_e7b_train_reference_inputs_v1.csv': reference_inputs_csv(rows),
             'gate_e7b_train_reference_predictions_v1.csv': predictions_csv(rows, P)}
    rep = dict(version=VERSION, train_edfs_sha256_verified=len(recs), train_windows=len(rows),
               train_rows_match_e6_window_table=True, second_fit_identical=True,
               train_prevalence=float(np.mean([r['y'] for r in rows])),
               note='Single in-sample fit on 18 SC_TRAIN persons; not comparable with E6 LOPO predictions.',
               data_boundary=dict(validation_eeg_accessed=False, validation_metadata_used='roster fields only',
                                  protected_test_st_muse_accessed=False))
    blobs['gate_e7b_prepare_report_v1.json'] = (json.dumps(rep, indent=2) + '\n').encode()
    for n, bl in blobs.items():
        if (out / n).exists() and (out / n).read_bytes() != bl:
            raise E7BStop('Frozen artifact conflict ' + n)
    for n, bl in blobs.items():
        write_once(out / n, bl)
    return rep


def read_reference_inputs_from_bytes(blob):
    tmp = io.StringIO(blob.decode())
    rr = list(csv.DictReader(tmp))
    return [dict(recording=r['recording'], person=r['person'], elapsed_s=float(r['elapsed_s']),
                 time_to_n2_s=float(r['time_to_n2_s']), anchor=int(r['anchor']), y=int(r['y']),
                 features=[float(r[f'f{j}']) for j in range(10)]) for r in rr]


def reproduce_reference(repo, e6):
    out = repo / OUT_REL
    M = json.loads((out / 'frozen_models_v1.json').read_text())
    rows = read_reference_inputs(out / 'gate_e7b_train_reference_inputs_v1.csv')
    F, T, _ = rows_matrices(rows, e6)
    P = predict_frozen(M, F, T, e6)
    saved = (out / 'gate_e7b_train_reference_predictions_v1.csv').read_bytes()
    regenerated = predictions_csv(rows, P)
    with (out / 'gate_e7b_train_reference_predictions_v1.csv').open(newline='') as f:
        sv = list(csv.DictReader(f))
    maxd = max(abs(float(s[f'p_{m}']) - P[m][i]) for i, s in enumerate(sv) for m in MODELS)
    return dict(byte_identical=regenerated == saved, max_abs_diff=float(maxd), n_rows=len(rows),
                passed=bool(regenerated == saved and maxd == 0.0)), M, rows, P


def manifest_files(repo):
    rel = [PROTOCOL_REL, AMENDMENT_REL, 'scripts/decision/h_eval/gate_e7b/gate_e7b.py', 'scripts/decision/h_eval/gate_e7b/test_gate_e7b.py',
           'scripts/decision/h_eval/gate_e6/gate_e6.py', 'scripts/decision/h_eval/gate_e5/gate_e5.py',
           'scripts/decision/h_eval/gate_e4/v3/gate_e4.py', 'scripts/decision/h_eval/gate_e7a/gate_e7a.py',
           'scripts/decision/h_eval/gate_e7a/gate_e7a_amendment01.py',
           'results/decision/h_eval/gate_e2/sleep_edf_origin_inventory.csv',
           'results/decision/h_eval/gate_e3/gate_e3_sc_group_split.json',
           'results/decision/h_eval/gate_e3/gate_e3_stage_diagnostics.csv',
           'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt',
           'docs/decision/h_eval/gate_e6/gate_e6_protocol.md']
    rel += [f'{OUT_REL}/{n}' for n in FROZEN_FILES]
    return {r: digest(repo / r) for r in rel}


def write_manifest(repo, dryrun_sha=None):
    man = dict(version=VERSION, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               protocol=PROTOCOL_REL, amendment=AMENDMENT_REL, supersedes='gate_e7b_freeze_manifest_v1.json', files=manifest_files(repo), environment=environment(),
               bootstrap=dict(seed=SEED, n_boot=N_BOOT, n_perm=N_PERM, n_time_matched_boot=N_TM_BOOT),
               donor=dict(salt=DONOR_SALT, draws=DRAWS, stratum_s=STRATUM_S),
               decision=dict(practical_threshold_R=PRACTICAL, nc1_band=list(NC1_BAND)),
               commands=dict(acquire='scripts/decision/h_eval/gate_e7b/gate_e7b.py acquire --repo . --manifest '
                                     f'{OUT_REL}/{MANIFEST_NAME}',
                             execute=f'scripts/decision/h_eval/gate_e7b/gate_e7b.py execute --repo . --manifest '
                                     f'{OUT_REL}/{MANIFEST_NAME} --manifest-sha256 <MANIFEST_SHA256> '
                                     f'--approval-ref <APPROVAL_REF> {CONFIRM_FLAG}'),
               dryrun_report_sha256=dryrun_sha,
               validation_eeg_accessed=False)
    blob = (json.dumps(man, indent=2) + '\n').encode()
    write_once(repo / OUT_REL / MANIFEST_NAME, blob)
    return hashlib.sha256(blob).hexdigest()


def verify_manifest(repo, manifest_path, expected_sha=None):
    if expected_sha is not None and digest(manifest_path) != expected_sha:
        raise E7BStop('Manifest SHA-256 differs from the approved value')
    man = json.loads(Path(manifest_path).read_text())
    bad = [r for r, h in man['files'].items() if digest(repo / r) != h]
    if bad:
        raise E7BStop('Frozen file hash mismatch: ' + ', '.join(bad))
    if man['environment'] != environment():
        raise E7BStop('Environment differs from frozen environment')
    return man


def dry_run(repo):
    """In-sample pipeline exercise on TRAIN reference inputs. NOT a result; no validation access."""
    X, e6, e7a, am = load_inputs(repo)
    repro, M, rows, P = reproduce_reference(repo, e6)
    if not repro['passed']:
        raise E7BStop('Reference reproduction failed')
    res, nc1_pass, nc2, s4 = score_cohort(rows, P, M, e6, e7a, am)
    dec = decide(res['primary'], res['S1_subject_equal']['delta'], s4, nc1_pass, nc2, repro['passed'])
    rep = dict(version=VERSION, status='DRY_RUN_IN_SAMPLE_TRAIN_STANDIN_NOT_A_RESULT', nc4_reproduction=repro,
               decision_function_output=dec, scores=res)
    blob = (json.dumps(rep, indent=2, default=float) + '\n').encode()
    write_once(repo / LOCAL_REL / 'dryrun_train_standin_v1.json', blob)
    return rep, hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------------ acquire / execute (approval required)
def roster_records(repo):
    return json.loads((repo / OUT_REL / 'gate_e7b_validation_roster_v1.json').read_text())['recordings']


def completeness(repo, recs):
    root = repo / 'data/raw/sleep_edfx_v1_0_0'
    status = {}
    for r in recs:
        p = root / r['psg_path']
        status[r['recording']] = ('VERIFIED' if p.is_file() and p.stat().st_size == r['bytes']
                                  and digest(p) == r['official_sha256'] else 'MISSING_OR_UNVERIFIED')
    return status


def acquire(repo, manifest):
    X, e6, e7a, am = load_inputs(repo)
    verify_manifest(repo, manifest)
    recs = roster_records(repo)
    root = repo / 'data/raw/sleep_edfx_v1_0_0'
    log = []
    for r in recs:
        assert_signal_allowed(r['subject_key'], r['psg_path'], X['split'], 'SC_VALIDATION',
                              {x['psg_path'] for x in recs})
        target = root / r['psg_path']
        st = None
        for base in (MIRROR, FALLBACK):                 # same .e4-part, so a fallback resumes the mirror's bytes
            try:
                st = X['sig'].resume_download(base + r['psg_path'], target, r['official_sha256'], r['bytes'])
                break
            except RuntimeError as exc:
                st = f'NETWORK_INCOMPLETE via {base}: ' + repr(exc)[:200]
            except ValueError as exc:                  # checksum / range / layout violation: never retried
                raise E7BStop(f'INVALID_DATA {r["psg_path"]}: {exc}') from exc
        log.append(dict(recording=r['recording'], role=r['role'], state=st))
    status = completeness(repo, recs)
    prim_ok = all(status[r['recording']] == 'VERIFIED' for r in recs if r['role'] == 'PRIMARY')
    sens_ok = all(status[r['recording']] == 'VERIFIED' for r in recs if r['role'] == 'SENSITIVITY')
    state = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), primary='READY' if prim_ok else 'INCOMPLETE',
                 sensitivity='READY' if sens_ok else 'SENSITIVITY_INCOMPLETE', files=status, log=log)
    d = repo / LOCAL_REL / 'acquire_states'
    d.mkdir(parents=True, exist_ok=True)
    write_once(d / f'acquire_{time.time_ns()}.json', (json.dumps(state, indent=2) + '\n').encode())
    return state


def one_shot_gate(local, out, approval_ref, manifest_sha, recovery_authorisation=None):
    """Exclusive one-shot marker. A second execution is refused, except (v1.1 amendment) a written recovery
    authorisation after a crash that left NO primary result file: it must name the existing marker's SHA-256."""
    local.mkdir(parents=True, exist_ok=True)
    marker = local / 'E7B_EXECUTION_MARKER.json'
    if marker.exists():
        if (out / 'gate_e7b_primary_result_v1.json').exists():
            raise E7BStop('E7B already executed and a primary result exists; re-execution is never permitted')
        if recovery_authorisation is None:
            raise E7BStop('E7B one-shot marker exists without a primary result: a written recovery authorisation is required')
        auth = json.loads(Path(recovery_authorisation).read_text())
        if auth.get('marker_sha256') != digest(marker) or not auth.get('authorised_by'):
            raise E7BStop('Recovery authorisation does not name the existing marker SHA-256 / authoriser')
        write_once(local / f'recovery_{time.time_ns()}.json', (json.dumps(dict(auth, used_utc=time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime()), authorisation_sha256=digest(recovery_authorisation))) + '\n').encode())
        return 'recovery'
    fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), approval_ref=approval_ref,
                       manifest_sha256=manifest_sha, version=VERSION), f)
        f.flush()
        os.fsync(f.fileno())
    return 'new'


def execute(repo, manifest, manifest_sha, approval_ref, recovery_authorisation=None):
    X, e6, e7a, am = load_inputs(repo)
    man = verify_manifest(repo, manifest, manifest_sha)
    repro, M, _, _ = reproduce_reference(repo, e6)
    if not repro['passed']:
        raise E7BStop('NC4-pre failed: frozen models do not reproduce TRAIN reference predictions')
    recs = roster_records(repo)
    roster_paths = {r['psg_path'] for r in recs}
    status = completeness(repo, recs)
    primary = [r for r in recs if r['role'] == 'PRIMARY']
    if not all(status[r['recording']] == 'VERIFIED' for r in primary):
        return dict(state='INCOMPLETE', files=status, note='no marker created; nothing extracted or scored')
    local = repo / LOCAL_REL
    out = repo / OUT_REL / 'execution_v1'
    one_shot_gate(local, out, approval_ref, manifest_sha, recovery_authorisation)
    rows = extract_cohort(repo, primary, X, 'SC_VALIDATION', roster_paths)
    F, T, _ = rows_matrices(rows, e6)
    P = predict_frozen(M, F, T, e6)
    pblob = predictions_csv(rows, P)
    write_once(local / 'validation_primary_predictions_v1.csv', pblob)
    res, nc1_pass, nc2, s4 = score_cohort(rows, P, M, e6, e7a, am)
    nc4_post = bool(verify_manifest(repo, manifest, manifest_sha) is not None)
    dec = decide(res['primary'], res['S1_subject_equal']['delta'], s4, nc1_pass, nc2, repro['passed'] and nc4_post)
    write_once(out / 'gate_e7b_primary_result_v1.json', (json.dumps(dict(
        version=VERSION, approval_ref=approval_ref, manifest_sha256=manifest_sha, decision=dec, scores=res,
        predictions_sha256=hashlib.sha256(pblob).hexdigest()), indent=2, default=float) + '\n').encode())
    sens = [r for r in recs if r['role'] == 'SENSITIVITY']
    if all(status[r['recording']] == 'VERIFIED' for r in sens):
        srows = rows + extract_cohort(repo, sens, X, 'SC_VALIDATION', roster_paths)
        Fs, Ts, _ = rows_matrices(srows, e6)
        sres, *_ = score_cohort(srows, predict_frozen(M, Fs, Ts, e6), M, e6, e7a, am)
        sout = dict(status='SENSITIVITY_ALL_27_DECISION_NEUTRAL', scores=sres)
    else:
        sout = dict(status='SENSITIVITY_INCOMPLETE')
    write_once(out / 'gate_e7b_sensitivity_result_v1.json', (json.dumps(sout, indent=2, default=float) + '\n').encode())
    return dict(state='EXECUTED', outcome=dec['outcome'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for c in ('prepare', 'verify', 'dry-run', 'acquire', 'execute', 'manifest'):
        p = sp.add_parser(c)
        p.add_argument('--repo', type=Path, required=True)
        if c in ('verify', 'acquire', 'execute'):
            p.add_argument('--manifest', type=Path, required=True)
        if c == 'execute':
            p.add_argument('--manifest-sha256', required=True)
            p.add_argument('--approval-ref', required=True)
            p.add_argument(CONFIRM_FLAG, dest='confirm', action='store_true')
            p.add_argument('--recovery-authorisation', default=None)
        if c == 'manifest':
            p.add_argument('--dryrun-sha256', default=None)
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    if a.cmd == 'prepare':
        print(json.dumps(prepare(repo), indent=2))
    elif a.cmd == 'manifest':
        print('manifest sha256', write_manifest(repo, a.dryrun_sha256))
    elif a.cmd == 'verify':
        _, e6, _, _ = load_inputs(repo)
        verify_manifest(repo, a.manifest)
        print(json.dumps(reproduce_reference(repo, e6)[0], indent=2))
    elif a.cmd == 'dry-run':
        rep, h = dry_run(repo)
        print(json.dumps(dict(status=rep['status'], decision_function_output=rep['decision_function_output']['outcome'],
                              nc4=rep['nc4_reproduction'], sha256=h), indent=2))
    elif a.cmd == 'acquire':
        print(json.dumps(acquire(repo, a.manifest), indent=2))
    elif a.cmd == 'execute':
        if not a.confirm:
            raise SystemExit(f'refusing: {CONFIRM_FLAG} required')
        print(json.dumps(execute(repo, a.manifest, a.manifest_sha256, a.approval_ref, a.recovery_authorisation), indent=2))


if __name__ == '__main__':
    main()
