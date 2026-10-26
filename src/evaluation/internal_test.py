#!/usr/bin/env python3
"""H-EVAL Gate T1: single protected SC test (docs/decision/h_eval/gate_t1/gate_t1_protocol_v1.md). NOT EXECUTED.

Commands: roster | manifest | verify | acquire | execute | sensitivity-complete.
acquire/execute/sensitivity-complete require an external approval record for protocol H-EVAL-T1-v1 naming the T1
manifest SHA-256 (fail-closed before any network or signal access). Reuses, read-only: E7B v1.2 safety primitives
(atomic publication, envelopes, file classification), the archived E7B scorer/decision function, and the CTH v2
M2_46_FINAL CPU inference. No GPU, no packages, no git operations, no overwrites.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = HERE.parents[1]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v12 = _imp(REPO_DEFAULT / 'scripts/decision/h_eval/gate_e7b/gate_e7b_v1_2.py', 'e7b_v12_for_t1')
base = v12.base
c2 = _imp(REPO_DEFAULT / 'scripts/decision/h_eval/cth_v2/cth_final_v2.py', 'cth_v2_for_t1')

VERSION = 'H-EVAL-T1-protected-test-v1-2026-10-23'
PROTOCOL_ID = 'H-EVAL-T1-v1'
PROTOCOL_REL = 'docs/decision/h_eval/gate_t1/gate_t1_protocol_v1.md'
SPLIT = 'SC_PROTECTED_MODEL_TEST'
OUT_REL = 'results/decision/h_eval/gate_t1'
LOCAL_REL = '.local/h_eval/gate_t1'
EXEC_SUBDIR = 'execution_v1'
MANIFEST_NAME = 'gate_t1_freeze_manifest_v1.json'
ROSTER_NAME = 'gate_t1_roster_v1.json'
MARKER_NAME = 'T1_EXECUTION_MARKER.json'
PRED_NAME = 'protected_primary_predictions_v1.csv'
CONF_NAME = 'gate_t1_confirmatory_result_v1.json'
SENS_NAME = 'gate_t1_sensitivity_result_v1.json'
SCOPES = ('acquire', 'execute', 'sensitivity-complete')
SEED2 = 20261026
PRACTICAL = 0.05
E7B_MODELS_REL = 'results/decision/h_eval/gate_e7b/frozen_models_v1.json'
M0_46_REL = 'results/decision/h_eval/cth_v1/frozen_m0_46_v1.json'
M2_DIR_REL = 'results/decision/h_eval/cth_v2/frozen_m2_46_final'
EXPECTED = {E7B_MODELS_REL: 'fc2572b477430fecd53d51a5876248bee185f6b466e74487aa017bca7da45d31',
            M0_46_REL: '125651e14a62ce33bcebd55b3e6498370536f9ee0031f2b45aab4e9755ec5c11',
            M2_DIR_REL + '/frozen_manifest.json': 'cf73e0d09f5c404273f6a7b7dc5355bce9dfe7294cd2387c2d8044ad69d92bbd'}
E7BStop = base.E7BStop
digest = base.digest
publish_atomic, envelope, read_envelope = v12.publish_atomic, v12.envelope, v12.read_envelope
classify_file, cohort_state, CrashInjected = v12.classify_file, v12.cohort_state, v12.CrashInjected


class ApprovalRefused(E7BStop):
    pass


# ------------------------------------------------------------------------ approval / roster / boundary
def load_approval(path, manifest_sha256, scope):
    if scope not in SCOPES:
        raise ApprovalRefused('unknown scope')
    if not path or not Path(path).is_file():
        raise ApprovalRefused('approval record missing')
    try:
        rec = json.loads(Path(path).read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise ApprovalRefused('approval record is not valid JSON') from exc
    if not isinstance(rec, dict):
        raise ApprovalRefused('approval record must be a JSON object')
    for k in ('approval_ref', 'approved_by', 'date'):
        if not isinstance(rec.get(k), str) or not rec[k].strip():
            raise ApprovalRefused(f'approval field {k!r} missing or empty')
    if rec.get('protocol') != PROTOCOL_ID:
        raise ApprovalRefused(f'approval protocol must be {PROTOCOL_ID!r} (E7B approvals are not valid here)')
    sha = rec.get('manifest_sha256')
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha) or sha != manifest_sha256:
        raise ApprovalRefused('approval manifest_sha256 missing, malformed or not the T1 manifest')
    if not isinstance(rec.get('scope'), list) or scope not in rec['scope']:
        raise ApprovalRefused(f'approval scope does not include {scope!r}')
    return dict(rec, approval_record_sha256=digest(path))


def build_roster(X):
    by = defaultdict(list)
    for r in X['inventory'].values():
        if (X['split'].get(r['subject_key']) == SPLIT and r['status'] == 'ELIGIBLE_ORIGIN_INDEPENDENT'
                and r['study'] == 'sleep-cassette' and X['stage'].get(r['recording']) == 'W'):
            by[r['subject_key']].append(r)
    entries = []
    for s in sorted(by):
        for i, r in enumerate(sorted(by[s], key=lambda r: (int(r['night']), r['recording']))):
            entries.append(dict(subject_key=s, recording=r['recording'], night=int(r['night']), psg_path=r['psg_path'],
                                bytes=int(r['psg_size_bytes']), official_sha256=X['sums'][r['psg_path']],
                                role='PRIMARY' if i == 0 else 'SENSITIVITY'))
    prim = [e for e in entries if e['role'] == 'PRIMARY']
    if len(prim) != 15 or len({e['subject_key'] for e in prim}) != 15:
        raise E7BStop('protected roster must contain 15 primary persons')
    return dict(version=VERSION, split=SPLIT, n_primary=len(prim), n_sensitivity=len(entries) - len(prim),
                primary_bytes=sum(e['bytes'] for e in prim), sensitivity_bytes=sum(e['bytes'] for e in entries if e['role'] != 'PRIMARY'),
                selection='E2 eligible, E3 stage W at LightsOff; primary = lowest night; sensitivity = remaining nights',
                outcome_fields_included=False, recordings=entries)


def extract(repo, recs, X, roster_paths):
    """Same windows/landmarks/E5 features as the archived E7B extractor, plus the raw [Fpz-Cz, Pz-Oz] windows."""
    root = Path(repo) / 'data/raw/sleep_edfx_v1_0_0'
    rows, raw = [], []
    for rec in recs:
        base.assert_signal_allowed(rec['subject_key'], rec['psg_path'], X['split'], SPLIT, roster_paths)
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
            labels = [ch[1]['label'] for ch in h['eeg_channels']]
            if labels != ['EEG Fpz-Cz', 'EEG Pz-Oz']:
                raise E7BStop('Unexpected channel order ' + str(labels))
            for t in landmarks:
                chans, feats = [], []
                for ch in h['eeg_channels']:
                    samples, _ = X['sig'].read_window(f, h, ch, t - 30, t)
                    chans.append(samples)
                    feats.extend(X['e5'].channel_features(samples))
                raw.append(np.asarray(chans, dtype=np.float32))
                rows.append(dict(recording=rec['recording'], person=rec['subject_key'], elapsed_s=t - off,
                                 time_to_n2_s=n2 - t, anchor=int(n2 - t <= 1200 + 1e-8), y=int(n2 - t <= 300 + 1e-8),
                                 features=feats))
    return rows, np.stack(raw)


# ------------------------------------------------------------------------ models and scoring
def m0_predict(M0, F, e6):
    return e6.logistic_predict_np(np.array(M0['weights']), e6.zapply(F, np.array(M0['eeg_mu']), np.array(M0['eeg_sd'])))


def predict_all(ctx, rows, raw):
    F, T, _ = base.rows_matrices(rows, ctx.e6)
    P = base.predict_frozen(ctx.M_e7b, F, T, ctx.e6)
    pm2, _ = c2.predict_raw(raw, ctx.m2_prep, ctx.m2_models)
    pm2 = np.clip(pm2, 1e-6, 1 - 1e-6)                    # same clipping as the frozen E7B models
    return P, dict(E7B_CONSTANT=P['CONSTANT'], E7B_EEG=P['EEG_LOGIT'], M0_46=m0_predict(ctx.M0, F, ctx.e6), M2_46=pm2)


def predictions_csv(rows, P, Q):
    return base.to_csv([dict(recording=r['recording'], person=r['person'], elapsed_s=r['elapsed_s'],
                             **{f'p_E7B_{m}': float(P[m][i]) for m in base.MODELS},
                             p_M0_46=float(Q['M0_46'][i]), p_M2_46=float(Q['M2_46'][i])) for i, r in enumerate(rows)])


def model_metrics(S, Q, m):
    y = S.y
    p = Q[m]
    b, ll = (p - y) ** 2, S.logloss(p)
    pe = [p[S.pidx == i] for i in range(len(S.persons))]
    ye = [y[S.pidx == i] for i in range(len(S.persons))]
    cal = c2.cx.calibration(p, y)
    wa = S.within_auroc(m) if any(0 < yy.sum() < len(yy) for yy in ye) else None
    def ll_person(a, c):
        a = np.clip(a, 1e-6, 1 - 1e-6)
        return float((-(c * np.log(a) + (1 - c) * np.log1p(-a))).mean())
    return dict(brier_ww=float(b.mean()), brier_se=float(np.mean([((a - c) ** 2).mean() for a, c in zip(pe, ye)])),
                logloss_ww=float(ll.mean()), logloss_se=float(np.mean([ll_person(a, c) for a, c in zip(pe, ye)])),
                **cal, within_recording_auroc=None if wa is None else dict(mean=wa['mean'], ci95=wa['ci95'], undefined=wa['undefined']),
                time_matched_auroc=S.time_matched(m))


def loo_R(S, ref, cand):
    br, bc = S._sums(S.brier(S.P[ref])), S._sums(S.brier(S.P[cand]))
    return float(min(((br.sum() - br[i]) - (bc.sum() - bc[i])) / (br.sum() - br[i]) for i in range(len(br))))


def contrast_block(S, ref, cand):
    c = S.contrast(S.brier(S.P[ref]), S.brier(S.P[cand]))
    pp = c['per_person_delta']
    worst = sorted(pp.items(), key=lambda kv: kv[1])[:3]
    el = np.array([r['elapsed_s'] for r in S.rows])
    lr, lc = S.brier(S.P[ref]), S.brier(S.P[cand])
    temporal = {k: (float(lr[m].mean() - lc[m].mean()) if m.any() else None)
                for k, m in (('elapsed_lt_30min', el < 1800), ('elapsed_ge_30min', el >= 1800))}
    return dict(c, loo_R_min=loo_R(S, ref, cand), persons_worse=int(sum(v < 0 for v in pp.values())),
                worst_three=[dict(person=p, delta=v) for p, v in worst], temporal_delta_ww=temporal,
                latency_tertiles=S.tertiles(ref, cand))


def h2_outcome(h1_outcome, h2, integrity_ok):
    if not integrity_ok:
        return 'INVALID'
    supported = bool(h2['R'] >= PRACTICAL and h2['delta_ww_ci95'][0] > 0)
    if h1_outcome != 'GO':
        return 'NOT_TESTED' + ('(would_be_SUPPORTED)' if supported else '(would_be_NOT_SUPPORTED)')
    return 'SUPPORTED' if supported else 'NOT_SUPPORTED'


def score_all(ctx, rows, raw, nc4_pass, integrity_ok):
    P, Q = predict_all(ctx, rows, raw)
    res1, nc1, nc2, s4 = base.score_cohort(rows, P, ctx.M_e7b, ctx.e6, ctx.e7a, ctx.am)
    dec1 = base.decide(res1['primary'], res1['S1_subject_equal']['delta'], s4, nc1, nc2, nc4_pass)
    S = base.Scorer(rows, Q, ctx.e7a, ctx.am, seed=SEED2)
    h2 = contrast_block(S, 'E7B_EEG', 'M2_46')
    h3 = contrast_block(S, 'M0_46', 'M2_46')
    donor = []
    for dr in range(base.DRAWS):
        plan, _ = base.donor_plan(rows, dr)
        donor.append(S.brier(np.clip(c2.predict_raw(raw[plan], ctx.m2_prep, ctx.m2_models)[0], 1e-6, 1 - 1e-6)))
    m2_nc = dict(permutation={k: v for k, v in S.permutation_null('M2_46').items() if k != 'null'},
                 donor_raw_vs_constant=S.contrast(S.brier(Q['E7B_CONSTANT']), np.mean(donor, 0)))
    m2_nc['donor_flag_gain_not_person_specific'] = bool(m2_nc['donor_raw_vs_constant']['delta_ww_ci95'][0] > 0)
    return dict(H1=dict(outcome=dec1['outcome'], decision=dec1, scores=res1),
                H2=dict(outcome=h2_outcome(dec1['outcome'], h2, integrity_ok), contrast='E7B_18 EEG_LOGIT -> M2_46_FINAL', **h2),
                H3=dict(status='DESCRIPTIVE_ONLY', contrast='M0_46 -> M2_46_FINAL', **h3),
                models={m: model_metrics(S, Q, m) for m in Q}, m2_negative_controls=m2_nc,
                n_persons=len(S.persons), n_windows=len(rows), n_positive=int(S.y.sum())), P, Q


# ------------------------------------------------------------------------ context and state machine
class Context:
    def __init__(self, repo, X, roster, M_e7b, M0, m2_prep, m2_models, e6, e7a, am, manifest_sha256, verify_manifest,
                 integrity, download=None, crash=None, local_dir=None, out_dir=None):
        self.repo = Path(repo)
        self.X, self.roster, self.M_e7b, self.M0 = X, roster, M_e7b, M0
        self.m2_prep, self.m2_models = m2_prep, m2_models
        self.e6, self.e7a, self.am = e6, e7a, am
        self.manifest_sha256, self.verify_manifest, self.integrity = manifest_sha256, verify_manifest, integrity
        self.download = download
        self.crash = crash or (lambda p: None)
        self.data_root = self.repo / 'data/raw/sleep_edfx_v1_0_0'
        self.local = Path(local_dir) if local_dir else self.repo / LOCAL_REL
        self.out = Path(out_dir) if out_dir else self.repo / OUT_REL / EXEC_SUBDIR
        self.extract_calls = 0

    def statuses(self, role):
        return {r['recording']: classify_file(self.data_root, r) for r in self.roster if r['role'] == role}

    def extract(self, recs):
        self.extract_calls += 1
        return extract(self.repo, recs, self.X, {r['psg_path'] for r in self.roster})


def acquire(ctx, approval_path):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'acquire')
    ctx.verify_manifest()
    paths = {r['psg_path'] for r in ctx.roster}
    log, invalid = [], None
    for r in ctx.roster:
        base.assert_signal_allowed(r['subject_key'], r['psg_path'], ctx.X['split'], SPLIT, paths)
        st = classify_file(ctx.data_root, r)
        if st == 'VERIFIED':
            log.append(dict(recording=r['recording'], state='already_verified'))
            continue
        if st == 'INVALID_DATA':
            invalid = r['psg_path']
            break
        out = None
        for b in (base.MIRROR, base.FALLBACK):
            try:
                out = ctx.download(b + r['psg_path'], ctx.data_root / r['psg_path'], r['official_sha256'], r['bytes'])
                break
            except RuntimeError as exc:
                out = f'NETWORK_INCOMPLETE via {b}: ' + repr(exc)[:200]
            except ValueError as exc:
                out, invalid = f'INVALID_DATA: {exc}'[:300], r['psg_path']
                break
        log.append(dict(recording=r['recording'], state=out))
        if invalid:
            break
    prim, sens = ctx.statuses('PRIMARY'), ctx.statuses('SENSITIVITY')
    state = dict(version=VERSION, approval_ref=approval['approval_ref'], approval_record_sha256=approval['approval_record_sha256'],
                 primary=cohort_state(prim), sensitivity=cohort_state(sens), files={**prim, **sens}, log=log,
                 invalid_data_file=invalid)
    publish_atomic(ctx.local / 'acquire_states' / f'acquire_{time.time_ns()}.json', envelope(state))
    if invalid:
        raise E7BStop('INVALID_DATA ' + invalid)
    return state


def _marker_gate(ctx, approval, recovery_path):
    ctx.local.mkdir(parents=True, exist_ok=True)
    marker = ctx.local / MARKER_NAME
    if marker.exists():
        if recovery_path is None:
            raise E7BStop('T1 marker exists without a confirmatory result: written recovery authorisation required')
        auth = json.loads(Path(recovery_path).read_text())
        if auth.get('marker_sha256') != digest(marker) or not str(auth.get('authorised_by', '')).strip():
            raise E7BStop('Recovery authorisation must name the marker SHA-256 and the authoriser')
        publish_atomic(ctx.local / f'recovery_{time.time_ns()}.json',
                       envelope(dict(auth, recovery_authorisation_sha256=digest(recovery_path))))
        return 'recovery'
    fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), version=VERSION,
                       approval_ref=approval['approval_ref'], approval_record_sha256=approval['approval_record_sha256'],
                       manifest_sha256=ctx.manifest_sha256), f)
        f.flush()
        os.fsync(f.fileno())
    v12.fsync_dir(ctx.local)
    return 'new'


def _publish_predictions(ctx, rows, raw, mode):
    P, Q = predict_all(ctx, rows, raw)
    blob = predictions_csv(rows, P, Q)
    path = ctx.local / PRED_NAME
    if path.exists():
        if mode != 'recovery' or path.read_bytes() != blob:
            raise E7BStop('INVALID: recomputed predictions differ from the published prediction file')
    else:
        publish_atomic(path, blob, ctx.crash)
    return hashlib.sha256(blob).hexdigest()


def _sensitivity(ctx, prim_rows, prim_raw, approval):
    st = cohort_state(ctx.statuses('SENSITIVITY'))
    if st != 'READY':
        publish_atomic(ctx.local / f'sensitivity_status_{time.time_ns()}.json', envelope(dict(status='SENSITIVITY_' + st)))
        return 'SENSITIVITY_' + st
    srows, sraw = ctx.extract([r for r in ctx.roster if r['role'] == 'SENSITIVITY'])
    res, _, _ = score_all(ctx, prim_rows + srows, np.concatenate([prim_raw, sraw]), True, True)
    publish_atomic(ctx.out / SENS_NAME, envelope(dict(version=VERSION, status='SENSITIVITY_ALL_30_DECISION_NEUTRAL',
                                                      approval_ref=approval['approval_ref'], scores=res)), ctx.crash)
    return 'SENSITIVITY_PUBLISHED'


def execute(ctx, approval_path, recovery_path=None):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'execute')
    ctx.verify_manifest()
    integ = ctx.integrity()
    if not integ['e7b_reference_ok']:
        raise E7BStop('NC4-pre failed: E7B frozen models do not reproduce their TRAIN reference')
    conf = ctx.out / CONF_NAME
    if conf.exists():
        read_envelope(conf)
        raise E7BStop('A valid confirmatory result exists: re-execution is never permitted')
    st = cohort_state(ctx.statuses('PRIMARY'))
    if st != 'READY':
        return dict(state=st, note='no marker created; nothing extracted or scored')
    mode = _marker_gate(ctx, approval, recovery_path)
    rows, raw = ctx.extract([r for r in ctx.roster if r['role'] == 'PRIMARY'])
    pred_sha = _publish_predictions(ctx, rows, raw, mode)
    try:
        ctx.verify_manifest()
        nc4_post = True
    except E7BStop:
        nc4_post = False
    res, _, _ = score_all(ctx, rows, raw, nc4_post, integ['m2_ok'] and integ['m0_46_ok'])
    publish_atomic(conf, envelope(dict(version=VERSION, approval_ref=approval['approval_ref'],
                                       approval_record_sha256=approval['approval_record_sha256'],
                                       manifest_sha256=ctx.manifest_sha256, run_mode=mode, integrity=integ,
                                       predictions_sha256=pred_sha, results=res)), ctx.crash)
    ctx.crash('after_confirmatory_published')
    return dict(state='EXECUTED', mode=mode, H1=res['H1']['outcome'], H2=res['H2']['outcome'],
                sensitivity=_sensitivity(ctx, rows, raw, approval))


def sensitivity_complete(ctx, approval_path, auth_path):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'sensitivity-complete')
    ctx.verify_manifest()
    conf, sens = ctx.out / CONF_NAME, ctx.out / SENS_NAME
    if not conf.exists():
        raise E7BStop('No confirmatory result: sensitivity-only completion not applicable')
    c = read_envelope(conf)
    if sens.exists():
        read_envelope(sens)
        raise E7BStop('Sensitivity result already published')
    if not auth_path or not Path(auth_path).is_file():
        raise E7BStop('Written sensitivity authorisation required')
    auth = json.loads(Path(auth_path).read_text())
    if (auth.get('marker_sha256') != digest(ctx.local / MARKER_NAME) or auth.get('confirmatory_result_sha256') != digest(conf)
            or not str(auth.get('authorised_by', '')).strip()):
        raise E7BStop('Sensitivity authorisation must name marker SHA-256, confirmatory result SHA-256 and authoriser')
    rows, raw = ctx.extract([r for r in ctx.roster if r['role'] == 'PRIMARY'])
    if _publish_predictions(ctx, rows, raw, 'recovery') != c['predictions_sha256']:
        raise E7BStop('INVALID: primary predictions differ from the confirmatory record')
    return dict(state='SENSITIVITY_COMPLETION', sensitivity=_sensitivity(ctx, rows, raw, approval))


# ------------------------------------------------------------------------ real context, manifest, CLI
def integrity_real(repo, e6):
    e7b_ok = base.reproduce_reference(repo, e6)[0]['passed']
    hashes_ok = all(digest(Path(repo) / k) == v for k, v in EXPECTED.items())
    v = c2.verify()
    m2_ok = bool(hashes_ok and not v['hash_failures'] and v['reference_byte_identical'] and v['repeat_identical']
                 and v['n_gradient_training_persons'] == 46)
    d = c2.cx.load()
    M0 = json.loads((Path(repo) / M0_46_REL).read_text())
    p0 = m0_predict(M0, d['e5'], e6)
    m0_ok = hashlib.sha256(np.asarray(p0, np.float64).tobytes()).hexdigest() == M0['train_reference_predictions_sha256']
    return dict(e7b_reference_ok=bool(e7b_ok), frozen_hashes_ok=bool(hashes_ok), m2_ok=m2_ok, m0_46_ok=bool(m0_ok))


def real_context(repo, manifest_path):
    repo = Path(repo).resolve()
    X, e6, e7a, am = base.load_inputs(repo)
    man_sha = digest(manifest_path)
    roster = json.loads((repo / OUT_REL / ROSTER_NAME).read_text())['recordings']
    prep, models = c2.load_final(repo / M2_DIR_REL)
    return Context(repo, X, roster, json.loads((repo / E7B_MODELS_REL).read_text()), json.loads((repo / M0_46_REL).read_text()),
                   prep, models, e6, e7a, am, man_sha, verify_manifest=lambda: verify_manifest(repo, manifest_path, man_sha),
                   integrity=lambda: integrity_real(repo, e6), download=X['sig'].resume_download)


def manifest_files(repo):
    rel = [PROTOCOL_REL, OUT_REL + '/' + ROSTER_NAME, 'scripts/decision/h_eval/gate_t1/gate_t1.py',
           'scripts/decision/h_eval/gate_t1/test_gate_t1.py', 'scripts/decision/h_eval/gate_e7b/gate_e7b_v1_2.py',
           'scripts/decision/h_eval/gate_e7b/gate_e7b.py', 'scripts/decision/h_eval/cth_v2/cth_final_v2.py',
           'scripts/decision/h_eval/cth_v1/cth_experiment.py', 'scripts/decision/h_eval/gate_e6/gate_e6.py',
           'scripts/decision/h_eval/gate_e5/gate_e5.py', 'scripts/decision/h_eval/gate_e4/v3/gate_e4.py',
           'scripts/decision/h_eval/gate_e7a/gate_e7a.py', 'scripts/decision/h_eval/gate_e7a/gate_e7a_amendment01.py',
           'results/decision/h_eval/gate_e2/sleep_edf_origin_inventory.csv', 'results/decision/h_eval/gate_e3/gate_e3_sc_group_split.json',
           'results/decision/h_eval/gate_e3/gate_e3_stage_diagnostics.csv', 'data/raw/sleep_edfx_v1_0_0/metadata/SHA256SUMS.txt',
           'docs/decision/h_eval/gate_e6/gate_e6_protocol.md', 'docs/decision/h_eval/gate_e7b/gate_e7b_protocol_v1.md',
           'results/decision/h_eval/gate_e7b/gate_e7b_train_reference_inputs_v1.csv',
           'results/decision/h_eval/gate_e7b/gate_e7b_train_reference_predictions_v1.csv',
           E7B_MODELS_REL, M0_46_REL, 'results/decision/h_eval/cth_v2/m2_epoch_schedule_v2.json']
    m2 = Path(repo) / M2_DIR_REL
    rel += [f'{M2_DIR_REL}/{p.name}' for p in sorted(m2.iterdir())]
    return {r: digest(Path(repo) / r) for r in rel}


def environment():
    return dict(base.environment(), torch=metadata.version('torch'), scikit_learn=metadata.version('scikit-learn'))


def write_manifest(repo):
    repo = Path(repo).resolve()
    man = dict(version=VERSION, protocol_id=PROTOCOL_ID, protocol=PROTOCOL_REL,
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=manifest_files(repo),
               environment=environment(), model_identities=dict(E7B_18=E7B_MODELS_REL, M0_46=M0_46_REL, M2_46_FINAL=M2_DIR_REL),
               seeds=dict(H1=base.SEED, H2_H3=SEED2), practical_threshold=PRACTICAL,
               approval_record_schema=dict(protocol=PROTOCOL_ID, manifest_sha256='<this file SHA-256>', approval_ref='<non-empty>',
                                           approved_by='<non-empty>', date='<non-empty>', scope=list(SCOPES)),
               protected_eeg_accessed=False)
    blob = (json.dumps(man, indent=2) + '\n').encode()
    publish_atomic(repo / OUT_REL / MANIFEST_NAME, blob)
    return hashlib.sha256(blob).hexdigest()


def verify_manifest(repo, manifest_path, expected_sha=None):
    if expected_sha is not None and digest(manifest_path) != expected_sha:
        raise E7BStop('T1 manifest SHA-256 mismatch')
    man = json.loads(Path(manifest_path).read_text())
    bad = [r for r, h in man['files'].items() if digest(Path(repo) / r) != h]
    if bad:
        raise E7BStop('T1 frozen file hash mismatch: ' + ', '.join(bad))
    if man['environment'] != environment():
        raise E7BStop('Environment differs from T1 freeze')
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for c in ('roster', 'manifest', 'verify', 'acquire', 'execute', 'sensitivity-complete'):
        p = sp.add_parser(c)
        p.add_argument('--repo', type=Path, required=True)
        if c not in ('roster', 'manifest'):
            p.add_argument('--manifest', type=Path, required=True)
        if c in ('acquire', 'execute', 'sensitivity-complete'):
            p.add_argument('--approval', type=Path, required=True)
        if c == 'execute':
            p.add_argument('--recovery-authorisation', type=Path, default=None)
        if c == 'sensitivity-complete':
            p.add_argument('--sensitivity-authorisation', type=Path, required=True)
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    if a.cmd == 'roster':
        X, *_ = base.load_inputs(repo)
        publish_atomic(repo / OUT_REL / ROSTER_NAME, (json.dumps(build_roster(X), indent=2) + '\n').encode())
        print('roster sha256', digest(repo / OUT_REL / ROSTER_NAME))
    elif a.cmd == 'manifest':
        print('T1 manifest sha256', write_manifest(repo))
    else:
        ctx = real_context(repo, a.manifest)
        if a.cmd == 'verify':
            ctx.verify_manifest()
            print(json.dumps(dict(manifest_sha256=ctx.manifest_sha256, integrity=ctx.integrity()), indent=2))
        elif a.cmd == 'acquire':
            print(json.dumps(acquire(ctx, a.approval), indent=2))
        elif a.cmd == 'execute':
            print(json.dumps(execute(ctx, a.approval, a.recovery_authorisation), indent=2, default=str))
        else:
            print(json.dumps(sensitivity_complete(ctx, a.approval, a.sensitivity_authorisation), indent=2, default=str))


if __name__ == '__main__':
    main()
