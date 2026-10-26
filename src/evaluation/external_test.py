#!/usr/bin/env python3
"""H-EVAL Gate X1: one-shot ST external replication (docs/decision/h_eval/gate_x1/gate_x1_protocol_v1.md). NOT EXECUTED.

Commands: roster | manifest | verify | acquire | execute.
acquire/execute require an external approval record for protocol H-EVAL-X1-v1 naming the X1 manifest SHA-256
(fail-closed before any network or signal access). Reuses, read-only and hash-pinned: the T1 executable (model loading,
prediction, contrast blocks, H2 rule), the E7B v1.2 safety primitives and the archived E7B scorer/decision function.
Frozen models E7B_18, M0_46 and M2_46_FINAL are used unchanged: no adaptation, retraining, recalibration or selection.
No GPU, no packages, no git operations, no overwrites.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = HERE.parents[1]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t1 = _imp(REPO_DEFAULT / 'scripts/decision/h_eval/gate_t1/gate_t1.py', 'gate_t1_for_x1')
v12, base, c2 = t1.v12, t1.base, t1.c2

VERSION = 'H-EVAL-X1-st-external-v1-2026-10-23'
PROTOCOL_ID = 'H-EVAL-X1-v1'
PROTOCOL_REL = 'docs/decision/h_eval/gate_x1/gate_x1_protocol_v1.md'
SPLIT = 'ST_EXTERNAL_RESERVED'
STUDY = 'sleep-telemetry'
CONDITIONS = ('placebo', 'temazepam')
N_PERSONS, N_RECORDINGS = 17, 28
OUT_REL = 'results/decision/h_eval/gate_x1'
LOCAL_REL = '.local/h_eval/gate_x1'
EXEC_SUBDIR = 'execution_v1'
MANIFEST_NAME = 'gate_x1_freeze_manifest_v1.json'
ROSTER_NAME = 'gate_x1_roster_v1.json'
MARKER_NAME = 'X1_EXECUTION_MARKER.json'
PRED_NAME = 'st_predictions_v1.csv'
RESULT_NAME = 'gate_x1_result_v1.json'
SCOPES = ('acquire', 'execute')
SEED_H1, SEED2 = base.SEED, t1.SEED2                       # 20261024 / 20261026, as E7B and T1
PRACTICAL = t1.PRACTICAL
T1_RESULT_REL = 'results/decision/h_eval/gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json'
T1_RESULT_SHA = '72e0a8ad334020a87531da508bd8aca39e138a4168dfd990064aa3f7dffc652f'
ST_METADATA_REL = 'data/raw/sleep_edfx_v1_0_0/metadata/ST-subjects.xls'
PSG_RE = re.compile(r'sleep-telemetry/(ST7\d\d)(\d)J0-PSG\.edf')
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
        raise ApprovalRefused(f'approval protocol must be {PROTOCOL_ID!r} (E7B and T1 approvals are not valid here)')
    sha = rec.get('manifest_sha256')
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha) or sha != manifest_sha256:
        raise ApprovalRefused('approval manifest_sha256 missing, malformed or not the X1 manifest')
    if not isinstance(rec.get('scope'), list) or scope not in rec['scope']:
        raise ApprovalRefused(f'approval scope does not include {scope!r}')
    return dict(rec, approval_record_sha256=digest(path))


def assert_st_allowed(subject, psg_path, split, roster_paths=None):
    if split.get(subject) != SPLIT:
        raise E7BStop(f'DATA BOUNDARY: {subject} is {split.get(subject)}, not {SPLIT}')
    m = PSG_RE.fullmatch(psg_path)
    if not m or m.group(1) != subject:
        raise E7BStop(f'DATA BOUNDARY: {psg_path} does not belong to {subject}')
    if roster_paths is not None and psg_path not in roster_paths:
        raise E7BStop(f'DATA BOUNDARY: {psg_path} not in frozen roster')


def build_roster(X):
    """Metadata only (split, E2 status, E3 stage at LightsOff, condition, night, size, official SHA-256).
    Landmark reconstruction is checked against E2 metadata; no outcome count is written."""
    recs = []
    for r in X['inventory'].values():
        if (X['split'].get(r['subject_key']) == SPLIT and r['status'] == 'ELIGIBLE_ORIGIN_INDEPENDENT'
                and r['study'] == STUDY and X['stage'].get(r['recording']) == 'W'):
            if r['condition'] not in CONDITIONS:
                raise E7BStop('Unknown ST condition ' + r['recording'])
            X['e5'].make_landmarks(float(r['lights_off_rel_s']), float(r['first_n2_after_off_s']), r)
            assert_st_allowed(r['subject_key'], r['psg_path'], X['split'])
            recs.append(r)
    recs.sort(key=lambda r: (r['subject_key'], int(r['night']), r['recording']))
    entries = [dict(subject_key=r['subject_key'], recording=r['recording'], night=int(r['night']), condition=r['condition'],
                    lights_off_source=r['metadata_source'], psg_path=r['psg_path'], bytes=int(r['psg_size_bytes']),
                    official_sha256=X['sums'][r['psg_path']], hypnogram_path=r['hypnogram_path'],
                    hypnogram_official_sha256=X['sums'][r['hypnogram_path']], role='PRIMARY') for r in recs]
    persons = sorted({e['subject_key'] for e in entries})
    if len(entries) != N_RECORDINGS or len(persons) != N_PERSONS:
        raise E7BStop(f'ST roster differs from protocol: {len(persons)} persons, {len(entries)} recordings')
    by = defaultdict(set)
    for e in entries:
        by[e['subject_key']].add(e['condition'])
    return dict(version=VERSION, split=SPLIT, n_persons=len(persons), n_recordings=len(entries),
                n_by_condition={c: sum(e['condition'] == c for e in entries) for c in CONDITIONS},
                n_persons_both_conditions=sum(len(v) == 2 for v in by.values()),
                total_bytes=sum(e['bytes'] for e in entries),
                selection='E3 ST_EXTERNAL_RESERVED persons; E2 ELIGIBLE_ORIGIN_INDEPENDENT; E3 stage W at LightsOff; '
                          'all eligible nights, both conditions; person is the cluster',
                landmark_reconstruction_ok=True, outcome_fields_included=False, recordings=entries)


# ------------------------------------------------------------------------ pre-flight (headers and checksums only)
def preflight(repo, roster, X):
    """Before the one-shot marker: EDF header (labels/order, uV, 100 Hz, size), signal duration covers the E2 timing,
    and local hypnogram checksums. Reads no EEG samples. Any failure => INVALID_DATA (no marker, no exclusion)."""
    root = Path(repo) / 'data/raw/sleep_edfx_v1_0_0'
    paths = {r['psg_path'] for r in roster}
    report, failures = {}, []
    for rec in roster:
        assert_st_allowed(rec['subject_key'], rec['psg_path'], X['split'], paths)
        inv = X['inventory'][rec['recording']]
        off, n2 = float(inv['lights_off_rel_s']), float(inv['first_n2_after_off_s'])
        try:
            lm = X['e5'].make_landmarks(off, n2, inv)
            path = root / rec['psg_path']
            with path.open('rb') as f:
                h = X['sig'].edf_header(f, path.stat().st_size)
            labels = [ch[1]['label'] for ch in h['eeg_channels']]
            if labels != ['EEG Fpz-Cz', 'EEG Pz-Oz']:
                raise ValueError('channel order ' + str(labels))
            if n2 > h['duration_s'] or min(lm) - 30 < 0:
                raise ValueError('E2 timing outside signal')
            hyp = root / rec['hypnogram_path']
            if not hyp.is_file() or digest(hyp) != rec['hypnogram_official_sha256']:
                raise ValueError('hypnogram missing or official SHA-256 mismatch')
            report[rec['recording']] = dict(ok=True, channels=labels, rates_hz=[ch[1]['rate_hz'] for ch in h['eeg_channels']],
                                            units=[ch[1]['unit'] for ch in h['eeg_channels']])
        except (ValueError, OSError) as exc:
            report[rec['recording']] = dict(ok=False, error=str(exc)[:300])
            failures.append(rec['recording'])
    return dict(ok=not failures, failures=failures, recordings=report)


def extract(repo, recs, X, roster_paths):
    """Same windows/landmarks/E5 features/raw windows as the frozen T1 extractor; ST boundary instead of SC."""
    root = Path(repo) / 'data/raw/sleep_edfx_v1_0_0'
    rows, raw = [], []
    for rec in recs:
        assert_st_allowed(rec['subject_key'], rec['psg_path'], X['split'], roster_paths)
        inv = X['inventory'][rec['recording']]
        off, n2 = float(inv['lights_off_rel_s']), float(inv['first_n2_after_off_s'])
        landmarks = X['e5'].make_landmarks(off, n2, inv)
        path = root / rec['psg_path']
        with path.open('rb') as f:
            h = X['sig'].edf_header(f, path.stat().st_size)
            if [ch[1]['label'] for ch in h['eeg_channels']] != ['EEG Fpz-Cz', 'EEG Pz-Oz']:
                raise E7BStop('Unexpected channel order ' + rec['recording'])
            for t in landmarks:
                chans, feats = [], []
                for ch in h['eeg_channels']:
                    samples, _ = X['sig'].read_window(f, h, ch, t - 30, t)
                    chans.append(samples)
                    feats.extend(X['e5'].channel_features(samples))
                raw.append(np.asarray(chans, dtype=np.float32))
                rows.append(dict(recording=rec['recording'], person=rec['subject_key'], condition=rec['condition'],
                                 elapsed_s=t - off, time_to_n2_s=n2 - t, anchor=int(n2 - t <= 1200 + 1e-8),
                                 y=int(n2 - t <= 300 + 1e-8), features=feats))
    return rows, np.stack(raw)


# ------------------------------------------------------------------------ scoring
def recording_auroc(rows, p, y, e7a, seed):
    """Within-recording AUROC: mean over recordings with both classes; 95 % CI by person-cluster bootstrap.
    Equals the E7B S4 quantity when every person contributes one recording."""
    recs = sorted({r['recording'] for r in rows})
    rid = np.array([r['recording'] for r in rows])
    owner = {r['recording']: r['person'] for r in rows}
    per = {k: e7a.auroc(p[rid == k], y[rid == k]) for k in recs}
    persons = sorted({r['person'] for r in rows})
    s, c = np.zeros(len(persons)), np.zeros(len(persons))
    for k, a in per.items():
        if a is not None:
            i = persons.index(owner[k])
            s[i] += a
            c[i] += 1
    J = np.random.default_rng(seed).integers(0, len(persons), size=(base.N_BOOT, len(persons)))
    with np.errstate(invalid='ignore'):
        boots = s[J].sum(1) / c[J].sum(1)
    return dict(definition='mean over recordings; person-cluster bootstrap', per_recording=per,
                n_defined=int(c.sum()), undefined=[k for k, a in per.items() if a is None],
                mean=float(s.sum() / c.sum()), ci95=[float(x) for x in np.nanpercentile(boots, [2.5, 97.5])])


def model_block(S, Q, m, rows):
    out = t1.model_metrics(S, Q, m)
    out['pooled_within_person_auroc'] = out.pop('within_recording_auroc')      # scorer groups by person
    w = recording_auroc(rows, Q[m], S.y, S.e7a, SEED2)
    out['within_recording_auroc'] = {k: w[k] for k in ('mean', 'ci95', 'n_defined', 'undefined')}
    if float(np.ptp(Q[m])) == 0.0:
        out['cal_slope'] = None
        out['cal_slope_note'] = 'undefined: constant predictor'
    return out


def m2_controls(S, Q, rows, raw, ctx):
    donor = []
    for dr in range(base.DRAWS):
        plan, _ = base.donor_plan(rows, dr)
        donor.append(S.brier(np.clip(c2.predict_raw(raw[plan], ctx.m2_prep, ctx.m2_models)[0], 1e-6, 1 - 1e-6)))
    nc = dict(permutation={k: v for k, v in S.permutation_null('M2_46').items() if k != 'null'},
              donor_raw_vs_constant=S.contrast(S.brier(Q['E7B_CONSTANT']), np.mean(donor, 0)))
    nc['donor_flag_gain_not_person_specific'] = bool(nc['donor_raw_vs_constant']['delta_ww_ci95'][0] > 0)
    return nc


def stratum(ctx, rows, Q, idx):
    sub = [rows[i] for i in idx]
    Qs = {m: v[idx] for m, v in Q.items()}
    S = base.Scorer(sub, Qs, ctx.e7a, ctx.am, seed=SEED2)
    b = lambda m: S.brier(Qs[m])  # noqa: E731
    return dict(n_persons=len(S.persons), n_recordings=len({r['recording'] for r in sub}), n_windows=len(sub),
                n_positive=int(S.y.sum()),
                E7B_EEG_vs_CONSTANT=S.contrast(b('E7B_CONSTANT'), b('E7B_EEG')),
                M2_vs_E7B_EEG=S.contrast(b('E7B_EEG'), b('M2_46')),
                M2_vs_M0_46=S.contrast(b('M0_46'), b('M2_46')),
                M0_46_vs_CONSTANT=S.contrast(b('E7B_CONSTANT'), b('M0_46')),
                M2_vs_CONSTANT=S.contrast(b('E7B_CONSTANT'), b('M2_46')),
                models={m: model_block(S, Qs, m, sub) for m in Qs})


def condition_paired(rows, Q, ref, cand, seed):
    """Persons with both conditions: placebo minus temazepam per-person Brier gain (ref - cand). Descriptive."""
    y = np.array([r['y'] for r in rows], float)
    lr, lc = (Q[ref] - y) ** 2, (Q[cand] - y) ** 2
    g = defaultdict(dict)
    for cond in CONDITIONS:
        for p in sorted({r['person'] for r in rows}):
            k = np.array([r['person'] == p and r['condition'] == cond for r in rows])
            if k.any():
                g[p][cond] = float(lr[k].mean() - lc[k].mean())
    both = sorted(p for p, v in g.items() if len(v) == 2)
    d = np.array([g[p]['placebo'] - g[p]['temazepam'] for p in both])
    J = np.random.default_rng(seed).integers(0, len(d), size=(base.N_BOOT, len(d)))
    return dict(persons=both, placebo_minus_temazepam_mean=float(d.mean()),
                ci95=[float(x) for x in np.percentile(d[J].mean(1), [2.5, 97.5])],
                per_person={p: g[p] for p in both})


def score_all(ctx, rows, raw, nc4_pass, integrity_ok):
    P, Q = t1.predict_all(ctx, rows, raw)
    res1, nc1, nc2, _ = base.score_cohort(rows, P, ctx.M_e7b, ctx.e6, ctx.e7a, ctx.am)
    y = np.array([r['y'] for r in rows], float)
    s4r = recording_auroc(rows, P['EEG_LOGIT'], y, ctx.e7a, SEED_H1)
    dec1 = base.decide(res1['primary'], res1['S1_subject_equal']['delta'], s4r, nc1, nc2, nc4_pass)
    res1['S4_pooled_within_person_auroc'] = res1.pop('S4_within_auroc')
    res1['S4R_within_recording_auroc'] = s4r
    ds = res1['drift_sensitivity']
    ds['pooled_within_person_auroc'] = ds.pop('within_auroc')
    S = base.Scorer(rows, Q, ctx.e7a, ctx.am, seed=SEED2)
    h2 = t1.contrast_block(S, 'E7B_EEG', 'M2_46')
    h3 = t1.contrast_block(S, 'M0_46', 'M2_46')
    cond = np.array([r['condition'] for r in rows])
    strata = {c: stratum(ctx, rows, Q, np.flatnonzero(cond == c)) for c in CONDITIONS}
    return dict(H1=dict(outcome=dec1['outcome'], decision=dec1, decision_auroc='S4R_within_recording_auroc', scores=res1),
                H2=dict(outcome=t1.h2_outcome(dec1['outcome'], h2, integrity_ok), contrast='E7B_18 EEG_LOGIT -> M2_46_FINAL', **h2),
                H3=dict(status='DESCRIPTIVE_ONLY', contrast='M0_46 -> M2_46_FINAL', **h3),
                models={m: model_block(S, Q, m, rows) for m in Q}, m2_negative_controls=m2_controls(S, Q, rows, raw, ctx),
                condition_strata_descriptive=strata,
                condition_paired_descriptive=dict(
                    E7B_EEG_vs_CONSTANT=condition_paired(rows, Q, 'E7B_CONSTANT', 'E7B_EEG', SEED2),
                    M2_vs_CONSTANT=condition_paired(rows, Q, 'E7B_CONSTANT', 'M2_46', SEED2)),
                n_persons=len(S.persons), n_recordings=len({r['recording'] for r in rows}), n_windows=len(rows),
                n_positive=int(S.y.sum())), P, Q


# ------------------------------------------------------------------------ context and state machine
class Context(t1.Context):
    def __init__(self, repo, *a, local_dir=None, out_dir=None, **k):
        super().__init__(repo, *a, local_dir=local_dir or Path(repo) / LOCAL_REL,
                         out_dir=out_dir or Path(repo) / OUT_REL / EXEC_SUBDIR, **k)

    def extract(self, recs):
        self.extract_calls += 1
        return extract(self.repo, recs, self.X, {r['psg_path'] for r in self.roster})


def acquire(ctx, approval_path):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'acquire')
    ctx.verify_manifest()
    paths = {r['psg_path'] for r in ctx.roster}
    log, invalid = [], None
    for r in ctx.roster:
        assert_st_allowed(r['subject_key'], r['psg_path'], ctx.X['split'], paths)
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
    files = ctx.statuses('PRIMARY')
    state = dict(version=VERSION, approval_ref=approval['approval_ref'], approval_record_sha256=approval['approval_record_sha256'],
                 cohort=cohort_state(files), files=files, log=log, invalid_data_file=invalid)
    publish_atomic(ctx.local / 'acquire_states' / f'acquire_{time.time_ns()}.json', envelope(state))
    if invalid:
        raise E7BStop('INVALID_DATA ' + invalid)
    return state


def _t1_published(ctx):
    p = ctx.repo / T1_RESULT_REL
    if not p.is_file() or digest(p) != T1_RESULT_SHA:
        raise E7BStop('Ordering safeguard: the T1 confirmatory result is missing or differs from the frozen hash')


def _marker_gate(ctx, approval, recovery_path):
    ctx.local.mkdir(parents=True, exist_ok=True)
    marker = ctx.local / MARKER_NAME
    if marker.exists():
        if recovery_path is None:
            raise E7BStop('X1 marker exists without a result: written recovery authorisation required')
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
    P, Q = t1.predict_all(ctx, rows, raw)
    blob = base.to_csv([dict(recording=r['recording'], person=r['person'], condition=r['condition'], elapsed_s=r['elapsed_s'],
                             **{f'p_E7B_{m}': float(P[m][i]) for m in base.MODELS},
                             p_M0_46=float(Q['M0_46'][i]), p_M2_46=float(Q['M2_46'][i])) for i, r in enumerate(rows)])
    path = ctx.local / PRED_NAME
    if path.exists():
        if mode != 'recovery' or path.read_bytes() != blob:
            raise E7BStop('INVALID: recomputed predictions differ from the published prediction file')
    else:
        publish_atomic(path, blob, ctx.crash)
    return hashlib.sha256(blob).hexdigest()


def execute(ctx, approval_path, recovery_path=None):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'execute')
    ctx.verify_manifest()
    _t1_published(ctx)
    integ = ctx.integrity()
    if not integ['e7b_reference_ok']:
        raise E7BStop('NC4-pre failed: E7B frozen models do not reproduce their TRAIN reference')
    res_path = ctx.out / RESULT_NAME
    if res_path.exists():
        read_envelope(res_path)
        raise E7BStop('A valid X1 result exists: re-execution is never permitted')
    st = cohort_state(ctx.statuses('PRIMARY'))
    if st != 'READY':
        return dict(state=st, note='no marker created; nothing extracted or scored')
    if not (ctx.local / MARKER_NAME).exists():
        pf = preflight(ctx.repo, ctx.roster, ctx.X)
        publish_atomic(ctx.local / f'preflight_{time.time_ns()}.json', envelope(pf))
        if not pf['ok']:
            return dict(state='INVALID_DATA', preflight_failures=pf['failures'],
                        note='header/montage/timing/annotation pre-flight failed; no marker; no exclusion or substitution')
    mode = _marker_gate(ctx, approval, recovery_path)
    rows, raw = ctx.extract(ctx.roster)
    pred_sha = _publish_predictions(ctx, rows, raw, mode)
    try:
        ctx.verify_manifest()
        nc4_post = True
    except E7BStop:
        nc4_post = False
    res, _, _ = score_all(ctx, rows, raw, nc4_post, integ['m2_ok'] and integ['m0_46_ok'])
    publish_atomic(res_path, envelope(dict(version=VERSION, protocol=PROTOCOL_ID, approval_ref=approval['approval_ref'],
                                           approval_record_sha256=approval['approval_record_sha256'],
                                           manifest_sha256=ctx.manifest_sha256, run_mode=mode, integrity=integ,
                                           predictions_sha256=pred_sha, results=res)), ctx.crash)
    ctx.crash('after_result_published')
    return dict(state='EXECUTED', mode=mode, X1_H1=res['H1']['outcome'], X1_H2=res['H2']['outcome'])


# ------------------------------------------------------------------------ real context, manifest, CLI
def real_context(repo, manifest_path):
    repo = Path(repo).resolve()
    X, e6, e7a, am = base.load_inputs(repo)
    man_sha = digest(manifest_path)
    roster = json.loads((repo / OUT_REL / ROSTER_NAME).read_text())['recordings']
    prep, models = c2.load_final(repo / t1.M2_DIR_REL)
    return Context(repo, X, roster, json.loads((repo / t1.E7B_MODELS_REL).read_text()), json.loads((repo / t1.M0_46_REL).read_text()),
                   prep, models, e6, e7a, am, man_sha, verify_manifest=lambda: verify_manifest(repo, manifest_path, man_sha),
                   integrity=lambda: t1.integrity_real(repo, e6), download=X['sig'].resume_download)


def manifest_files(repo):
    f = t1.manifest_files(repo)
    for r in (PROTOCOL_REL, OUT_REL + '/' + ROSTER_NAME, 'scripts/decision/h_eval/gate_x1/gate_x1.py',
              'scripts/decision/h_eval/gate_x1/test_gate_x1.py', 'scripts/decision/h_eval/gate_t1/gate_t1.py',
              'results/decision/h_eval/gate_t1/gate_t1_freeze_manifest_v1.json', T1_RESULT_REL, ST_METADATA_REL):
        f[r] = digest(Path(repo) / r)
    return f


def write_manifest(repo):
    repo = Path(repo).resolve()
    man = dict(version=VERSION, protocol_id=PROTOCOL_ID, protocol=PROTOCOL_REL,
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=manifest_files(repo),
               environment=t1.environment(),
               model_identities=dict(E7B_18=t1.E7B_MODELS_REL, M0_46=t1.M0_46_REL, M2_46_FINAL=t1.M2_DIR_REL),
               frozen_model_sha256=t1.EXPECTED, seeds=dict(H1=SEED_H1, H2_H3_strata=SEED2), practical_threshold=PRACTICAL,
               prerequisite=dict(t1_confirmatory_result=T1_RESULT_REL, sha256=T1_RESULT_SHA),
               approval_record_schema=dict(protocol=PROTOCOL_ID, manifest_sha256='<this file SHA-256>', approval_ref='<non-empty>',
                                           approved_by='<non-empty>', date='<non-empty>', scope=list(SCOPES)),
               st_signal_accessed=False)
    blob = (json.dumps(man, indent=2) + '\n').encode()
    publish_atomic(repo / OUT_REL / MANIFEST_NAME, blob)
    return hashlib.sha256(blob).hexdigest()


def verify_manifest(repo, manifest_path, expected_sha=None):
    if expected_sha is not None and digest(manifest_path) != expected_sha:
        raise E7BStop('X1 manifest SHA-256 mismatch')
    man = json.loads(Path(manifest_path).read_text())
    bad = [r for r, h in man['files'].items() if digest(Path(repo) / r) != h]
    if bad:
        raise E7BStop('X1 frozen file hash mismatch: ' + ', '.join(bad))
    if man['environment'] != t1.environment():
        raise E7BStop('Environment differs from X1 freeze')
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for c in ('roster', 'manifest', 'verify', 'acquire', 'execute'):
        p = sp.add_parser(c)
        p.add_argument('--repo', type=Path, required=True)
        if c not in ('roster', 'manifest'):
            p.add_argument('--manifest', type=Path, required=True)
        if c in ('acquire', 'execute'):
            p.add_argument('--approval', type=Path, required=True)
        if c == 'execute':
            p.add_argument('--recovery-authorisation', type=Path, default=None)
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    if a.cmd == 'roster':
        X, *_ = base.load_inputs(repo)
        publish_atomic(repo / OUT_REL / ROSTER_NAME, (json.dumps(build_roster(X), indent=2) + '\n').encode())
        print('roster sha256', digest(repo / OUT_REL / ROSTER_NAME))
    elif a.cmd == 'manifest':
        print('X1 manifest sha256', write_manifest(repo))
    else:
        ctx = real_context(repo, a.manifest)
        if a.cmd == 'verify':
            ctx.verify_manifest()
            _t1_published(ctx)
            print(json.dumps(dict(manifest_sha256=ctx.manifest_sha256, t1_result_ok=True, integrity=ctx.integrity()), indent=2))
        elif a.cmd == 'acquire':
            print(json.dumps(acquire(ctx, a.approval), indent=2))
        else:
            print(json.dumps(execute(ctx, a.approval, a.recovery_authorisation), indent=2, default=str))


if __name__ == '__main__':
    main()
