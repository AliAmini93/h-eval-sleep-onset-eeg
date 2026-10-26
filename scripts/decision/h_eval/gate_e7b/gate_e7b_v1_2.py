#!/usr/bin/env python3
"""H-EVAL Gate E7B v1.2 executable (protocol v1 + v1.1 amendment + v1.2 safety amendment).

v1.2 repairs two execution-safety defects of v1.1 and keeps all science unchanged:
  * fail-closed external APPROVAL RECORD required by `acquire`, `execute` and `sensitivity-complete`, checked before
    any network request or validation-signal access;
  * crash-atomic, no-overwrite publication (stage -> fsync -> hard-link publish -> fsync dir) with self-verifying
    result envelopes, post-marker recovery, and an authorised sensitivity-only completion procedure;
  * ABSENT / PARTIAL (=> INCOMPLETE) vs complete-but-wrong (=> INVALID_DATA) EDF classification.
Roster, frozen models, estimands, controls, thresholds and decision rules come unchanged from the archived v1.1 module
`gate_e7b.py` (imported read-only and hashed in the v1.2 manifest). No GPU, no packages, no git operations.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('gate_e7b_v1_1_archived', HERE / 'gate_e7b.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

VERSION = 'H-EVAL-E7B-frozen-validation-v1.2-2026-10-23'
PROTOCOL_ID = 'H-EVAL-E7B-v1.2'
AMENDMENT_V12_REL = 'docs/decision/h_eval/gate_e7b/gate_e7b_protocol_v1_2_amendment.md'
MANIFEST_NAME = 'gate_e7b_freeze_manifest_v1_2.json'
EXEC_SUBDIR = 'execution_v1_2'
PRED_NAME = 'validation_primary_predictions_v1_2.csv'
PRIMARY_NAME = 'gate_e7b_primary_result_v1_2.json'
SENS_NAME = 'gate_e7b_sensitivity_result_v1_2.json'
MARKER_NAME = 'E7B_V1_2_EXECUTION_MARKER.json'
SCOPES = ('acquire', 'execute', 'sensitivity-complete')
E7BStop = base.E7BStop


class ApprovalRefused(E7BStop):
    """Missing, empty or mismatched approval: nothing is accessed."""


class CrashInjected(RuntimeError):
    """Test-only simulated crash."""


def digest(p):
    return base.digest(p)


def fsync_dir(d):
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ------------------------------------------------------------------------ approval gate
def load_approval(path, manifest_sha256, scope):
    """Fail closed. The approval record is an external JSON file written by the supervisor:
    {"protocol": "H-EVAL-E7B-v1.2", "manifest_sha256": <64-hex of the v1.2 manifest>, "approval_ref": <non-empty>,
     "approved_by": <non-empty>, "scope": [...subset of acquire/execute/sensitivity-complete...], "date": <non-empty>}"""
    if scope not in SCOPES:
        raise ApprovalRefused('unknown scope ' + repr(scope))
    if not path:
        raise ApprovalRefused('no approval record supplied')
    p = Path(path)
    if not p.is_file():
        raise ApprovalRefused(f'approval record not found: {p}')
    try:
        rec = json.loads(p.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise ApprovalRefused('approval record is not valid JSON') from exc
    if not isinstance(rec, dict):
        raise ApprovalRefused('approval record must be a JSON object')
    for k in ('approval_ref', 'approved_by', 'date'):
        if not isinstance(rec.get(k), str) or not rec[k].strip():
            raise ApprovalRefused(f'approval field {k!r} missing or empty')
    if rec.get('protocol') != PROTOCOL_ID:
        raise ApprovalRefused(f'approval protocol must be {PROTOCOL_ID!r}')
    sha = rec.get('manifest_sha256')
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
        raise ApprovalRefused('approval manifest_sha256 missing or malformed')
    if not isinstance(manifest_sha256, str) or sha != manifest_sha256:
        raise ApprovalRefused('approval manifest_sha256 does not match the v1.2 freeze manifest')
    sc = rec.get('scope')
    if not isinstance(sc, list) or scope not in sc:
        raise ApprovalRefused(f'approval scope does not include {scope!r}')
    return dict(rec, approval_record_sha256=digest(p))


# ------------------------------------------------------------------------ data classification
def classify_file(data_root, rec):
    """VERIFIED | ABSENT | PARTIAL (=> INCOMPLETE) | INVALID_DATA (complete final file with wrong size or SHA)."""
    final = Path(data_root) / rec['psg_path']
    part = final.with_suffix('.edf.e4-part')
    if final.is_file():
        if final.stat().st_size != rec['bytes'] or digest(final) != rec['official_sha256']:
            return 'INVALID_DATA'
        return 'VERIFIED'
    return 'PARTIAL' if part.is_file() else 'ABSENT'


def cohort_state(statuses):
    v = list(statuses.values())
    if any(s == 'INVALID_DATA' for s in v):
        return 'INVALID_DATA'
    return 'READY' if v and all(s == 'VERIFIED' for s in v) else 'INCOMPLETE'


# ------------------------------------------------------------------------ atomic publication
def envelope(content):
    canon = json.dumps(content, sort_keys=True, default=float)
    return (json.dumps(dict(content_sha256=hashlib.sha256(canon.encode()).hexdigest(), content=json.loads(canon)),
                       indent=2) + '\n').encode()


def read_envelope(path):
    """Returns content if the published file is a complete, self-consistent envelope; raises otherwise."""
    try:
        doc = json.loads(Path(path).read_text())
        canon = json.dumps(doc['content'], sort_keys=True, default=float)
        if hashlib.sha256(canon.encode()).hexdigest() != doc['content_sha256']:
            raise ValueError('content hash mismatch')
        return doc['content']
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise E7BStop(f'CORRUPT published file {path}: manual review required ({exc})') from exc


def publish_atomic(final, data: bytes, crash=lambda point: None):
    """Durable, no-overwrite publication: the final name appears only after all bytes are fsynced.
    Stage file lives in the same directory (same filesystem); publication is a hard link, which fails if the final
    name exists, so nothing is ever replaced. A crash leaves either no final file or a complete one."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise E7BStop(f'refusing to publish over existing file {final}')
    stage = final.parent / f'.{final.name}.staging-{os.getpid()}-{time.time_ns()}'
    with stage.open('xb') as f:
        f.write(data)
        crash('after_stage_write:' + final.name)
        f.flush()
        os.fsync(f.fileno())
    crash('after_stage_fsync:' + final.name)
    os.link(stage, final)
    crash('after_link:' + final.name)
    fsync_dir(final.parent)
    stage.unlink()
    fsync_dir(final.parent)
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------ context
class Context:
    """Everything execute/acquire need. RealContext binds the repository; tests build synthetic ones."""

    def __init__(self, repo, X, roster, M, e6, e7a, am, manifest_sha256, verify_manifest, reproduce_reference,
                 download=None, crash=None, local_dir=None, out_dir=None):
        self.repo = Path(repo)
        self.X, self.roster, self.M = X, roster, M
        self.e6, self.e7a, self.am = e6, e7a, am
        self.manifest_sha256 = manifest_sha256
        self.verify_manifest, self.reproduce_reference = verify_manifest, reproduce_reference
        self.download = download
        self.crash = crash or (lambda point: None)
        self.data_root = self.repo / 'data/raw/sleep_edfx_v1_0_0'
        self.local = Path(local_dir) if local_dir else self.repo / base.LOCAL_REL
        self.out = Path(out_dir) if out_dir else self.repo / base.OUT_REL / EXEC_SUBDIR
        self.extract_calls = 0

    def statuses(self, role):
        return {r['recording']: classify_file(self.data_root, r) for r in self.roster if r['role'] == role}

    def extract(self, recs):
        self.extract_calls += 1
        return base.extract_cohort(self.repo, recs, self.X, 'SC_VALIDATION', {r['psg_path'] for r in self.roster})


def real_context(repo, manifest_path):
    repo = Path(repo).resolve()
    X, e6, e7a, am = base.load_inputs(repo)
    man_sha = digest(manifest_path)
    out = repo / base.OUT_REL
    M = json.loads((out / 'frozen_models_v1.json').read_text())
    roster = base.roster_records(repo)
    return Context(repo, X, roster, M, e6, e7a, am, man_sha,
                   verify_manifest=lambda: base.verify_manifest(repo, manifest_path, man_sha),
                   reproduce_reference=lambda: base.reproduce_reference(repo, e6)[0]['passed'],
                   download=X['sig'].resume_download)


# ------------------------------------------------------------------------ acquire
def acquire(ctx, approval_path):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'acquire')     # before any network request
    ctx.verify_manifest()
    roster_paths = {r['psg_path'] for r in ctx.roster}
    log, invalid = [], None
    for r in ctx.roster:
        base.assert_signal_allowed(r['subject_key'], r['psg_path'], ctx.X['split'], 'SC_VALIDATION', roster_paths)
        state = classify_file(ctx.data_root, r)
        if state == 'VERIFIED':
            log.append(dict(recording=r['recording'], role=r['role'], state='already_verified'))
            continue
        if state == 'INVALID_DATA':
            invalid = r['psg_path']
            log.append(dict(recording=r['recording'], role=r['role'], state='INVALID_DATA_existing'))
            break
        outcome = None
        for b in (base.MIRROR, base.FALLBACK):
            try:
                outcome = ctx.download(b + r['psg_path'], ctx.data_root / r['psg_path'], r['official_sha256'], r['bytes'])
                break
            except RuntimeError as exc:
                outcome = f'NETWORK_INCOMPLETE via {b}: ' + repr(exc)[:200]
            except ValueError as exc:
                outcome = f'INVALID_DATA: {exc}'[:300]
                invalid = r['psg_path']
                break
        log.append(dict(recording=r['recording'], role=r['role'], state=outcome))
        if invalid:
            break
    prim, sens = ctx.statuses('PRIMARY'), ctx.statuses('SENSITIVITY')
    state = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), version=VERSION,
                 approval_ref=approval['approval_ref'], approval_record_sha256=approval['approval_record_sha256'],
                 primary=cohort_state(prim), sensitivity=cohort_state(sens), files={**prim, **sens}, log=log,
                 invalid_data_file=invalid)
    publish_atomic(ctx.local / 'acquire_states_v1_2' / f'acquire_{time.time_ns()}.json', envelope(state))
    if invalid:
        raise E7BStop(f'INVALID_DATA: {invalid} (evidence preserved; acquisition stopped)')
    return state


# ------------------------------------------------------------------------ execute
def _marker_gate(ctx, approval, recovery_path):
    ctx.local.mkdir(parents=True, exist_ok=True)
    marker = ctx.local / MARKER_NAME
    if marker.exists():
        if recovery_path is None:
            raise E7BStop('E7B v1.2 marker exists without a primary result: a written recovery authorisation is required')
        auth = json.loads(Path(recovery_path).read_text())
        if auth.get('marker_sha256') != digest(marker) or not str(auth.get('authorised_by', '')).strip():
            raise E7BStop('Recovery authorisation must name the existing marker SHA-256 and the authoriser')
        publish_atomic(ctx.local / f'recovery_v1_2_{time.time_ns()}.json',
                       envelope(dict(auth, recovery_authorisation_sha256=digest(recovery_path),
                                     approval_record_sha256=approval['approval_record_sha256'])))
        return 'recovery'
    fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), version=VERSION,
                       approval_ref=approval['approval_ref'], approval_record_sha256=approval['approval_record_sha256'],
                       manifest_sha256=ctx.manifest_sha256), f)
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(ctx.local)
    return 'new'


def _predict_and_publish(ctx, rows, existing_ok):
    F, T, _ = base.rows_matrices(rows, ctx.e6)
    P = base.predict_frozen(ctx.M, F, T, ctx.e6)
    blob = base.predictions_csv(rows, P)
    path = ctx.local / PRED_NAME
    if path.exists():
        if not existing_ok or path.read_bytes() != blob:
            raise E7BStop('INVALID: recomputed validation predictions differ from the published prediction file')
    else:
        publish_atomic(path, blob, ctx.crash)
    return P, hashlib.sha256(blob).hexdigest()


def _sensitivity(ctx, primary_rows, approval):
    sst = ctx.statuses('SENSITIVITY')
    state = cohort_state(sst)
    if state != 'READY':
        publish_atomic(ctx.local / f'sensitivity_status_{time.time_ns()}.json',
                       envelope(dict(status='SENSITIVITY_' + state, files=sst)))
        return 'SENSITIVITY_' + state
    sens = [r for r in ctx.roster if r['role'] == 'SENSITIVITY']
    srows = primary_rows + ctx.extract(sens)
    Fs, Ts, _ = base.rows_matrices(srows, ctx.e6)
    sres, *_ = base.score_cohort(srows, base.predict_frozen(ctx.M, Fs, Ts, ctx.e6), ctx.M, ctx.e6, ctx.e7a, ctx.am)
    publish_atomic(ctx.out / SENS_NAME, envelope(dict(version=VERSION, status='SENSITIVITY_ALL_27_DECISION_NEUTRAL',
                                                      approval_ref=approval['approval_ref'], scores=sres)), ctx.crash)
    return 'SENSITIVITY_PUBLISHED'


def execute(ctx, approval_path, recovery_path=None):
    approval = load_approval(approval_path, ctx.manifest_sha256, 'execute')     # before any validation access
    ctx.verify_manifest()
    if not ctx.reproduce_reference():
        raise E7BStop('NC4-pre failed: frozen models do not reproduce the TRAIN reference predictions')
    primary_path = ctx.out / PRIMARY_NAME
    if primary_path.exists():
        read_envelope(primary_path)                                             # raises CORRUPT for a bad file
        raise E7BStop('A valid primary result exists: re-execution/rescoring is never permitted '
                      '(use sensitivity-complete if the sensitivity result is missing)')
    pst = ctx.statuses('PRIMARY')
    state = cohort_state(pst)
    if state != 'READY':
        return dict(state=state, files=pst, note='no marker created; nothing extracted or scored')
    mode = _marker_gate(ctx, approval, recovery_path)
    prim = [r for r in ctx.roster if r['role'] == 'PRIMARY']
    rows = ctx.extract(prim)
    P, pred_sha = _predict_and_publish(ctx, rows, existing_ok=(mode == 'recovery'))
    res, nc1_pass, nc2, s4 = base.score_cohort(rows, P, ctx.M, ctx.e6, ctx.e7a, ctx.am)
    try:                                                                        # NC4-post
        ctx.verify_manifest()
        nc4_post = True
    except E7BStop:
        nc4_post = False                                                        # => decision INVALID (published)
    dec = base.decide(res['primary'], res['S1_subject_equal']['delta'], s4, nc1_pass, nc2, nc4_post)
    publish_atomic(primary_path, envelope(dict(version=VERSION, approval_ref=approval['approval_ref'],
                                               approval_record_sha256=approval['approval_record_sha256'],
                                               manifest_sha256=ctx.manifest_sha256, run_mode=mode, decision=dec,
                                               scores=res, predictions_sha256=pred_sha)), ctx.crash)
    ctx.crash('after_primary_published')
    sens_state = _sensitivity(ctx, rows, approval)
    return dict(state='EXECUTED', mode=mode, outcome=dec['outcome'], sensitivity=sens_state)


def sensitivity_complete(ctx, approval_path, sensitivity_auth_path):
    """Authorised completion of the pre-specified sensitivity analysis when a valid primary result exists but the
    sensitivity result was interrupted. Never rescores or republishes the primary result."""
    approval = load_approval(approval_path, ctx.manifest_sha256, 'sensitivity-complete')
    ctx.verify_manifest()
    if not ctx.reproduce_reference():
        raise E7BStop('NC4-pre failed')
    primary_path, sens_path = ctx.out / PRIMARY_NAME, ctx.out / SENS_NAME
    if not primary_path.exists():
        raise E7BStop('No primary result: sensitivity-only completion is not applicable')
    primary = read_envelope(primary_path)
    if sens_path.exists():
        read_envelope(sens_path)
        raise E7BStop('Sensitivity result already published')
    if not sensitivity_auth_path or not Path(sensitivity_auth_path).is_file():
        raise E7BStop('A written sensitivity-completion authorisation is required')
    auth = json.loads(Path(sensitivity_auth_path).read_text())
    if (auth.get('marker_sha256') != digest(ctx.local / MARKER_NAME) or auth.get('primary_result_sha256') != digest(primary_path)
            or not str(auth.get('authorised_by', '')).strip()):
        raise E7BStop('Sensitivity authorisation must name the marker SHA-256, the primary result SHA-256 and the authoriser')
    if cohort_state(ctx.statuses('PRIMARY')) != 'READY':
        raise E7BStop('Primary EDFs no longer verify')
    prim = [r for r in ctx.roster if r['role'] == 'PRIMARY']
    rows = ctx.extract(prim)
    _, pred_sha = _predict_and_publish(ctx, rows, existing_ok=True)            # must equal the published predictions
    if pred_sha != primary['predictions_sha256']:
        raise E7BStop('INVALID: primary predictions differ from those recorded in the primary result')
    publish_atomic(ctx.local / f'sensitivity_completion_{time.time_ns()}.json',
                   envelope(dict(auth, authorisation_sha256=digest(sensitivity_auth_path),
                                 approval_record_sha256=approval['approval_record_sha256'])))
    return dict(state='SENSITIVITY_COMPLETION', sensitivity=_sensitivity(ctx, rows, approval))


# ------------------------------------------------------------------------ manifest
def manifest_files(repo):
    files = base.manifest_files(repo)       # protocol v1, v1.1 code/tests, upstream code/data, frozen artifacts
    for rel in (base.AMENDMENT_REL, AMENDMENT_V12_REL, 'scripts/decision/h_eval/gate_e7b/gate_e7b_v1_2.py',
                'scripts/decision/h_eval/gate_e7b/test_gate_e7b_v1_2.py',
                'results/decision/h_eval/gate_e7b/gate_e7b_freeze_manifest_v1_1.json'):
        files[rel] = digest(repo / rel)
    return files


def write_manifest(repo, dryrun_sha):
    repo = Path(repo).resolve()
    man = dict(version=VERSION, protocol_id=PROTOCOL_ID, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               protocol=base.PROTOCOL_REL, amendments=[base.AMENDMENT_REL, AMENDMENT_V12_REL],
               supersedes=['gate_e7b_freeze_manifest_v1.json', 'gate_e7b_freeze_manifest_v1_1.json'],
               files=manifest_files(repo), environment=base.environment(),
               bootstrap=dict(seed=base.SEED, n_boot=base.N_BOOT, n_perm=base.N_PERM, n_time_matched_boot=base.N_TM_BOOT),
               donor=dict(salt=base.DONOR_SALT, draws=base.DRAWS, stratum_s=base.STRATUM_S),
               decision=dict(practical_threshold_R=base.PRACTICAL, nc1_band=list(base.NC1_BAND)),
               approval_record_schema=dict(protocol=PROTOCOL_ID, manifest_sha256='<this file SHA-256>',
                                           approval_ref='<non-empty>', approved_by='<non-empty>', date='<non-empty>',
                                           scope=list(SCOPES)),
               dryrun_report_sha256=dryrun_sha, validation_eeg_accessed=False)
    blob = (json.dumps(man, indent=2) + '\n').encode()
    path = repo / base.OUT_REL / MANIFEST_NAME
    if path.exists():
        if path.read_bytes() != blob:
            raise E7BStop('v1.2 manifest exists and differs')
    else:
        publish_atomic(path, blob)
    return hashlib.sha256(blob).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for c in ('manifest', 'verify', 'acquire', 'execute', 'sensitivity-complete'):
        p = sp.add_parser(c)
        p.add_argument('--repo', type=Path, required=True)
        if c != 'manifest':
            p.add_argument('--manifest', type=Path, required=True)
        if c in ('acquire', 'execute', 'sensitivity-complete'):
            p.add_argument('--approval', type=Path, required=True)
        if c == 'execute':
            p.add_argument('--recovery-authorisation', type=Path, default=None)
        if c == 'sensitivity-complete':
            p.add_argument('--sensitivity-authorisation', type=Path, required=True)
        if c == 'manifest':
            p.add_argument('--dryrun-sha256', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'manifest':
        print('v1.2 manifest sha256', write_manifest(a.repo, a.dryrun_sha256))
        return
    ctx = real_context(a.repo, a.manifest)
    if a.cmd == 'verify':
        ctx.verify_manifest()
        print(json.dumps(dict(manifest_sha256=ctx.manifest_sha256, reference_reproduction=ctx.reproduce_reference()), indent=2))
    elif a.cmd == 'acquire':
        print(json.dumps(acquire(ctx, a.approval), indent=2))
    elif a.cmd == 'execute':
        print(json.dumps(execute(ctx, a.approval, a.recovery_authorisation), indent=2))
    elif a.cmd == 'sensitivity-complete':
        print(json.dumps(sensitivity_complete(ctx, a.approval, a.sensitivity_authorisation), indent=2))


if __name__ == '__main__':
    main()
