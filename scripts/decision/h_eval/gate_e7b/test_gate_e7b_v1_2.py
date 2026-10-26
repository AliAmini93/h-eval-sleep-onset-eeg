"""Gate E7B v1.2 tests: network-free, synthetic end-to-end. Synthetic EDFs stand in for SC_VALIDATION; no real
validation file is downloaded or read, and the real acquire/execute CLI is never invoked."""
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
_s = importlib.util.spec_from_file_location('gate_e7b_v1_2_test', HERE / 'gate_e7b_v1_2.py')
v12 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(v12)
base = v12.base
e6, e7a, am = base.modules(REPO)
_t = importlib.util.spec_from_file_location('e6_tests_for_v12', REPO / 'scripts/decision/h_eval/gate_e6/test_gate_e6.py')
e6t = importlib.util.module_from_spec(_t)
_t.loader.exec_module(e6t)
E4 = e6.import_py(REPO / 'scripts/decision/h_eval/gate_e4/v3/gate_e4.py', 'e4_for_v12_tests')
E5 = e6.import_py(REPO / 'scripts/decision/h_eval/gate_e5/gate_e5.py', 'e5_for_v12_tests')
MAN_SHA = 'ab' * 32


def _landmark_counts(off, n2):
    first, last = math.ceil((off + 30) / 30 - 1e-9), math.floor(n2 / 30 + 1e-9)
    tt = [k * 30. for k in range(first, last + 1) if k * 30 < n2 - 1e-8]
    return len(tt), sum(n2 - t <= 1200 + 1e-8 for t in tt), sum(n2 - t <= 300 + 1e-8 for t in tt)


class World:
    """Synthetic repository with SC_VALIDATION-like EDFs (4 persons, 2 with a second night)."""

    def __init__(self, tmp, seed=0):
        self.tmp = Path(tmp)
        self.root = self.tmp / 'data/raw/sleep_edfx_v1_0_0'
        self.store = self.tmp / 'store'               # where "downloads" come from
        rng = np.random.default_rng(seed)
        self.inventory, self.split, self.roster = {}, {'SC499': 'SC_PROTECTED_MODEL_TEST'}, []
        for j in range(4):
            subj = f'SC49{j}'
            self.split[subj] = 'SC_VALIDATION'
            for night in ((1, 2) if j < 2 else (1,)):
                rid, psg = f'SC49{j}{night}', f'sleep-cassette/SC49{j}{night}E0-PSG.edf'
                off, n2 = 60., 60. + 30. * int(rng.integers(14, 30))
                dur = int(n2 + 120)
                t = np.arange(dur * 100) / 100.
                amp = 150 + 300 * (t > n2 - 300)
                ch0 = np.clip(rng.normal(0, 1, len(t)) * amp, -2000, 2000).astype(int).tolist()
                ch1 = np.clip(rng.normal(0, 1, len(t)) * amp, -2000, 2000).astype(int).tolist()
                f = self.store / psg
                f.parent.mkdir(parents=True, exist_ok=True)
                e6t.write_edf(f, ch0, ch1)
                n, a, pos = _landmark_counts(off, n2)
                self.inventory[rid] = dict(recording=rid, subject_key=subj, lights_off_rel_s=str(off),
                                           first_n2_after_off_s=str(n2), landmark_n=str(n), anchored_lb1200_n=str(a),
                                           positive_h300_n=str(pos))
                self.roster.append(dict(subject_key=subj, recording=rid, night=night, psg_path=psg, bytes=f.stat().st_size,
                                        official_sha256=base.digest(f), role='PRIMARY' if night == 1 else 'SENSITIVITY'))
        rows = []
        for j in range(6):
            for k in range(40):
                y = int(k >= 30)
                rows.append(dict(recording=f'T{j}', person=f'T{j}', elapsed_s=30. * (k + 1), time_to_n2_s=30. * (40 - k),
                                 anchor=1, y=y, features=list(rng.normal(size=10) + y)))
        self.M = base.fit_frozen(rows, e6)
        self.manifest_ok = True
        self.downloads = []

    def install(self, which='all'):
        for r in self.roster:
            if which == 'all' or r['role'] == which:
                (self.root / r['psg_path']).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.store / r['psg_path'], self.root / r['psg_path'])

    def verify(self):
        if not self.manifest_ok:
            raise base.E7BStop('manifest mismatch (simulated)')

    def download(self, url, target, sha, nbytes):
        self.downloads.append(url)
        if self.fail_mirror and url.startswith(base.MIRROR):
            raise RuntimeError('simulated network failure')
        if self.bad_sha_for and target.name.startswith(self.bad_sha_for):
            raise ValueError('Downloaded EDF SHA FAILED (simulated)')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.store / target.relative_to(self.root), target)
        return 'downloaded_sha_verified'

    fail_mirror = False
    bad_sha_for = None

    def ctx(self, crash=None):
        X = dict(inventory=self.inventory, split=self.split, sig=E4, e5=E5)
        return v12.Context(self.tmp, X, self.roster, self.M, e6, e7a, am, MAN_SHA, self.verify, lambda: True,
                           download=self.download, crash=crash)

    def approval(self, name='approval.json', **over):
        rec = dict(protocol=v12.PROTOCOL_ID, manifest_sha256=MAN_SHA, approval_ref='TEST-REF', approved_by='tester',
                   date='2026-10-23', scope=list(v12.SCOPES))
        rec.update(over)
        p = self.tmp / name
        p.write_text(json.dumps(rec))
        return p


def crash_at(point):
    def f(p):
        if p == point:
            raise v12.CrashInjected(p)
    return f


class TestApprovalGate(unittest.TestCase):
    def test_all_refusal_states(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            bad = [None, w.tmp / 'nope.json']
            (w.tmp / 'notjson.json').write_text('{')
            (w.tmp / 'list.json').write_text('[]')
            bad += [w.tmp / 'notjson.json', w.tmp / 'list.json']
            bad += [w.approval('a1.json', approval_ref=''), w.approval('a2.json', approval_ref='   '),
                    w.approval('a3.json', approved_by=''), w.approval('a4.json', date=''),
                    w.approval('a5.json', manifest_sha256='cd' * 32), w.approval('a6.json', manifest_sha256='xyz'),
                    w.approval('a7.json', protocol='H-EVAL-E7B-v1.1'), w.approval('a8.json', scope=['execute']),
                    w.approval('a9.json', scope='acquire')]
            for b in bad:
                with self.assertRaises(v12.ApprovalRefused, msg=str(b)):
                    v12.load_approval(b, MAN_SHA, 'acquire')
            ok = v12.load_approval(w.approval(), MAN_SHA, 'acquire')
            self.assertEqual(ok['approval_ref'], 'TEST-REF')
            with self.assertRaises(v12.ApprovalRefused):
                v12.load_approval(w.approval(), None, 'acquire')

    def test_refusal_happens_before_any_access(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            ctx = w.ctx()
            bad = w.approval('bad.json', manifest_sha256='cd' * 32)
            with self.assertRaises(v12.ApprovalRefused):
                v12.acquire(ctx, bad)
            self.assertEqual(w.downloads, [])
            w.install()
            for fn in (lambda: v12.execute(ctx, bad), lambda: v12.execute(ctx, None),
                       lambda: v12.sensitivity_complete(ctx, bad, None)):
                with self.assertRaises(v12.ApprovalRefused):
                    fn()
            self.assertEqual(ctx.extract_calls, 0)
            self.assertFalse((ctx.local / v12.MARKER_NAME).exists())


class TestAcquireAndCompleteness(unittest.TestCase):
    def test_acquire_fallback_and_ready(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.fail_mirror = True
            st = v12.acquire(w.ctx(), w.approval())
            self.assertEqual((st['primary'], st['sensitivity']), ('READY', 'READY'))
            self.assertTrue(any(u.startswith(base.FALLBACK) for u in w.downloads))

    def test_acquire_network_failure_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.download = lambda *a: (_ for _ in ()).throw(RuntimeError('down'))
            st = v12.acquire(w.ctx(), w.approval())
            self.assertEqual(st['primary'], 'INCOMPLETE')

    def test_acquire_invalid_sha_stops(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.bad_sha_for = 'SC4911'
            with self.assertRaises(base.E7BStop):
                v12.acquire(w.ctx(), w.approval())

    def test_boundary_refuses_protected_subject(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.roster.insert(0, dict(subject_key='SC499', recording='SC4991', night=1, psg_path='sleep-cassette/SC4991E0-PSG.edf',
                                    bytes=1, official_sha256='0' * 64, role='PRIMARY'))
            with self.assertRaises(base.E7BStop):
                v12.acquire(w.ctx(), w.approval())
            self.assertEqual(w.downloads, [])

    def test_classification(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            r = w.roster[0]
            self.assertEqual(v12.classify_file(w.root, r), 'ABSENT')
            final = w.root / r['psg_path']
            final.parent.mkdir(parents=True)
            final.with_suffix('.edf.e4-part').write_bytes(b'xx')
            self.assertEqual(v12.classify_file(w.root, r), 'PARTIAL')
            final.write_bytes(b'short')
            self.assertEqual(v12.classify_file(w.root, r), 'INVALID_DATA')            # complete final, wrong size
            final.write_bytes(b'\0' * r['bytes'])
            self.assertEqual(v12.classify_file(w.root, r), 'INVALID_DATA')            # right size, wrong SHA
            shutil.copy(w.store / r['psg_path'], final)
            self.assertEqual(v12.classify_file(w.root, r), 'VERIFIED')

    def test_execute_incomplete_and_invalid_create_no_marker(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.install()
            ctx = w.ctx()
            (w.root / w.roster[0]['psg_path']).unlink()
            self.assertEqual(v12.execute(ctx, w.approval())['state'], 'INCOMPLETE')
            (w.root / w.roster[0]['psg_path']).write_bytes(b'\0' * w.roster[0]['bytes'])
            self.assertEqual(v12.execute(ctx, w.approval())['state'], 'INVALID_DATA')
            self.assertFalse((ctx.local / v12.MARKER_NAME).exists())
            self.assertEqual(ctx.extract_calls, 0)


class TestAtomicPublication(unittest.TestCase):
    def test_crash_injection_each_boundary(self):
        for point, final_exists in (('after_stage_write', False), ('after_stage_fsync', False), ('after_link', True)):
            with tempfile.TemporaryDirectory() as d:
                final = Path(d) / 'r.json'
                with self.assertRaises(v12.CrashInjected):
                    v12.publish_atomic(final, v12.envelope({'a': 1}), crash_at(f'{point}:r.json'))
                self.assertEqual(final.exists(), final_exists)
                if final_exists:
                    self.assertEqual(v12.read_envelope(final), {'a': 1})
                    with self.assertRaises(base.E7BStop):
                        v12.publish_atomic(final, b'x')               # never replaced
                else:
                    v12.publish_atomic(final, v12.envelope({'a': 2}))
                    self.assertEqual(v12.read_envelope(final), {'a': 2})

    def test_truncated_file_is_corrupt_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_bytes(v12.envelope({'a': 1})[:20])
            with self.assertRaises(base.E7BStop):
                v12.read_envelope(p)


class TestExecuteEndToEnd(unittest.TestCase):
    def test_happy_path_and_one_shot(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.install()
            ctx = w.ctx()
            r = v12.execute(ctx, w.approval())
            self.assertEqual((r['state'], r['mode'], r['sensitivity']), ('EXECUTED', 'new', 'SENSITIVITY_PUBLISHED'))
            prim = v12.read_envelope(ctx.out / v12.PRIMARY_NAME)
            self.assertIn(prim['decision']['outcome'], ('GO', 'NO-GO', 'INCONCLUSIVE', 'INVALID'))
            self.assertEqual(prim['predictions_sha256'], base.digest(ctx.local / v12.PRED_NAME))
            v12.read_envelope(ctx.out / v12.SENS_NAME)
            with self.assertRaises(base.E7BStop):
                v12.execute(ctx, w.approval())
            with self.assertRaises(base.E7BStop):
                v12.sensitivity_complete(ctx, w.approval(), None)

    def test_crash_before_primary_then_authorised_recovery(self):
        for point in ('after_stage_write', 'after_stage_fsync'):
            with tempfile.TemporaryDirectory() as d:
                w = World(d)
                w.install()
                with self.assertRaises(v12.CrashInjected):
                    v12.execute(w.ctx(crash_at(f'{point}:{v12.PRIMARY_NAME}')), w.approval())
                ctx = w.ctx()
                self.assertFalse((ctx.out / v12.PRIMARY_NAME).exists())
                with self.assertRaises(base.E7BStop):
                    v12.execute(ctx, w.approval())                               # no recovery authorisation
                bad = w.tmp / 'rec_bad.json'
                bad.write_text(json.dumps(dict(marker_sha256='0' * 64, authorised_by='x')))
                with self.assertRaises(base.E7BStop):
                    v12.execute(ctx, w.approval(), bad)
                good = w.tmp / 'rec.json'
                good.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / v12.MARKER_NAME), authorised_by='sup')))
                r = v12.execute(ctx, w.approval(), good)
                self.assertEqual((r['state'], r['mode']), ('EXECUTED', 'recovery'))

    def test_recovery_rejects_tampered_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.install()
            with self.assertRaises(v12.CrashInjected):
                v12.execute(w.ctx(crash_at(f'after_stage_fsync:{v12.PRIMARY_NAME}')), w.approval())
            ctx = w.ctx()
            pf = ctx.local / v12.PRED_NAME
            data = pf.read_bytes()
            pf.unlink()
            pf.write_bytes(data.replace(b'0.', b'1.', 1))
            good = w.tmp / 'rec.json'
            good.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / v12.MARKER_NAME), authorised_by='sup')))
            with self.assertRaises(base.E7BStop):
                v12.execute(ctx, w.approval(), good)

    def test_primary_published_sensitivity_interrupted_then_completed(self):
        for point in (f'after_link:{v12.PRIMARY_NAME}', 'after_primary_published', f'after_stage_write:{v12.SENS_NAME}'):
            with tempfile.TemporaryDirectory() as d:
                w = World(d)
                w.install()
                with self.assertRaises(v12.CrashInjected):
                    v12.execute(w.ctx(crash_at(point)), w.approval())
                ctx = w.ctx()
                primary_before = base.digest(ctx.out / v12.PRIMARY_NAME)
                self.assertFalse((ctx.out / v12.SENS_NAME).exists())
                with self.assertRaises(base.E7BStop):
                    v12.execute(ctx, w.approval())                               # no rescoring ever
                with self.assertRaises(base.E7BStop):
                    v12.sensitivity_complete(ctx, w.approval(), None)
                bad = w.tmp / 's_bad.json'
                bad.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / v12.MARKER_NAME),
                                               primary_result_sha256='0' * 64, authorised_by='sup')))
                with self.assertRaises(base.E7BStop):
                    v12.sensitivity_complete(ctx, w.approval(), bad)
                auth = w.tmp / 's.json'
                auth.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / v12.MARKER_NAME),
                                                primary_result_sha256=primary_before, authorised_by='sup')))
                r = v12.sensitivity_complete(ctx, w.approval(), auth)
                self.assertEqual(r['sensitivity'], 'SENSITIVITY_PUBLISHED')
                self.assertEqual(base.digest(ctx.out / v12.PRIMARY_NAME), primary_before)   # primary untouched
                with self.assertRaises(base.E7BStop):
                    v12.sensitivity_complete(ctx, w.approval(), auth)

    def test_sensitivity_incomplete_then_completed(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.install('PRIMARY')
            ctx = w.ctx()
            r = v12.execute(ctx, w.approval())
            self.assertEqual(r['sensitivity'], 'SENSITIVITY_INCOMPLETE')
            w.install('SENSITIVITY')
            auth = w.tmp / 's.json'
            auth.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / v12.MARKER_NAME),
                                            primary_result_sha256=base.digest(ctx.out / v12.PRIMARY_NAME), authorised_by='sup')))
            self.assertEqual(v12.sensitivity_complete(ctx, w.approval(), auth)['sensitivity'], 'SENSITIVITY_PUBLISHED')

    def test_corrupt_primary_refused(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.install()
            ctx = w.ctx()
            ctx.out.mkdir(parents=True)
            (ctx.out / v12.PRIMARY_NAME).write_bytes(b'{"content_sha256": "ab", "cont')
            with self.assertRaises(base.E7BStop) as cm:
                v12.execute(ctx, w.approval())
            self.assertIn('CORRUPT', str(cm.exception))
            with self.assertRaises(base.E7BStop):
                v12.sensitivity_complete(ctx, w.approval(), None)

    def test_nc4_post_failure_publishes_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.install('PRIMARY')
            calls = {'n': 0}

            def verify():
                calls['n'] += 1
                if calls['n'] >= 2:
                    raise base.E7BStop('manifest changed (simulated)')
            ctx = w.ctx()
            ctx.verify_manifest = verify
            r = v12.execute(ctx, w.approval())
            self.assertEqual(r['outcome'], 'INVALID')


class TestFrozenReferenceUnchanged(unittest.TestCase):
    def test_train_reference_predictions_byte_identical(self):
        rep = base.reproduce_reference(REPO, e6)[0]
        self.assertTrue(rep['passed'])


if __name__ == '__main__':
    unittest.main()
