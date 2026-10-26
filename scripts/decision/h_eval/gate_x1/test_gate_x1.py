"""Gate X1 tests: network-free; synthetic ST-like EDFs only (no ST file is downloaded or read)."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


x1 = _imp(HERE / 'gate_x1.py', 'gate_x1_test')
TT = _imp(REPO / 'scripts/decision/h_eval/gate_t1/test_gate_t1.py', 't1_tests_for_x1')
W = TT.W
base = x1.base
e6, e7a, am = base.modules(REPO)
MAN = 'ef' * 32
PLAN = [(0, ((1, 'placebo'), (2, 'temazepam'))), (1, ((1, 'temazepam'), (2, 'placebo'))),
        (2, ((2, 'placebo'),)), (3, ((1, 'temazepam'),))]


def crash_at(point):
    def f(p):
        if p == point:
            raise x1.CrashInjected(p)
    return f


class STWorld(TT.T1World):
    """Synthetic ST: 4 persons, 6 recordings, crossover conditions; plus one SC person that must never be touched."""

    def __init__(self, tmp):
        super().__init__(tmp)
        rng = np.random.default_rng(7)
        shutil.rmtree(self.store, ignore_errors=True)
        self.inventory, self.roster = {}, []
        self.split = {'SC499': 'SC_PROTECTED_MODEL_TEST', 'SC488': 'SC_VALIDATION'}
        for j, nights in PLAN:
            subj = f'ST79{j}'
            self.split[subj] = x1.SPLIT
            for night, cond in nights:
                rid = f'ST79{j}{night}'
                psg, hyp = f'sleep-telemetry/{rid}J0-PSG.edf', f'sleep-telemetry/{rid}JX-Hypnogram.edf'
                off, n2 = 60., 60. + 30. * int(rng.integers(14, 30))
                t = np.arange(int(n2 + 120) * 100) / 100.
                amp = 150 + 300 * (t > n2 - 300)
                chans = [np.clip(rng.normal(0, 1, len(t)) * amp, -2000, 2000).astype(int).tolist() for _ in range(2)]
                f = self.store / psg
                f.parent.mkdir(parents=True, exist_ok=True)
                W.e6t.write_edf(f, *chans)
                (self.root / hyp).parent.mkdir(parents=True, exist_ok=True)
                (self.root / hyp).write_bytes(b'synthetic hypnogram ' + rid.encode())
                n, a, pos = W._landmark_counts(off, n2)
                self.inventory[rid] = dict(recording=rid, subject_key=subj, lights_off_rel_s=str(off), first_n2_after_off_s=str(n2),
                                           landmark_n=str(n), anchored_lb1200_n=str(a), positive_h300_n=str(pos))
                self.roster.append(dict(subject_key=subj, recording=rid, night=night, condition=cond, psg_path=psg,
                                        bytes=f.stat().st_size, official_sha256=base.digest(f), hypnogram_path=hyp,
                                        hypnogram_official_sha256=base.digest(self.root / hyp), role='PRIMARY'))
        self.t1_result = self.tmp / x1.T1_RESULT_REL
        self.t1_result.parent.mkdir(parents=True, exist_ok=True)
        self.t1_result.write_text('t1')

    def xctx(self, crash=None, t1_sha=None):
        X = dict(inventory=self.inventory, split=self.split, sig=W.E4, e5=W.E5)
        ctx = x1.Context(self.tmp, X, self.roster, self.M, self.M0, self.m2_prep, self.m2_models, e6, e7a, am, MAN,
                         self.verify, lambda: dict(self.integ), download=self.download, crash=crash)
        return ctx

    def xapproval(self, name='x1.json', **over):
        rec = dict(protocol=x1.PROTOCOL_ID, manifest_sha256=MAN, approval_ref='X1-TEST', approved_by='tester',
                   date='2026-10-23', scope=list(x1.SCOPES))
        rec.update(over)
        p = self.tmp / name
        p.write_text(json.dumps(rec))
        return p


def patched_t1_sha(w):
    """The real ordering safeguard pins the real T1 result hash; tests point it at the synthetic stand-in."""
    x1.T1_RESULT_SHA = base.digest(w.t1_result)


class TestApprovalAndBoundary(unittest.TestCase):
    def test_foreign_approvals_refused_before_access(self):
        with tempfile.TemporaryDirectory() as d:
            w = STWorld(d)
            ctx = w.xctx()
            t1_like = w.t1approval('t1.json', manifest_sha256=MAN)
            for bad in (None, t1_like, w.xapproval('a.json', manifest_sha256='ab' * 32),
                        w.xapproval('b.json', scope=['acquire']), w.xapproval('c.json', approved_by='')):
                with self.assertRaises(x1.ApprovalRefused):
                    x1.execute(ctx, bad)
            with self.assertRaises(x1.ApprovalRefused):
                x1.acquire(ctx, t1_like)
            self.assertEqual((w.downloads, ctx.extract_calls), ([], 0))

    def test_boundary(self):
        split = {'ST701': x1.SPLIT, 'SC400': 'SC_TRAIN', 'ST702': 'SC_TRAIN'}
        x1.assert_st_allowed('ST701', 'sleep-telemetry/ST7011J0-PSG.edf', split)
        for subj, path in (('ST701', 'sleep-telemetry/ST7021J0-PSG.edf'), ('SC400', 'sleep-cassette/SC4001E0-PSG.edf'),
                           ('ST702', 'sleep-telemetry/ST7021J0-PSG.edf'), ('ST701', 'sleep-telemetry/ST7011J0-Hypnogram.edf')):
            with self.assertRaises(base.E7BStop):
                x1.assert_st_allowed(subj, path, split)
        with self.assertRaises(base.E7BStop):
            x1.assert_st_allowed('ST701', 'sleep-telemetry/ST7011J0-PSG.edf', split, {'other'})

    def test_real_roster_metadata_only(self):
        X = e6.load_inputs(REPO)
        r = x1.build_roster(X)
        self.assertEqual((r['n_persons'], r['n_recordings'], r['n_persons_both_conditions']), (17, 28, 11))
        self.assertEqual(r['n_by_condition'], {'placebo': 15, 'temazepam': 13})
        for e in r['recordings']:
            self.assertEqual(X['split'][e['subject_key']], x1.SPLIT)
            self.assertNotIn('first_n2_after_off_s', e)


class TestExecute(unittest.TestCase):
    def test_ordering_incomplete_invalid_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            w = STWorld(d)
            ctx = w.xctx()
            x1.T1_RESULT_SHA = '0' * 64
            with self.assertRaises(base.E7BStop):
                x1.execute(ctx, w.xapproval())                       # T1 not published (hash mismatch)
            patched_t1_sha(w)
            self.assertEqual(x1.execute(ctx, w.xapproval())['state'], 'INCOMPLETE')
            w.install()
            (w.root / w.roster[0]['psg_path']).write_bytes(b'\0' * w.roster[0]['bytes'])
            self.assertEqual(x1.execute(ctx, w.xapproval())['state'], 'INVALID_DATA')
            w.install()
            (w.root / w.roster[2]['hypnogram_path']).write_bytes(b'tampered')
            r = x1.execute(ctx, w.xapproval())
            self.assertEqual((r['state'], r['preflight_failures']), ('INVALID_DATA', [w.roster[2]['recording']]))
            self.assertFalse((ctx.local / x1.MARKER_NAME).exists())
            self.assertEqual(ctx.extract_calls, 0)

    def test_acquire_and_end_to_end_one_shot(self):
        with tempfile.TemporaryDirectory() as d:
            w = STWorld(d)
            patched_t1_sha(w)
            ctx = w.xctx()
            st = x1.acquire(ctx, w.xapproval())
            self.assertEqual(st['cohort'], 'READY')
            self.assertTrue(all(u.startswith(base.MIRROR + 'sleep-telemetry/ST79') for u in w.downloads))
            r = x1.execute(ctx, w.xapproval())
            self.assertEqual(r['state'], 'EXECUTED')
            c = x1.read_envelope(ctx.out / x1.RESULT_NAME)['results']
            self.assertEqual((c['n_persons'], c['n_recordings']), (4, 6))
            self.assertEqual(c['H1']['decision_auroc'], 'S4R_within_recording_auroc')
            self.assertIn('S4_pooled_within_person_auroc', c['H1']['scores'])
            self.assertNotIn('S4_within_auroc', c['H1']['scores'])
            self.assertEqual(c['H1']['scores']['S4R_within_recording_auroc']['n_defined'], 6)
            self.assertIsNone(c['models']['E7B_CONSTANT']['cal_slope'])
            for cond, n in (('placebo', 3), ('temazepam', 3)):
                self.assertEqual(c['condition_strata_descriptive'][cond]['n_recordings'], n)
            self.assertEqual(c['condition_paired_descriptive']['E7B_EEG_vs_CONSTANT']['persons'], ['ST790', 'ST791'])
            if c['H1']['outcome'] != 'GO':
                self.assertTrue(c['H2']['outcome'].startswith('NOT_TESTED'))
            with self.assertRaises(base.E7BStop):
                x1.execute(ctx, w.xapproval())                       # one-shot

    def test_crash_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            w = STWorld(d)
            patched_t1_sha(w)
            w.install()
            with self.assertRaises(x1.CrashInjected):
                x1.execute(w.xctx(crash_at(f'after_stage_fsync:{x1.RESULT_NAME}')), w.xapproval())
            ctx = w.xctx()
            with self.assertRaises(base.E7BStop):
                x1.execute(ctx, w.xapproval())                       # marker without authorisation
            rec = w.tmp / 'rec.json'
            rec.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / x1.MARKER_NAME), authorised_by='sup')))
            self.assertEqual(x1.execute(ctx, w.xapproval(), rec)['mode'], 'recovery')


class TestRecordingAuroc(unittest.TestCase):
    def test_equals_person_mean_with_one_night_and_differs_from_pooled(self):
        rows = [dict(recording=f'R{p}{n}', person=f'P{p}') for p in range(3) for n in range(2) for _ in range(4)]
        y = np.array([0, 0, 1, 1] * 6, float)
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9] * 3)        # perfect within night, offset between nights
        r = x1.recording_auroc(rows, p, y, e7a, 1)
        self.assertAlmostEqual(r['mean'], 1.0)
        pooled = e7a.auroc(p[:8], y[:8])
        self.assertLess(pooled, 1.0)


if __name__ == '__main__':
    unittest.main()
