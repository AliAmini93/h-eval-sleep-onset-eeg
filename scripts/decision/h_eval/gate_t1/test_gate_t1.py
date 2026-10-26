"""Gate T1 tests: network-free; synthetic EDFs stand in for the protected cohort (no protected file is read)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t1 = _imp(HERE / 'gate_t1.py', 'gate_t1_test')
W = _imp(REPO / 'scripts/decision/h_eval/gate_e7b/test_gate_e7b_v1_2.py', 'v12_tests_for_t1')
base, c2 = t1.base, t1.c2
e6, e7a, am = base.modules(REPO)
MAN = 'cd' * 32


def crash_at(point):
    def f(p):
        if p == point:
            raise t1.CrashInjected(p)
    return f


class T1World(W.World):
    def __init__(self, tmp):
        super().__init__(tmp)
        for s in list(self.split):
            if self.split[s] == 'SC_VALIDATION':
                self.split[s] = t1.SPLIT
        self.split['SC488'] = 'SC_VALIDATION'
        rng = np.random.default_rng(1)
        self.M0 = dict(eeg_mu=[0.0] * 10, eeg_sd=[1.0] * 10, weights=list(rng.normal(0, 0.05, 11)))
        d = self.tmp / 'm2'
        d.mkdir()
        for s in (0, 1, 2):
            torch.manual_seed(s)
            m = c2.cx.CTHNet(18, 32, cin=2, **c2.ARCH)
            torch.save(m.state_dict(), d / f'M2_46_FINAL_seed{s}.pt')
        (d / 'preprocessing_and_config.json').write_text(json.dumps(dict(cfg=dict(width=32), arch=c2.ARCH, raw_scale_uv=150.0)))
        self.m2_prep, self.m2_models = c2.load_final(d)
        self.integ = dict(e7b_reference_ok=True, frozen_hashes_ok=True, m2_ok=True, m0_46_ok=True)

    def t1ctx(self, crash=None):
        X = dict(inventory=self.inventory, split=self.split, sig=W.E4, e5=W.E5)
        return t1.Context(self.tmp, X, self.roster, self.M, self.M0, self.m2_prep, self.m2_models, e6, e7a, am, MAN,
                          self.verify, lambda: dict(self.integ), download=self.download, crash=crash)

    def t1approval(self, name='t1.json', **over):
        rec = dict(protocol=t1.PROTOCOL_ID, manifest_sha256=MAN, approval_ref='T1-TEST', approved_by='tester',
                   date='2026-10-23', scope=list(t1.SCOPES))
        rec.update(over)
        p = self.tmp / name
        p.write_text(json.dumps(rec))
        return p


class TestApproval(unittest.TestCase):
    def test_e7b_approval_and_bad_records_refused_before_access(self):
        with tempfile.TemporaryDirectory() as d:
            w = T1World(d)
            ctx = w.t1ctx()
            e7b_like = w.approval('e7b.json', manifest_sha256=MAN)            # protocol H-EVAL-E7B-v1.2
            for bad in (None, e7b_like, w.t1approval('a.json', manifest_sha256='ab' * 32),
                        w.t1approval('b.json', scope=['acquire']), w.t1approval('c.json', approval_ref=' ')):
                with self.assertRaises(t1.ApprovalRefused):
                    t1.execute(ctx, bad)
            with self.assertRaises(t1.ApprovalRefused):
                t1.acquire(ctx, e7b_like)
            self.assertEqual(w.downloads, [])
            self.assertEqual(ctx.extract_calls, 0)


class TestRosterAndBoundary(unittest.TestCase):
    def test_real_roster_metadata_only(self):
        X = e6.load_inputs(REPO)
        r = t1.build_roster(X)
        self.assertEqual((r['n_primary'], r['n_sensitivity']), (15, 15))
        for e in r['recordings']:
            self.assertEqual(X['split'][e['subject_key']], t1.SPLIT)
            self.assertEqual(set(e), {'subject_key', 'recording', 'night', 'psg_path', 'bytes', 'official_sha256', 'role'})

    def test_validation_subject_refused(self):
        with tempfile.TemporaryDirectory() as d:
            w = T1World(d)
            w.roster.insert(0, dict(subject_key='SC488', recording='SC4881', night=1, psg_path='sleep-cassette/SC4881E0-PSG.edf',
                                    bytes=1, official_sha256='0' * 64, role='PRIMARY'))
            with self.assertRaises(base.E7BStop):
                t1.acquire(w.t1ctx(), w.t1approval())
            self.assertEqual(w.downloads, [])


class TestExecute(unittest.TestCase):
    def test_incomplete_and_invalid_no_marker(self):
        with tempfile.TemporaryDirectory() as d:
            w = T1World(d)
            w.install()
            ctx = w.t1ctx()
            (w.root / w.roster[0]['psg_path']).unlink()
            self.assertEqual(t1.execute(ctx, w.t1approval())['state'], 'INCOMPLETE')
            (w.root / w.roster[0]['psg_path']).write_bytes(b'\0' * w.roster[0]['bytes'])
            self.assertEqual(t1.execute(ctx, w.t1approval())['state'], 'INVALID_DATA')
            self.assertFalse((ctx.local / t1.MARKER_NAME).exists())
            self.assertEqual(ctx.extract_calls, 0)

    def test_end_to_end_hierarchy_and_one_shot(self):
        with tempfile.TemporaryDirectory() as d:
            w = T1World(d)
            w.install()
            ctx = w.t1ctx()
            r = t1.execute(ctx, w.t1approval())
            self.assertEqual((r['state'], r['sensitivity']), ('EXECUTED', 'SENSITIVITY_PUBLISHED'))
            c = t1.read_envelope(ctx.out / t1.CONF_NAME)['results']
            if c['H1']['outcome'] != 'GO':
                self.assertTrue(c['H2']['outcome'].startswith('NOT_TESTED'))
            for k in ('E7B_CONSTANT', 'E7B_EEG', 'M0_46', 'M2_46'):
                self.assertTrue(np.isfinite(c['models'][k]['logloss_ww']))
            self.assertIn('donor_raw_vs_constant', c['m2_negative_controls'])
            with self.assertRaises(base.E7BStop):
                t1.execute(ctx, w.t1approval())

    def test_crash_recovery_and_sensitivity_completion(self):
        with tempfile.TemporaryDirectory() as d:
            w = T1World(d)
            w.install()
            with self.assertRaises(t1.CrashInjected):
                t1.execute(w.t1ctx(crash_at(f'after_stage_fsync:{t1.CONF_NAME}')), w.t1approval())
            ctx = w.t1ctx()
            with self.assertRaises(base.E7BStop):
                t1.execute(ctx, w.t1approval())
            rec = w.tmp / 'rec.json'
            rec.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / t1.MARKER_NAME), authorised_by='sup')))
            self.assertEqual(t1.execute(ctx, w.t1approval(), rec)['mode'], 'recovery')
        with tempfile.TemporaryDirectory() as d:
            w = T1World(d)
            w.install()
            with self.assertRaises(t1.CrashInjected):
                t1.execute(w.t1ctx(crash_at('after_confirmatory_published')), w.t1approval())
            ctx = w.t1ctx()
            before = base.digest(ctx.out / t1.CONF_NAME)
            auth = w.tmp / 's.json'
            auth.write_text(json.dumps(dict(marker_sha256=base.digest(ctx.local / t1.MARKER_NAME),
                                            confirmatory_result_sha256=before, authorised_by='sup')))
            self.assertEqual(t1.sensitivity_complete(ctx, w.t1approval(), auth)['sensitivity'], 'SENSITIVITY_PUBLISHED')
            self.assertEqual(base.digest(ctx.out / t1.CONF_NAME), before)


class TestH2Rule(unittest.TestCase):
    def test_h2_outcomes(self):
        ok = dict(R=0.08, delta_ww_ci95=[0.001, 0.02])
        weak = dict(R=0.03, delta_ww_ci95=[0.001, 0.02])
        ci0 = dict(R=0.08, delta_ww_ci95=[-0.001, 0.02])
        self.assertEqual(t1.h2_outcome('GO', ok, True), 'SUPPORTED')
        self.assertEqual(t1.h2_outcome('GO', weak, True), 'NOT_SUPPORTED')
        self.assertEqual(t1.h2_outcome('GO', ci0, True), 'NOT_SUPPORTED')
        self.assertTrue(t1.h2_outcome('NO-GO', ok, True).startswith('NOT_TESTED'))
        self.assertEqual(t1.h2_outcome('GO', ok, False), 'INVALID')


if __name__ == '__main__':
    unittest.main()
