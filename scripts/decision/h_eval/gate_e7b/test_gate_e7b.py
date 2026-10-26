"""Gate E7B preparation tests. Synthetic unit tests, metadata-only roster checks, and post-freeze integrity checks.
No test reads, downloads or scores SC_VALIDATION EEG; acquire/execute are never called."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
spec = importlib.util.spec_from_file_location('gate_e7b_test', HERE / 'gate_e7b.py')
e7b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e7b)
e6, e7a, am = e7b.modules(REPO)


def synth_rows(n_persons=8, seed=0, signal=0.8):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_persons):
        n = int(rng.integers(8, 70))
        for k in range(n):
            ttn = 30. * (n - k)
            y = int(ttn <= 300)
            rows.append(dict(recording=f'R{j}', person=f'P{j:02d}', elapsed_s=30. * (k + 1), time_to_n2_s=ttn,
                             anchor=int(ttn <= 1200), y=y, features=list(rng.normal(size=10) + signal * y)))
    return rows


class TestRosterAndBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = e6.load_inputs(REPO)

    def test_roster_metadata_only_and_sizes(self):
        r = e7b.build_roster(self.X)
        self.assertEqual((r['n_primary'], r['n_sensitivity']), (15, 12))
        allowed = {'subject_key', 'recording', 'night', 'psg_path', 'bytes', 'official_sha256', 'role'}
        for e in r['recordings']:
            self.assertEqual(set(e), allowed)                  # no latency / label / outcome fields
            self.assertEqual(self.X['split'][e['subject_key']], 'SC_VALIDATION')
        prim = [e for e in r['recordings'] if e['role'] == 'PRIMARY']
        for p in prim:
            nights = [e['night'] for e in r['recordings'] if e['subject_key'] == p['subject_key']]
            self.assertEqual(p['night'], min(nights))
        self.assertEqual(r['primary_bytes'], 745_754_880)

    def test_boundary_guard(self):
        sp = self.X['split']
        with self.assertRaises(e7b.E7BStop):
            e7b.assert_signal_allowed('SC404', 'sleep-cassette/SC4041E0-PSG.edf', sp, 'SC_TRAIN')
        with self.assertRaises(e7b.E7BStop):
            e7b.assert_signal_allowed('SC404', 'sleep-cassette/SC4131E0-PSG.edf', sp, 'SC_VALIDATION')
        with self.assertRaises(e7b.E7BStop):
            e7b.assert_signal_allowed('SC404', 'sleep-cassette/SC4041E0-PSG.edf', sp, 'SC_VALIDATION', {'x'})
        with self.assertRaises(e7b.E7BStop):
            e7b.assert_signal_allowed('SC403', 'sleep-cassette/SC4031E0-PSG.edf', sp, 'SC_VALIDATION')   # protected test

    def test_extract_refuses_validation_before_any_file_access(self):
        rec = dict(subject_key='SC404', recording='SC4041', psg_path='sleep-cassette/SC4041E0-PSG.edf')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(e7b.E7BStop):
                e7b.extract_cohort(Path(d), [rec], self.X, 'SC_TRAIN')


class TestFrozenModels(unittest.TestCase):
    def test_fit_serialise_reload_byte_identical(self):
        rows = synth_rows()
        M = e7b.fit_frozen(rows, e6)
        M2 = json.loads(json.dumps(M))
        F, T, _ = e7b.rows_matrices(rows, e6)
        self.assertEqual(e7b.predictions_csv(rows, e7b.predict_frozen(M, F, T, e6)),
                         e7b.predictions_csv(rows, e7b.predict_frozen(M2, F, T, e6)))

    def test_fit_deterministic_and_reference_roundtrip(self):
        rows = synth_rows(seed=1)
        M = e7b.fit_frozen(rows, e6)
        M_b = e7b.fit_frozen(e7b.read_reference_inputs_from_bytes(e7b.reference_inputs_csv(rows)), e6)
        self.assertEqual(json.dumps(M, sort_keys=True), json.dumps(M_b, sort_keys=True))

    def test_drift_residuals_orthogonal_to_time_on_train(self):
        rows = synth_rows(seed=2)
        M = e7b.fit_frozen(rows, e6)
        F, T, _ = e7b.rows_matrices(rows, e6)
        R = F - np.column_stack([np.ones(len(T)), T]) @ np.array(M['drift']['beta'])
        self.assertTrue(np.allclose(np.column_stack([np.ones(len(T)), T]).T @ R, 0, atol=1e-7))


class TestDonor(unittest.TestCase):
    def test_deterministic_never_self_label_independent(self):
        rows = synth_rows(seed=3)
        a, _ = e7b.donor_plan(rows, 0)
        b, _ = e7b.donor_plan(rows, 0)
        self.assertEqual(a, b)
        self.assertTrue(all(rows[j]['person'] != rows[i]['person'] for i, j in enumerate(a)))
        flipped = [dict(r, y=1 - r['y'], time_to_n2_s=0.) for r in rows]
        self.assertEqual(e7b.donor_plan(flipped, 0)[0], a)
        self.assertNotEqual(e7b.donor_plan(rows, 1)[0], a)

    def test_fallback_to_nearest_stratum(self):
        rows = [dict(person='A', elapsed_s=30.), dict(person='A', elapsed_s=1000.), dict(person='B', elapsed_s=60.)]
        plan, info = e7b.donor_plan(rows, 0)
        self.assertEqual(plan, [2, 2, 0])
        self.assertEqual(info, dict(fallback_windows=1, max_stratum_distance=3))


class TestScoringAndDecision(unittest.TestCase):
    def test_contrast_matches_brute_force_with_duplicates(self):
        rows = synth_rows(n_persons=5, seed=4)
        P = {m: np.clip(np.random.default_rng(5).uniform(size=len(rows)), 1e-6, 1) for m in e7b.MODELS}
        S = e7b.Scorer(rows, P, e7a, am, n_boot=50)
        lr, lc = S.brier(P['CONSTANT']), S.brier(P['EEG_LOGIT'])
        c = S.contrast(lr, lc)
        pid = S.pidx
        ww = [np.concatenate([lr[pid == i] for i in r]).mean() - np.concatenate([lc[pid == i] for i in r]).mean() for r in S.I]
        self.assertTrue(np.allclose(c['delta_ww_ci95'], np.percentile(ww, [2.5, 97.5])))
        self.assertAlmostEqual(c['delta_ww'], lr.mean() - lc.mean())
        self.assertAlmostEqual(c['R'], (lr.mean() - lc.mean()) / lr.mean())

    def _prim(self, d, lo, hi, R, rlo, rhi):
        return dict(delta_ww=d, delta_ww_ci95=[lo, hi], R=R, R_ci95=[rlo, rhi])

    def test_decision_rules(self):
        s4ok, s4bad = dict(ci95=[.6, .9]), dict(ci95=[.4, .9])
        nc2ok, nc2bad = dict(delta_ww_ci95=[-.01, .01]), dict(delta_ww_ci95=[.001, .02])
        go = e7b.decide(self._prim(.01, .002, .02, .08, .01, .15), .01, s4ok, True, nc2ok, True)
        self.assertEqual(go['outcome'], 'GO')
        self.assertFalse(go['flags']['PRACTICALLY_ESTABLISHED'])
        stat_only = e7b.decide(self._prim(.002, .0005, .004, .02, .005, .04), .01, s4ok, True, nc2ok, True)
        self.assertEqual(stat_only['outcome'], 'NO-GO')        # statistically positive, practically excluded
        self.assertTrue(stat_only['flags']['STATISTICALLY_POSITIVE'])
        self.assertEqual(e7b.decide(self._prim(.01, -.002, .02, .08, -.01, .15), .01, s4ok, True, nc2ok, True)['outcome'],
                         'INCONCLUSIVE')
        self.assertEqual(e7b.decide(self._prim(-.01, -.02, .0, -.05, -.1, .0), -.01, s4ok, True, nc2ok, True)['outcome'],
                         'NO-GO')
        self.assertEqual(e7b.decide(self._prim(.01, .002, .02, .08, .01, .15), .01, s4bad, True, nc2ok, True)['outcome'],
                         'INCONCLUSIVE')
        self.assertEqual(e7b.decide(self._prim(.01, .002, .02, .08, .01, .15), .01, s4ok, True, nc2bad, True)['outcome'],
                         'INCONCLUSIVE')
        self.assertEqual(e7b.decide(self._prim(.01, .002, .02, .08, .01, .15), .01, s4ok, False, nc2ok, True)['outcome'],
                         'INVALID')
        self.assertEqual(e7b.decide(self._prim(.01, .002, .02, .08, .01, .15), .01, s4ok, True, nc2ok, False)['outcome'],
                         'INVALID')

    def test_score_cohort_end_to_end_synthetic(self):
        rows = synth_rows(n_persons=9, seed=6, signal=1.2)
        M = e7b.fit_frozen(rows, e6)
        F, T, _ = e7b.rows_matrices(rows, e6)
        P = e7b.predict_frozen(M, F, T, e6)
        res, nc1, nc2, s4 = e7b.score_cohort(rows, P, M, e6, e7a, am)
        self.assertTrue(nc1)
        self.assertEqual(res['n_persons'], 9)
        self.assertEqual(len(res['NC2_donor']['donor_fallback']), e7b.DRAWS)


class TestCompleteness(unittest.TestCase):
    def test_missing_files_are_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            recs = [dict(recording='SC4041', psg_path='sleep-cassette/SC4041E0-PSG.edf', bytes=3, official_sha256='0' * 64)]
            self.assertEqual(e7b.completeness(Path(d), recs), {'SC4041': 'MISSING_OR_UNVERIFIED'})
            p = Path(d) / 'data/raw/sleep_edfx_v1_0_0/sleep-cassette'
            p.mkdir(parents=True)
            (p / 'SC4041E0-PSG.edf').write_bytes(b'abc')
            self.assertEqual(e7b.completeness(Path(d), recs), {'SC4041': 'MISSING_OR_UNVERIFIED'})   # wrong SHA
            recs[0]['official_sha256'] = e7b.digest(p / 'SC4041E0-PSG.edf')
            self.assertEqual(e7b.completeness(Path(d), recs), {'SC4041': 'VERIFIED'})


class TestFrozenArtifactsIfPresent(unittest.TestCase):
    """Post-freeze checks on the real frozen artifacts (TRAIN only)."""

    def test_reference_reproduction_and_manifest(self):
        out = REPO / e7b.OUT_REL
        if not (out / 'frozen_models_v1.json').exists():
            self.skipTest('frozen artifacts not yet prepared')
        rep = e7b.reproduce_reference(REPO, e6)[0]
        self.assertTrue(rep['passed'])
        self.assertEqual(rep['n_rows'], 1271)
        if (out / e7b.MANIFEST_NAME).exists():
            e7b.verify_manifest(REPO, out / e7b.MANIFEST_NAME)


class TestOneShotGate(unittest.TestCase):
    def test_marker_exclusive_and_recovery_rules(self):
        with tempfile.TemporaryDirectory() as d:
            local, out = Path(d) / 'local', Path(d) / 'out'
            self.assertEqual(e7b.one_shot_gate(local, out, 'REF', 'abc'), 'new')
            with self.assertRaises(e7b.E7BStop):
                e7b.one_shot_gate(local, out, 'REF', 'abc')                       # second run refused
            bad = Path(d) / 'bad.json'
            bad.write_text(json.dumps(dict(marker_sha256='0' * 64, authorised_by='x')))
            with self.assertRaises(e7b.E7BStop):
                e7b.one_shot_gate(local, out, 'REF', 'abc', bad)                  # wrong marker hash
            good = Path(d) / 'good.json'
            good.write_text(json.dumps(dict(marker_sha256=e7b.digest(local / 'E7B_EXECUTION_MARKER.json'),
                                            authorised_by='supervisor')))
            self.assertEqual(e7b.one_shot_gate(local, out, 'REF', 'abc', good), 'recovery')
            out.mkdir()
            (out / 'gate_e7b_primary_result_v1.json').write_text('{}')
            with self.assertRaises(e7b.E7BStop):
                e7b.one_shot_gate(local, out, 'REF', 'abc', good)                 # never after a primary result


if __name__ == '__main__':
    unittest.main()
