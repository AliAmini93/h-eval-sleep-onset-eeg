"""Gate E6 tests: synthetic unit tests plus one integration test on the six local, SHA-verified E5 EDFs.

No network, no protected data: the integration test reads only the six SC_TRAIN EDFs already used by E5."""
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
spec = importlib.util.spec_from_file_location('gate_e6_test', HERE / 'gate_e6.py')
e6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e6)
e5 = e6.import_py(REPO / 'scripts/decision/h_eval/gate_e5/gate_e5.py', 'e5_for_e6_tests')
e4 = e6.import_py(REPO / 'scripts/decision/h_eval/gate_e4/v3/gate_e4.py', 'e4_for_e6_tests')


def write_edf(path, ch0, ch1, rate=100):
    """Minimal EDF readable by the E4 reader: two EEG channels, 1 s records."""
    n_rec = len(ch0) // rate
    ns = 2
    hdr = 256 * (ns + 1)

    def f(s, w):
        return s.encode('ascii').ljust(w)[:w]
    fixed = f('0', 8) + f('', 80) + f('', 80) + f('01.01.01', 8) + f('00.00.00', 8) + f(str(hdr), 8) + f('', 44) \
        + f(str(n_rec), 8) + f('1', 8) + f(str(ns), 4)
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    fields = [['EEG Fpz-Cz', 'EEG Pz-Oz'], [''] * 2, ['uV'] * 2, ['-200'] * 2, ['200'] * 2, ['-2048'] * 2,
              ['2047'] * 2, [''] * 2, [str(rate)] * 2, [''] * 2]
    sig = b''.join(f(v, w) for vals, w in zip(fields, widths) for v in vals)
    data = b''
    for r in range(n_rec):
        for ch in (ch0, ch1):
            data += struct.pack('<%dh' % rate, *ch[r * rate:(r + 1) * rate])
    Path(path).write_bytes(fixed + sig + data)


def synth_rows(n_persons=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_persons):
        n = int(rng.integers(15, 60))
        for k in range(n):
            t = 30. * (k + 1)
            ttn = 30. * (n - k)
            rows.append(dict(recording=f'R{j}', subject_key=f'S{j:02d}', origin='E6_HASH' if j >= 2 else 'E5_RETAINED',
                             elapsed_s=t, landmark_end_s=t, time_to_n2_s=ttn, anchor=bool(ttn <= 1200),
                             y=int(ttn <= 300), features=list(rng.normal(size=10) + (ttn <= 300) * 0.8)))
    return rows


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.split = {f'SC4{i:02d}': 'SC_TRAIN' for i in range(20)}
        self.split['SC499'] = 'SC_VALIDATION'
        inv, stage, sums = {}, {}, {}
        for i in list(range(20)) + [99]:
            for night in (2, 1):
                rid = f'SC4{i:02d}{night}'
                psg = f'sleep-cassette/SC4{i:02d}{night}E0-PSG.edf'
                inv[rid] = dict(recording=rid, subject_key=f'SC4{i:02d}', night=str(night), psg_path=psg,
                                psg_size_bytes=str(50_000_000 + i), status='ELIGIBLE_ORIGIN_INDEPENDENT',
                                study='sleep-cassette', latency_s=str(100 * i + night))
                stage[rid] = 'W' if i != 7 else '1'
                sums[psg] = 'ab' * 32
        self.inv, self.stage, self.sums = inv, stage, sums

    def select(self, inv=None):
        return e6.select_subjects(inv or self.inv, self.split, self.stage, self.sums, ['SC4001', 'SC4011'])

    def test_deterministic_lowest_night_and_exclusions(self):
        _, sel, cand, total, _ = self.select()
        self.assertEqual(len(sel), 12)
        self.assertTrue(all(c['night'] == 1 for c in sel))
        keys = {c['subject_key'] for c in cand}
        self.assertNotIn('SC499', keys)          # validation subject never a candidate
        self.assertNotIn('SC407', keys)          # not W at LightsOff
        self.assertNotIn('SC400', keys)          # retained
        self.assertEqual([c['hash_key'] for c in cand], sorted(c['hash_key'] for c in cand))
        self.assertEqual(sel, cand[:12])

    def test_selection_is_outcome_independent(self):
        inv2 = {k: dict(v, latency_s='1', landmark_n='999', first_n2_after_off_s='1') for k, v in self.inv.items()}
        a = [c['recording'] for c in self.select()[1]]
        b = [c['recording'] for c in self.select(inv2)[1]]
        self.assertEqual(a, b)

    def test_budget_prefix_rule(self):
        old = e6.BUDGET_BYTES
        try:
            e6.BUDGET_BYTES = 160_000_100
            _, sel, cand, total, reason = self.select()
            self.assertEqual(len(sel), 3)
            self.assertEqual(sel, cand[:3])
            self.assertTrue(reason.startswith('budget_prefix_stop'))
            self.assertLessEqual(total, e6.BUDGET_BYTES)
        finally:
            e6.BUDGET_BYTES = old

    def test_guard_refuses_non_train_signal(self):
        with self.assertRaises(e6.CriticalStop):
            e6.assert_train_signal('SC499', 'sleep-cassette/SC4991E0-PSG.edf', self.split)
        with self.assertRaises(e6.CriticalStop):
            e6.assert_train_signal('SC401', 'sleep-cassette/SC4021E0-PSG.edf', self.split)
        e6.assert_train_signal('SC401', 'sleep-cassette/SC4011E0-PSG.edf', self.split)


class TestEstimator(unittest.TestCase):
    def test_numpy_logistic_matches_e5(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + rng.normal(size=60) > 0).astype(int)
        w_np = e6.logistic_fit_np(X, y)
        w_e5 = e5.logistic_fit(X.tolist(), y.tolist())
        self.assertTrue(np.allclose(w_np, w_e5, atol=1e-10, rtol=0))
        p_np = e6.logistic_predict_np(w_np, X)
        p_e5 = [e5.predict_logit(w_e5, row) for row in X.tolist()]
        self.assertTrue(np.allclose(p_np, p_e5, atol=1e-12))

    def test_sigmoid_extremes(self):
        s = e6.sigmoid_np(np.array([-1e4, -61, 0, 61, 1e4]))
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertAlmostEqual(s[2], .5)

    def test_standardisation_matches_e5_and_is_train_only(self):
        tr = np.array([[2.] * 10, [4.] * 10])
        ho = np.array([[10.] * 10])
        mu, sd = e6.zfit(tr)
        self.assertEqual(e6.zapply(ho, mu, sd).tolist(), [[5.] * 10])
        t, h = e5.preprocess([{'features': [2.] * 10}, {'features': [4.] * 10}], [{'features': [10.] * 10}])
        self.assertEqual(h[0][1], e6.zapply(ho, mu, sd)[0].tolist())

    def test_time_features(self):
        self.assertEqual(e6.bin_idx(300), 1)
        self.assertEqual(e6.time_onehot(7300).tolist(), [0.] * 6 + [1.])
        self.assertTrue(np.allclose(e6.time_cont_raw(3600), [math.log1p(60), 1.]))


class TestLOPO(unittest.TestCase):
    def test_no_person_leakage_and_complete_predictions(self):
        rows = synth_rows()
        pred, audit = e6.lopo_predictions(rows)
        for a in audit:
            self.assertNotIn(a['held'], a['train_persons'])
        for ts in e6.SUPPORTS:
            for m in e6.MODELS:
                self.assertFalse(np.isnan(pred[ts][m]).any())
        anchored_train = [a for a in audit if a['train_support'] == 'ANCHOR']
        full_train = [a for a in audit if a['train_support'] == 'FULL']
        self.assertTrue(all(x['n_train'] < y['n_train'] for x, y in zip(anchored_train, full_train)))

    def test_held_out_prediction_unaffected_by_own_labels(self):
        rows = synth_rows(seed=4)
        pred, _ = e6.lopo_predictions(rows)
        rows2 = [dict(r, y=1 - r['y']) if r['subject_key'] == 'S00' else r for r in rows]
        pred2, _ = e6.lopo_predictions(rows2)
        k = np.array([r['subject_key'] == 'S00' for r in rows])
        for m in e6.MODELS:
            self.assertTrue(np.array_equal(pred['FULL'][m][k], pred2['FULL'][m][k]))

    def test_matched_support_and_tables(self):
        rows = synth_rows()
        pred, _ = e6.lopo_predictions(rows)
        pt = e6.person_table(rows, pred)
        agg = e6.aggregate(pt)
        self.assertEqual(len(pt), 6 * 4 * len(e6.MODELS))
        for es in e6.SUPPORTS:
            n = {a['total_eval_windows'] for a in agg if a['eval_support'] == es}
            self.assertEqual(len(n), 1)          # every model scored on exactly the same windows


class TestBootstrapAndDecision(unittest.TestCase):
    def setUp(self):
        self.ptab = []
        for i, (dm, n) in enumerate([(.02, 10), (.01, 20), (-.01, 30), (.03, 40)]):
            for m, b in (('A', .2), ('B', .2 - dm)):
                self.ptab.append(dict(person=f'P{i}', train_support='FULL', eval_support='FULL', model=m,
                                      n_windows=n, brier=b, brier_sum=b * n, logloss=b, logloss_sum=b * n))

    def test_cluster_bootstrap_matches_brute_force(self):
        B = e6.Bootstrap(['P0', 'P1', 'P2', 'P3'], n=200, seed=1)
        c = B.contrast(self.ptab, 'FULL', 'FULL', 'A', 'B')
        d = np.array([.02, .01, -.01, .03])
        n = np.array([10, 20, 30, 40.])
        se = [d[r].mean() for r in B.I]
        ww = [(d[r] * n[r]).sum() / n[r].sum() for r in B.I]
        self.assertTrue(np.allclose(c['subject_equal']['ci95'], np.percentile(se, [2.5, 97.5])))
        self.assertTrue(np.allclose(c['window_weighted']['ci95'], np.percentile(ww, [2.5, 97.5])))
        self.assertAlmostEqual(c['subject_equal']['delta'], d.mean())
        self.assertAlmostEqual(c['window_weighted']['delta'], (d * n).sum() / n.sum())
        self.assertEqual(c['persons_favouring_candidate'], 3)
        self.assertEqual(c['leave_one_person_out']['sign_flips'], 0)

    def test_duplicate_cluster_counts_each_copy(self):
        B = e6.Bootstrap(['P0', 'P1', 'P2', 'P3'], n=1, seed=0)
        B.I = np.array([[3, 3, 0, 0]])
        c = B.contrast(self.ptab, 'FULL', 'FULL', 'A', 'B')
        self.assertAlmostEqual(c['window_weighted']['ci95'][0], (2 * .03 * 40 + 2 * .02 * 10) / 100)

    def test_draw_matrix_predeclared(self):
        B = e6.Bootstrap([f'S{i}' for i in range(18)])
        self.assertTrue(np.array_equal(B.I, np.random.default_rng(20261023).integers(0, 18, size=(10000, 18))))

    def _c(self, d, lo, hi, flips=0):
        return dict(subject_equal=dict(delta=d, ci95=[lo, hi]), leave_one_person_out=dict(sign_flips=flips))

    def test_classification_rules(self):
        good = self._c(.01, .002, .02)
        self.assertEqual(e6.classify(good, self._c(.01, 0, 0), [good] * 4, .01)['label'],
                         'EEG_INCREMENT_SUPPORTED_EXPLORATORY')
        self.assertEqual(e6.classify(good, self._c(.01, 0, 0), [good] * 3 + [self._c(.001, -.01, .02)], .01)['label'],
                         'INCONCLUSIVE')
        self.assertEqual(e6.classify(good, self._c(.01, 0, 0), [good] * 4, -.01)['label'], 'INCONCLUSIVE')
        self.assertEqual(e6.classify(self._c(-.02, -.03, -.01), self._c(-.01, 0, 0), [good] * 4, -.01)['label'],
                         'EEG_INCREMENT_NOT_SUPPORTED')
        self.assertEqual(e6.classify(good, self._c(.01, 0, 0), [good] * 3 + [self._c(-.05, -.08, -.01)], .01)['label'],
                         'EEG_INCREMENT_NOT_SUPPORTED')

    def test_kendall_tau(self):
        self.assertEqual(e6.kendall_tau(['a', 'b', 'c'], ['a', 'b', 'c']), 1.)
        self.assertEqual(e6.kendall_tau(['a', 'b', 'c'], ['c', 'b', 'a']), -1.)


class TestCausalFeatures(unittest.TestCase):
    def test_features_use_only_pre_landmark_signal(self):
        rng = np.random.default_rng(0)
        n = 100 * 120
        ch0 = rng.integers(-500, 500, size=n).tolist()
        ch1 = rng.integers(-500, 500, size=n).tolist()
        t = 60.
        with tempfile.TemporaryDirectory() as d:
            def feats(c0, c1):
                p = Path(d) / f'x{len(list(Path(d).iterdir()))}.edf'
                write_edf(p, c0, c1)
                with p.open('rb') as f:
                    h = e4.edf_header(f, p.stat().st_size)
                    return [v for ch in h['eeg_channels'] for v in e5.channel_features(e4.read_window(f, h, ch, t - 30, t)[0])]
            base = feats(ch0, ch1)
            after = [v if i < int(t * 100) else 2000 for i, v in enumerate(ch0)]
            self.assertEqual(base, feats(after, ch1))                     # future samples changed: identical
            before = list(ch0)
            before[int(t * 100) - 1] = 2000
            self.assertNotEqual(base, feats(before, ch1))                 # last pre-landmark sample matters


class TestImmutable(unittest.TestCase):
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'o'
            self.assertEqual(e6.immut(p, b'a'), 'written')
            self.assertEqual(e6.immut(p, b'a'), 'identical_existing')
            with self.assertRaises(e6.CriticalStop):
                e6.immut(p, b'b')


class TestIntegrationE5Reproduction(unittest.TestCase):
    """Reads only the six SC_TRAIN EDFs already verified in E4/E5 (no network)."""

    def test_reproduces_e5(self):
        X = e6.load_inputs(REPO)
        recs = []
        for r in X['e4qc']['sampled_train_recordings']:
            p = REPO / 'data/raw/sleep_edfx_v1_0_0' / r['psg_path']
            if not p.exists():
                self.skipTest('E5 EDFs not present locally')
            recs.append(dict(subject_key=r['subject_key'], recording=r['recording'], psg_path=r['psg_path'],
                             origin='E5_RETAINED'))
        rows, _ = e6.extract(REPO, recs, X['inventory'], X['split'], X['sig'], X['e5'], lambda m: None)
        rep = e6.reproduce_e5(rows, X['e5_summary'])
        self.assertTrue(rep['passed'], json.dumps(rep))
        self.assertEqual(rep['n_landmarks'], 336)


if __name__ == '__main__':
    unittest.main()
