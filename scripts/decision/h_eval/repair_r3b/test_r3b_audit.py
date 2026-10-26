"""Synthetic tests for R3B (no held-out data)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('r3b', Path(__file__).resolve().parent / 'r3b_audit.py')
r3b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3b)


class TestReferences(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.t = rng.uniform(30, 3000, 2000)
        self.s1 = (rng.random(2000) < 0.3).astype(float)
        p = 1 / (1 + np.exp(-(-2.5 + 2.0 * self.s1 + 0.3 * np.log(self.t / 300))))
        self.y = (rng.random(2000) < p).astype(float)
        self.person = np.array([f'P{i % 20:02d}' for i in range(2000)])

    def test_A_is_laplace_state_prevalence(self):
        A = r3b.fit_A(self.y, self.s1)
        m = self.s1 == 1
        self.assertAlmostEqual(A['p_stage1'], (self.y[m].sum() + 1) / (m.sum() + 2))
        self.assertTrue(np.array_equal(r3b.predict_A(A, self.s1) == A['p_stage1'], m))

    def test_B_bounds_and_constant_extrapolation(self):
        ref = r3b.fit_B(self.t, self.s1, self.y, (3, 1.0, True))
        far = r3b.predict_B(ref, np.array([1e5, 1e6]), np.array([1.0, 1.0]))
        edge = r3b.predict_B(ref, np.exp(np.array([ref['logt_hi']])), np.array([1.0]))
        self.assertAlmostEqual(far[0], edge[0], places=12)
        self.assertAlmostEqual(far[1], edge[0], places=12)
        low = r3b.predict_B(ref, np.array([1.0]), np.array([0.0]))
        self.assertTrue(0 < low[0] < 1)

    def test_B_stored_params_reproduce_fit(self):
        from sklearn.linear_model import LogisticRegression
        ref = r3b.fit_B(self.t, self.s1, self.y, (5, 10.0, False))
        X = r3b.design(self.t, self.s1, ref['logt_lo'], ref['logt_hi'], 5, False)
        clf = LogisticRegression(C=10.0, max_iter=5000).fit(X, self.y)
        self.assertTrue(np.allclose(clf.predict_proba(X)[:, 1], r3b.predict_B(ref, self.t, self.s1), atol=1e-9))

    def test_spline_partition_of_unity(self):
        B = r3b.spline_basis(np.log(self.t), np.log(30), np.log(3000), 5)
        self.assertTrue(np.allclose(B.sum(1), 1.0))

    def test_selection_deterministic(self):
        a = r3b.select_B(self.t, self.s1, self.y, self.person)
        b = r3b.select_B(self.t, self.s1, self.y, self.person)
        self.assertEqual(a[0], b[0])


class TestStatistics(unittest.TestCase):
    def test_paired_identity_and_values(self):
        y = np.array([0, 1, 0, 1, 0, 0.0] * 5)
        person = np.repeat([f'P{i}' for i in range(10)], 3)
        ref, cand = np.full(30, 0.3), np.where(y == 1, 0.6, 0.1)
        I = np.random.default_rng(0).integers(0, 10, size=(200, 10))
        out = r3b.paired(ref, ref, y, person, I)
        self.assertEqual(out['delta_ww'], 0.0)
        out = r3b.paired(ref, cand, y, person, I)
        self.assertAlmostEqual(out['delta_ww'], ((ref - y) ** 2).mean() - ((cand - y) ** 2).mean(), places=14)
        self.assertEqual(out['interval_label'], 'R3B re-bootstrap')

    def test_ci_suppressed_small(self):
        y = np.array([0, 1.0] * 6)
        person = np.repeat(['a', 'b', 'c'], 4)
        out = r3b.paired(np.full(12, .5), np.full(12, .4), y, person, None)
        self.assertEqual(out['ci'], 'not_reported_lt_8_persons')

    def test_matched_auroc_cross_person_only_and_constant_is_half(self):
        y = np.array([1, 0, 1, 0], float)
        person = np.array(['a', 'a', 'b', 'b'])
        k = np.array([1, 1, 1, 1])
        pid = np.searchsorted(['a', 'b'], person)
        pi, ni = r3b.matched_pairs(k, pid, y.astype(int))
        self.assertTrue(all(pid[i] != pid[j] for i, j in zip(pi, ni)))
        out = r3b.matched_auroc(np.full(4, .2), y, person, k, np.random.default_rng(0))
        self.assertEqual(out['auroc'], 0.5)

    def test_calibration_slope_undefined_for_constant(self):
        cil, slope, why = r3b.calib(np.full(10, .3), np.array([0, 1.0] * 5))
        self.assertIsNone(slope)
        self.assertIn('prediction constant', why)

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'
            r3b.write_new(p, b'1')
            with self.assertRaises(FileExistsError):
                r3b.write_new(p, b'2')

    def test_contrast_list_complete(self):
        ids = [c[0] for c in r3b.CONTRASTS]
        for need in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '3s', '4s'):
            self.assertIn(need, ids)

    def test_no_eeg_or_hypnogram_access_in_code(self):
        src = (Path(__file__).resolve().parent / 'r3b_audit.py').read_text()
        for bad in ('read_window', 'decode_edfplus', 'torch', 'Hypnogram', 'PSG.edf'):
            self.assertNotIn(bad, src)


if __name__ == '__main__':
    unittest.main()
