"""Gate E7A tests: synthetic unit tests and one integration test that re-verifies the frozen E6 outputs."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
spec = importlib.util.spec_from_file_location('gate_e7a_test', HERE / 'gate_e7a.py')
e7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e7)


def synth_diag(n_persons=9, seed=0):
    rng = np.random.default_rng(seed)
    rows, pred = [], {ts: {m: [] for m in e7.MODELS4} for ts in ('FULL', 'ANCHOR')}
    for j in range(n_persons):
        n = int(rng.integers(9, 80))
        for k in range(n):
            ttn = 30. * (n - k)
            rows.append(dict(recording=f'R{j}', subject_key=f'P{j}', origin='E6_HASH', elapsed_s=30. * (k + 1),
                             time_to_n2_s=ttn, anchor=ttn <= 1200, y=int(ttn <= 300)))
            for ts in pred:
                for m in e7.MODELS4:
                    pred[ts][m].append(float(np.clip(rng.uniform(.05, .6) + .2 * (ttn <= 300) * (m != 'CONSTANT'), 1e-6, 1)))
    pred = {ts: {m: np.array(v) for m, v in d.items()} for ts, d in pred.items()}
    return rows, pred


class TestDecomposition(unittest.TestCase):
    def test_level_plus_within_equals_brier(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(2, 60))
            p, y = rng.uniform(size=n), (rng.uniform(size=n) < .3).astype(int)
            L, W = e7.level_within(p, y)
            self.assertAlmostEqual(L + W, ((p - y) ** 2).mean(), places=12)

    def test_constant_predictor_has_zero_within(self):
        L, W = e7.level_within(np.full(10, .3), np.array([0] * 7 + [1] * 3))
        self.assertAlmostEqual(W, 0.)
        self.assertAlmostEqual(L, .21)

    def test_aligned_variation_has_negative_within(self):
        y = np.array([0] * 8 + [1] * 2)
        self.assertLess(e7.level_within(.2 + .5 * y, y)[1], 0)
        self.assertGreater(e7.level_within(.7 - .5 * y, y)[1], 0)

    def test_covariance_identity_and_tables(self):
        rows, pred = synth_diag()
        diag = e7.person_diagnostics(rows, pred)
        dec = e7.decomposition(diag)            # raises if the identity fails
        c = dec['train_FULL__eval_FULL']['C1']
        self.assertAlmostEqual(c['se_minus_ww_gap'], c['brier']['subject_equal'] - c['brier']['window_weighted'], places=12)
        self.assertAlmostEqual(sum(c['gap_contributions'].values()), c['se_minus_ww_gap'], places=12)
        self.assertAlmostEqual(c['level']['subject_equal'] + c['within']['subject_equal'], c['brier']['subject_equal'], places=12)
        st = e7.strata(diag)['train_FULL__eval_FULL']
        self.assertEqual(sum(len(g['persons']) for g in st.values()), 9)
        inf = e7.influence(diag)['train_FULL__eval_FULL']['C1']
        self.assertEqual(inf['all']['n_persons'], 9)


class TestDiscrimination(unittest.TestCase):
    def test_auroc_basic_and_undefined(self):
        self.assertEqual(e7.auroc([.1, .2, .9], [0, 0, 1]), 1.)
        self.assertEqual(e7.auroc([.9, .2, .1], [0, 0, 1]), 0.)
        self.assertEqual(e7.auroc([.5, .5], [0, 1]), .5)
        self.assertIsNone(e7.auroc([.1, .2], [1, 1]))

    def test_rank_formula_matches_auroc(self):
        rng = np.random.default_rng(2)
        p = np.round(rng.uniform(size=40), 1)
        y = (rng.uniform(size=40) < .3).astype(int)
        r = e7.tie_ranks(p)
        npos = y.sum()
        self.assertAlmostEqual((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (40 - npos)), e7.auroc(p, y), places=12)

    def test_monotone_time_predictor_is_structurally_perfect(self):
        rows, _ = synth_diag()
        for j in range(9):
            rr = [r for r in rows if r['subject_key'] == f'P{j}']
            a = e7.auroc([r['elapsed_s'] for r in rr], [r['y'] for r in rr])
            self.assertTrue(a is None or a == 1.)

    def test_single_class_reported_not_imputed(self):
        rows, pred = synth_diag(seed=3)
        table, summ = e7.within_discrimination(rows, pred)
        for t in table:
            if t['n_negative'] == 0:
                self.assertEqual(t['auroc'], '')
                self.assertTrue(t['status'].startswith('undefined'))
        s = summ['train_FULL__eval_FULL']['EEG_LOGIT']
        self.assertEqual(s['n_persons_defined'] + s['n_persons_undefined'], 9)


class TestClassification(unittest.TestCase):
    def _dec(self, lv, wi):
        return {c: dict(level=dict(subject_equal_ci95=lv), within=dict(subject_equal_ci95=wi)) for c in ('C1', 'C2')}

    def test_rules(self):
        self.assertEqual(e7.classify_source(self._dec([.01, .02], [-.01, .01]))['C1']['label'], 'primarily between-person level')
        self.assertEqual(e7.classify_source(self._dec([-.01, .02], [.01, .02]))['C1']['label'], 'primarily within-person')
        self.assertEqual(e7.classify_source(self._dec([.01, .02], [.01, .02]))['C1']['label'], 'both')
        self.assertEqual(e7.classify_source(self._dec([-.01, .02], [-.02, .01]))['C1']['label'], 'neither established')


class TestIntegrationE6Verification(unittest.TestCase):
    """Reads only frozen E6 outputs (no EEG)."""

    def test_e6_outputs_verify_and_recompute(self):
        X = e7.load_and_verify(REPO)
        rows, pred, ptab, rep = e7.recompute_and_compare(X)
        self.assertTrue(rep['passed'])
        self.assertEqual(rep['bootstrap_contrasts_recomputed'], 72)
        self.assertEqual(len(rows), 1271)


if __name__ == '__main__':
    unittest.main()


_spec2 = importlib.util.spec_from_file_location('gate_e7a_amend_test', HERE / 'gate_e7a_amendment01.py')
am = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(am)


class TestAmendment01(unittest.TestCase):
    def setUp(self):
        self.p = np.array([.9, .1, .8, .2, .5, .5])
        self.y = np.array([1, 0, 1, 0, 1, 0])
        self.person = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        self.stratum = np.array([0, 0, 0, 0, 1, 1])
        self.persons = ['A', 'B', 'C']

    def test_same_person_pairs_excluded_and_brute_force(self):
        win, pairs, used = am.pair_matrices(self.p, self.y, self.person, self.stratum, self.persons)
        self.assertEqual(pairs[0, 0] + pairs[1, 1] + pairs[2, 2], 0)
        self.assertEqual(pairs.sum(), 2)          # stratum 0: A+ vs B-, B+ vs A-; stratum 1 only within C
        self.assertEqual(used, 1)
        self.assertEqual(am.weighted_auc(win, pairs, np.ones(3)), 1.)

    def test_duplicate_weighting(self):
        win, pairs, _ = am.pair_matrices(np.array([.9, .1, .1, .9]), np.array([1, 0, 1, 0]),
                                         np.array(['A', 'A', 'B', 'B']), np.zeros(4), ['A', 'B'])
        self.assertEqual(am.weighted_auc(win, pairs, np.ones(2)), .5)
        self.assertEqual(am.weighted_auc(win, pairs, np.array([2., 1.])), .5)
        self.assertTrue(np.isnan(am.weighted_auc(win, pairs, np.array([2., 0.]))))
