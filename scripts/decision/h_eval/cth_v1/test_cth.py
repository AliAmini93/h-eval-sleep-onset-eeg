"""CTH v1 tests (synthetic; no data access)."""
import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cth_exp_test', HERE / 'cth_experiment.py')
cx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cx)
spec2 = importlib.util.spec_from_file_location('cth_data_test', HERE / 'cth_data.py')
cd = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cd)


def synth(n_rec=12, seed=0):
    rng = np.random.default_rng(seed)
    d = {k: [] for k in ('tau', 'y', 'elapsed', 'rec_idx', 'person', 'recording', 'night')}
    for r in range(n_rec):
        n = int(rng.integers(5, 40))
        for k in range(n):
            tau = 30. * (n - k)
            d['tau'].append(tau); d['y'].append(int(tau <= 300)); d['elapsed'].append(30. * (k + 1))
            d['rec_idx'].append(r); d['person'].append(f'P{r // 2:02d}'); d['recording'].append(f'R{r}'); d['night'].append(r % 2 + 1)
    d = {k: np.array(v) for k, v in d.items()}
    N = len(d['y'])
    d['raw'] = rng.normal(size=(N, 2, 3000)).astype(np.float32) * 20
    d['rich'] = rng.normal(size=(N, 18))
    d['e5'] = rng.normal(size=(N, 10))
    hist = np.full((N, 10), -1)
    for i in range(N):
        for k in range(10):
            j = i - k
            if j >= 0 and d['rec_idx'][j] == d['rec_idx'][i]:
                hist[i, 9 - k] = j
            else:
                break
    d['hist'] = hist
    return d


class TestHazard(unittest.TestCase):
    def test_event_interval(self):
        np.testing.assert_array_equal(cx.event_interval([30, 31, 300, 301, 600]), [1, 2, 10, 11, 11])

    def test_loss_equals_discrete_time_likelihood(self):
        rng = np.random.default_rng(0)
        z = torch.tensor(rng.normal(size=(6, 10)), dtype=torch.float64)
        K = torch.tensor([1, 3, 10, 11, 11, 5])
        h = torch.sigmoid(z).numpy()
        manual = []
        for i, k in enumerate(K.tolist()):
            if k <= 10:
                manual.append(-(np.log(1 - h[i, :k - 1]).sum() + np.log(h[i, k - 1])))
            else:
                manual.append(-np.log(1 - h[i]).sum())
        self.assertAlmostEqual(float(cx.hazard_loss(z, K)), float(np.mean(manual)), places=10)

    def test_p300_is_cumulative_incidence(self):
        z = torch.tensor([[0.3] * 10])
        h = torch.sigmoid(z)
        self.assertAlmostEqual(float(cx.p300_from(z, 'hazard')), float(1 - torch.prod(1 - h)), places=6)

    def test_horizon_label_consistency(self):
        tau = np.array([30, 150, 300, 330, 1000])
        K = cx.event_interval(tau)
        self.assertTrue(np.array_equal(K <= 10, tau <= 300))


class TestCausality(unittest.TestCase):
    def test_history_indices_past_same_recording_and_masked(self):
        d = synth()
        for i in range(len(d['y'])):
            idx = d['hist'][i][d['hist'][i] >= 0]
            self.assertTrue(np.all(idx <= i))
            self.assertTrue(np.all(d['rec_idx'][idx] == d['rec_idx'][i]))
            self.assertEqual(d['hist'][i, 9], i)
            first = d['elapsed'][i] == 30.
            if first:
                self.assertTrue(np.all(d['hist'][i, :9] == -1))

    def test_features_independent_of_future_windows_and_labels(self):
        d = synth()
        tr = np.arange(len(d['y']))
        T1 = cx.Tensors(d, tr)
        f1 = cx.m1_features(d)
        d2 = {k: v.copy() for k, v in d.items()}
        i = 20
        later = np.where((d['rec_idx'] == d['rec_idx'][i]) & (np.arange(len(d['y'])) > i))[0]
        d2['raw'][later] += 999
        d2['rich'][later] += 999
        d2['tau'][:] = 1.0
        d2['y'][:] = 1
        f2 = cx.m1_features(d2)
        np.testing.assert_array_equal(f1[i], f2[i])
        T2 = cx.Tensors(d2, np.setdiff1d(tr, later))
        T1b = cx.Tensors(d, np.setdiff1d(tr, later))
        self.assertTrue(torch.equal(T1b.seq[i], T2.seq[i]))
        self.assertTrue(torch.equal(T1b.raw[i], T2.raw[i]))
        self.assertIsNotNone(T1)

    def test_normalisation_uses_training_rows_only(self):
        d = synth()
        tr = np.arange(100)
        T = cx.Tensors(d, tr)
        np.testing.assert_allclose(T.mu, d['rich'][tr].mean(0))


class TestFoldsAndComparison(unittest.TestCase):
    def test_person_disjoint_folds(self):
        d = synth(n_rec=40)
        f = cx.person_folds(d['person'])
        pf = np.array([f[p] for p in d['person']])
        for p in set(d['person']):
            self.assertEqual(len(set(pf[d['person'] == p])), 1)
        self.assertEqual(set(pf), set(range(5)))

    def test_paired_bootstrap_and_credible_rule(self):
        d = synth(n_rec=20)
        y = d['y'].astype(float)
        ref = np.full(len(y), y.mean())
        good = np.clip(0.7 * y + 0.15, 0, 1)
        r = cx.paired(ref, good, d)
        self.assertTrue(r['credible'])
        self.assertGreater(r['R'], 0.05)
        r2 = cx.paired(ref, ref, d)
        self.assertFalse(r2['credible'])
        persons = sorted(set(d['person']))
        idx = np.searchsorted(persons, d['person'])
        sr = np.bincount(idx, weights=(ref - y) ** 2)
        sc = np.bincount(idx, weights=(good - y) ** 2)
        l2o = min(((sr.sum() - sr[i] - sr[j]) - (sc.sum() - sc[i] - sc[j])) / (sr.sum() - sr[i] - sr[j])
                  for i, j in itertools.combinations(range(len(persons)), 2))
        self.assertAlmostEqual(r['l2o_R_min'], l2o)


class TestModelsRun(unittest.TestCase):
    def test_forward_shapes_all_archs(self):
        d = synth()
        tr = np.arange(len(d['y']))
        for name, arch in cx.ARCHS.items():
            T = cx.Tensors(d, tr, **cx.SPECIAL.get(name, {}))
            m = cx.CTHNet(T.n_feat, 16, cin=T.cin, **arch).to(cx.DEV)
            z = m(T.raw[:4], T.seq[:4], T.tf[:4])
            self.assertEqual(z.shape[1], 10 if arch['head'] == 'hazard' else 1)

    def test_rich_features_finite_and_alpha_sensitive(self):
        t = np.arange(3000) / 100
        a = cd.rich_features(20 * np.sin(2 * np.pi * 10 * t))
        th = cd.rich_features(20 * np.sin(2 * np.pi * 6 * t))
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertGreater(a[6], th[6])          # log(alpha/theta) higher for 10 Hz than 6 Hz


if __name__ == '__main__':
    unittest.main()
