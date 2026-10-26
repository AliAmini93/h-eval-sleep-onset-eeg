"""INTERP-v1 tests: perturbation correctness, model geometry, training-loop equivalence, data boundary."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('interp_v1_test', HERE / 'interp_v1.py')
iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv)
cx = iv.cx


class ZeroRng:
    def uniform(self, lo, hi, size):
        return np.zeros(size)


def sig(n=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(iv.N) / iv.FS
    x = rng.normal(0, 20, (n, 2, iv.N)) + 30 * np.sin(2 * np.pi * 10 * t) + 5.0
    return x


class TestPerturbations(unittest.TestCase):
    def test_band_gain_and_energy_controls(self):
        x = sig()
        y = iv.band_gain(x, 8, 12, 1.0)
        Y = np.fft.rfft(y, axis=-1)
        m = (iv.FREQS >= 8) & (iv.FREQS < 12)
        self.assertLess(np.abs(Y[..., m]).max(), 1e-8)
        np.testing.assert_allclose(Y[..., ~m], np.fft.rfft(x, axis=-1)[..., ~m], atol=1e-7)
        np.testing.assert_allclose(iv.energy(iv.rescale_energy(y, iv.energy(x))), iv.energy(x), rtol=1e-10)
        u = iv.rescale_energy(x, iv.energy(y))
        np.testing.assert_allclose(iv.energy(u), iv.energy(y), rtol=1e-10)
        np.testing.assert_allclose(u.mean(-1), x.mean(-1), atol=1e-9)

    def test_fourier_surrogates_preserve_spectra(self):
        x = sig()
        s = iv.fourier_surrogate(x, np.random.default_rng(1))
        X, S = np.fft.rfft(x, axis=-1), np.fft.rfft(s, axis=-1)
        np.testing.assert_allclose(np.abs(S), np.abs(X), rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(S[:, 0] * np.conj(S[:, 1]), X[:, 0] * np.conj(X[:, 1]), rtol=1e-6, atol=1e-3)
        self.assertGreater(np.abs(s - x).max(), 1.0)
        m = (iv.FREQS >= 4) & (iv.FREQS < 8)
        b = iv.fourier_surrogate(x, np.random.default_rng(2), mask=m)
        B = np.fft.rfft(b, axis=-1)
        np.testing.assert_allclose(np.abs(B), np.abs(X), rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(B[..., ~m], X[..., ~m], atol=1e-6)

    def test_local_surrogate_identity_and_power(self):
        x = np.random.default_rng(0).normal(0, 20, (40, 2, iv.N))
        x[:, :, 1500:] *= 3.0                                           # non-stationary power: must be kept locally
        s = iv.local_surrogate(x, np.random.default_rng(3))
        self.assertGreater(np.abs(s - x).max(), 1.0)
        self.assertLess(abs(np.corrcoef(s[:, 0].ravel(), x[:, 0].ravel())[0, 1]), 0.2)   # waveform destroyed
        for lo, hi in ((0.5, 4), (8, 12), (15, 30)):
            def bp(z):
                Z = np.abs(np.fft.rfft(z.reshape(len(z), 2, 15, 200) * iv._HANN200, axis=-1)) ** 2
                f = np.fft.rfftfreq(200, 0.01)
                return Z[..., (f >= lo) & (f < hi)].sum(-1)
            r = bp(s) / bp(x)
            self.assertLess(abs(np.median(r[..., 2:6]) - 1), 0.15)     # quiet half
            self.assertLess(abs(np.median(r[..., 9:13]) - 1), 0.15)    # loud half: local power tracked
        r = iv.energy(s) / iv.energy(x)
        self.assertLess(abs(np.mean(r) - 1), 0.1)

    def test_splice_and_artifacts(self):
        x, dn = sig(2, 0), sig(2, 9)
        y = iv.splice(x, dn, 2)
        np.testing.assert_array_equal(y[..., :1000], x[..., :1000])
        np.testing.assert_array_equal(y[..., 1500:], x[..., 1500:])
        inner = slice(1010, 1490)
        np.testing.assert_allclose(y[..., inner] - y[..., 1000:1500].mean(-1, keepdims=True),
                                   dn[..., inner] - dn[..., 1000:1500].mean(-1, keepdims=True), atol=1e-6 + 2.0)
        b = iv.artifacts(x, 'blink', 100, 5)
        np.testing.assert_array_equal(b[:, 1], x[:, 1])
        self.assertGreaterEqual((b[:, 0] - x[:, 0]).max(), 99)          # blinks may overlap
        self.assertLessEqual((b[:, 0] - x[:, 0]).max(), 300.001)
        p = iv.artifacts(x, 'pop', 100, 5)
        np.testing.assert_array_equal(p[:, 0], x[:, 0])
        e = iv.artifacts(x, 'emg', 10, 5) - x
        burst = np.abs(e[0, 0]) > 0
        self.assertAlmostEqual(np.sqrt((e[0, 0, burst] ** 2).mean()), 10, delta=0.5)
        self.assertTrue(296 <= int(burst.sum()) <= 301)


class TestModels(unittest.TestCase):
    def test_stft_geometry_and_capacity(self):
        x = torch.randn(3, 2, iv.N)
        self.assertEqual(tuple(iv.stft_logpow(x).shape), (3, 2, 60, 57))
        n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        self.assertEqual((n(iv.STFTNet(16)), n(iv.STFTNet(32))), (31465, 117105))
        m2 = lambda w: cx.CTHNet(18, w, cin=2, **cx.ARCHS['M2_raw_cnn_binary'])  # noqa: E731
        self.assertEqual((n(m2(16)), n(m2(32))), (32881, 114817))
        self.assertEqual(tuple(iv.STFTNet(16)(iv.stft_logpow(x)).shape), (3, 1))

    def test_train_generic_equals_cx_train_net(self):
        rng = np.random.default_rng(0)
        n = 96
        d = dict(raw=rng.normal(0, 30, (n, 2, iv.N)).astype(np.float32), rich=rng.normal(size=(n, 18)),
                 hist=np.tile(np.arange(-9, 1), (n, 1)), elapsed=np.arange(n) * 30.0, tau=rng.uniform(30, 900, n))
        d['y'] = (d['tau'] <= 300).astype(int)
        tr, va = np.arange(64), np.arange(64, 96)
        T = cx.Tensors(d, tr)
        cfg = dict(lr=1e-3, width=16)
        arch = cx.ARCHS['M2_raw_cnn_binary']
        m1, i1 = cx.train_net(T, tr, va, cfg, arch, seed=0, max_epochs=3)
        m2, i2 = iv.train_generic(T, tr, va, cfg, lambda w: cx.CTHNet(T.n_feat, w, cin=2, **arch), seed=0, max_epochs=3)
        p1, p2 = cx.predict_net(m1, T, va, 'binary'), cx.predict_net(m2, T, va, 'binary')
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        self.assertEqual(i1['epochs'], i2['epochs'])

    def test_replace_bins(self):
        x, xd = torch.zeros(2, 2, iv.N), torch.ones(2, 2, iv.N)
        out = iv._replace_bins(x, xd, np.array([[0, 31], [59, 5]]))
        self.assertEqual(float(out[0, 0, :100].sum()), 100)
        self.assertEqual(float(out[0, 1, 100:200].sum()), 100)
        self.assertEqual(float(out[1, 1, 2900:].sum()), 100)
        self.assertEqual(float(out.sum()), 400)


class TestData(unittest.TestCase):
    def test_train_only_and_e5_exact(self):
        d = iv.check_inputs()
        idx = np.random.default_rng(0).choice(len(d['y']), 300, replace=False)
        self.assertLess(np.abs(iv.e5_features(d['raw'][idx].astype(np.float64)) - d['e5'][idx]).max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
