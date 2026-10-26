"""Models: spectral features + SpecLR, the exact RawCNN, scored-stage references, wake-to-N1 temporal CNNs.

Golden values were produced by the original research implementation on these synthetic fixtures; the new code
reproduced them bitwise on the author's workstation (tolerances below only absorb cross-platform BLAS/oneDNN noise).
"""
import math

import numpy as np
import pytest
import torch

from heval.models import raw_cnn, references, spectral, temporal_cnn

GOLD_FEATURES = [16.08178841590858, 12.69908824537284, 17.510263504302547, 14.30420914376612, 2.863536245250488]
GOLD_SPECLR = [-1.3860263969231725, -0.08586637907055046, 0.7244869440836786, -0.8154904787959432, 0.06620731790628794,
               0.765449748128483, -0.12847861450148726, 0.7295491007343967, -0.8836925873533733, -0.007684170498418015,
               0.7046515218274002]
GOLD_RAWCNN_INIT = [0.10768657177686691, 0.1023995503783226, 0.09805270284414291, 0.11177288740873337]
GOLD_RAWCNN_ENSEMBLE = [0.30887624621391296, 0.31366108854611713, 0.31140321493148804, 0.3083902597427368]
GOLD_RAWCNN_SEED0 = [0.31500497460365295, 0.3164849877357483, 0.3146156966686249, 0.3099958300590515]


# ------------------------------------------------------------------------ spectral features and SpecLR
def test_spectral_features_values_shape_finite(nights):
    t = np.arange(3000) / 100
    x = 20 * np.sin(2 * np.pi * 10 * t) + 10 * np.sin(2 * np.pi * 3 * t) + np.random.default_rng(0).normal(0, 5, 3000)
    f = spectral.channel_features(x.tolist())
    assert np.allclose(f, GOLD_FEATURES, rtol=1e-12, atol=0)
    assert f[2] == max(f[:4])                    # the 10 Hz component dominates the alpha band (8-13 Hz)
    F = spectral.window_features(nights['raw'][0])
    assert len(F) == 10 and all(math.isfinite(v) for v in F)


def test_spectral_features_reject_invalid_windows():
    with pytest.raises(ValueError, match='30s, 100Hz'):
        spectral.channel_features([0.0] * 2999)
    with pytest.raises(ValueError, match='30s, 100Hz'):
        spectral.channel_features([0.0] * 2999 + [float('nan')])
    with pytest.raises(ValueError, match='window'):
        spectral.window_features(np.zeros((3, 3000)))
    with pytest.raises(ValueError, match='power of 2'):
        spectral.fft_real([1.0] * 3)


def test_speclr_fit_predict_train_only(nights):
    F = np.array([spectral.window_features(w) for w in nights['raw']])
    M = spectral.fit_speclr(F, nights['elapsed_s'], nights['y'])
    assert M['n_windows'] == 322 and M['constant_prior'] == pytest.approx(0.3117283950617284, abs=1e-15)
    assert np.allclose(M['weights']['EEG_LOGIT'], GOLD_SPECLR, rtol=1e-9, atol=1e-12)
    P = spectral.predict_speclr(M, F, nights['elapsed_s'])
    assert set(P) == set(spectral.MODELS)
    for p in P.values():
        assert p.shape == (322,) and p.min() >= 1e-6 and p.max() <= 1 - 1e-6
    assert np.allclose(P['EEG_LOGIT'][:3], [0.002985759577247496, 0.0052328062017899446, 0.005914822895037349], rtol=1e-8)
    # inference uses only the frozen TRAIN statistics: predicting any subset gives the same per-window values
    assert np.array_equal(spectral.predict_speclr(M, F[:5], nights['elapsed_s'][:5])['EEG_LOGIT'], P['EEG_LOGIT'][:5])
    with pytest.raises(ValueError, match='binary'):
        spectral.fit_speclr(F, nights['elapsed_s'], nights['y'] * 2)
    with pytest.raises(ValueError, match='shape'):
        spectral.fit_speclr(F[:, :9], nights['elapsed_s'], nights['y'])


# ------------------------------------------------------------------------ RawCNN
def test_rawcnn_exact_architecture():
    m = raw_cnn.RawCNN()
    assert raw_cnn.count_parameters(m) == 114_817 == raw_cnn.N_PARAMETERS
    assert [s.kernel_size[0] for s in m.raw.stems] == [7, 25, 101] and all(s.stride[0] == 2 for s in m.raw.stems)
    convs = [l for l in m.raw.body if isinstance(l, torch.nn.Conv1d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size[0], c.stride[0]) for c in convs] == [(96, 64, 7, 4), (64, 64, 7, 4), (64, 64, 7, 4)]
    assert sum(isinstance(l, torch.nn.GroupNorm) for l in m.raw.body) == 3
    assert (m.raw.out.in_features, m.raw.out.out_features) == (128, 32)
    assert (m.head.in_features, m.head.out_features) == (32, 1)          # one logit: P(first N2 within 300 s)


def test_rawcnn_forward_backward_optimizer_step(nights):
    torch.manual_seed(0)
    m = raw_cnn.RawCNN().eval()
    x = torch.tensor(raw_cnn.normalise(nights['raw'][:4], 10.0))
    assert x.shape == (4, 2, 3000) and float(x.abs().max()) <= 10
    with torch.no_grad():
        z = m(x)
    assert z.shape == (4, 1)
    assert np.allclose(z[:, 0].numpy(), GOLD_RAWCNN_INIT, atol=1e-5)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    before = [p.detach().clone() for p in m.parameters()]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x[:2])[:, 0], torch.tensor([0., 1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step()
    assert sum(not torch.equal(a, b) for a, b in zip(before, m.parameters())) == len(before)


def test_rawcnn_training_and_three_seed_ensemble(nights):
    raw, y = nights['raw'][:64], nights['y'][:64]
    scale = raw_cnn.train_scale(raw)
    assert scale == pytest.approx(10.585420608520508, rel=1e-6)
    x = raw_cnn.normalise(raw, scale)
    models = [raw_cnn.fit_seed(x, y, s, epochs=1)[0] for s in (0, 1, 2)]
    prior = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    m0 = raw_cnn.fit_seed(x, y, 0, epochs=0)[0]
    assert float(m0.head.bias.detach()) == pytest.approx(math.log(prior / (1 - prior)), abs=1e-6)   # bias starts at TRAIN prevalence
    ens, per = raw_cnn.predict_ensemble(nights['raw'][:4], scale, models)
    assert np.allclose(per[0], GOLD_RAWCNN_SEED0, atol=1e-4) and np.allclose(ens, GOLD_RAWCNN_ENSEMBLE, atol=1e-4)
    assert np.array_equal(ens, np.mean(per, 0)) and not np.array_equal(per[0], per[1])
    again = raw_cnn.fit_seed(x, y, 0, epochs=1)[0]                                       # reproducible per seed
    assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), again.state_dict().values()))
    assert raw_cnn.epoch_schedule([3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6]) == 4


def test_rawcnn_rejects_invalid_input(nights):
    with pytest.raises(ValueError, match='N, 2, 3000'):
        raw_cnn.normalise(np.zeros((2, 3000)), 10.0)
    with pytest.raises(ValueError, match='non-finite'):
        raw_cnn.normalise(np.full((1, 2, 3000), np.nan), 10.0)
    with pytest.raises(ValueError, match='positive'):
        raw_cnn.normalise(nights['raw'][:1], 0.0)
    with pytest.raises(ValueError, match='length N'):
        raw_cnn.fit_seed(raw_cnn.normalise(nights['raw'][:4], 10.0), [0, 1], 0, epochs=1)


# ------------------------------------------------------------------------ Stage-only and Stage+time references
def test_stage_only_laplace_smoothing():
    y = np.array([1, 0, 0, 0, 1, 1, 0])
    s1 = np.array([0, 0, 0, 0, 1, 1, 1])
    A = references.fit_A(y, s1)
    assert A == dict(p_wake=(1 + 1) / (4 + 2), p_stage1=(2 + 1) / (3 + 2))
    assert list(references.predict_A(A, [0, 1])) == [A['p_wake'], A['p_stage1']]
    assert references.fit_A([0, 0], [1, 1]) == dict(p_wake=0.5, p_stage1=1 / 4)             # empty stratum -> 1/2


def test_stage_time_reference_train_fit_and_inference():
    from heval.synthetic import synthetic_nights
    d = synthetic_nights(n_persons=16, seed=2, with_raw=False)
    s1, y = (d['stage'] == '1').astype(float), d['y'].astype(float)
    R = references.fit_references(d['elapsed_s'], s1, y, d['person'])
    assert R['A'] == dict(p_wake=0.1371308016877637, p_stage1=0.9897959183673469)
    assert (R['B_selection']['knots'], R['B_selection']['C'], R['B_selection']['interaction']) == (3, 10.0, True)
    assert R['B']['intercept'] == pytest.approx(-2.1882295706435166, rel=1e-6)
    p = references.predict_B(R['B'], np.array([10., 600., 1e5]), np.array([0., 1., 1.]))
    assert np.allclose(p, [0.007630364619445936, 0.9886159058484237, 0.9987526819826912], rtol=1e-6)
    lo, hi = math.exp(R['B']['logt_lo']), math.exp(R['B']['logt_hi'])
    assert lo == pytest.approx(d['elapsed_s'].min()) and hi == pytest.approx(d['elapsed_s'].max())
    clamp = references.predict_B(R['B'], np.array([lo, lo / 10, hi, hi * 10]), np.zeros(4))
    assert clamp[0] == clamp[1] and clamp[2] == clamp[3]                                    # time clamped to TRAIN support
    with pytest.raises(ValueError, match='positive'):
        references.fit_B(np.array([0., 30.]), np.array([0., 1.]), np.array([0., 1.]), (3, 1.0, False))
    with pytest.raises(ValueError, match='0/1'):
        references.fit_A(np.array([2, 0]), np.array([0, 1]))


# ------------------------------------------------------------------------ wake-to-N1 temporal CNNs (smoke)
@pytest.mark.parametrize('name', list(temporal_cnn.MODELS))
def test_temporal_cnn_families_forward(nights, name):
    n = 24
    hist = np.full((n, 10), -1)
    for i in range(n):
        for k in range(min(i, 9) + 1):
            hist[i, 9 - k] = i - k
    d = dict(hist=hist, tau=nights['time_to_n2_s'][:n], k1=np.full(n, 11), e1=np.ones(n, bool), stage=nights['stage'][:n])
    cfg = temporal_cnn.MODELS[name]
    torch.manual_seed(0)
    D = temporal_cnn.HistoryInputs(d, nights['raw'][:n], np.arange(n), cfg)
    net = temporal_cnn.HazardNet(cfg['enc'], cfg['state']).eval()
    with torch.no_grad():
        (l1, m1), (l2, m2), s = net(D.batch(torch.arange(4)))
    p1, p2 = 1 - torch.exp(m1.sum(1)), 1 - torch.exp(m2.sum(1))     # untrained: P(event within 300 s) can round to 1
    assert l1.shape == (4, 10) and all(torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in (p1, p2))
    assert torch.all(l1 <= 0) and torch.all(m2 <= 0)                     # log-hazards and log-survivals
    assert (s is None) == (not cfg['state'])
    assert list(temporal_cnn.event_interval([0, 30, 31, 300, 301])) == [1, 1, 2, 10, 11]
