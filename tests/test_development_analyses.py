"""Development-data analyses: architecture ablation (Fig. 7), representation and perturbation analysis (Fig. 8).

Golden values were produced by the original research implementation on these synthetic fixtures (CPU); the new code
reproduced them bitwise on the authors' workstation (tolerances only absorb cross-platform float noise).
"""
import itertools

import numpy as np
import pytest
import torch

from heval.analysis import architecture_ablation as AA, perturbation as P, representation as RP
from heval.models import ablation_networks as AN, raw_cnn, stft_cnn as SC

GOLD_RICH = [17.76360912750589, 0.18603493221505435, 0.006316917443787459, 0.7734931757149519, 0.006496558437810848,
             0.027658416188395325, 4.807685504433377, 3.3827031207516596, 2.863536245250488]
GOLD_ABLATION = {'M2_raw_cnn_binary': [0.2799796164035797, 0.28187450766563416, 0.2808723747730255],
                 'M3_cthnet': [0.3014812469482422, 0.30433422327041626, 0.2989943027496338],
                 'A2_no_raw': [0.29693228006362915, 0.29482150077819824, 0.2966257929801941]}
GOLD_STFT = [0.29388555884361267, 0.2915874719619751, 0.2782977521419525]


@pytest.fixture(scope='module')
def dev(nights):
    rich = np.array([[v for c in w for v in AN.rich_features(c.astype(float))] for w in nights['raw']])
    return AA.development_table(nights['raw'], P.spectral_features_np(nights['raw'].astype(np.float64)), rich, nights['elapsed_s'],
                                nights['time_to_n2_s'], nights['y'], nights['person'], nights['recording'])


def _signal(n=3, seed=0):
    t = np.arange(3000) / 100
    return np.random.default_rng(seed).normal(0, 20, (n, 2, 3000)) + 30 * np.sin(2 * np.pi * 10 * t) + 5.0


# ------------------------------------------------------------------------ networks and inputs (Fig. 7)
def test_variant_parameter_counts_and_rich_features():
    counts = {k: sum(p.numel() for p in AN.CTHNet(18 if k not in ('A6a_fpz_only', 'A6b_pz_only') else 9, 32,
                                                  cin=1 if k.startswith('A6') else 2, **a).parameters()) for k, a in AN.ARCHS.items()}
    assert counts == dict(M2_raw_cnn_binary=114817, M3_cthnet=136938, A1_no_trajectory=115114, A2_no_raw=22186, A3_binary_head=136641,
                          A4_history_30s=136938, A5_plus_time=137002, A6a_fpz_only=129226, A6b_pz_only=129226)
    t = np.arange(3000) / 100
    x = 20 * np.sin(2 * np.pi * 10 * t) + 10 * np.sin(2 * np.pi * 3 * t) + np.random.default_rng(0).normal(0, 5, 3000)
    assert np.allclose(AN.rich_features(x), GOLD_RICH, rtol=1e-12)
    with pytest.raises(ValueError, match='3000-sample'):
        AN.rich_features(np.zeros(10))


def test_hazard_loss_is_discrete_time_likelihood():
    assert list(AN.event_interval([30, 31, 300, 301, 600])) == [1, 2, 10, 11, 11]
    z = torch.tensor(np.random.default_rng(0).normal(size=(6, 10)), dtype=torch.float64)
    K = torch.tensor([1, 3, 10, 11, 11, 5])
    h = torch.sigmoid(z).numpy()
    manual = [-(np.log(1 - h[i, :k - 1]).sum() + np.log(h[i, k - 1])) if k <= 10 else -np.log(1 - h[i]).sum() for i, k in enumerate(K.tolist())]
    assert float(AN.hazard_loss(z, K)) == pytest.approx(float(np.mean(manual)), abs=1e-10)
    zz = torch.tensor([[0.3] * 10])
    assert float(AN.p300_from(zz, 'hazard')) == pytest.approx(float(1 - torch.prod(1 - torch.sigmoid(zz))), abs=1e-6)


def test_history_is_causal_and_inputs_use_training_rows_only(dev):
    for i in range(len(dev['y'])):
        idx = dev['hist'][i][dev['hist'][i] >= 0]
        assert dev['hist'][i, 9] == i and np.all(idx <= i) and np.all(dev['rec_idx'][idx] == dev['rec_idx'][i])
    tr = np.arange(150)
    T = AN.AblationInputs(dev, tr)
    assert np.allclose(T.mu, dev['rich'][tr].mean(0)) and T.scale == pytest.approx(raw_cnn.train_scale(dev['raw'][tr]))
    d2 = {k: v.copy() for k, v in dev.items()}
    i = 20
    later = np.where((dev['rec_idx'] == dev['rec_idx'][i]) & (np.arange(len(dev['y'])) > i))[0]
    d2['raw'][later] += 999
    d2['rich'][later] += 999
    T1, T2 = AN.AblationInputs(dev, np.setdiff1d(tr, later)), AN.AblationInputs(d2, np.setdiff1d(tr, later))
    assert torch.equal(T1.seq[i], T2.seq[i]) and torch.equal(T1.raw[i], T2.raw[i])     # no future window leaks in


def test_folds_paired_and_credible_rule(dev):
    folds = AA.person_folds(dev['person'])
    pf = np.array([folds[p] for p in dev['person']])
    assert set(pf) == set(range(5)) and all(len(set(pf[dev['person'] == p])) == 1 for p in set(dev['person']))
    te, tr, itr, iva = AA.fold_rows(pf, 0)
    assert set(dev['person'][iva]) == {p for p, f in folds.items() if f == 1} and not set(itr) & set(iva) and not set(tr) & set(te)
    y = dev['y'].astype(float)
    ref, good = np.full(len(y), y.mean()), np.clip(0.7 * y + 0.15, 0, 1)
    r = AA.paired(ref, good, dev)
    assert r['credible'] and r['R'] > 0.05 and not AA.paired(ref, ref, dev)['credible']
    pu = sorted(set(dev['person']))
    idx = np.searchsorted(pu, dev['person'])
    sr, sc = np.bincount(idx, weights=(ref - y) ** 2), np.bincount(idx, weights=(good - y) ** 2)
    l2o = min(((sr.sum() - sr[i] - sr[j]) - (sc.sum() - sc[i] - sc[j])) / (sr.sum() - sr[i] - sr[j]) for i, j in itertools.combinations(range(len(pu)), 2))
    assert r['l2o_R_min'] == pytest.approx(l2o, rel=1e-12)


def test_ablation_cv_and_summary(dev):
    archs = ['M2_raw_cnn_binary', 'M3_cthnet', 'A2_no_raw']
    oof, seed_oof, log, folds = AA.ablation_cv(dev, archs=archs, folds_to_run=[0], max_epochs=1, seeds=(0,), grid=AA.GRID[:1])
    pf = np.array([folds[p] for p in dev['person']])
    te = np.where(pf == 0)[0]
    assert len(te) == 64 and all(np.isfinite(oof[m][te]).all() and np.isnan(oof[m][pf != 0]).all() for m in archs)
    for m in archs:
        assert np.allclose(oof[m][te[:3]], GOLD_ABLATION[m], atol=1e-5), m
    s = AA.summarise_ablation(oof, dev, n_boot=200)
    assert set(s['comparisons']) == {'M2_raw_cnn_binary__vs__M3_cthnet', 'A2_no_raw__vs__M3_cthnet'} and s['n_windows'] == 64
    with pytest.raises(ValueError, match='unknown variants'):
        AA.ablation_cv(dev, archs=['M9'])


# ------------------------------------------------------------------------ representation and perturbation (Fig. 8)
def test_stft_cnn_geometry_and_matched_cv(dev):
    x = torch.randn(3, 2, 3000)
    assert tuple(SC.stft_logpow(x).shape) == (3, 2, 60, 57)
    assert (SC.count_parameters(SC.STFTNet(16)), SC.count_parameters(SC.STFTNet(32))) == (31465, 117105)
    assert tuple(SC.STFTNet(16)(SC.stft_logpow(x)).shape) == (3, 1)
    oof, _, _, store = RP.matched_cv(dev, folds_to_run=[0], max_epochs=1, seeds=(0,), grid=AA.GRID[:1])
    te = np.where(np.isfinite(oof['STFT_cnn']))[0]
    # Golden CPU predictions from the original implementation differ by up to 2.3e-5
    # across PyTorch CPU builds (2.9.1 vs 2.9.1+cpu). Keep a tight absolute
    # cross-build tolerance while preserving the original frozen reference values.
    assert np.allclose(oof['STFT_cnn'][te[:3]], GOLD_STFT, atol=5e-5, rtol=0)
    assert np.allclose(oof['M2_raw_cnn'][te[:3]], GOLD_ABLATION['M2_raw_cnn_binary'], atol=1e-5)    # same replay as Fig. 7 M2
    assert len(np.unique(oof['T0_constant'][te])) == 1 and set(store[0]) >= {'M2_raw_cnn', 'STFT_cnn', 'm0_outer', 'm2_scale'}


def test_perturbations_keep_what_they_claim():
    x = _signal()
    y = P.band_gain(x, 8, 12, 1.0)
    Y, m = np.fft.rfft(y, axis=-1), (P.FREQS >= 8) & (P.FREQS < 12)
    assert np.abs(Y[..., m]).max() < 1e-8 and np.allclose(Y[..., ~m], np.fft.rfft(x, axis=-1)[..., ~m], atol=1e-7)
    assert np.allclose(P.energy(P.rescale_energy(y, P.energy(x))), P.energy(x), rtol=1e-10)                  # P4
    u = P.rescale_energy(x, P.energy(y))                                                                     # P5 control
    assert np.allclose(P.energy(u), P.energy(y), rtol=1e-10) and np.allclose(u.mean(-1), x.mean(-1), atol=1e-9)
    s = P.fourier_surrogate(x, np.random.default_rng(1))
    X, S = np.fft.rfft(x, axis=-1), np.fft.rfft(s, axis=-1)
    assert np.allclose(np.abs(S), np.abs(X), rtol=1e-8, atol=1e-6) and np.abs(s - x).max() > 1.0
    assert np.allclose(S[:, 0] * np.conj(S[:, 1]), X[:, 0] * np.conj(X[:, 1]), rtol=1e-6, atol=1e-3)      # cross-spectrum kept
    z = np.random.default_rng(0).normal(0, 20, (40, 2, 3000))
    z[:, :, 1500:] *= 3.0
    loc = P.local_surrogate(z, np.random.default_rng(3))
    assert abs(np.corrcoef(loc[:, 0].ravel(), z[:, 0].ravel())[0, 1]) < 0.2 and abs(np.mean(P.energy(loc) / P.energy(z)) - 1) < 0.1
    sp = P.splice(x, _signal(3, 9), 2)
    assert np.array_equal(sp[..., :1000], x[..., :1000]) and np.array_equal(sp[..., 1500:], x[..., 1500:])
    b = P.artifacts(x, 'blink', 100, 5)
    assert np.array_equal(b[:, 1], x[:, 1]) and 99 <= (b[:, 0] - x[:, 0]).max() <= 300.001
    with pytest.raises(ValueError):
        P.artifacts(x, 'unknown', 1, 0)


def test_surrogates_golden(nights):
    xs = nights['raw'][:4].astype(np.float64)
    assert (P.fourier_surrogate(xs, np.random.default_rng(P.PERT_SEED)) ** 2).sum() == pytest.approx(2696050.503496304, rel=1e-12)
    assert P.local_surrogate(xs, np.random.default_rng(P.PERT_SEED + 1))[0, 0, 0] == pytest.approx(10.925216132528538, rel=1e-12)


def test_perturbation_battery_and_retention(dev):
    oof, _, _, store = RP.matched_cv(dev, folds_to_run=[0, 1], max_epochs=1, seeds=(0,), grid=AA.GRID[:1])
    folds = AA.person_folds(dev['person'])
    pf_all = np.array([folds[p] for p in dev['person']])
    m = np.isin(pf_all, [0, 1])
    sub = dict(raw=dev['raw'][m], y=dev['y'][m], person=dev['person'][m], rec_idx=dev['rec_idx'][m])
    fm = {f: P.fold_predictors(s) for f, s in store.items()}
    donor = P.fold_donors(sub['person'], dev['elapsed'][m], pf_all[m])
    conds = P.conditions(sub['raw'].astype(np.float64), donor)
    assert len(conds) == 79 and len(P.followup_conditions()) == 6
    keep = [c for c in conds if c[0] in ('P3_delta_s1.0', 'P5_delta_uniform_control', 'P7_scale_2.0')]
    res, per_person, base = P.run_battery(sub, fm, pf_all[m], oof['T0_constant'][m], keep)
    for c in res.values():
        for k in ('M2', 'STFT', 'M0'):
            assert np.isfinite(c[k]['retention']) and c[k]['retention_ci95'][0] <= c[k]['retention_ci95'][1]
    po = base['M2'].mean(0)
    y = sub['y'].astype(float)
    B = P.boot_indices(sub['person'], n_boot=100)
    assert P.score_condition(po, po, oof['T0_constant'][m], y, sub['rec_idx'], B)['retention'] == pytest.approx(1.0)
    one = np.isin(dev['person'], dev['person'][:1])
    with pytest.raises(ValueError, match='single person'):
        P.fold_donors(dev['person'][one], dev['elapsed'][one], np.zeros(one.sum(), int))


def test_score_condition_golden(dev):
    rng = np.random.default_rng(5)
    y = dev['y'].astype(float)
    po = np.clip(0.5 * y + 0.25 + rng.normal(0, .05, len(y)), .01, .99)
    pp = np.clip(0.3 * y + 0.35 + rng.normal(0, .05, len(y)), .01, .99)
    out = P.score_condition(po, pp, np.full(len(y), y.mean()), y, dev['rec_idx'], P.boot_indices(dev['person']))
    assert out['retention'] == pytest.approx(0.6109807630273175, rel=1e-12)
    assert np.allclose(out['retention_ci95'], [0.5613834163246856, 0.6615978331729426], rtol=1e-12)
    assert out['delta_brier_ww'] == pytest.approx(0.057820538525512735, rel=1e-12)


def test_attribution_checks(nights):
    torch.manual_seed(4)
    model = raw_cnn.RawCNN().eval()
    x = torch.tensor(raw_cnn.normalise(nights['raw'][:3], 10.0))
    ig = RP.integrated_gradients(model, x, steps=8, batch=8)
    assert np.allclose(RP.bin_map(ig)[0, :4].numpy(), [0.1681646853685379, 0.12462121993303299, 0.0738576278090477, 0.06859878450632095], atol=1e-5)
    ig64 = RP.integrated_gradients(model, x, steps=64, batch=8)                   # completeness: sum IG ~ f(x) - f(0)
    with torch.no_grad():
        gap = (model(x) - model(torch.zeros_like(x)))[:, 0]
    assert torch.allclose(ig64.sum((1, 2)), gap, atol=0.05 * float(gap.abs().max()) + 1e-3)
    r = RP.randomise(model, 1)
    assert not torch.equal(r.head.weight, model.head.weight) and torch.equal(r.fuse[0].weight, model.fuse[0].weight)
    xd = torch.ones(2, 2, 3000)
    out = RP.replace_bins(torch.zeros(2, 2, 3000), xd, np.array([[0, 31], [59, 5]]))
    assert float(out.sum()) == 400 and float(out[0, 1, 100:200].sum()) == 100
    assert np.allclose(RP.spearman_rows(np.arange(6.).reshape(1, 6), np.arange(6.).reshape(1, 6)), 1.0)
