#!/usr/bin/env python3
"""H-EVAL INTERP-v1 (protocol docs/decision/h_eval/interp_v1/interp_stft_protocol_v1.md). POST-HOC EXPLORATORY.

SC_TRAIN only (.local/h_eval/cth_v1/dataset_v1.npz). Never reads held-out EEG, never modifies frozen artefacts.
  interp_v1.py cv        Part B: matched TRAIN 5-fold CV of M0, M2 (CTH v1 replay) and STFTNet; saves fold models
  interp_v1.py perturb   Part A: perturbation battery on out-of-fold models (M2, STFTNet, M0) + frozen M2 in-sample
  interp_v1.py explain   Part A: IG seed agreement, randomisation sanity, deletion fidelity, cross-method agreement
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cx = _imp(REPO / 'scripts/decision/h_eval/cth_v1/cth_experiment.py', 'cx_for_interp')
c2 = _imp(REPO / 'scripts/decision/h_eval/cth_v2/cth_final_v2.py', 'c2_for_interp')
base = _imp(REPO / 'scripts/decision/h_eval/gate_e7b/gate_e7b.py', 'e7b_for_interp')

PROTOCOL = REPO / 'docs/decision/h_eval/interp_v1/interp_stft_protocol_v1.md'
PROTOCOL_SHA = 'd825b79d499bf773868c1df8be93e404f18f97b00ec98ab28828c8ca57541d20'
AMENDMENT = REPO / 'docs/decision/h_eval/interp_v1/interp_stft_amendment_01.md'
AMENDMENT_SHA = 'cd298919d08436190209a086df54763fbf3f3038b91fbb0dd47a493f28517ed4'
OUT = REPO / 'results/decision/h_eval/interp_v1'
LOC = REPO / '.local/h_eval/interp_v1'
DEV = cx.DEV
FS, N = 100, 3000
BANDS = dict(delta=(0.5, 4.0), theta=(4.0, 8.0), alpha=(8.0, 12.0), sigma=(12.0, 15.0), beta=(15.0, 30.0), gamma=(30.0, 50.01))
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
SCALES = (0.5, 0.7, 0.85, 1.2, 1.4, 2.0)
BOOT_B, SEED_B = 10_000, 20261101
BOOT_A, SEED_A = 2_000, 20261103
PERT_SEED = 20261102
FREQ_KEEP = slice(1, 61)                      # 0.5..30.0 Hz at 0.5 Hz resolution
CH_MULT = (1, 2, 2.5, 2)
FROZEN_FINAL = REPO / 'results/decision/h_eval/cth_v2/frozen_m2_46_final'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_inputs():
    if digest(PROTOCOL) != PROTOCOL_SHA or digest(AMENDMENT) != AMENDMENT_SHA:
        sys.exit('protocol or amendment changed after freeze')
    d = cx.load()
    split = json.loads((REPO / 'results/decision/h_eval/gate_e3/gate_e3_sc_group_split.json').read_text())['by_subject']
    bad = sorted({p for p in d['person'] if split.get(p) != 'SC_TRAIN'})
    if bad:
        sys.exit('DATA BOUNDARY: non-TRAIN persons in dataset: ' + ', '.join(bad))
    return d


# ------------------------------------------------------------------------ E5 features (exact numpy re-implementation)
_HANN400 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)
_HZ = np.arange(257) * 100 / 512
_BIDX = np.full(257, -1)
for _b, (_lo, _hi) in enumerate([(0.5, 4), (4, 8), (8, 13), (13, 30)]):
    _BIDX[(_HZ >= _lo) & (_HZ < _hi) & (np.arange(257) >= 1) & (np.arange(257) < 154)] = _b


def e5_channel(raw):
    """raw (n, 3000) uV -> (n, 5) = log1p band power (delta, theta, alpha, beta) + log1p AC RMS, as gate_e5."""
    P = np.zeros((len(raw), 4))
    for st in (0, 600, 1200, 1800, 2400):
        fr = raw[:, st:st + 400].astype(np.float64)
        fr = (fr - fr.mean(1, keepdims=True)) * _HANN400
        pw = np.abs(np.fft.rfft(fr, 512)) ** 2
        for b in range(4):
            P[:, b] += pw[:, _BIDX == b].sum(1)
    r = raw.astype(np.float64)
    rms = np.sqrt(((r - r.mean(1, keepdims=True)) ** 2).mean(1))
    return np.column_stack([np.log1p(P), np.log1p(rms)])


def e5_features(raw):
    return np.hstack([e5_channel(raw[:, 0]), e5_channel(raw[:, 1])])


# ------------------------------------------------------------------------ perturbations (uV in, uV out; float64)
FREQS = np.fft.rfftfreq(N, 1 / FS)


def _ac(x):
    return x - x.mean(-1, keepdims=True)


def band_gain(x, lo, hi, s):
    X = np.fft.rfft(x, axis=-1)
    X[..., (FREQS >= lo) & (FREQS < hi)] *= (1 - s)
    return np.fft.irfft(X, n=N, axis=-1)


def energy(x):
    return (_ac(x) ** 2).sum(-1, keepdims=True)


def rescale_energy(x, target_energy):
    m = x.mean(-1, keepdims=True)
    return m + _ac(x) * np.sqrt(target_energy / np.maximum(energy(x), 1e-12))


def _phases(rng, n, k):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, 1, k)))


def fourier_surrogate(x, rng, mask=None):
    """Multivariate: the same random phase per frequency added to both channels (preserves auto- and cross-spectra)."""
    X = np.fft.rfft(x, axis=-1)
    ph = _phases(rng, len(x), X.shape[-1])
    keep = np.zeros(X.shape[-1], bool)
    keep[0] = keep[-1] = True                                   # DC and Nyquist stay real
    if mask is not None:
        keep |= ~mask
    ph[..., keep] = 1.0
    return np.fft.irfft(X * ph, n=N, axis=-1)


_HANN200 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(200) / 200)   # periodic Hann (tests only)
_SINE200 = np.sin(np.pi * np.arange(200) / 200)                    # sqrt(periodic Hann): sum of squares = 1 at hop 100


def local_surrogate(x, rng):
    """2 s frames, 50 % overlap, random phases per frame (shared by channels); sine window at analysis and synthesis
    with gain sqrt(2) = 1/sqrt(mean(w^2)) (amendment 01): expected short-time band power preserved, sub-2 s
    waveform/phase structure destroyed."""
    pad = np.concatenate([x[..., 100:0:-1], x, x[..., -2:-102:-1]], -1)
    out = np.zeros_like(pad)
    for st in range(0, pad.shape[-1] - 199, 100):
        fr = pad[..., st:st + 200] * _SINE200
        Fz = np.fft.rfft(fr, axis=-1)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(len(x), 1, Fz.shape[-1])))
        ph[..., 0] = ph[..., -1] = 1.0
        out[..., st:st + 200] += np.sqrt(2.0) * np.fft.irfft(Fz * ph, n=200, axis=-1) * _SINE200
    return out[..., 100:100 + N]


def splice(x, seg, j, seglen=500, fade=10):
    """Replace segment j of x with seg's segment j (level-matched), raised-cosine crossfade of `fade` samples."""
    out = x.copy()
    a, b = j * seglen, (j + 1) * seglen
    s = seg[..., a:b] - seg[..., a:b].mean(-1, keepdims=True) + x[..., a:b].mean(-1, keepdims=True)
    w = np.ones(seglen)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    if a > 0:
        w[:fade] = ramp
    if b < N:
        w[-fade:] = ramp[::-1]
    out[..., a:b] = x[..., a:b] * (1 - w) + s * w
    return out


def artifacts(x, kind, amp, seed0):
    out = x.copy()
    t = np.arange(N) / FS
    for i in range(len(x)):
        rng = np.random.default_rng(seed0 + i)
        if kind == 'blink':
            for t0 in rng.uniform(0, 29.6, 3):
                k = (t >= t0) & (t < t0 + 0.4)
                out[i, 0, k] += amp * np.sin(np.pi * (t[k] - t0) / 0.4)
        elif kind == 'emg':
            t0 = rng.uniform(0, 27)
            k = (t >= t0) & (t < t0 + 3)
            for c in range(2):
                n = rng.normal(size=N)
                Xn = np.fft.rfft(n)
                Xn[(FREQS < 20) | (FREQS > 45)] = 0
                n = np.fft.irfft(Xn, n=N)
                seg = n[k] * np.hanning(k.sum())
                out[i, c, k] += seg * amp / np.sqrt((seg ** 2).mean())
        elif kind == 'pop':
            t0 = rng.uniform(1, 28)
            out[i, 1] += amp * np.exp(-(t - t0) / 1.0) * (t >= t0)
        elif kind == 'white':
            out[i] += rng.normal(0, amp, size=(2, N))
        else:
            raise ValueError(kind)
    return out


def conditions(d, donor):
    """Ordered (name, family, fn(raw)->raw). All randomness seeded and identical across models."""
    med_rms = np.median(np.sqrt((_ac(d['raw'].astype(np.float64)) ** 2).mean(-1)), axis=0)   # per channel
    C = [('P1_fourier_surrogate', 'P1', lambda x: fourier_surrogate(x, np.random.default_rng(PERT_SEED))),
         ('P2_local_surrogate_2s', 'P2', lambda x: local_surrogate(x, np.random.default_rng(PERT_SEED + 1)))]
    for b, (lo, hi) in BANDS.items():
        for s in STRENGTHS:
            C.append((f'P3_{b}_s{s}', 'P3', lambda x, lo=lo, hi=hi, s=s: band_gain(x, lo, hi, s)))
        C.append((f'P4_{b}_energy_matched', 'P4', lambda x, lo=lo, hi=hi: rescale_energy(band_gain(x, lo, hi, 1.0), energy(x))))
        C.append((f'P5_{b}_uniform_control', 'P5', lambda x, lo=lo, hi=hi: rescale_energy(x, energy(band_gain(x, lo, hi, 1.0)))))
        C.append((f'P6_{b}_phase', 'P6', lambda x, lo=lo, hi=hi: fourier_surrogate(
            x, np.random.default_rng(PERT_SEED + 2), mask=(FREQS >= lo) & (FREQS < hi))))
    for a in SCALES:
        C.append((f'P7_scale_{a}', 'P7', lambda x, a=a: x.mean(-1, keepdims=True) + a * _ac(x)))
    C.append(('P7_rms_normalised', 'P7', lambda x: x.mean(-1, keepdims=True) + _ac(x) * (
        med_rms[None, :, None] / np.maximum(np.sqrt((_ac(x) ** 2).mean(-1, keepdims=True)), 1e-9))))
    for c, nm in ((0, 'fpz'), (1, 'pz')):
        def dsub(x, c=c):
            y = x.copy()
            y[:, c] = x[donor, c]
            return y

        def zero(x, c=c):
            y = x.copy()
            y[:, c] = 0.0
            return y
        C.append((f'P8_{nm}_donor', 'P8', dsub))
        C.append((f'P8_{nm}_zero_ood', 'P8', zero))
    for j in range(6):
        C.append((f'P9a_seg{j}_donor', 'P9a', lambda x, j=j: splice(x, x[donor], j)))
        C.append((f'P9b_seg{j}_local_surrogate', 'P9b', lambda x, j=j: splice(
            x, local_surrogate(x, np.random.default_rng(PERT_SEED + 1)), j)))
    for kind, amps in (('blink', (50, 100, 200)), ('emg', (5, 10, 20)), ('pop', (50, 100, 200)), ('white', (5, 10, 20))):
        for a in amps:
            C.append((f'P10_{kind}_{a}uV', 'P10', lambda x, kind=kind, a=a: artifacts(x, kind, a, PERT_SEED + 10)))
    return C


# ------------------------------------------------------------------------ STFT model
def stft_logpow(xn):
    """xn: torch (n, 2, 3000) normalised raw -> (n, 2, 60, 57) log power, 0.5..30 Hz."""
    n = xn.shape[0]
    S = torch.stft(xn.reshape(-1, N), 200, 50, window=torch.hann_window(200, device=xn.device), center=False,
                   return_complex=True)
    P = torch.log(S.abs() ** 2 + 1e-6)[:, FREQ_KEEP]
    return P.reshape(n, 2, P.shape[1], P.shape[2])


class STFTNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        c = [2] + [int(round(width * k)) for k in CH_MULT]
        layers = []
        for i in range(len(CH_MULT)):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.GELU(), nn.GroupNorm(4, c[i + 1])]
            if i < len(CH_MULT) - 1:
                layers.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(2 * c[-1], 32)
        self.fuse = nn.Sequential(nn.Linear(32, 32), nn.GELU())
        self.head = nn.Linear(32, 1)

    def forward(self, spec, seq=None, tf=None):
        h = self.body(spec).flatten(2)
        h = F.gelu(self.out(torch.cat([h.mean(-1), torch.log(h.var(-1) + 1e-6)], 1)))
        return self.head(self.fuse(h))


def normalise_raw(raw_uv, scale):
    x = np.asarray(raw_uv, np.float32)
    return np.clip((x - x.mean(-1, keepdims=True)) / np.float32(scale), -10, 10)


class SpecStats:
    def __init__(self, raw_uv, rows, scale):
        self.scale = scale
        P = self._logpow(raw_uv[rows])
        self.mu = P.mean((0, 3), keepdim=True)
        self.sd = P.std((0, 3), keepdim=True) + 1e-6

    def _logpow(self, raw_uv):
        out = []
        for i in range(0, len(raw_uv), 512):
            x = torch.tensor(normalise_raw(raw_uv[i:i + 512], self.scale), device=DEV)
            out.append(stft_logpow(x))
        return torch.cat(out)

    def spec(self, raw_uv):
        return (self._logpow(raw_uv) - self.mu) / self.sd


class STFTTensors:
    """cx.Tensors-compatible container whose .raw holds the standardised spectrogram."""

    def __init__(self, d, tr):
        T = cx.Tensors(d, tr)
        self.__dict__.update(T.__dict__)
        self.stats = SpecStats(d['raw'], tr, T.scale)
        self.raw = self.stats.spec(d['raw'])


def train_generic(T, tr, va, cfg, make_model, seed, max_epochs=cx.MAX_EPOCHS):
    """Line-for-line the optimisation of cx.train_net (binary head), with an injected model constructor."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(cfg['width']).to(DEV)
    prior = float(T.y[torch.tensor(tr, device=DEV)].mean().clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        model.head.bias.fill_(float(np.log(prior / (1 - prior))))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    yv = T.y[torch.tensor(va, device=DEV)].cpu().numpy()
    best, best_state, bad, ep_run = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), cx.BATCH):
            b = torch.tensor(perm[s:s + cx.BATCH], device=DEV)
            z = model(T.raw[b], T.seq[b], T.tf[b])
            loss = F.binary_cross_entropy_with_logits(z[:, 0], T.y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ep_run = ep + 1
        score = float(((cx.predict_net(model, T, va, 'binary') - yv) ** 2).mean())
        if score < best - 1e-7:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cx.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, dict(inner_val_brier_ww=best, epochs=ep_run, train_s=round(time.time() - t0, 1),
                       n_params=int(sum(p.numel() for p in model.parameters())))


# ------------------------------------------------------------------------ fold model wrappers (predict from uV)
def m0_fit(X, y, tr):
    mu, sd = cx.e6.zfit(X[tr])
    w = cx.e6.logistic_fit_np(cx.e6.zapply(X[tr], mu, sd), y[tr].astype(float))
    return dict(mu=mu, sd=sd, w=w)


def m0_predict(M, X):
    return cx.e6.logistic_predict_np(M['w'], cx.e6.zapply(X, M['mu'], M['sd']))


def _logits(model, x):
    with torch.no_grad():
        return torch.cat([model(x[i:i + 512], None, None)[:, 0] for i in range(0, len(x), 512)])


def m2_logits(models, scale, raw_uv):
    x = torch.tensor(normalise_raw(raw_uv, scale), device=DEV)
    dummy_seq = torch.zeros((len(x), cx.H, 37), device=DEV)
    dummy_tf = torch.zeros((len(x), 2), device=DEV)
    out = []
    for m in models:
        m.eval()
        with torch.no_grad():
            out.append(torch.cat([m(x[i:i + 512], dummy_seq[i:i + 512], dummy_tf[i:i + 512])[:, 0]
                                  for i in range(0, len(x), 512)]).cpu().numpy().astype(np.float64))
    return np.array(out)


def stft_logits(models, stats, raw_uv):
    s = stats.spec(raw_uv)
    out = []
    for m in models:
        m.eval()
        out.append(_logits(m, s).cpu().numpy().astype(np.float64))
    return np.array(out)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------------ Part B: matched CV
def cmd_cv():
    d = check_inputs()
    folds = cx.person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    y = d['y']
    names = ['T0_constant', 'M0_outer_train', 'M0_inner_train', 'M2_raw_cnn', 'STFT_cnn']
    oof = {m: np.full(len(y), np.nan) for m in names}
    seed_oof = {m: {s: np.full(len(y), np.nan) for s in cx.SEEDS} for m in ('M2_raw_cnn', 'STFT_cnn')}
    log, fold_store = [], {}
    LOC.mkdir(parents=True, exist_ok=True)
    for f in range(cx.N_FOLDS):
        te = np.where(pf == f)[0]
        tr = np.where(pf != f)[0]
        inner_va = np.where(pf == (f + 1) % cx.N_FOLDS)[0]
        inner_tr = np.setdiff1d(tr, inner_va)
        oof['T0_constant'][te] = cx.fit_constant(tr, d)(te)
        M0o, M0i = m0_fit(d['e5'], y, tr), m0_fit(d['e5'], y, inner_tr)
        oof['M0_outer_train'][te] = m0_predict(M0o, d['e5'][te])
        assert np.allclose(oof['M0_outer_train'][te], cx.fit_m0(tr, d)(te))
        oof['M0_inner_train'][te] = m0_predict(M0i, d['e5'][te])
        Tm2 = cx.Tensors(d, inner_tr)
        Ts = STFTTensors(d, inner_tr)
        store = dict(m0_outer=M0o, m0_inner=M0i, m2_scale=Tm2.scale, constant=float(oof['T0_constant'][te][0]))
        for name, T, make in (('M2_raw_cnn', Tm2, None), ('STFT_cnn', Ts, STFTNet)):
            grid = []
            for cfg in cx.GRID:
                if make is None:
                    _, info = cx.train_net(T, inner_tr, inner_va, cfg, cx.ARCHS['M2_raw_cnn_binary'], seed=0)
                else:
                    _, info = train_generic(T, inner_tr, inner_va, cfg, make, seed=0)
                grid.append((info['inner_val_brier_ww'], cfg))
                log.append(dict(fold=f, model=name, stage='grid', cfg=cfg, **info))
            cfg = min(grid, key=lambda g: g[0])[1]
            states, preds = [], []
            for s in cx.SEEDS:
                if make is None:
                    model, info = cx.train_net(T, inner_tr, inner_va, cfg, cx.ARCHS['M2_raw_cnn_binary'], seed=s)
                else:
                    model, info = train_generic(T, inner_tr, inner_va, cfg, make, seed=s)
                p = cx.predict_net(model, T, te, 'binary')
                seed_oof[name][s][te] = p
                preds.append(p)
                states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
                log.append(dict(fold=f, model=name, stage='seed', seed=s, cfg=cfg, **info))
            oof[name][te] = np.mean(preds, 0)
            store[name] = dict(cfg=cfg, states=states)
        store['stft_mu'], store['stft_sd'] = Ts.stats.mu.cpu(), Ts.stats.sd.cpu()
        torch.save(store, LOC / f'fold{f}_models.pt')
        fold_store[f] = dict(te=te.tolist())
        print(f'[INTERP cv] fold {f} done', flush=True)
    np.savez(LOC / 'oof_v1.npz', **{k: v for k, v in oof.items()},
             **{f'{m}_seed{s}': v for m, ss in seed_oof.items() for s, v in ss.items()})
    res = summarise_cv(oof, seed_oof, log, d)
    write_json(OUT / 'interp_partB_cv_results_v1.json', res)
    print(json.dumps({m: round(v['brier_ww'], 5) for m, v in res['models'].items()}, indent=2))


def paired_boot(p_ref, p_cand, d, n_boot=BOOT_B, seed=SEED_B):
    old = cx.BOOT_N, cx.BOOT_SEED
    cx.BOOT_N, cx.BOOT_SEED = n_boot, seed
    try:
        out = cx.paired(p_ref, p_cand, d)
    finally:
        cx.BOOT_N, cx.BOOT_SEED = old
    out.pop('credible', None)
    return out


def summarise_cv(oof, seed_oof, log, d):
    logged = json.loads((REPO / 'results/decision/h_eval/cth_v1/cth_cv_results_v1.json').read_text())
    res = dict(version='H-EVAL-INTERP-v1-partB', protocol_sha256=PROTOCOL_SHA, status='POST_HOC_EXPLORATORY_TRAIN_ONLY',
               models={m: cx.metrics(p, d) for m, p in oof.items()},
               seeds={m: {str(s): cx.metrics(p, d)['brier_ww'] for s, p in ss.items()} for m, ss in seed_oof.items()},
               comparisons={}, per_fold_selected={}, compute={})
    for a, b in (('M2_raw_cnn', 'STFT_cnn'), ('M0_outer_train', 'STFT_cnn'), ('M0_outer_train', 'M2_raw_cnn'),
                 ('M0_inner_train', 'STFT_cnn'), ('M0_inner_train', 'M2_raw_cnn'), ('T0_constant', 'STFT_cnn'),
                 ('T0_constant', 'M2_raw_cnn'), ('T0_constant', 'M0_outer_train'), ('M0_inner_train', 'M0_outer_train')):
        res['comparisons'][f'{a}__vs__{b}'] = paired_boot(oof[a], oof[b], d)
    for r in log:
        if r['stage'] == 'seed':
            res['per_fold_selected'].setdefault(r['model'], {})[str(r['fold'])] = r['cfg']
            c = res['compute'].setdefault(r['model'], dict(n_params=[], epochs=[], train_s=[]))
            c['n_params'].append(r['n_params'])
            c['epochs'].append(r['epochs'])
            c['train_s'].append(r['train_s'])
    res['grid_log'] = [r for r in log if r['stage'] == 'grid']
    res['m2_replay_fidelity'] = dict(
        logged_cth_v1_m2_brier_ww=logged['models']['M2_raw_cnn_binary']['brier_ww'],
        replay_m2_brier_ww=res['models']['M2_raw_cnn']['brier_ww'],
        abs_diff=abs(logged['models']['M2_raw_cnn_binary']['brier_ww'] - res['models']['M2_raw_cnn']['brier_ww']),
        logged_m0=logged['models']['M0_e5_logistic']['brier_ww'], replay_m0=res['models']['M0_outer_train']['brier_ww'])
    res['data'] = dict(n_windows=int(len(d['y'])), n_positive=int(d['y'].sum()), n_persons=len(set(d['person'])))
    return res


# ------------------------------------------------------------------------ Part A: perturbations
def load_fold_models(f):
    st = torch.load(LOC / f'fold{f}_models.pt', map_location='cpu', weights_only=False)
    m2 = []
    for sd in st['M2_raw_cnn']['states']:
        m = cx.CTHNet(18, st['M2_raw_cnn']['cfg']['width'], cin=2, **cx.ARCHS['M2_raw_cnn_binary'])
        m.load_state_dict(sd)
        m2.append(m.to(DEV).eval())
    sf = []
    for sd in st['STFT_cnn']['states']:
        m = STFTNet(st['STFT_cnn']['cfg']['width'])
        m.load_state_dict(sd)
        sf.append(m.to(DEV).eval())
    stats = SpecStats.__new__(SpecStats)
    stats.scale, stats.mu, stats.sd = st['m2_scale'], st['stft_mu'].to(DEV), st['stft_sd'].to(DEV)
    return st, m2, sf, stats


def oof_predict(d, raw, fold_models, pf):
    """Out-of-fold predictions for perturbed raw: seed-level logits and ensemble p for M2, STFT; M0 p."""
    n = len(raw)
    out = dict(M2=np.full((3, n), np.nan), STFT=np.full((3, n), np.nan), M0=np.full(n, np.nan))
    for f, (st, m2, sf, stats) in fold_models.items():
        te = np.where(pf == f)[0]
        r = raw[te]
        out['M2'][:, te] = sigmoid(m2_logits(m2, st['m2_scale'], r))
        out['STFT'][:, te] = sigmoid(stft_logits(sf, stats, r))
        out['M0'][te] = m0_predict(st['m0_outer'], e5_features(r))
    return out


def boot_indices(d, n_boot=BOOT_A, seed=SEED_A):
    pu = sorted(set(d['person']))
    idx = np.searchsorted(pu, d['person'])
    n = np.bincount(idx, minlength=len(pu)).astype(float)
    I = np.random.default_rng(seed).integers(0, len(pu), size=(n_boot, len(pu)))
    return pu, idx, n, I


def score_condition(p_orig, p_pert, p_const, d, B):
    pu, idx, n, I = B
    y = d['y'].astype(float)
    sums = lambda p: np.bincount(idx, weights=(p - y) ** 2, minlength=len(pu))  # noqa: E731
    so, sp, sc = sums(p_orig), sums(p_pert), sums(p_const)
    No = n[I].sum(1)
    bo, bp, bc = so[I].sum(1) / No, sp[I].sum(1) / No, sc[I].sum(1) / No
    rho_b = (bc - bp) / (bc - bo)
    Bo, Bp, Bc = so.sum() / n.sum(), sp.sum() / n.sum(), sc.sum() / n.sum()
    rec = [cx.auroc(p_pert[d['rec_idx'] == r], y[d['rec_idx'] == r]) for r in np.unique(d['rec_idx'])]
    return dict(brier_ww=float(Bp), delta_brier_ww=float(Bp - Bo), delta_brier_ci95=[float(x) for x in np.percentile(bp - bo, [2.5, 97.5])],
                retention=float((Bc - Bp) / (Bc - Bo)), retention_ci95=[float(x) for x in np.percentile(rho_b, [2.5, 97.5])],
                delta_mean_p=float(p_pert.mean() - p_orig.mean()),
                within_rec_auroc=float(np.mean([a for a in rec if a is not None])))


def cmd_perturb():
    d = check_inputs()
    oofz = np.load(LOC / 'oof_v1.npz')
    folds = cx.person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    fold_models = {f: load_fold_models(f) for f in range(cx.N_FOLDS)}
    raw = d['raw'].astype(np.float64)
    donor = np.empty(len(raw), int)
    for f in range(cx.N_FOLDS):
        te = np.where(pf == f)[0]
        plan, _ = base.donor_plan([dict(person=d['person'][i], elapsed_s=float(d['elapsed'][i])) for i in te], 0)
        donor[te] = te[np.array(plan)]
    base_pred = oof_predict(d, raw, fold_models, pf)
    # identity check: perturbation harness reproduces Part B OOF predictions
    fid = dict(M2=float(np.abs(base_pred['M2'].mean(0) - oofz['M2_raw_cnn']).max()),
               STFT=float(np.abs(base_pred['STFT'].mean(0) - oofz['STFT_cnn']).max()),
               M0=float(np.abs(base_pred['M0'] - oofz['M0_outer_train']).max()))
    e5_check = float(np.abs(e5_features(raw) - d['e5']).max())
    p_const = oofz['T0_constant']
    fprep, fmodels = c2.load_final(FROZEN_FINAL)
    fmodels = [m.to(DEV).eval() for m in fmodels]
    frozen_orig = sigmoid(m2_logits(fmodels, fprep['raw_scale_uv'], raw)).mean(0)
    B = boot_indices(d)
    y = d['y'].astype(float)
    res = dict(version='H-EVAL-INTERP-v1-partA-perturbation', protocol_sha256=PROTOCOL_SHA,
               status='POST_HOC_EXPLORATORY_TRAIN_ONLY', harness_identity_max_abs_diff=fid, e5_numpy_max_abs_diff=e5_check,
               donor_rule='E7B elapsed-time-matched donor (draw 0) restricted to other persons in the same test fold',
               conditions={}, band_specificity={}, seed_retention_range={}, frozen_m2_in_sample={})
    per_person = {}
    for name, fam, fn in conditions(d, donor):
        t0 = time.time()
        xp = fn(raw)
        pr = oof_predict(d, xp, fold_models, pf)
        row = {}
        for m, po in (('M2', base_pred['M2'].mean(0)), ('STFT', base_pred['STFT'].mean(0)), ('M0', base_pred['M0'])):
            pp = pr[m].mean(0) if m != 'M0' else pr[m]
            row[m] = score_condition(po, pp, p_const, d, B)
            pu, idx, n, _ = B
            per_person.setdefault(name, {})[m] = (np.bincount(idx, weights=(pp - y) ** 2, minlength=len(pu)) / n).tolist()
        for m in ('M2', 'STFT'):
            rr = []
            for s in range(3):
                bo, bp = ((base_pred[m][s] - y) ** 2).mean(), ((pr[m][s] - y) ** 2).mean()
                bc = ((p_const - y) ** 2).mean()
                rr.append(float((bc - bp) / (bc - bo)))
            res['seed_retention_range'].setdefault(name, {})[m] = rr
        fp = sigmoid(m2_logits(fmodels, fprep['raw_scale_uv'], xp)).mean(0)
        res['frozen_m2_in_sample'][name] = dict(delta_brier_ww=float(((fp - y) ** 2).mean() - ((frozen_orig - y) ** 2).mean()),
                                                delta_mean_p=float(fp.mean() - frozen_orig.mean()))
        row['family'] = fam
        res['conditions'][name] = row
        print(f'[INTERP perturb] {name}: M2 rho {row["M2"]["retention"]:.3f} STFT {row["STFT"]["retention"]:.3f} '
              f'M0 {row["M0"]["retention"]:.3f} ({time.time() - t0:.1f}s)', flush=True)
    # band specificity: P3 (s=1) vs matched-energy uniform control P5, paired per person
    pu, idx, n, I = B
    for b in BANDS:
        for m in ('M2', 'STFT', 'M0'):
            a = np.array(per_person[f'P3_{b}_s1.0'][m])
            c = np.array(per_person[f'P5_{b}_uniform_control'][m])
            diff_b = ((a - c) * n)[I].sum(1) / n[I].sum(1)
            res['band_specificity'].setdefault(b, {})[m] = dict(
                delta_brier_P3_minus_P5=float(((a - c) * n).sum() / n.sum()),
                ci95=[float(x) for x in np.percentile(diff_b, [2.5, 97.5])])
    # natural quality: 30-50 Hz power fraction tertiles
    X = np.abs(np.fft.rfft(_ac(raw), axis=-1)) ** 2
    hf = (X[..., FREQS >= 30].sum(-1) / X[..., FREQS >= 0.5].sum(-1)).mean(1)
    cuts = np.quantile(hf, [1 / 3, 2 / 3])
    terc = np.digitize(hf, cuts)
    res['natural_quality_hf_tertiles'] = {}
    for t in range(3):
        k = terc == t
        bc = ((p_const[k] - y[k]) ** 2).mean()
        res['natural_quality_hf_tertiles'][str(t)] = dict(
            hf_fraction_range=[float(hf[k].min()), float(hf[k].max())], n_windows=int(k.sum()),
            skill_vs_constant={m: float(1 - ((p[k] - y[k]) ** 2).mean() / bc) for m, p in
                               (('M2', base_pred['M2'].mean(0)), ('STFT', base_pred['STFT'].mean(0)), ('M0', base_pred['M0']))})
    write_json(OUT / 'interp_partA_perturbation_v1.json', res)
    np.savez(LOC / 'perturb_per_person_v1.npz', persons=np.array(B[0]),
             **{f'{c}__{m}': np.array(v) for c, mv in per_person.items() for m, v in mv.items()})


# ------------------------------------------------------------------------ Part A: explanations (frozen M2_46_FINAL)
def integrated_gradients(model, x, steps=32, batch=256):
    """x: torch (n, 2, 3000) preprocessed; baseline 0. Returns IG (n, 2, 3000) for the logit."""
    model.eval()
    out = torch.zeros_like(x)
    dseq = torch.zeros((batch, cx.H, 37), device=x.device)
    dtf = torch.zeros((batch, 2), device=x.device)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        g = torch.zeros_like(xb)
        for a in (torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps:
            xi = (a * xb).detach().requires_grad_(True)
            z = model(xi, dseq[:len(xb)], dtf[:len(xb)])[:, 0].sum()
            g += torch.autograd.grad(z, xi)[0]
        out[i:i + batch] = xb * g / steps
    return out


def bin_map(ig):
    return ig.abs().reshape(len(ig), 2, 30, 100).sum(-1).reshape(len(ig), 60)


def spearman_rows(A, Bm):
    ra = np.argsort(np.argsort(A, 1), 1).astype(float)
    rb = np.argsort(np.argsort(Bm, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)


def person_mean_ci(v, d, n_boot=BOOT_A, seed=SEED_A):
    pu = sorted(set(d['person']))
    idx = np.searchsorted(pu, d['person'])
    pm = np.bincount(idx, weights=v, minlength=len(pu)) / np.bincount(idx, minlength=len(pu))
    I = np.random.default_rng(seed).integers(0, len(pu), size=(n_boot, len(pu)))
    return dict(mean_of_person_means=float(pm.mean()), ci95=[float(x) for x in np.percentile(pm[I].mean(1), [2.5, 97.5])],
                n_persons=len(pu))


def randomise(model, stage, seed=999):
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    groups = [['head'], ['fuse'], ['raw.out'], ['raw.body'], ['raw.stems']]
    names = [n for g in groups[:stage] for n in g]
    for nm, mod in m.named_modules():
        if any(nm == g or nm.startswith(g + '.') for g in names) and hasattr(mod, 'reset_parameters'):
            mod.reset_parameters()
    return m.eval()


def label_randomised_model(d, scale):
    T = cx.Tensors(d, np.arange(len(d['y'])))
    T.y = T.y[torch.tensor(np.random.default_rng(7).permutation(len(d['y'])), device=DEV)]
    torch.manual_seed(0)
    np.random.seed(0)
    model = cx.CTHNet(T.n_feat, 32, cin=2, **cx.ARCHS['M2_raw_cnn_binary']).to(DEV)
    c2.init_bias(model, T, np.arange(len(d['y'])))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(4):
        c2._epoch(model, opt, T, np.arange(len(d['y'])), rng)
    assert abs(T.scale - scale) < 1e-3
    return model.eval()


def cmd_explain():
    d = check_inputs()
    raw = d['raw'].astype(np.float64)
    prep, models = c2.load_final(FROZEN_FINAL)
    models = [m.to(DEV).eval() for m in models]
    x = torch.tensor(normalise_raw(raw, prep['raw_scale_uv']), device=DEV)
    maps = [bin_map(integrated_gradients(m, x)).cpu().numpy() for m in models]
    absx = x.abs().reshape(len(x), 2, 30, 100).sum(-1).reshape(len(x), 60).cpu().numpy()
    res = dict(version='H-EVAL-INTERP-v1-partA-explanations', protocol_sha256=PROTOCOL_SHA,
               status='POST_HOC_EXPLORATORY_TRAIN_ONLY; frozen M2_46_FINAL evaluated IN-SAMPLE on TRAIN',
               method='Integrated Gradients, baseline 0 (mean-removed flat input), 32 midpoint steps, |IG| in 60 bins (2 ch x 30 s)')
    # E2 seed agreement
    res['E2_seed_agreement'] = {f'{a}-{b}': person_mean_ci(spearman_rows(maps[a], maps[b]), d) for a, b in ((0, 1), (0, 2), (1, 2))}
    res['IG_vs_abs_input'] = {f'seed{s}': person_mean_ci(spearman_rows(maps[s], absx), d) for s in range(3)}
    # E3 randomisation sanity
    res['E3_cascading_randomisation'] = {}
    for stage, label in enumerate(['head', '+fuse', '+raw.out', '+raw.body', '+raw.stems (full)'], start=1):
        mr = randomise(models[0], stage)
        mp = bin_map(integrated_gradients(mr, x)).cpu().numpy()
        res['E3_cascading_randomisation'][label] = dict(ig_similarity_to_trained=person_mean_ci(spearman_rows(maps[0], mp), d),
                                                        ig_similarity_to_abs_input=person_mean_ci(spearman_rows(mp, absx), d))
    ml = label_randomised_model(d, prep['raw_scale_uv'])
    mp = bin_map(integrated_gradients(ml, x)).cpu().numpy()
    res['E3_label_randomised_model'] = dict(ig_similarity_to_trained=person_mean_ci(spearman_rows(maps[0], mp), d),
                                            ig_similarity_to_abs_input=person_mean_ci(spearman_rows(mp, absx), d))
    # E4 deletion fidelity (all seeds; donor bins from an elapsed-time-matched other person)
    plan, _ = base.donor_plan([dict(person=d['person'][i], elapsed_s=float(d['elapsed'][i])) for i in range(len(raw))], 0)
    xd = torch.tensor(normalise_raw(raw[np.array(plan)], prep['raw_scale_uv']), device=DEV)
    res['E4_deletion_fidelity'] = {}
    for s, m in enumerate(models):
        z0 = _logits_m2(m, x)
        order = np.argsort(-maps[s], 1)
        rng = np.random.default_rng(SEED_A + s)
        per_k = {}
        for k in (3, 6, 12, 24):
            dz = {}
            for label, sel in (('top', order[:, :k]), ('bottom', order[:, -k:])):
                dz[label] = np.abs(_logits_m2(m, _replace_bins(x, xd, sel)) - z0)
            dz['random'] = np.mean([np.abs(_logits_m2(m, _replace_bins(x, xd, np.argsort(rng.random((len(x), 60)), 1)[:, :k])) - z0)
                                    for _ in range(5)], 0)
            per_k[str(k)] = dict(top=person_mean_ci(dz['top'], d), random=person_mean_ci(dz['random'], d),
                                 bottom=person_mean_ci(dz['bottom'], d),
                                 top_minus_random=person_mean_ci(dz['top'] - dz['random'], d),
                                 random_minus_bottom=person_mean_ci(dz['random'] - dz['bottom'], d))
        res['E4_deletion_fidelity'][f'seed{s}'] = per_k
    # E5 cross-method agreement (seed-ensemble IG; P8 / P9a donor responses of the frozen ensemble)
    ens_map = np.mean(maps, 0)
    zf = np.mean([_logits_m2(m, x) for m in models], 0)
    ig_ch = ens_map[:, :30].sum(1) / ens_map.sum(1)
    occ = {}
    for c in (0, 1):
        xp = raw.copy()
        xp[:, c] = raw[np.array(plan), c]
        xt = torch.tensor(normalise_raw(xp, prep['raw_scale_uv']), device=DEV)
        occ[c] = np.abs(np.mean([_logits_m2(m, xt) for m in models], 0) - zf)
    occ_ch = occ[0] / (occ[0] + occ[1] + 1e-12)
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    res['E5_channel_agreement'] = dict(
        spearman_across_windows=float(np.corrcoef(rk(ig_ch), rk(occ_ch))[0, 1]),
        ig_fpz_share=person_mean_ci(ig_ch, d), occlusion_fpz_share=person_mean_ci(occ_ch, d))
    seg_ig = ens_map.reshape(len(x), 2, 6, 5).sum((1, 3))
    seg_occ = np.zeros((len(x), 6))
    for j in range(6):
        xt = torch.tensor(normalise_raw(splice(raw, raw[np.array(plan)], j), prep['raw_scale_uv']), device=DEV)
        seg_occ[:, j] = np.abs(np.mean([_logits_m2(m, xt) for m in models], 0) - zf)
    res['E5_time_agreement'] = dict(per_window_spearman=person_mean_ci(spearman_rows(seg_ig, seg_occ), d),
                                    ig_segment_share=[person_mean_ci(seg_ig[:, j] / seg_ig.sum(1), d) for j in range(6)],
                                    occlusion_abs_dlogit=[person_mean_ci(seg_occ[:, j], d) for j in range(6)])
    res['IG_channel_time_profile'] = dict(fpz=ens_map[:, :30].mean(0).tolist(), pz=ens_map[:, 30:].mean(0).tolist())
    # spectral-model comparison: standardised weights of frozen E7B_18 and M0_46
    e7b = json.loads((REPO / 'results/decision/h_eval/gate_e7b/frozen_models_v1.json').read_text())
    m0 = json.loads((REPO / 'results/decision/h_eval/cth_v1/frozen_m0_46_v1.json').read_text())
    feat = [f'{c}_{b}' for c in ('fpz', 'pz') for b in ('logpow_delta', 'logpow_theta', 'logpow_alpha', 'logpow_beta', 'log_rms')]
    res['spectral_models_standardised_weights'] = dict(
        features=feat, E7B_18=dict(zip(['intercept'] + feat, e7b['weights']['EEG_LOGIT'])),
        M0_46=dict(zip(['intercept'] + feat, m0['weights'])))
    write_json(OUT / 'interp_partA_explanations_v1.json', res)
    np.save(LOC / 'ig_maps_frozen_v1.npy', np.array(maps))
    print(json.dumps(dict(E2={k: round(v['mean_of_person_means'], 3) for k, v in res['E2_seed_agreement'].items()},
                          E3={k: round(v['ig_similarity_to_trained']['mean_of_person_means'], 3)
                              for k, v in res['E3_cascading_randomisation'].items()}), indent=2))


def _logits_m2(model, x):
    dseq = torch.zeros((512, cx.H, 37), device=x.device)
    dtf = torch.zeros((512, 2), device=x.device)
    with torch.no_grad():
        return torch.cat([model(x[i:i + 512], dseq[:len(x[i:i + 512])], dtf[:len(x[i:i + 512])])[:, 0]
                          for i in range(0, len(x), 512)]).cpu().numpy().astype(np.float64)


def _replace_bins(x, xd, sel):
    """Replace 1 s bins (index = channel*30 + second) with the donor's bins (preprocessed domain)."""
    out = x.clone()
    n = len(x)
    rows = np.repeat(np.arange(n), sel.shape[1])
    ch, sec = (sel // 30).ravel(), (sel % 30).ravel()
    xv = out.view(n, 2, 30, 100)
    dv = xd.view(n, 2, 30, 100)
    r = torch.tensor(rows, device=x.device)
    c = torch.tensor(ch, device=x.device)
    s = torch.tensor(sec, device=x.device)
    xv[r, c, s] = dv[r, c, s]
    return out


# ------------------------------------------------------------------------ POST-HOC follow-ups F1/F2 (see followup note)
FOLLOWUP_NOTE = REPO / 'docs/decision/h_eval/interp_v1/interp_followup_F1_F2_note.md'


def band_gain_ch(x, lo, hi, s, ch):
    y = x.copy()
    y[:, ch] = band_gain(x[:, ch:ch + 1], lo, hi, s)[:, 0]
    return y


def cmd_followup():
    d = check_inputs()
    oofz = np.load(LOC / 'oof_v1.npz')
    folds = cx.person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    fold_models = {f: load_fold_models(f) for f in range(cx.N_FOLDS)}
    raw = d['raw'].astype(np.float64)
    base_pred = oof_predict(d, raw, fold_models, pf)
    p_const = oofz['T0_constant']
    B = boot_indices(d)
    lo_d, hi_d = BANDS['delta']
    fpz_phase = lambda x: np.concatenate([fourier_surrogate(x, np.random.default_rng(PERT_SEED + 2),  # noqa: E731
                                                            mask=(FREQS >= lo_d) & (FREQS < hi_d))[:, :1], x[:, 1:]], 1)
    conds = [('F2a_infraslow_0.03-0.5Hz', lambda x: band_gain(x, 0.03, 0.5, 1.0)),
             ('F2b_delta_low_0.5-2Hz', lambda x: band_gain(x, 0.5, 2.0, 1.0)),
             ('F2c_delta_high_2-4Hz', lambda x: band_gain(x, 2.0, 4.0, 1.0)),
             ('F2d_delta_fpz_only', lambda x: band_gain_ch(x, lo_d, hi_d, 1.0, 0)),
             ('F2e_delta_pz_only', lambda x: band_gain_ch(x, lo_d, hi_d, 1.0, 1)),
             ('F2f_delta_phase_fpz_only', fpz_phase)]
    res = dict(version='H-EVAL-INTERP-v1-followup-F1-F2', status='POST_HOC (defined after Part A results)',
               note_sha256=digest(FOLLOWUP_NOTE), F2={}, F1={})
    for name, fn in conds:
        pr = oof_predict(d, fn(raw), fold_models, pf)
        res['F2'][name] = {m: score_condition(po, pr[m].mean(0) if m != 'M0' else pr[m], p_const, d, B)
                           for m, po in (('M2', base_pred['M2'].mean(0)), ('STFT', base_pred['STFT'].mean(0)), ('M0', base_pred['M0']))}
        print(name, {m: round(v['retention'], 3) for m, v in res['F2'][name].items()}, flush=True)
    # F1: deletion with model-agnostic rankings (frozen M2_46_FINAL seed 0, in-sample)
    prep, models = c2.load_final(FROZEN_FINAL)
    m = models[0].to(DEV).eval()
    x = torch.tensor(normalise_raw(raw, prep['raw_scale_uv']), device=DEV)
    plan, _ = base.donor_plan([dict(person=d['person'][i], elapsed_s=float(d['elapsed'][i])) for i in range(len(raw))], 0)
    xd = torch.tensor(normalise_raw(raw[np.array(plan)], prep['raw_scale_uv']), device=DEV)
    ig = np.load(LOC / 'ig_maps_frozen_v1.npy')[0]
    absx = x.abs().reshape(len(x), 2, 30, 100).sum(-1).reshape(len(x), 60).cpu().numpy()
    ig_rand = bin_map(integrated_gradients(randomise(m, 5), x)).cpu().numpy()
    z0 = _logits_m2(m, x)
    rng = np.random.default_rng(SEED_A)
    for k in (3, 6, 12, 24):
        dz = {lab: np.abs(_logits_m2(m, _replace_bins(x, xd, np.argsort(-mp, 1)[:, :k])) - z0)
              for lab, mp in (('ig_trained', ig), ('abs_input', absx), ('ig_randomised_model', ig_rand))}
        dz['random'] = np.mean([np.abs(_logits_m2(m, _replace_bins(x, xd, np.argsort(rng.random((len(x), 60)), 1)[:, :k])) - z0)
                                for _ in range(5)], 0)
        res['F1'][str(k)] = {lab: person_mean_ci(v, d) for lab, v in dz.items()}
        res['F1'][str(k)]['ig_minus_abs_input'] = person_mean_ci(dz['ig_trained'] - dz['abs_input'], d)
        res['F1'][str(k)]['ig_minus_ig_randomised'] = person_mean_ci(dz['ig_trained'] - dz['ig_randomised_model'], d)
    write_json(OUT / 'interp_followup_F1_F2_v1.json', res)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f'refusing to overwrite {path}')
    path.write_text(json.dumps(obj, indent=2, default=float) + '\n')
    print('wrote', path.relative_to(REPO), digest(path))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cv', 'perturb', 'explain', 'followup'])
    a = ap.parse_args(argv)
    dict(cv=cmd_cv, perturb=cmd_perturb, explain=cmd_explain, followup=cmd_followup)[a.cmd]()


if __name__ == '__main__':
    main()
