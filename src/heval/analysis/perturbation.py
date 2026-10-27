"""Perturbation analysis (paper Fig. 8c-d; development data only): out-of-fold skill retention under input perturbations.

Every TRAIN window is perturbed and re-scored by the models of its own CV fold (RawCNN and STFT-CNN seed ensembles,
SpecLR on recomputed spectral features). Skill retention rho = (B_const - B_pert) / (B_const - B_orig) (1 = unchanged,
< 0 = worse than the constant), with a 95% person-cluster bootstrap (2,000 replicates, seed 20261103).
Battery (all random draws seeded, identical across models):
- P1, 30 s multivariate Fourier surrogate (phase randomised, auto- and cross-spectra kept);
- P2, 2 s local surrogate (sine-windowed frames, 50% overlap, gain sqrt(2));
- P3, band removal per band at strengths 0.25-1;
- P4, band removal rescaled to the original energy;
- P5, uniform energy reduction matched to P3 (the control for P3);
- P6, in-band phase randomisation;
- P7, amplitude scaling and RMS normalisation;
- P8, channel donor / zeroing;
- P9, 5 s segment donor / local-surrogate splices;
- P10, synthetic blink, EMG, pop and white-noise artefacts.
Band specificity compares P3 (full removal) with P5 by the paired person-level Brier difference. The post-hoc
follow-up F2 splits the delta band and restricts removal / phase randomisation to one channel.

Lineage: INTERP v1 (perturbations, conditions, oof_predict, score_condition, band specificity, follow-up F2) of the
research archive, tag v0.1.0-paper-review. Arithmetic and seeds unchanged; the device is a parameter.
"""
import numpy as np
import torch

from ..evaluation.heldout import donor_plan
from ..evaluation.metrics import auroc
from ..models import raw_cnn
from ..models.ablation_networks import ARCHS, CTHNet
from ..models.stft_cnn import SpecStats, STFTNet, logits
from .representation import m0_predict

FS, N = 100, 3000
BANDS = dict(delta=(0.5, 4.0), theta=(4.0, 8.0), alpha=(8.0, 12.0), sigma=(12.0, 15.0), beta=(15.0, 30.0), gamma=(30.0, 50.01))
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
SCALES = (0.5, 0.7, 0.85, 1.2, 1.4, 2.0)
PERT_SEED = 20261102
BOOT_A, SEED_A = 2_000, 20261103
FREQS = np.fft.rfftfreq(N, 1 / FS)
FIGURE_CONDITIONS = ['P1_fourier_surrogate', 'P2_local_surrogate_2s'] + [f'P3_{b}_s1.0' for b in BANDS] + ['P6_delta_phase']

# exact NumPy form of the 10 spectral features (used on perturbed windows; matches the pure-Python reference to ~1e-7)
_HANN400 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)
_HZ = np.arange(257) * 100 / 512
_BIDX = np.full(257, -1)
for _b, (_lo, _hi) in enumerate([(0.5, 4), (4, 8), (8, 13), (13, 30)]):
    _BIDX[(_HZ >= _lo) & (_HZ < _hi) & (np.arange(257) >= 1) & (np.arange(257) < 154)] = _b
_SINE200 = np.sin(np.pi * np.arange(200) / 200)                    # sqrt(periodic Hann): sum of squares = 1 at hop 100


def spectral_channel_np(raw):
    """raw (n, 3000) uV -> (n, 5) = log1p band power (delta, theta, alpha, beta) + log1p AC RMS."""
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


def spectral_features_np(raw):
    return np.hstack([spectral_channel_np(raw[:, 0]), spectral_channel_np(raw[:, 1])])


# ------------------------------------------------------------------------ perturbations (uV in, uV out; float64)
def _ac(x):
    return x - x.mean(-1, keepdims=True)


def band_gain(x, lo, hi, s):
    X = np.fft.rfft(x, axis=-1)
    X[..., (FREQS >= lo) & (FREQS < hi)] *= (1 - s)
    return np.fft.irfft(X, n=N, axis=-1)


def band_gain_ch(x, lo, hi, s, ch):
    y = x.copy()
    y[:, ch] = band_gain(x[:, ch:ch + 1], lo, hi, s)[:, 0]
    return y


def energy(x):
    return (_ac(x) ** 2).sum(-1, keepdims=True)


def rescale_energy(x, target_energy):
    m = x.mean(-1, keepdims=True)
    return m + _ac(x) * np.sqrt(target_energy / np.maximum(energy(x), 1e-12))


def fourier_surrogate(x, rng, mask=None):
    """The same random phase per frequency for both channels (keeps auto- and cross-spectra); mask limits the bins."""
    X = np.fft.rfft(x, axis=-1)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(len(x), 1, X.shape[-1])))
    keep = np.zeros(X.shape[-1], bool)
    keep[0] = keep[-1] = True                                   # DC and Nyquist stay real
    if mask is not None:
        keep |= ~mask
    ph[..., keep] = 1.0
    return np.fft.irfft(X * ph, n=N, axis=-1)


def local_surrogate(x, rng):
    """2 s frames, 50% overlap, random phases per frame (shared by channels), sine window at analysis and synthesis
    with gain sqrt(2): short-time band power kept, sub-2 s waveform and phase structure destroyed."""
    pad = np.concatenate([x[..., 100:0:-1], x, x[..., -2:-102:-1]], -1)
    out = np.zeros_like(pad)
    for st in range(0, pad.shape[-1] - 199, 100):
        Fz = np.fft.rfft(pad[..., st:st + 200] * _SINE200, axis=-1)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(len(x), 1, Fz.shape[-1])))
        ph[..., 0] = ph[..., -1] = 1.0
        out[..., st:st + 200] += np.sqrt(2.0) * np.fft.irfft(Fz * ph, n=200, axis=-1) * _SINE200
    return out[..., 100:100 + N]


def splice(x, seg, j, seglen=500, fade=10):
    """Replace 5 s segment j of x with seg's segment j (level-matched), raised-cosine crossfade of `fade` samples."""
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


def conditions(raw, donor):
    """Ordered (name, family, fn(raw) -> raw). raw: all windows (uV); donor: donor window index for each window."""
    med_rms = np.median(np.sqrt((_ac(np.asarray(raw, np.float64)) ** 2).mean(-1)), axis=0)   # per channel
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


def followup_conditions():
    """Post-hoc follow-up F2 (defined after the main battery): delta sub-bands and single-channel delta perturbations."""
    lo_d, hi_d = BANDS['delta']

    def fpz_phase(x):
        return np.concatenate([fourier_surrogate(x, np.random.default_rng(PERT_SEED + 2),
                                                 mask=(FREQS >= lo_d) & (FREQS < hi_d))[:, :1], x[:, 1:]], 1)
    return [('F2a_infraslow_0.03-0.5Hz', 'F2', lambda x: band_gain(x, 0.03, 0.5, 1.0)),
            ('F2b_delta_low_0.5-2Hz', 'F2', lambda x: band_gain(x, 0.5, 2.0, 1.0)),
            ('F2c_delta_high_2-4Hz', 'F2', lambda x: band_gain(x, 2.0, 4.0, 1.0)),
            ('F2d_delta_fpz_only', 'F2', lambda x: band_gain_ch(x, lo_d, hi_d, 1.0, 0)),
            ('F2e_delta_pz_only', 'F2', lambda x: band_gain_ch(x, lo_d, hi_d, 1.0, 1)),
            ('F2f_delta_phase_fpz_only', 'F2', fpz_phase)]


# ------------------------------------------------------------------------ out-of-fold scoring
def fold_predictors(store, device='cpu'):
    """A matched_cv fold store (or one rebuilt from saved state dicts) -> (store, RawCNN models, STFT models, stats)."""
    def build(entry, make):
        if 'models' in entry:
            return [m.to(device).eval() for m in entry['models']]
        out = []
        for sd in entry['states']:
            m = make(entry['cfg']['width'])
            m.load_state_dict(sd)
            out.append(m.to(device).eval())
        return out
    m2 = build(store['M2_raw_cnn'], lambda w: CTHNet(18, w, cin=2, **ARCHS['M2_raw_cnn_binary']))
    sf = build(store['STFT_cnn'], STFTNet)
    stats = SpecStats.from_state(store['m2_scale'], store['stft_mu'], store['stft_sd'], device)
    return store, m2, sf, stats


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def oof_predict(raw, fold_models, pf, device='cpu'):
    """Out-of-fold probabilities for (perturbed) raw: per-seed for RawCNN and STFT-CNN (n_seeds, n), SpecLR (n)."""
    n, k = len(raw), len(next(iter(fold_models.values()))[1])
    out = dict(M2=np.full((k, n), np.nan), STFT=np.full((k, n), np.nan), M0=np.full(n, np.nan))
    for f, (st, m2, sf, stats) in fold_models.items():
        te = np.where(pf == f)[0]
        r = raw[te]
        out['M2'][:, te] = sigmoid(logits(m2, torch.tensor(raw_cnn.normalise(r, st['m2_scale']), device=device)))
        out['STFT'][:, te] = sigmoid(logits(sf, stats.spec(r)))
        out['M0'][te] = m0_predict(st['m0_outer'], spectral_features_np(r))
    return out


def boot_indices(person, n_boot=BOOT_A, seed=SEED_A):
    pu = sorted(set(person))
    idx = np.searchsorted(pu, person)
    n = np.bincount(idx, minlength=len(pu)).astype(float)
    return pu, idx, n, np.random.default_rng(seed).integers(0, len(pu), size=(n_boot, len(pu)))


def score_condition(p_orig, p_pert, p_const, y, rec_idx, B):
    pu, idx, n, I = B
    y = np.asarray(y, float)
    sums = lambda p: np.bincount(idx, weights=(p - y) ** 2, minlength=len(pu))  # noqa: E731
    so, sp, sc = sums(p_orig), sums(p_pert), sums(p_const)
    No = n[I].sum(1)
    bo, bp, bc = so[I].sum(1) / No, sp[I].sum(1) / No, sc[I].sum(1) / No
    rho_b = (bc - bp) / (bc - bo)
    Bo, Bp, Bc = so.sum() / n.sum(), sp.sum() / n.sum(), sc.sum() / n.sum()
    rec = [auroc(p_pert[rec_idx == r], y[rec_idx == r]) for r in np.unique(rec_idx)]
    return dict(brier_ww=float(Bp), delta_brier_ww=float(Bp - Bo),
                delta_brier_ci95=[float(x) for x in np.percentile(bp - bo, [2.5, 97.5])],
                retention=float((Bc - Bp) / (Bc - Bo)), retention_ci95=[float(x) for x in np.percentile(rho_b, [2.5, 97.5])],
                delta_mean_p=float(p_pert.mean() - p_orig.mean()), within_rec_auroc=float(np.mean([a for a in rec if a is not None])))


def fold_donors(person, elapsed, pf):
    """Elapsed-time-matched donor (draw 0) among other persons of the same CV test fold."""
    donor = np.empty(len(person), int)
    for f in np.unique(pf):
        te = np.where(pf == f)[0]
        if len({person[i] for i in te}) < 2:
            raise ValueError(f'fold {f} contains a single person: donor perturbations need at least two persons per fold')
        plan, _ = donor_plan([person[i] for i in te], [float(elapsed[i]) for i in te], 0)
        donor[te] = te[np.array(plan)]
    return donor


def run_battery(d, fold_models, pf, p_const, conds, device='cpu'):
    """Score each perturbation for the three models; returns (results by condition, per-person Brier by condition)."""
    raw = d['raw'].astype(np.float64)
    y = d['y'].astype(float)
    base = oof_predict(raw, fold_models, pf, device)
    orig = {'M2': base['M2'].mean(0), 'STFT': base['STFT'].mean(0), 'M0': base['M0']}
    B = boot_indices(d['person'])
    pu, idx, n, _ = B
    res, per_person = {}, {}
    for name, fam, fn in conds:
        pr = oof_predict(fn(raw), fold_models, pf, device)
        row = {}
        for m, po in orig.items():
            pp = pr[m].mean(0) if m != 'M0' else pr[m]
            row[m] = score_condition(po, pp, p_const, y, d['rec_idx'], B)
            per_person.setdefault(name, {})[m] = np.bincount(idx, weights=(pp - y) ** 2, minlength=len(pu)) / n
        res[name] = dict(row, family=fam)
    return res, per_person, base


def band_specificity(per_person, person, n_boot=BOOT_A, seed=SEED_A):
    """Per band and model: person-weighted Brier difference of full band removal (P3) minus its energy control (P5)."""
    pu, idx, n, I = boot_indices(person, n_boot, seed)
    out = {}
    for b in BANDS:
        for m in ('M2', 'STFT', 'M0'):
            a, c = np.array(per_person[f'P3_{b}_s1.0'][m]), np.array(per_person[f'P5_{b}_uniform_control'][m])
            diff_b = ((a - c) * n)[I].sum(1) / n[I].sum(1)
            out.setdefault(b, {})[m] = dict(delta_brier_P3_minus_P5=float(((a - c) * n).sum() / n.sum()),
                                            ci95=[float(x) for x in np.percentile(diff_b, [2.5, 97.5])])
    return out
