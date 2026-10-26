"""SpecLR: logistic regression on ten log band-power / amplitude features of the two EEG channels.

Features per channel (30 s at 100 Hz): five 4 s Hann frames (starts 0, 6, 12, 18, 24 s), each mean-removed and
zero-padded to 512 points; summed periodogram power in 0.5-4, 4-8, 8-13 and 13-30 Hz; features log(1 + band power)
and log(1 + AC RMS of the whole window). The 10-vector is channel_features(Fpz-Cz) + channel_features(Pz-Oz),
i.e. the order of FEATURE_NAMES.
Estimator: training-set z-score (population SD, floor 1e-6, clipped to +-5) and full-batch gradient-descent logistic
regression (350 steps, learning rate 0.3 / (1 + step / 200), L2 0.025 on non-intercept weights, intercept started
at the Laplace-smoothed prevalence); predictions clipped to [1e-6, 1 - 1e-6].

Lineage (research archive, tag v0.1.0-paper-review): fft_real / channel_features from E5 (pure Python, kept
verbatim so features are bit-identical); NumPy estimator from E6 (checked there against E5); the five-model frozen
family (constant, time, SpecLR, SpecLR + time, drift-residual SpecLR) from E7B. SpecLR-46 is the same recipe fitted
on all 46 TRAIN persons.
"""
import math
import statistics

import numpy as np

FEATURE_NAMES = ('fpz_log_delta', 'fpz_log_theta', 'fpz_log_alpha', 'fpz_log_beta', 'fpz_log_rms',
                 'pz_log_delta', 'pz_log_theta', 'pz_log_alpha', 'pz_log_beta', 'pz_log_rms')
MODELS = ('CONSTANT', 'TIME_CONT', 'EEG_LOGIT', 'TIME_CONT_EEG_LOGIT', 'EEG_RESID_LOGIT')


# ------------------------------------------------------------------------ features (exact pure-Python reference)
def fft_real(x):
    n = len(x)
    if n == 0 or n & (n - 1):
        raise ValueError('FFT length must be a power of 2')
    out = [complex(v) for v in x]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    length = 2
    while length <= n:
        wlen = complex(math.cos(-2 * math.pi / length), math.sin(-2 * math.pi / length))
        half = length >> 1
        for i in range(0, n, length):
            w = 1 + 0j
            for j in range(i, i + half):
                u, v = out[j], out[j + half] * w
                out[j], out[j + half] = u + v, u - v
                w *= wlen
        length *= 2
    return out


def channel_features(samples):
    """One channel, 3000 samples (30 s, 100 Hz, microvolts) -> [log1p delta, theta, alpha, beta power, log1p AC RMS]."""
    if len(samples) != 3000 or any(not math.isfinite(x) for x in samples):
        raise ValueError('Invalid fixed 30s, 100Hz EEG window')
    bpower = [0., 0., 0., 0.]
    w = [.5 - .5 * math.cos(2 * math.pi * i / 399) for i in range(400)]
    for start in (0, 600, 1200, 1800, 2400):
        frame = samples[start:start + 400]
        mean = statistics.fmean(frame)
        x = [(v - mean) * wi for v, wi in zip(frame, w)] + [0.] * 112
        f = fft_real(x)
        for k in range(1, 154):
            hz = k * 100. / 512.
            idx = 0 if 0.5 <= hz < 4 else 1 if 4 <= hz < 8 else 2 if 8 <= hz < 13 else 3 if 13 <= hz < 30 else -1
            if idx >= 0:
                bpower[idx] += f[k].real * f[k].real + f[k].imag * f[k].imag
    avg = statistics.fmean(samples)
    rms = math.sqrt(statistics.fmean((v - avg) ** 2 for v in samples))
    return [math.log(1. + p) for p in bpower] + [math.log(1. + rms)]


def window_features(window):
    """(2, 3000) [Fpz-Cz, Pz-Oz] window -> 10 features."""
    window = np.asarray(window, float)
    if window.shape != (2, 3000):
        raise ValueError('expected a (2, 3000) [Fpz-Cz, Pz-Oz] window at 100 Hz')
    return channel_features(window[0].tolist()) + channel_features(window[1].tolist())


# ------------------------------------------------------------------------ estimator
def time_cont_raw(t):
    return np.array([math.log1p(t / 60.), t / 3600.])


def zfit(X):
    mu = X.mean(axis=0)
    sd = np.sqrt(((X - mu) ** 2).mean(axis=0))
    return mu, np.maximum(sd, 1e-6)


def zapply(X, mu, sd):
    return np.clip((X - mu) / sd, -5., 5.)


def sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1. / (1. + np.exp(-np.minimum(60., x[pos])))
    z = np.exp(np.maximum(-60., x[~pos]))
    out[~pos] = z / (1. + z)
    return out


def logistic_fit_np(X, y, steps=350, penalty=.025, lr=.3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) == 0 or len(X) != len(y):
        raise ValueError('Logistic feature matrix invalid')
    n, d = X.shape
    p = (y.sum() + 1) / (n + 2)
    w = np.zeros(d + 1)
    w[0] = math.log(p / (1 - p))
    for step in range(steps):
        err = sigmoid_np(w[0] + X @ w[1:]) - y
        g0, g = err.sum(), X.T @ err
        eta = lr / (1. + step / 200.)
        w[0] -= eta * g0 / n
        w[1:] -= eta * (g / n + penalty * w[1:])
    return w


def logistic_predict_np(w, X):
    return np.clip(sigmoid_np(w[0] + np.asarray(X, dtype=float) @ w[1:]), 1e-6, 1 - 1e-6)


# ------------------------------------------------------------------------ frozen five-model family
def _matrices(features, elapsed_s, y=None):
    F = np.asarray(features, float)
    if F.ndim != 2 or F.shape[1] != 10:
        raise ValueError('features must have shape (n, 10)')
    if not np.isfinite(F).all():
        raise ValueError('features contain non-finite values')
    T = np.array([time_cont_raw(t) for t in np.asarray(elapsed_s, float)]).reshape(len(F), 2)
    if len(T) != len(F) or (y is not None and len(y) != len(F)):
        raise ValueError('features, elapsed_s and y must have the same length')
    return F, T


def fit_speclr(features, elapsed_s, y):
    """Fit the frozen model family on training windows. 'EEG_LOGIT' is SpecLR; 'CONSTANT' is the smoothed prevalence."""
    y = np.asarray(y, float)
    if not set(np.unique(y)) <= {0., 1.}:
        raise ValueError('y must be binary 0/1')
    F, T = _matrices(features, elapsed_s, y)
    prior = (y.sum() + 1) / (len(y) + 2)
    cmu, csd = zfit(T)
    mu, sd = zfit(F)
    C, Z = zapply(T, cmu, csd), zapply(F, mu, sd)
    D = np.column_stack([np.ones(len(T)), T])
    beta = np.linalg.lstsq(D, F, rcond=None)[0]
    R = F - D @ beta
    rmu, rsd = zfit(R)
    RZ = zapply(R, rmu, rsd)
    w = {'TIME_CONT': logistic_fit_np(C, y), 'EEG_LOGIT': logistic_fit_np(Z, y),
         'TIME_CONT_EEG_LOGIT': logistic_fit_np(np.hstack([C, Z]), y), 'EEG_RESID_LOGIT': logistic_fit_np(RZ, y)}
    L = lambda a: [float(x) for x in np.ravel(a)]  # noqa: E731
    return dict(constant_prior=float(prior), time_features=dict(mu=L(cmu), sd=L(csd)), eeg_features=dict(mu=L(mu), sd=L(sd)),
                drift=dict(beta=[L(beta[i]) for i in range(beta.shape[0])], resid_mu=L(rmu), resid_sd=L(rsd)),
                weights={k: L(v) for k, v in w.items()}, n_windows=int(len(y)), n_positive=int(y.sum()))


def predict_speclr(model, features, elapsed_s):
    """Apply a fitted family (fit_speclr output) to windows -> {model name: probabilities}."""
    F, T = _matrices(features, elapsed_s)
    a = lambda v: np.array(v, float)  # noqa: E731
    C = zapply(T, a(model['time_features']['mu']), a(model['time_features']['sd']))
    Z = zapply(F, a(model['eeg_features']['mu']), a(model['eeg_features']['sd']))
    R = F - np.column_stack([np.ones(len(T)), T]) @ a(model['drift']['beta'])
    RZ = zapply(R, a(model['drift']['resid_mu']), a(model['drift']['resid_sd']))
    W = {k: a(v) for k, v in model['weights'].items()}
    return {'CONSTANT': np.full(len(F), model['constant_prior']),
            'TIME_CONT': logistic_predict_np(W['TIME_CONT'], C),
            'EEG_LOGIT': logistic_predict_np(W['EEG_LOGIT'], Z),
            'TIME_CONT_EEG_LOGIT': logistic_predict_np(W['TIME_CONT_EEG_LOGIT'], np.hstack([C, Z])),
            'EEG_RESID_LOGIT': logistic_predict_np(W['EEG_RESID_LOGIT'], RZ)}
