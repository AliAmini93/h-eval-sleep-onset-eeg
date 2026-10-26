"""TRAIN-only exploratory wake-to-first-N1 analysis (estimands E1-E3, references, recalibration, warnings).

Estimands (all oracle-defined by scorer labels; TRAIN windows only):
  E1  wake -> first N1: windows with t < T1 (first scored sleep, always Stage 1 here), label 1{T1 - t <= 300 s}
  E2  wake input -> first N2: windows with S(t) = W (pre-sleep t < T1 and returning wake t >= T1), label as E0
  E3  N1 -> N2: windows with S(t) = 1, label as E0
References: TRAIN-prevalence constant and a time-only cubic-spline logistic model of log elapsed time (knots {3, 5} x
C {0.1, 1, 10} chosen by 4-fold person-disjoint inner CV); both fitted within the estimand's population.
Outer evaluation: 5 person-disjoint folds (two salted partitions A and B), paired Brier skill with a person-cluster
bootstrap (seed 20261024, 10,000 replicates).
Recalibration of E1 scores (nested, leakage-safe): K1 intercept, K2 intercept + slope (penalty towards slope 1),
K3 = K2 + standardised log elapsed time; penalty chosen by leave-one-inner-fold-out Brier.

Lineage (research archive, tag v0.1.0-paper-review): W2S table builder (features, estimands), W2S evaluation and
classical families, P0a time-matched AUROC diagnostic and P0b calibration / warnings. Unchanged arithmetic.
"""
import hashlib
import math

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

FS = 100.0
BOOT_N, BOOT_SEED = 10_000, 20261024
N_FOLDS, N_INNER = 5, 4
FOLD_SALTS = {'A': 'H-EVAL-CTH-v1-folds', 'B': 'H-EVAL-W2S-folds-B'}
INNER_SALT = 'H-EVAL-W2S-inner'
ESTIMANDS = {'E1': ('e1', 'y1'), 'E2': ('e2', 'y'), 'E3': ('e3', 'y')}
EVAL_SUBSETS = {'E1': ['e1'], 'E2': ['e2', 'e2pre', 'e2ret'], 'E3': ['e3']}
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TIME_GRID = [(k, c) for k in (3, 5) for c in (0.1, 1.0, 10.0)]
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
LEAD_BINS = ((0, 60), (60, 120), (120, 180), (180, 300))


# ------------------------------------------------------------------------ estimands
def estimands(stage, elapsed, first_sleep_s, tau, y):
    """Per-window masks and labels. stage: scored input stage ('W'/'1'); first_sleep_s: per-window first-sleep time
    of its recording (s after lights-off); tau: time to first N2; y: E0 label."""
    stage, el, fs = np.asarray(stage), np.asarray(elapsed, float), np.asarray(first_sleep_s, float)
    e1 = el < fs - 1e-6
    y1 = (e1 & (fs - el <= 300 + 1e-8)).astype(int)
    k1 = np.where(e1 & (fs - el <= 300 + 1e-8), np.clip(np.ceil((fs - el) / 30 - 1e-9), 1, 10), 11).astype(int)
    e2 = stage == 'W'
    return dict(e1=e1, y1=y1, k1=k1, e2=e2, e2pre=e2 & e1, e2ret=e2 & ~e1, e3=stage == '1', y=np.asarray(y),
                tau=np.asarray(tau, float), elapsed=el, first_sleep_s=fs)


# ------------------------------------------------------------------------ window features of the classical families
def perm_entropy(x, order=5, delay=1):
    n = len(x) - (order - 1) * delay
    emb = np.stack([x[i * delay:i * delay + n] for i in range(order)], 1)
    codes = (np.argsort(emb, 1) * (order ** np.arange(order))).sum(1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def dfa(x, scales=(16, 25, 40, 63, 100, 160, 250, 400)):
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        n = len(y) // s
        seg = y[:n * s].reshape(n, s)
        A = np.vstack([np.arange(s), np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        F.append(np.sqrt(((seg.T - A @ coef) ** 2).mean()))
    return float(np.polyfit(np.log(scales), np.log(np.array(F) + 1e-12), 1)[0])


def hjorth(x):
    d1, d2 = np.diff(x), np.diff(x, 2)
    v0, v1, v2 = x.var() + 1e-12, d1.var() + 1e-12, d2.var() + 1e-12
    mob = np.sqrt(v1 / v0)
    return float(mob), float(np.sqrt(v2 / v1) / mob)


def channel_features(x):
    """18 aperiodic/periodic, relative-band, entropy, Hjorth and DFA features of one 30 s channel."""
    x = x - x.mean()
    f, P = welch(x, fs=FS, nperseg=200, noverlap=100)
    band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum()   # noqa: E731
    tot = band(0.5, 30) + 1e-12
    fit = (f >= 2) & (f <= 30) & ~((f >= 7) & (f <= 14))
    lf, lp = np.log10(f[fit]), np.log10(P[fit] + 1e-12)
    slope, off = np.polyfit(lf, lp, 1)
    pred = off + slope * lf
    r2 = 1 - ((lp - pred) ** 2).sum() / (((lp - lp.mean()) ** 2).sum() + 1e-12)
    ap = lambda m: off + slope * np.log10(f[m])           # noqa: E731
    th, al = (f >= 4) & (f < 8), (f >= 8) & (f < 12)
    th_res = float(np.max(np.log10(P[th] + 1e-12) - ap(th)))
    al_res = float(np.max(np.log10(P[al] + 1e-12) - ap(al)))
    m = (f >= 0.5) & (f <= 30)
    pn = P[m] / (P[m].sum() + 1e-12)
    sent = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    cum = np.cumsum(P[m]) / (P[m].sum() + 1e-12)
    sef95 = float(f[m][np.searchsorted(cum, 0.95)])
    mob, comp = hjorth(x)
    return [-slope, off, r2, th_res, al_res, band(0.5, 4) / tot, band(4, 8) / tot, band(8, 12) / tot,
            band(12, 15) / tot, band(15, 30) / tot, np.log(tot), sent, sef95, mob, comp,
            perm_entropy(x, 5, 1), perm_entropy(x, 5, 4), dfa(x)]


FEAT_NAMES = ['aperiodic_exponent', 'aperiodic_offset', 'aperiodic_r2', 'theta_periodic_resid', 'alpha_periodic_resid',
              'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta', 'log_total_power', 'spectral_entropy',
              'sef95', 'hjorth_mobility', 'hjorth_complexity', 'perm_entropy_d1', 'perm_entropy_d4', 'dfa_alpha']
SOS = {b: butter(4, lims, btype='band', fs=FS, output='sos') for b, lims in (('theta', (4, 8)), ('alpha', (8, 12)))}


def plv(x1, x2, band):
    """Phase-locking value between channels; zero-phase filtering is applied within the 30 s window only."""
    a, b = sosfiltfilt(SOS[band], x1), sosfiltfilt(SOS[band], x2)
    return float(np.abs(np.exp(1j * (np.angle(hilbert(a)) - np.angle(hilbert(b)))).mean()))


def window_features(raw):
    """(N, 2, 3000) windows -> (N, 38): 18 features per channel + theta and alpha PLV."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (2, 3000):
        raise ValueError('expected (N, 2, 3000) windows')
    out = np.zeros((len(raw), 2 * len(FEAT_NAMES) + 2))
    for i, w in enumerate(raw):
        x0, x1 = w[0].astype(float), w[1].astype(float)
        out[i] = channel_features(x0) + channel_features(x1) + [plv(x0, x1, 'theta'), plv(x0, x1, 'alpha')]
    return out


def history_features(F, hist):
    """Causal: mean of the previous <= 4 windows and OLS slope over <= 10 windows (current included); 0 if unavailable."""
    n, d = F.shape
    mean_prev, slope = np.zeros((n, d)), np.zeros((n, d))
    for i in range(n):
        idx = hist[i][hist[i] >= 0]
        prev = idx[:-1][-4:]
        if len(prev):
            mean_prev[i] = F[prev].mean(0)
        if len(idx) >= 3:
            t = np.arange(len(idx), dtype=float)
            t -= t.mean()
            slope[i] = (t[:, None] * (F[idx] - F[idx].mean(0))).sum(0) / (t ** 2).sum()
    return mean_prev, slope


# ------------------------------------------------------------------------ folds and references
def folds(persons, salt):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{salt}|{p}'.encode()).hexdigest())
    f = {p: i % N_FOLDS for i, p in enumerate(ranked)}
    return np.array([f[p] for p in persons])


def inner_folds(persons):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{INNER_SALT}|{p}'.encode()).hexdigest())
    f = {p: i % N_INNER for i, p in enumerate(ranked)}
    return np.array([f[p] for p in persons])


def time_feats(t):
    t = np.asarray(t, float)
    return np.column_stack([np.log1p(t / 60), t / 3600])


def fit_const(y):
    p = (y.sum() + 1) / (len(y) + 2)
    return lambda n: np.full(n, p)


def spline_logit(t_tr, y_tr, t_te, knots, C, extra_tr=None, extra_te=None):
    st = SplineTransformer(n_knots=knots, degree=3, extrapolation='constant').fit(np.log(t_tr)[:, None])
    Xtr, Xte = st.transform(np.log(t_tr)[:, None]), st.transform(np.log(t_te)[:, None])
    if extra_tr is not None:
        Xtr, Xte = np.hstack([Xtr, extra_tr]), np.hstack([Xte, extra_te])
    if y_tr.min() == y_tr.max():
        return np.full(len(t_te), (y_tr.sum() + 1) / (len(y_tr) + 2))
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y_tr)
    return np.clip(clf.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)


def select_inner(fit_predict, configs, idx, y, persons):
    """Configuration with the lowest inner-CV squared error (ties: first in list)."""
    fi = inner_folds(persons[idx])
    best, best_s = None, np.inf
    for c in configs:
        s = 0.0
        for g in range(N_INNER):
            a, b = idx[fi != g], idx[fi == g]
            s += ((fit_predict(a, b, c) - y[b]) ** 2).sum()
        if s < best_s - 1e-12:
            best, best_s = c, s
    return best


def classical_oof(d, part):
    """Out-of-fold predictions of CONST_pop, TIME_pop, ORACLE_state (diagnostic), C1 logistic and C2 boosting
    for each estimand. d needs person, elapsed, first_sleep_s, the estimand masks/labels and feat_cur,
    feat_mean_prev, feat_slope (window_features / history_features)."""
    pf = folds(d['person'], FOLD_SALTS[part])
    X = np.hstack([d['feat_cur'], d['feat_mean_prev'], d['feat_slope'], time_feats(d['elapsed'])])
    t = d['elapsed']
    post = (t >= d['first_sleep_s']).astype(float)
    oracle_extra = np.column_stack([post, np.log1p(np.maximum(t - d['first_sleep_s'], 0) / 60)])
    out = {}
    for E, (mk, yk) in ESTIMANDS.items():
        pop, y = d[mk].astype(bool), d[yk].astype(float)
        P = {k: np.full(len(y), np.nan) for k in ('CONST_pop', 'TIME_pop', 'ORACLE_state', 'C1_logistic', 'C2_hgb')}
        for f in range(N_FOLDS):
            tr, te = np.where(pop & (pf != f))[0], np.where(pop & (pf == f))[0]
            P['CONST_pop'][te] = fit_const(y[tr])(len(te))
            cfg = select_inner(lambda a, b, c: spline_logit(t[a], y[a], t[b], *c), TIME_GRID, tr, y, d['person'])
            P['TIME_pop'][te] = spline_logit(t[tr], y[tr], t[te], *cfg)
            P['ORACLE_state'][te] = spline_logit(t[tr], y[tr], t[te], 3, 1.0, oracle_extra[tr], oracle_extra[te])

            def c1(a, b, C):
                mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
                clf = LogisticRegression(C=C, max_iter=5000).fit((X[a] - mu) / sd, y[a])
                return np.clip(clf.predict_proba((X[b] - mu) / sd)[:, 1], 1e-6, 1 - 1e-6)
            P['C1_logistic'][te] = c1(tr, te, select_inner(c1, C_GRID, tr, y, d['person']))
            hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                 min_samples_leaf=40, early_stopping=False, random_state=0)
            P['C2_hgb'][te] = np.clip(hgb.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        out[E] = P
    return out


# ------------------------------------------------------------------------ evaluation
def calib(p, y):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = None
    if 0 < y.sum() < len(y) and lp.std() > 1e-9:
        slope = float(LogisticRegression(C=1e6, max_iter=2000).fit(lp[:, None], y).coef_[0, 0])
    return float(p.mean() - y.mean()), slope


def model_metrics(p, y, person):
    pu = sorted(set(person))
    b = (p - y) ** 2
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pc) + (1 - y) * np.log1p(-pc))
    cil, slope = calib(p, y)
    return dict(brier_ww=float(b.mean()), brier_se=float(np.mean([b[person == q].mean() for q in pu])),
                logloss_ww=float(ll.mean()), cal_in_large=cil, cal_slope=slope)


def paired(p_ref, p_c, y, person):
    pu = sorted(set(person))
    k = np.searchsorted(pu, person)
    n = np.bincount(k, minlength=len(pu)).astype(float)
    sr, sc = np.bincount(k, (p_ref - y) ** 2, len(pu)), np.bincount(k, (p_c - y) ** 2, len(pu))
    per = (sr - sc) / n
    I = np.random.default_rng(BOOT_SEED).integers(0, len(pu), size=(BOOT_N, len(pu)))
    Br, Bc = sr[I].sum(1), sc[I].sum(1)
    q = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]   # noqa: E731
    loo = [((sr.sum() - sr[i]) - (sc.sum() - sc[i])) / (sr.sum() - sr[i]) for i in range(len(pu))]
    return dict(R=float((sr.sum() - sc.sum()) / sr.sum()), R_ci95=q((Br - Bc) / Br),
                delta_ww=float((sr.sum() - sc.sum()) / n.sum()), delta_se=float(per.mean()),
                delta_se_ci95=q(per[I].mean(1)), persons_favouring=int((per > 0).sum()), n_persons=len(pu),
                loo_R_min=float(min(loo)))


def evaluate(name, preds, refs, d, y, mask):
    """preds/refs: {model: full-length out-of-fold arrays}; refs must contain 'CONST_pop' and 'TIME_pop'."""
    pm = d['person'][mask]
    out = dict(subset=name, n_windows=int(mask.sum()), n_positive=int(y[mask].sum()), n_persons=len(set(pm)),
               n_recordings=len(set(d['recording'][mask])), models={}, vs_const={}, vs_time={})
    for k, p in {**refs, **preds}.items():
        out['models'][k] = model_metrics(p[mask], y[mask], pm)
    for k, p in preds.items():
        out['vs_const'][k] = paired(refs['CONST_pop'][mask], p[mask], y[mask], pm)
        out['vs_time'][k] = paired(refs['TIME_pop'][mask], p[mask], y[mask], pm)
    return out


def warnings(p, theta, y, rec, el, mask):
    """Fixed-threshold alarms (p >= theta) on mask: alarm windows per hour, false-alarm EPISODES per hour (maximal
    runs of consecutive alarms with no positive window), and the share of events with at least one alarmed positive."""
    alarm = (p >= theta) & mask
    hours = mask.sum() * 30 / 3600
    ep_fa, det, nev = 0, 0, 0
    for r in np.unique(rec[mask]):
        idx = np.where(mask & (rec == r))[0]
        idx = idx[np.argsort(el[idx])]
        a = alarm[idx]
        run = []
        for j, flag in enumerate(a):
            if flag:
                run.append(idx[j])
            if (not flag or j == len(a) - 1) and run:
                if y[run].sum() == 0:
                    ep_fa += 1
                run = []
        if y[idx].sum():
            nev += 1
            det += int(alarm[idx][y[idx] == 1].any())
    return dict(alarm_positive_windows_per_hour=float((alarm & (y == 0)).sum() / hours),
                false_alarm_episodes_per_hour=float(ep_fa / hours), event_sensitivity=det / nev if nev else None,
                n_events=nev)


def pair_index(strata, fold, pid, y, within_fold):
    """Cross-person (positive, negative) pairs in the same stratum (and the same outer fold if within_fold)."""
    pi, ni = [], []
    key = strata if not within_fold else strata * 10 + fold
    for s in np.unique(key):
        idx = np.where(key == s)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if not len(pos) or not len(neg):
            continue
        P, N = np.meshgrid(pos, neg, indexing='ij')
        keep = pid[P] != pid[N]
        pi.append(P[keep])
        ni.append(N[keep])
    if not pi:
        return np.array([], int), np.array([], int)
    return np.concatenate(pi), np.concatenate(ni)


def auc_pairs(p, pi, ni):
    return float(((p[pi] > p[ni]) + 0.5 * (p[pi] == p[ni])).mean())


def lead_time_auroc(p, d, m, pf, tte):
    """AUROC of positives in each lead-time bin against far negatives (> 600 s before first N1), using within-fold
    exact-landmark cross-person pairs."""
    per, el = d['person'], d['elapsed']
    out = {}
    far = m & (tte > 600)
    for lo, hi in LEAD_BINS:
        pos = m & (tte > lo) & (tte <= hi)
        sub = pos | far
        yy = pos[sub].astype(int)
        pid2 = np.searchsorted(sorted(set(per[sub])), per[sub])
        a, b = pair_index((el[sub] // 30).astype(int), pf[sub], pid2, yy, True)
        out[f'({lo},{hi}]'] = dict(n_windows=int(pos.sum()), n_persons=len(set(per[pos])), mean_pred=float(p[pos].mean()),
                                   auroc_vs_far=(auc_pairs(p[sub], a, b) if len(a) else None), n_pairs=int(len(a)))
    return out


def level_within_pooled(p, y, person, m):
    """Window-averaged level (person mean miscalibration) and within-person components of the Brier score."""
    lev = wit = 0.0
    for q in sorted(set(person[m])):
        i = person[m] == q
        pm, ym = p[m][i], y[m][i]
        pb, yb = pm.mean(), ym.mean()
        lev += i.sum() * (pb - yb) ** 2
        wit += (((pm - pb) - (ym - yb)) ** 2).sum()
    return lev / m.sum(), wit / m.sum()


# ------------------------------------------------------------------------ nested recalibration
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_cal(kind, s, u, y, lam):
    """Penalised log-loss calibrator on scores s (logits) and standardised log time u -> (a, b, c)."""
    if kind == 'K1':
        f = lambda th: -np.mean(y * -np.logaddexp(0, -(s + th[0])) + (1 - y) * -np.logaddexp(0, s + th[0]))  # noqa: E731
        th = minimize(f, [0.0], method='BFGS').x
        return th[0], 1.0, 0.0

    def obj(th):
        a, b = th[0], th[1]
        c = th[2] if kind == 'K3' else 0.0
        z = a + b * s + c * u
        return np.mean(np.logaddexp(0, z) - y * z) + lam * ((b - 1) ** 2 + c ** 2) / len(y)
    th = minimize(obj, [0.0, 1.0, 0.0] if kind == 'K3' else [0.0, 1.0], method='BFGS').x
    return th[0], th[1], (th[2] if kind == 'K3' else 0.0)


def apply_cal(params, s, u):
    a, b, c = params
    return np.clip(1 / (1 + np.exp(-(a + b * s + c * u))), 1e-6, 1 - 1e-6)


def calibrate_fold(kind, s_cal, u_cal, y_cal, g_cal, s_te, u_te):
    """Choose lambda by leave-one-inner-fold-out Brier on the calibration set (K2/K3), refit, apply to the test fold."""
    if kind not in ('K1', 'K2', 'K3'):
        raise ValueError('calibrator kind must be K1, K2 or K3')
    lam = None
    if kind in ('K2', 'K3'):
        best = np.inf
        for L in LAMBDAS:
            sc = 0.0
            for g in np.unique(g_cal):
                a, b = g_cal != g, g_cal == g
                sc += ((apply_cal(fit_cal(kind, s_cal[a], u_cal[a], y_cal[a], L), s_cal[b], u_cal[b]) - y_cal[b]) ** 2).sum()
            if sc < best - 1e-12:
                best, lam = sc, L
    params = fit_cal(kind, s_cal, u_cal, y_cal, lam or 0.0)
    return apply_cal(params, s_te, u_te), dict(lambda_=lam, a=float(params[0]), b=float(params[1]), c=float(params[2]))


def recalibrate_e1(outer, inner, d, pf, kind):
    """Nested recalibration of E1 out-of-fold scores. outer: full-length outer OOF probabilities; inner: {fold f:
    full-length inner-OOF probabilities on the outer-training persons of fold f}."""
    m, y, el = d['e1'].astype(bool), d['y1'].astype(float), d['elapsed']
    u_all = np.log1p(el / 60)
    out, params = np.full(len(y), np.nan), {}
    for f in range(N_FOLDS):
        cal, te = m & (pf != f), m & (pf == f)
        pin = inner[f]
        if not np.isfinite(pin[cal]).all():
            raise ValueError(f'inner predictions missing for calibration windows of fold {f}')
        uc = u_all[cal]
        mu, sd = uc.mean(), uc.std() + 1e-9
        out[te], params[f] = calibrate_fold(kind, logit(pin[cal]), (uc - mu) / sd, y[cal], pf[cal],
                                            logit(outer[te]), (u_all[te] - mu) / sd)
    return out, params
