"""Constant, Stage-only and Stage+time references.

The Stage-only and Stage+time references use the retrospectively SCORED stage of the input epoch (W or Stage 1).
They are oracle diagnostics for interpreting EEG skill, not deployable forecasters and not performance ceilings.

Stage-only (A): Laplace-smoothed event rate per scored state, (events + 1) / (windows + 2).
Stage+time (B): L2 logistic regression (scikit-learn, max_iter 5000) on [Stage-1 indicator, cubic B-spline of
log elapsed time with uniform knots on the TRAIN log-time range and constant extrapolation (time clamped to the TRAIN
support)] with optional spline x stage interaction. Configuration chosen from knots {3, 5} x C {0.1, 1, 10} x
interaction {no, yes} by 4-fold person-disjoint inner cross-validation on TRAIN (sum of squared errors; ties go to the
earlier grid entry). The paper's TRAIN fit selected 5 knots, C = 10, no interaction.

Lineage: post-hoc reference audit R3B (research archive, tag v0.1.0-paper-review), functions unchanged.
"""
import hashlib

import numpy as np
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression

GRID = [(k, C, inter) for k in (3, 5) for C in (0.1, 1.0, 10.0) for inter in (False, True)]
SEL_SALT, N_SEL = 'H-EVAL-R3B-select', 4


def fit_constant(y):
    y = np.asarray(y, float)
    return float((y.sum() + 1) / (len(y) + 2))


def _check(t=None, s1=None, y=None):
    for name, v in (('elapsed time', t), ('stage indicator', s1), ('label', y)):
        if v is None:
            continue
        v = np.asarray(v, float)
        if not np.isfinite(v).all():
            raise ValueError(f'{name} contains non-finite values')
        if name == 'elapsed time' and (v <= 0).any():
            raise ValueError('elapsed time must be positive (seconds after lights-off)')
        if name != 'elapsed time' and not set(np.unique(v)) <= {0., 1.}:
            raise ValueError(f'{name} must be 0/1')


def spline_basis(logt, lo, hi, k):
    """Cubic B-spline basis, uniform knots on [lo, hi], constant extrapolation (x clipped to [lo, hi])."""
    step = (hi - lo) / (k - 1)
    knots = np.r_[lo - 3 * step + np.arange(3) * step, np.linspace(lo, hi, k), hi + step * np.arange(1, 4)]
    x = np.clip(np.asarray(logt, float), lo, hi)
    return BSpline.design_matrix(x, knots, 3).toarray()


def design(t, s1, lo, hi, k, inter):
    S = spline_basis(np.log(t), lo, hi, k)
    X = [np.asarray(s1, float)[:, None], S]
    if inter:
        X.append(S * np.asarray(s1, float)[:, None])
    return np.hstack(X)


def fit_A(y, s1):
    _check(s1=s1, y=y)
    y, s1 = np.asarray(y, float), np.asarray(s1)
    return dict(p_wake=float((y[s1 == 0].sum() + 1) / ((s1 == 0).sum() + 2)),
                p_stage1=float((y[s1 == 1].sum() + 1) / ((s1 == 1).sum() + 2)))


def predict_A(ref, s1):
    _check(s1=s1)
    s1 = np.asarray(s1)
    return np.where(s1 == 1, ref['p_stage1'], ref['p_wake'])


def fit_B(t, s1, y, cfg, lo=None, hi=None):
    _check(t, s1, y)
    k, C, inter = cfg
    lt = np.log(t)
    lo = float(lt.min()) if lo is None else lo
    hi = float(lt.max()) if hi is None else hi
    clf = LogisticRegression(C=C, max_iter=5000).fit(design(t, s1, lo, hi, k, inter), y)
    return dict(knots=k, C=C, interaction=bool(inter), logt_lo=lo, logt_hi=hi, intercept=float(clf.intercept_[0]),
                coef=[float(c) for c in clf.coef_[0]])


def predict_B(ref, t, s1):
    _check(t, s1)
    X = design(t, s1, ref['logt_lo'], ref['logt_hi'], ref['knots'], ref['interaction'])
    z = ref['intercept'] + X @ np.asarray(ref['coef'])
    return np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)


def select_B(t, s1, y, person, salt=SEL_SALT):
    """Inner person-disjoint selection of the Stage+time configuration -> (cfg, {cfg name: inner squared error})."""
    t, s1, y, person = np.asarray(t, float), np.asarray(s1, float), np.asarray(y, float), np.asarray(person)
    ranked = sorted(set(person), key=lambda p: hashlib.sha256(f'{salt}|{p}'.encode()).hexdigest())
    if len(ranked) < N_SEL:
        raise ValueError(f'need at least {N_SEL} persons for the inner selection')
    fold = {p: i % N_SEL for i, p in enumerate(ranked)}
    fi = np.array([fold[p] for p in person])
    scores = {}
    for cfg in GRID:
        s = 0.0
        for h in range(N_SEL):
            a, b = fi != h, fi == h
            ref = fit_B(t[a], s1[a], y[a], cfg)
            s += float(((predict_B(ref, t[b], s1[b]) - y[b]) ** 2).sum())
        scores[cfg] = s
    best = None
    for cfg in GRID:                                   # ties -> earlier grid entry
        if best is None or scores[cfg] < scores[best] - 1e-12:
            best = cfg
    return best, {f'k{c[0]}_C{c[1]}_int{int(c[2])}': v for c, v in scores.items()}


def fit_references(t, s1, y, person):
    """TRAIN fit of both references (A and the selected B), as frozen before any held-out use."""
    cfg, scores = select_B(t, s1, y, person)
    return dict(A=fit_A(y, s1), B=fit_B(np.asarray(t, float), np.asarray(s1, float), np.asarray(y, float), cfg),
                B_selection=dict(knots=cfg[0], C=cfg[1], interaction=cfg[2], inner_squared_error=scores))
