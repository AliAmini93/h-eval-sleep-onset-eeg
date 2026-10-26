"""H-EVAL W2S: shared folds, references and evaluation (design w2s_design_v1.md §3, §5)."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

REPO = Path(__file__).resolve().parents[4]
TABLE = REPO / '.local/h_eval/discovery_w2s/w2s_table_v1.npz'
BOOT_N, BOOT_SEED = 10_000, 20261024
N_FOLDS, N_INNER = 5, 4
ESTIMANDS = {'E1': ('e1', 'y1'), 'E2': ('e2', 'y'), 'E3': ('e3', 'y')}   # population mask, label
EVAL_SUBSETS = {'E1': ['e1'], 'E2': ['e2', 'e2pre', 'e2ret'], 'E3': ['e3']}


def load():
    z = np.load(TABLE, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['person'] = d['person'].astype(str)
    d['recording'] = d['recording'].astype(str)
    d['e2ret'] = d['e2'] & ~d['e1']
    return d


def folds(persons, salt):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{salt}|{p}'.encode()).hexdigest())
    f = {p: i % N_FOLDS for i, p in enumerate(ranked)}
    return np.array([f[p] for p in persons])


FOLD_SALTS = {'A': 'H-EVAL-CTH-v1-folds', 'B': 'H-EVAL-W2S-folds-B'}
INNER_SALT = 'H-EVAL-W2S-inner'


def inner_folds(persons):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{INNER_SALT}|{p}'.encode()).hexdigest())
    f = {p: i % N_INNER for i, p in enumerate(ranked)}
    return np.array([f[p] for p in persons])


def time_feats(t):
    t = np.asarray(t, float)
    return np.column_stack([np.log1p(t / 60), t / 3600])


# ------------------------------------------------------------------------ references
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
    """Return the config with the lowest inner-CV window-weighted Brier (ties: first in list)."""
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


# ------------------------------------------------------------------------ metrics
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
    Br, Bc, N = sr[I].sum(1), sc[I].sum(1), n[I].sum(1)
    q = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]   # noqa: E731
    loo = [((sr.sum() - sr[i]) - (sc.sum() - sc[i])) / (sr.sum() - sr[i]) for i in range(len(pu))]
    return dict(R=float((sr.sum() - sc.sum()) / sr.sum()), R_ci95=q((Br - Bc) / Br),
                delta_ww=float((sr.sum() - sc.sum()) / n.sum()), delta_se=float(per.mean()),
                delta_se_ci95=q(per[I].mean(1)), persons_favouring=int((per > 0).sum()), n_persons=len(pu),
                loo_R_min=float(min(loo)))


def warning(p, theta, y, d, m, horizon_label):
    """Event-level warning behaviour on population mask m (fixed threshold theta per window)."""
    rec, el = d['recording'], d['elapsed']
    det, leads, n_ev = 0, [], 0
    fa = ((p >= theta) & (y == 0) & m).sum()
    hours = m.sum() * 30 / 3600
    for r in sorted(set(rec[m])):
        idx = np.where(m & (rec == r))[0]
        pos = idx[y[idx] == 1]
        if len(pos) == 0:
            continue
        n_ev += 1
        alarmed = pos[p[pos] >= theta[pos]]
        if len(alarmed):
            det += 1
            leads.append(float(horizon_label[alarmed].max()))
    return dict(n_events=n_ev, sensitivity=(det / n_ev if n_ev else None),
                median_first_alarm_time_to_event_s=(float(np.median(leads)) if leads else None),
                false_alarms_per_hour=float(fa / hours) if hours else None)


def evaluate(name, preds, refs, d, y, mask, theta=None, tte=None):
    """preds/refs: dict model -> full-length OOF arrays (nan outside population)."""
    pm = d['person'][mask]
    out = dict(subset=name, n_windows=int(mask.sum()), n_positive=int(y[mask].sum()),
               n_persons=len(set(pm)), n_recordings=len(set(d['recording'][mask])), models={}, vs_const={}, vs_time={})
    allm = {**refs, **preds}
    for k, p in allm.items():
        out['models'][k] = model_metrics(p[mask], y[mask], pm)
        if theta is not None and tte is not None and k in preds:
            out['models'][k]['warning'] = warning(p, theta, y, d, mask, tte)
    for k, p in preds.items():
        out['vs_const'][k] = paired(refs['CONST_pop'][mask], p[mask], y[mask], pm)
        out['vs_time'][k] = paired(refs['TIME_pop'][mask], p[mask], y[mask], pm)
    return out
