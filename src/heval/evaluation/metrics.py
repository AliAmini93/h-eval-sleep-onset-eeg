"""Window-level scores, calibration and AUROC variants.

Lineage (research archive, tag v0.1.0-paper-review): brier / logloss / calibration from CTH v1 (also used for the
held-out model metrics in T1 and X1); calibration_in_stratum, matched_pairs and matched_auroc from R3B; auroc,
level_within and tie_ranks from E7A; pair_matrices and weighted_auc from the E7A amendment; recording_auroc from X1.
Formulas unchanged.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

MIN_PERSONS_FOR_CI = 8
AUC_BOOT_N = 2_000


def _arrays(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError('p and y must be 1-D arrays of equal length')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('predictions must be finite probabilities in [0, 1]')
    if not set(np.unique(y)) <= {0., 1.}:
        raise ValueError('labels must be 0/1')
    return p, y


def brier(p, y):
    p, y = _arrays(p, y)
    return (p - y) ** 2


def logloss(p, y):
    p, y = _arrays(p, y)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def calibration(p, y):
    """Calibration-in-the-large (mean p - mean y), logistic recalibration slope and 10-bin ECE."""
    p, y = _arrays(p, y)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]) if 0 < y.sum() < len(y) else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(p))
    return dict(cal_in_large=float(p.mean() - y.mean()), cal_slope=slope, ece10=ece)


def calibration_in_stratum(p, y):
    """(calibration-in-the-large, slope or None, reason when undefined) as used for the stage-stratified tables."""
    p, y = _arrays(p, y)
    cil = float(p.mean() - y.mean())
    if y.min() == y.max():
        return cil, None, 'undefined: outcome constant in stratum'
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if lp.std() < 1e-9:
        return cil, None, 'undefined: prediction constant in stratum'
    return cil, float(LogisticRegression(C=1e6, max_iter=2000).fit(lp[:, None], y).coef_[0, 0]), None


def auroc(p, y):
    """Mann-Whitney AUROC with ties counted 0.5; None when only one class is present."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def level_within(p, y):
    """Exact decomposition of one person's Brier score: B = L + W (population moments)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    pb, yb = p.mean(), y.mean()
    L = (pb - yb) ** 2 + yb * (1 - yb)
    W = ((p - pb) ** 2).mean() - 2 * ((p - pb) * (y - yb)).mean()
    return L, W


def tie_ranks(p):
    order = np.argsort(p, kind='mergesort')
    ranks = np.empty(len(p))
    sp = np.asarray(p)[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def within_recording_auroc(p, y, recording):
    """AUROC inside each recording (both classes required); mean / median over the defined recordings."""
    p, y, recording = np.asarray(p, float), np.asarray(y, int), np.asarray(recording)
    vals = [auroc(p[recording == r], y[recording == r]) for r in sorted(set(recording))]
    d = [v for v in vals if v is not None]
    return dict(mean=(float(np.mean(d)) if d else None), median=(float(np.median(d)) if d else None),
                n_defined=len(d), n_undefined=len(vals) - len(d))


def recording_auroc(p, y, recording, person, seed, n_boot=10_000):
    """Within-recording AUROC averaged over recordings; 95% CI from a person-cluster bootstrap (EXT decision AUROC)."""
    p, y, recording, person = np.asarray(p, float), np.asarray(y, int), np.asarray(recording), np.asarray(person)
    recs = sorted(set(recording))
    owner = {r: q for r, q in zip(recording, person)}
    per = {k: auroc(p[recording == k], y[recording == k]) for k in recs}
    persons = sorted(set(person))
    s, c = np.zeros(len(persons)), np.zeros(len(persons))
    for k, a in per.items():
        if a is not None:
            i = persons.index(owner[k])
            s[i] += a
            c[i] += 1
    if c.sum() == 0:
        raise ValueError('no recording contains both classes')
    J = np.random.default_rng(seed).integers(0, len(persons), size=(n_boot, len(persons)))
    with np.errstate(invalid='ignore'):
        boots = s[J].sum(1) / c[J].sum(1)
    return dict(n_defined=int(c.sum()), n_undefined=sum(a is None for a in per.values()), mean=float(s.sum() / c.sum()),
                ci95=[float(x) for x in np.nanpercentile(boots, [2.5, 97.5])])


def matched_pairs(kidx, pid, y):
    """Cross-person (positive, negative) index pairs sharing the same time key (e.g. the same 30 s landmark)."""
    pi, ni = [], []
    for kk in np.unique(kidx):
        idx = np.where(kidx == kk)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(pos) and len(neg):
            P, N = np.meshgrid(pos, neg, indexing='ij')
            keep = pid[P] != pid[N]
            pi.append(P[keep])
            ni.append(N[keep])
    if not pi:
        return np.array([], int), np.array([], int)
    return np.concatenate(pi), np.concatenate(ni)


def matched_auroc(p, y, person, kidx, rng, min_persons=MIN_PERSONS_FOR_CI, n_boot=AUC_BOOT_N):
    """Cross-person AUROC over time-matched pairs; person-cluster bootstrap CI only with >= 8 persons per class."""
    p, y, person, kidx = np.asarray(p, float), np.asarray(y), np.asarray(person), np.asarray(kidx)
    pu = sorted(set(person))
    pid = np.searchsorted(pu, person)
    pi, ni = matched_pairs(kidx, pid, y.astype(int))
    if not len(pi):
        return dict(auroc=None, n_pairs=0, reason='no matched cross-person pairs')
    c = (p[pi] > p[ni]) + 0.5 * (p[pi] == p[ni])
    out = dict(auroc=float(c.mean()), n_pairs=int(len(pi)), n_persons_pos=len(set(pid[pi])), n_persons_neg=len(set(pid[ni])))
    if out['n_persons_pos'] >= min_persons and out['n_persons_neg'] >= min_persons:
        bs = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, len(pu), len(pu)), minlength=len(pu)).astype(float)
            ww = w[pid[pi]] * w[pid[ni]]
            bs.append((c * ww).sum() / ww.sum() if ww.sum() else np.nan)
        out['ci95'] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
    else:
        out['ci95'] = 'not_reported_lt_8_persons'
    return out


def pair_matrices(p, y, person, stratum, persons):
    """WIN[a, b], PAIRS[a, b]: concordance of positives of person a against negatives of person b (a != b),
    summed over elapsed-time strata."""
    idx = {s: i for i, s in enumerate(persons)}
    S = len(persons)
    win, pairs = np.zeros((S, S)), np.zeros((S, S))
    n_strata_used = 0
    for st in np.unique(stratum):
        k = stratum == st
        pos, neg = np.where(k & (y == 1))[0], np.where(k & (y == 0))[0]
        used = False
        for i in pos:
            for j in neg:
                a, b = idx[person[i]], idx[person[j]]
                if a == b:
                    continue
                pairs[a, b] += 1
                win[a, b] += 1. if p[i] > p[j] else .5 if p[i] == p[j] else 0.
                used = True
        n_strata_used += used
    return win, pairs, n_strata_used


def weighted_auc(win, pairs, c):
    w = np.outer(c, c)
    np.fill_diagonal(w, 0)
    den = (w * pairs).sum()
    return float((w * win).sum() / den) if den > 0 else float('nan')
