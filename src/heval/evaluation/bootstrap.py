"""Paired Brier-score contrasts with 95% person-cluster percentile bootstrap intervals.

For a reference r and candidate c on the same windows: delta_WW = mean over windows of (p_r - y)^2 - (p_c - y)^2,
Brier skill R = delta_WW / mean (p_r - y)^2, delta_SE = mean over persons of the per-person mean difference.
Bootstrap: persons (sorted) are resampled with replacement (index matrix I of shape (n_boot, n_persons)); the
replicate statistics are ratio estimators over the resampled persons' window sums. Intervals are not reported for
fewer than 8 persons.

Lineage (research archive, tag v0.1.0-paper-review): stratum_contrast and boot_index from R2 (seed 20261025);
reference_contrast from R3B (seed 20261026, NaN-tolerant percentiles). Unchanged.
"""
import numpy as np

BOOT_N = 10_000
MIN_CI_PERSONS = 8
SEED_STAGE_AUDIT, SEED_REFERENCE_AUDIT = 20261025, 20261026


def _validate(p_ref, p_cand, y, person):
    p_ref, p_cand, y = (np.asarray(a, float) for a in (p_ref, p_cand, y))
    person = np.asarray(person)
    n = len(y)
    if not (p_ref.shape == p_cand.shape == y.shape == person.shape == (n,)) or n == 0:
        raise ValueError('p_ref, p_cand, y and person must be non-empty 1-D arrays of equal length')
    for name, p in (('p_ref', p_ref), ('p_cand', p_cand)):
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f'{name} must contain finite probabilities in [0, 1]')
    if not set(np.unique(y)) <= {0., 1.}:
        raise ValueError('y must be 0/1')
    return p_ref, p_cand, y, person


def boot_index(n_persons, seed=SEED_STAGE_AUDIT, n=BOOT_N):
    if n_persons < 1:
        raise ValueError('need at least one person')
    return np.random.default_rng(seed).integers(0, n_persons, size=(n, n_persons))


def stratum_contrast(p_ref, p_cand, y, person, I=None):
    """Paired statistics on one subset (arrays already masked). I: bootstrap index over sorted persons or None."""
    p_ref, p_cand, y, person = _validate(p_ref, p_cand, y, person)
    pu = sorted(set(person))
    k = np.searchsorted(pu, person)
    br, bc = (p_ref - y) ** 2, (p_cand - y) ** 2
    pr, pc = np.clip(p_ref, 1e-6, 1 - 1e-6), np.clip(p_cand, 1e-6, 1 - 1e-6)
    lr = -(y * np.log(pr) + (1 - y) * np.log1p(-pr))
    lc = -(y * np.log(pc) + (1 - y) * np.log1p(-pc))
    n = np.bincount(k, minlength=len(pu)).astype(float)
    sr, sc = np.bincount(k, br, len(pu)), np.bincount(k, bc, len(pu))
    per = (sr - sc) / n
    ref_ww = sr.sum() / n.sum()
    out = dict(n_windows=int(n.sum()), n_persons=len(pu), brier_ref_ww=float(ref_ww), brier_cand_ww=float(sc.sum() / n.sum()),
               delta_ww=float((sr.sum() - sc.sum()) / n.sum()),
               R=(float((sr.sum() - sc.sum()) / sr.sum()) if ref_ww > 1e-12 else None),
               delta_se=float(per.mean()), delta_logloss_ww=float((lr.sum() - lc.sum()) / n.sum()),
               persons_favouring_candidate=int((per > 0).sum()))
    if len(pu) < MIN_CI_PERSONS or I is None:
        out['ci'] = 'not_reported_lt_8_persons' if len(pu) < MIN_CI_PERSONS else 'not_computed'
        return out
    if I.shape[1] != len(pu):
        raise ValueError('bootstrap index width must equal the number of persons in the subset')
    Br, Bc, N = sr[I].sum(1), sc[I].sum(1), n[I].sum(1)
    ok = Br / N > 1e-12
    q = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]   # noqa: E731
    out.update(delta_ww_ci95=q((Br - Bc) / N), R_ci95=q(((Br - Bc) / Br)[ok]), R_boot_undefined=int((~ok).sum()),
               delta_se_ci95=q(per[I].mean(1)))
    return out


def reference_contrast(p_ref, p_c, y, person, I):
    """Paired statistics as used for the scored-stage reference tables (CI suppressed below 8 persons or I None)."""
    p_ref, p_c, y, person = _validate(p_ref, p_c, y, person)
    pu = sorted(set(person))
    k = np.searchsorted(pu, person)
    n = np.bincount(k, minlength=len(pu)).astype(float)
    sr, sc = np.bincount(k, (p_ref - y) ** 2, len(pu)), np.bincount(k, (p_c - y) ** 2, len(pu))
    lr = -(y * np.log(np.clip(p_ref, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p_ref, 1e-6, 1)))
    lc = -(y * np.log(np.clip(p_c, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p_c, 1e-6, 1)))
    per = (sr - sc) / n
    out = dict(n_persons=len(pu), n_windows=int(n.sum()), brier_ref_ww=float(sr.sum() / n.sum()),
               brier_cand_ww=float(sc.sum() / n.sum()), delta_ww=float((sr.sum() - sc.sum()) / n.sum()),
               R=float((sr.sum() - sc.sum()) / sr.sum()) if sr.sum() > 0 else None, delta_se=float(per.mean()),
               delta_logloss_ww=float((lr.sum() - lc.sum()) / n.sum()), persons_favouring_candidate=int((per > 0).sum()),
               interval_label='R3B re-bootstrap')
    if len(pu) < MIN_CI_PERSONS or I is None:
        out['ci'] = 'not_reported_lt_8_persons'
        return out
    Br, Bc, N = sr[I].sum(1), sc[I].sum(1), n[I].sum(1)
    q = lambda a: [float(x) for x in np.nanpercentile(a, [2.5, 97.5])]   # noqa: E731
    with np.errstate(divide='ignore', invalid='ignore'):
        out.update(delta_ww_ci95=q((Br - Bc) / N), R_ci95=q(np.where(Br > 0, (Br - Bc) / Br, np.nan)),
                   delta_se_ci95=q(per[I].mean(1)))
    return out
