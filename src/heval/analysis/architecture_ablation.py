"""Architecture ablation study (paper Fig. 7): 5-fold person-disjoint cross-validation on TRAIN windows only.

Folds: persons ordered by sha256('H-EVAL-CTH-v1-folds|' + person), fold = rank mod 5. For outer fold f, the inner
validation fold is (f + 1) mod 5 and the remaining persons form the inner training set. Per fold, the learning rate
and width of RawCNN (M2) and of the full network (M3) are chosen from {1e-3, 3e-4} x {16, 32} by inner-validation
Brier (seed 0); every variant is then trained with seeds 0, 1, 2 (variants use the M3 configuration) and the fold's
out-of-fold prediction is the seed mean. Each variant is compared with the full network by the paired window-weighted
Brier skill R = (B_variant - B_full) / B_variant with a 95% person-cluster bootstrap (10,000 replicates, seed
20261023), leave-one-person-out and leave-two-persons-out minima.

Lineage: CTH v1 experiment (person_folds, run_cv, metrics, paired) of the research archive, tag v0.1.0-paper-review.
The classical comparators of that run (constant, time hazard, spectral and dynamics logistic models) only fed its
model-selection rule and do not change the network predictions; they are not part of this module.
"""
import hashlib
import itertools

import numpy as np

from ..evaluation.metrics import auroc, calibration
from ..models.ablation_networks import ARCHS, SPECIAL, AblationInputs, history_index, predict_net, train_net

FOLD_SALT = 'H-EVAL-CTH-v1-folds'
N_FOLDS = 5
SEEDS = (0, 1, 2)
GRID = [dict(lr=lr, width=w) for lr in (1e-3, 3e-4) for w in (16, 32)]
BOOT_N, BOOT_SEED = 10_000, 20261023
FIGURE_ROWS = [('Remove raw-waveform branch', 'A2_no_raw'), ('Remove 5-min trajectory', 'A1_no_trajectory'),
               ('30-s instead of 5-min history', 'A4_history_30s'), ('Binary instead of hazard head', 'A3_binary_head'),
               ('Add elapsed-time fusion', 'A5_plus_time'), ('Fpz-Cz only', 'A6a_fpz_only'), ('Pz-Oz only', 'A6b_pz_only'),
               ('Plain multiscale CNN (RawCNN)', 'M2_raw_cnn_binary')]


def development_table(raw, spectral, rich, elapsed, tau, y, person, recording):
    """Window arrays (rows grouped by recording, elapsed order) -> the dict used by the development analyses."""
    recording = np.asarray(recording).astype(str)
    order = list(dict.fromkeys(recording))
    rec_idx = np.array([order.index(r) for r in recording])
    if np.any(np.diff(rec_idx) < 0):
        raise ValueError('rows must be grouped by recording')
    night = np.array([int(r[5]) if len(r) > 5 and r[5].isdigit() else 1 for r in recording])
    d = dict(raw=np.asarray(raw), e5=np.asarray(spectral, float), rich=np.asarray(rich, float), elapsed=np.asarray(elapsed, float),
             tau=np.asarray(tau, float), y=np.asarray(y).astype(int), person=np.asarray(person).astype(str),
             recording=recording, rec_idx=rec_idx, night=night)
    d['hist'] = history_index(rec_idx, d['elapsed'])
    if len({len(v) for v in d.values()}) != 1:
        raise ValueError('all window arrays must have the same length')
    return d


def person_folds(persons):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{FOLD_SALT}|{p}'.encode()).hexdigest())
    if len(ranked) < N_FOLDS:
        raise ValueError(f'need at least {N_FOLDS} persons for {N_FOLDS}-fold cross-validation')
    return {p: i % N_FOLDS for i, p in enumerate(ranked)}


def fold_rows(pf, f):
    """(test, outer-train, inner-train, inner-validation) row indices of outer fold f."""
    te, tr = np.where(pf == f)[0], np.where(pf != f)[0]
    inner_va = np.where(pf == (f + 1) % N_FOLDS)[0]
    return te, tr, np.setdiff1d(tr, inner_va), inner_va


def cv_metrics(p, d):
    """Window-weighted / subject-equal Brier and log loss, first-night Brier, within-recording AUROC, calibration."""
    y = d['y'].astype(float)
    persons = d['person']
    pu = sorted(set(persons))
    b = (p - y) ** 2
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pc) + (1 - y) * np.log1p(-pc))
    rec_auc = [auroc(p[d['rec_idx'] == r], y[d['rec_idx'] == r]) for r in np.unique(d['rec_idx'])]
    defined = [a for a in rec_auc if a is not None]
    first = np.array([n == min(d['night'][persons == s]) for n, s in zip(d['night'], persons)])
    return dict(brier_ww=float(b.mean()), brier_se=float(np.mean([b[persons == s].mean() for s in pu])), logloss_ww=float(ll.mean()),
                logloss_se=float(np.mean([ll[persons == s].mean() for s in pu])), brier_ww_first_night=float(b[first].mean()),
                within_recording_auroc_mean=float(np.mean(defined)), n_recordings_auroc_defined=len(defined),
                n_recordings_auroc_undefined=len(rec_auc) - len(defined), **calibration(p, y))


def paired(p_ref, p_cand, d, n_boot=BOOT_N, seed=BOOT_SEED):
    """Paired person-cluster bootstrap of the window-weighted Brier difference and R = delta / B_ref, with
    leave-one- and leave-two-persons-out R minima; 'credible' = R >= 0.05, CI lower > 0 and both minima > 0."""
    y = d['y'].astype(float)
    persons = d['person']
    pu = sorted(set(persons))
    idx = np.searchsorted(pu, persons)
    sr = np.bincount(idx, weights=(p_ref - y) ** 2, minlength=len(pu))
    sc = np.bincount(idx, weights=(p_cand - y) ** 2, minlength=len(pu))
    n = np.bincount(idx, minlength=len(pu)).astype(float)
    I = np.random.default_rng(seed).integers(0, len(pu), size=(n_boot, len(pu)))
    Br, Bc = sr[I].sum(1) / n[I].sum(1), sc[I].sum(1) / n[I].sum(1)
    R = (sr.sum() - sc.sum()) / sr.sum()
    loo = [((sr.sum() - sr[i]) - (sc.sum() - sc[i])) / (sr.sum() - sr[i]) for i in range(len(pu))]
    l2o = [((sr.sum() - sr[i] - sr[j]) - (sc.sum() - sc[i] - sc[j])) / (sr.sum() - sr[i] - sr[j])
           for i, j in itertools.combinations(range(len(pu)), 2)]
    per_person = (sr - sc) / n
    out = dict(delta_ww=float((sr.sum() - sc.sum()) / n.sum()), delta_ww_ci95=[float(x) for x in np.percentile(Br - Bc, [2.5, 97.5])],
               R=float(R), R_ci95=[float(x) for x in np.percentile((Br - Bc) / Br, [2.5, 97.5])],
               loo_R_min=float(min(loo)), l2o_R_min=float(min(l2o)), persons_favouring_candidate=int((per_person > 0).sum()),
               n_persons=len(pu), per_person_delta={p: float(v) for p, v in zip(pu, per_person)})
    out['credible'] = bool(out['R'] >= 0.05 and out['R_ci95'][0] > 0 and out['loo_R_min'] > 0 and out['l2o_R_min'] > 0)
    return out


def ablation_cv(d, archs=None, folds_to_run=None, max_epochs=40, seeds=SEEDS, grid=GRID, device='cpu'):
    """Out-of-fold predictions of the network variants. Returns (oof, seed_oof, log, folds)."""
    archs = list(archs or ARCHS)
    unknown = set(archs) - set(ARCHS)
    if unknown:
        raise ValueError(f'unknown variants: {sorted(unknown)}')
    folds = person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    oof = {m: np.full(len(d['y']), np.nan) for m in archs}
    seed_oof = {m: {s: np.full(len(d['y']), np.nan) for s in seeds} for m in archs}
    log = []
    for f in (folds_to_run if folds_to_run is not None else range(N_FOLDS)):
        te, _, inner_tr, inner_va = fold_rows(pf, f)
        Tfull = AblationInputs(d, inner_tr, device=device)
        selected = {}
        for name in ('M2_raw_cnn_binary', 'M3_cthnet'):
            if name not in archs:
                continue
            scores = []
            for cfg in grid:
                _, info = train_net(Tfull, inner_tr, inner_va, cfg, ARCHS[name], seed=0, max_epochs=max_epochs)
                scores.append((info['inner_val_brier_ww'], cfg))
                log.append(dict(fold=f, model=name, stage='grid', cfg=cfg, **info))
            selected[name] = min(scores, key=lambda s: s[0])[1]
        for name in archs:
            cfg = selected['M2_raw_cnn_binary'] if name == 'M2_raw_cnn_binary' else selected.get('M3_cthnet', grid[0])
            T = AblationInputs(d, inner_tr, device=device, **SPECIAL[name]) if name in SPECIAL else Tfull
            preds = []
            for s in seeds:
                model, info = train_net(T, inner_tr, inner_va, cfg, ARCHS[name], seed=s, max_epochs=max_epochs)
                p = predict_net(model, T, te, ARCHS[name]['head'])
                seed_oof[name][s][te] = p
                preds.append(p)
                log.append(dict(fold=f, model=name, stage='seed', seed=s, cfg=cfg, **info))
            oof[name][te] = np.mean(preds, 0)
    return oof, seed_oof, log, folds


def summarise_ablation(oof, d, n_boot=BOOT_N, seed=BOOT_SEED):
    """Figure 7 comparisons (each variant as reference vs the full network M3) and per-model CV metrics."""
    if 'M3_cthnet' not in oof:
        raise ValueError('the full network M3_cthnet is required as the comparator')
    m = np.isfinite(oof['M3_cthnet'])
    sub = {k: d[k][m] for k in ('y', 'person', 'rec_idx', 'night')}
    out = dict(models={k: cv_metrics(p[m], sub) for k, p in oof.items()}, comparisons={}, n_windows=int(m.sum()),
               n_persons=len(set(sub['person'])))
    for label, k in FIGURE_ROWS:
        if k in oof:
            c = paired(oof[k][m], oof['M3_cthnet'][m], sub, n_boot=n_boot, seed=seed)
            c.pop('per_person_delta')
            out['comparisons'][f'{k}__vs__M3_cthnet'] = dict(c, label=label)
    return out
