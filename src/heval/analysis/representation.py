"""Representation analysis (paper Fig. 8a-b and the attribution checks reported with it; development data only).

Matched CV: on the same person folds, inner splits and configuration grid as the architecture ablation, it fits
- the TRAIN-prevalence constant;
- the SpecLR recipe on the outer-training and on the inner-training persons;
- RawCNN (a replay of the ablation-study M2);
- the capacity-matched STFT-CNN.
Seed means give out-of-fold predictions, compared pairwise with a 95% person-cluster bootstrap (10,000 replicates,
seed 20261101). The fitted fold models are returned for the perturbation analysis.

Attribution checks, on a fitted RawCNN ensemble evaluated in-sample:
- Integrated Gradients: baseline 0, 32 midpoint steps, |IG| summed in 60 one-second bins (2 channels x 30 s).
- Model-randomisation sanity check: cascading re-initialisation (seed 999) and a label-permuted refit.
- Deletion test: bins ranked by IG, by |input| or by IG of a randomised model, or chosen at random; the top k are
  replaced by an elapsed-time-matched donor's bins and the change in logit is recorded.
- Person-mean summaries carry a person-cluster bootstrap (2,000 replicates, seed 20261103).

Lineage: INTERP v1 (Part B cv, summarise_cv; Part A explanations; follow-up F1) of the research archive,
tag v0.1.0-paper-review.
"""
import copy

import numpy as np
import torch

from ..evaluation.heldout import donor_plan
from ..models import raw_cnn
from ..models.ablation_networks import ARCHS, AblationInputs, predict_net, train_net
from ..models.spectral import logistic_fit_np, logistic_predict_np, zapply, zfit
from ..models.stft_cnn import STFTInputs, STFTNet
from .architecture_ablation import GRID, N_FOLDS, SEEDS, cv_metrics, fold_rows, paired, person_folds

BOOT_B, SEED_B = 10_000, 20261101
BOOT_A, SEED_A = 2_000, 20261103
CV_PAIRS = (('M2_raw_cnn', 'STFT_cnn'), ('M0_outer_train', 'STFT_cnn'), ('M0_outer_train', 'M2_raw_cnn'),
            ('M0_inner_train', 'STFT_cnn'), ('M0_inner_train', 'M2_raw_cnn'), ('T0_constant', 'STFT_cnn'),
            ('T0_constant', 'M2_raw_cnn'), ('T0_constant', 'M0_outer_train'), ('M0_inner_train', 'M0_outer_train'))


# ------------------------------------------------------------------------ matched cross-validation
def m0_fit(X, y, tr):
    mu, sd = zfit(X[tr])
    return dict(mu=mu, sd=sd, w=logistic_fit_np(zapply(X[tr], mu, sd), y[tr].astype(float)))


def m0_predict(M, X):
    return logistic_predict_np(M['w'], zapply(X, M['mu'], M['sd']))


def matched_cv(d, folds_to_run=None, max_epochs=40, seeds=SEEDS, grid=GRID, device='cpu'):
    """Returns (oof, seed_oof, log, fold_models) where fold_models[f] holds the fold's fitted models."""
    folds = person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    y = d['y']
    names = ['T0_constant', 'M0_outer_train', 'M0_inner_train', 'M2_raw_cnn', 'STFT_cnn']
    oof = {m: np.full(len(y), np.nan) for m in names}
    seed_oof = {m: {s: np.full(len(y), np.nan) for s in seeds} for m in ('M2_raw_cnn', 'STFT_cnn')}
    log, fold_models = [], {}
    for f in (folds_to_run if folds_to_run is not None else range(N_FOLDS)):
        te, tr, inner_tr, inner_va = fold_rows(pf, f)
        prior = (y[tr].sum() + 1) / (len(tr) + 2)
        oof['T0_constant'][te] = np.full(len(te), prior)
        M0o, M0i = m0_fit(d['e5'], y, tr), m0_fit(d['e5'], y, inner_tr)
        oof['M0_outer_train'][te] = m0_predict(M0o, d['e5'][te])
        oof['M0_inner_train'][te] = m0_predict(M0i, d['e5'][te])
        Tm2 = AblationInputs(d, inner_tr, device=device)
        Ts = STFTInputs(d, inner_tr, device=device)
        store = dict(m0_outer=M0o, m0_inner=M0i, m2_scale=Tm2.scale, constant=float(prior),
                     stft_mu=Ts.stats.mu.cpu(), stft_sd=Ts.stats.sd.cpu())
        for name, T, make in (('M2_raw_cnn', Tm2, None), ('STFT_cnn', Ts, STFTNet)):
            fit = lambda cfg, s: train_net(T, inner_tr, inner_va, cfg, ARCHS['M2_raw_cnn_binary'], seed=s,  # noqa: E731
                                           max_epochs=max_epochs, make_model=make)
            scores = []
            for cfg in grid:
                _, info = fit(cfg, 0)
                scores.append((info['inner_val_brier_ww'], cfg))
                log.append(dict(fold=f, model=name, stage='grid', cfg=cfg, **info))
            cfg = min(scores, key=lambda g: g[0])[1]
            models, preds = [], []
            for s in seeds:
                model, info = fit(cfg, s)
                p = predict_net(model, T, te, 'binary')
                seed_oof[name][s][te] = p
                preds.append(p)
                models.append(model.eval())
                log.append(dict(fold=f, model=name, stage='seed', seed=s, cfg=cfg, **info))
            oof[name][te] = np.mean(preds, 0)
            store[name] = dict(cfg=cfg, models=models)
        fold_models[f] = store
    return oof, seed_oof, log, fold_models


def summarise_matched_cv(oof, d, n_boot=BOOT_B, seed=SEED_B):
    """Per-model CV metrics (Fig. 8b) and the pairwise comparisons (Fig. 8a: candidate = second name)."""
    m = np.isfinite(oof['STFT_cnn'])
    sub = {k: d[k][m] for k in ('y', 'person', 'rec_idx', 'night')}
    res = dict(models={k: cv_metrics(p[m], sub) for k, p in oof.items()}, comparisons={})
    for a, b in CV_PAIRS:
        c = paired(oof[a][m], oof[b][m], sub, n_boot=n_boot, seed=seed)
        c.pop('credible')
        c.pop('per_person_delta')
        res['comparisons'][f'{a}__vs__{b}'] = c
    return res


# ------------------------------------------------------------------------ attribution checks (fitted RawCNN ensemble)
def integrated_gradients(model, x, steps=32, batch=256):
    """x: torch (n, 2, 3000) normalised input; baseline 0. Returns IG (n, 2, 3000) for the logit."""
    model.eval()
    out = torch.zeros_like(x)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        g = torch.zeros_like(xb)
        for a in (torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps:
            xi = (a * xb).detach().requires_grad_(True)
            z = model(xi)[:, 0].sum()
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


def person_mean_ci(v, person, n_boot=BOOT_A, seed=SEED_A):
    pu = sorted(set(person))
    idx = np.searchsorted(pu, person)
    pm = np.bincount(idx, weights=v, minlength=len(pu)) / np.bincount(idx, minlength=len(pu))
    I = np.random.default_rng(seed).integers(0, len(pu), size=(n_boot, len(pu)))
    return dict(mean_of_person_means=float(pm.mean()), ci95=[float(x) for x in np.percentile(pm[I].mean(1), [2.5, 97.5])],
                n_persons=len(pu))


def randomise(model, stage, seed=999):
    """Cascading re-initialisation from the output: stage 1 = head, 2 = + fuse, ... 5 = all layers."""
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    groups = [['head'], ['fuse'], ['raw.out'], ['raw.body'], ['raw.stems']]
    names = [n for g in groups[:stage] for n in g]
    for nm, mod in m.named_modules():
        if any(nm == g or nm.startswith(g + '.') for g in names) and hasattr(mod, 'reset_parameters'):
            mod.reset_parameters()
    return m.eval()


def model_logits(model, x):
    with torch.no_grad():
        return torch.cat([model(x[i:i + 512])[:, 0] for i in range(0, len(x), 512)]).cpu().numpy().astype(np.float64)


def replace_bins(x, xd, sel):
    """Replace 1 s bins (index = channel * 30 + second) of x by the donor input's bins (normalised domain)."""
    out = x.clone()
    n = len(x)
    rows = np.repeat(np.arange(n), sel.shape[1])
    ch, sec = (sel // 30).ravel(), (sel % 30).ravel()
    xv, dv = out.view(n, 2, 30, 100), xd.view(n, 2, 30, 100)
    r, c, s = (torch.tensor(v, device=x.device) for v in (rows, ch, sec))
    xv[r, c, s] = dv[r, c, s]
    return out


def donor_windows(person, elapsed):
    """Elapsed-time-matched donor window of another person for every window (draw 0 of the held-out donor rule)."""
    return np.array(donor_plan(list(person), [float(e) for e in elapsed], 0)[0])


def randomisation_check(models, raw_uv, scale, y, person, device='cpu'):
    """IG similarity (row-wise Spearman of the 60-bin maps) between the trained seed-0 model and cascading randomised
    copies, and a model refitted on permuted labels (permutation seed 7, 4 epochs, seed 0); also vs |input|."""
    x = torch.tensor(raw_cnn.normalise(raw_uv, scale), device=device)
    ref = bin_map(integrated_gradients(models[0], x)).cpu().numpy()
    absx = x.abs().reshape(len(x), 2, 30, 100).sum(-1).reshape(len(x), 60).cpu().numpy()
    out = {}
    for stage, label in enumerate(['head', '+fuse', '+raw.out', '+raw.body', '+raw.stems (full)'], start=1):
        mp = bin_map(integrated_gradients(randomise(models[0], stage), x)).cpu().numpy()
        out[label] = dict(ig_similarity_to_trained=person_mean_ci(spearman_rows(ref, mp), person),
                          ig_similarity_to_abs_input=person_mean_ci(spearman_rows(mp, absx), person))
    perm = np.random.default_rng(7).permutation(len(y))
    ml = raw_cnn.fit_seed(raw_cnn.normalise(raw_uv, raw_cnn.train_scale(raw_uv)), np.asarray(y)[perm], 0, epochs=4,
                          device=device)[0]
    mp = bin_map(integrated_gradients(ml, x)).cpu().numpy()
    out['label_randomised_model'] = dict(ig_similarity_to_trained=person_mean_ci(spearman_rows(ref, mp), person),
                                         ig_similarity_to_abs_input=person_mean_ci(spearman_rows(mp, absx), person))
    return out, ref


def deletion_check(model, raw_uv, scale, person, elapsed, ig_map, ks=(3, 6, 12, 24), device='cpu'):
    """|change in logit| after replacing the top-k bins ranked by trained-model IG, by |input| and by IG of a fully
    randomised model, or k random bins (5 draws, seed 20261103), with donor bins."""
    x = torch.tensor(raw_cnn.normalise(raw_uv, scale), device=device)
    plan = donor_windows(person, elapsed)
    xd = torch.tensor(raw_cnn.normalise(np.asarray(raw_uv)[plan], scale), device=device)
    absx = x.abs().reshape(len(x), 2, 30, 100).sum(-1).reshape(len(x), 60).cpu().numpy()
    ig_rand = bin_map(integrated_gradients(randomise(model, 5), x)).cpu().numpy()
    z0 = model_logits(model, x)
    rng = np.random.default_rng(SEED_A)
    out = {}
    for k in ks:
        dz = {lab: np.abs(model_logits(model, replace_bins(x, xd, np.argsort(-mp, 1)[:, :k])) - z0)
              for lab, mp in (('ig_trained', ig_map), ('abs_input', absx), ('ig_randomised_model', ig_rand))}
        dz['random'] = np.mean([np.abs(model_logits(model, replace_bins(x, xd, np.argsort(rng.random((len(x), 60)), 1)[:, :k])) - z0)
                                for _ in range(5)], 0)
        out[str(k)] = {lab: person_mean_ci(v, person) for lab, v in dz.items()}
        out[str(k)]['ig_minus_abs_input'] = person_mean_ci(dz['ig_trained'] - dz['abs_input'], person)
        out[str(k)]['ig_minus_ig_randomised'] = person_mean_ci(dz['ig_trained'] - dz['ig_randomised_model'], person)
    return out
