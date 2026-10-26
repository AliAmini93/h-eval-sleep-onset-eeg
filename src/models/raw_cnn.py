#!/usr/bin/env python3
"""CTH v1: TRAIN-only cross-validated comparison of CTH-Net, comparators and ablations (protocol
docs/decision/h_eval/cth_v1/cth_v1_protocol.md). Reads only .local/h_eval/cth_v1/dataset_v1.npz (SC_TRAIN).

Usage:
  cth_experiment.py feasibility        one fold, reduced epochs: technical check only
  cth_experiment.py cv                 full predefined 5-fold comparison + ablations
  cth_experiment.py freeze             refit the selected model on all TRAIN persons and freeze it
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / '.local/h_eval/cth_v1/dataset_v1.npz'
RES = REPO / 'results/decision/h_eval/cth_v1'
LOC = REPO / '.local/h_eval/cth_v1'
FOLD_SALT = 'H-EVAL-CTH-v1-folds'
N_FOLDS = 5
H = 10
SEEDS = (0, 1, 2)
GRID = [dict(lr=lr, width=w) for lr in (1e-3, 3e-4) for w in (16, 32)]
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 256
BOOT_N, BOOT_SEED = 10_000, 20261023
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e6 = _imp(REPO / 'scripts/decision/h_eval/gate_e6/gate_e6.py', 'e6_for_cth')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------ data / folds / labels
def load():
    z = np.load(DATA, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['person'] = d['person'].astype(str)
    d['recording'] = d['recording'].astype(str)
    return d


def person_folds(persons):
    ranked = sorted(set(persons), key=lambda p: hashlib.sha256(f'{FOLD_SALT}|{p}'.encode()).hexdigest())
    return {p: i % N_FOLDS for i, p in enumerate(ranked)}


def event_interval(tau):
    """K in 1..10 if tau <= 300 else 11 (horizon-censored)."""
    k = np.ceil(np.asarray(tau) / 30.0 - 1e-9).astype(int)
    return np.where(np.asarray(tau) <= 300 + 1e-8, np.clip(k, 1, H), H + 1)


def time_feats(elapsed):
    e = np.asarray(elapsed, float)
    return np.stack([np.log1p(e / 60.0), e / 3600.0], 1)


# ------------------------------------------------------------------------ metrics
def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def calibration(p, y):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]) if 0 < y.sum() < len(y) else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(p))
    return dict(cal_in_large=float(p.mean() - y.mean()), cal_slope=slope, ece10=ece)


def metrics(p, d):
    y = d['y'].astype(float)
    persons = d['person']
    pu = sorted(set(persons))
    b, ll = brier(p, y), logloss(p, y)
    se_b = np.mean([b[persons == s].mean() for s in pu])
    se_ll = np.mean([ll[persons == s].mean() for s in pu])
    rec_auc = [auroc(p[d['rec_idx'] == r], y[d['rec_idx'] == r]) for r in np.unique(d['rec_idx'])]
    defined = [a for a in rec_auc if a is not None]
    first = np.array([n == min(d['night'][persons == s]) for n, s in zip(d['night'], persons)])
    return dict(brier_ww=float(b.mean()), brier_se=float(se_b), logloss_ww=float(ll.mean()), logloss_se=float(se_ll),
                brier_ww_first_night=float(b[first].mean()), within_recording_auroc_mean=float(np.mean(defined)),
                n_recordings_auroc_defined=len(defined), n_recordings_auroc_undefined=len(rec_auc) - len(defined),
                **calibration(p, y))


def paired(p_ref, p_cand, d):
    """Paired person-cluster bootstrap of WW Brier difference and R = delta / B_ref; LOO and leave-two-out R."""
    y = d['y'].astype(float)
    persons = d['person']
    pu = sorted(set(persons))
    idx = np.searchsorted(pu, persons)
    sr = np.bincount(idx, weights=brier(p_ref, y), minlength=len(pu))
    sc = np.bincount(idx, weights=brier(p_cand, y), minlength=len(pu))
    n = np.bincount(idx, minlength=len(pu)).astype(float)
    I = np.random.default_rng(BOOT_SEED).integers(0, len(pu), size=(BOOT_N, len(pu)))
    Br, Bc = sr[I].sum(1) / n[I].sum(1), sc[I].sum(1) / n[I].sum(1)
    R_b = (Br - Bc) / Br
    R = (sr.sum() - sc.sum()) / sr.sum()
    loo = [((sr.sum() - sr[i]) - (sc.sum() - sc[i])) / (sr.sum() - sr[i]) for i in range(len(pu))]
    l2o = [((sr.sum() - sr[i] - sr[j]) - (sc.sum() - sc[i] - sc[j])) / (sr.sum() - sr[i] - sr[j])
           for i, j in itertools.combinations(range(len(pu)), 2)]
    per_person = (sr - sc) / n
    out = dict(delta_ww=float((sr.sum() - sc.sum()) / n.sum()),
               delta_ww_ci95=[float(x) for x in np.percentile(Br - Bc, [2.5, 97.5])],
               R=float(R), R_ci95=[float(x) for x in np.percentile(R_b, [2.5, 97.5])],
               loo_R_min=float(min(loo)), l2o_R_min=float(min(l2o)),
               persons_favouring_candidate=int((per_person > 0).sum()), n_persons=len(pu),
               per_person_delta={p: float(v) for p, v in zip(pu, per_person)})
    out['credible'] = bool(out['R'] >= 0.05 and out['R_ci95'][0] > 0 and out['loo_R_min'] > 0 and out['l2o_R_min'] > 0)
    return out


# ------------------------------------------------------------------------ classical models
def fit_constant(tr, d):
    prior = (d['y'][tr].sum() + 1) / (len(tr) + 2)
    return lambda te: np.full(len(te), prior)


def fit_time_cont(tr, d):
    T = time_feats(d['elapsed'])
    mu, sd = e6.zfit(T[tr])
    w = e6.logistic_fit_np(e6.zapply(T[tr], mu, sd), d['y'][tr].astype(float))
    return lambda te: e6.logistic_predict_np(w, e6.zapply(T[te], mu, sd))


def th_design(elapsed, k):
    tk = elapsed + 30.0 * (k - 1)
    oh = np.zeros((len(elapsed), H))
    oh[:, k - 1] = 1
    return np.hstack([oh, np.log1p(tk / 60.0)[:, None], (tk / 3600.0)[:, None]])


def fit_time_hazard(tr, d):
    K = event_interval(d['tau'][tr])
    el = d['elapsed'][tr]
    Xs, ys = [], []
    for k in range(1, H + 1):
        at = K >= k
        Xs.append(th_design(el[at], k))
        ys.append((K[at] == k).astype(int))
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack(Xs), np.concatenate(ys))

    def pred(te):
        surv = np.ones(len(te))
        for k in range(1, H + 1):
            surv *= 1 - clf.predict_proba(th_design(d['elapsed'][te], k))[:, 1]
        return 1 - surv
    return pred


def fit_m0(tr, d):
    X = d['e5']
    mu, sd = e6.zfit(X[tr])
    w = e6.logistic_fit_np(e6.zapply(X[tr], mu, sd), d['y'][tr].astype(float))
    return lambda te: e6.logistic_predict_np(w, e6.zapply(X[te], mu, sd))


def m1_features(d, chans=(0, 1)):
    cols = [c for ch in chans for c in range(ch * 9, ch * 9 + 9)]
    R = d['rich'][:, cols]
    hist = d['hist']
    n = len(R)
    out = np.zeros((n, R.shape[1] * 4 + 1))
    for i in range(n):
        idx = hist[i][hist[i] >= 0]
        seq = R[idx]
        cur = R[i]
        if len(idx) >= 2:
            x = np.arange(len(idx)) - (len(idx) - 1) / 2
            slope = (x[:, None] * (seq - seq.mean(0))).sum(0) / (x ** 2).sum()
            sd = seq.std(0)
            diff = cur - seq[0]
        else:
            slope = sd = diff = np.zeros(R.shape[1])
        out[i] = np.concatenate([cur, slope, sd, diff, [len(idx) / H]])
    return out


def fit_m1(tr, inner_tr, inner_va, d, feats):
    def fit(rows, C):
        mu, sd = feats[rows].mean(0), feats[rows].std(0) + 1e-9
        clf = LogisticRegression(C=C, max_iter=3000).fit((feats[rows] - mu) / sd, d['y'][rows])
        return lambda te: clf.predict_proba((feats[te] - mu) / sd)[:, 1]
    best = min((1e-2, 1e-1, 1.0), key=lambda C: brier(fit(inner_tr, C)(inner_va), d['y'][inner_va]).mean())
    return fit(tr, best), dict(C=best)


# ------------------------------------------------------------------------ neural models
class RawBranch(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        self.stems = nn.ModuleList([nn.Conv1d(cin, w, k, stride=2, padding=k // 2) for k in (7, 25, 101)])
        self.body = nn.Sequential(
            nn.GELU(), nn.GroupNorm(4, 3 * w),
            nn.Conv1d(3 * w, 2 * w, 7, stride=4, padding=3), nn.GELU(), nn.GroupNorm(4, 2 * w),
            nn.Conv1d(2 * w, 2 * w, 7, stride=4, padding=3), nn.GELU(), nn.GroupNorm(4, 2 * w),
            nn.Conv1d(2 * w, 2 * w, 7, stride=4, padding=3), nn.GELU())
        self.out = nn.Linear(4 * w, 32)

    def forward(self, x):
        h = self.body(torch.cat([s(x) for s in self.stems], 1))
        return F.gelu(self.out(torch.cat([h.mean(-1), torch.log(h.var(-1) + 1e-6)], 1)))


class CTHNet(nn.Module):
    def __init__(self, n_feat, width=32, use_raw=True, use_traj=True, head='hazard', use_time=False, cin=2):
        super().__init__()
        self.use_raw, self.use_traj, self.use_time, self.head_type = use_raw, use_traj, use_time, head
        dim = 0
        if use_raw:
            self.raw = RawBranch(cin, width)
            dim += 32
        if use_traj:
            self.gru = nn.GRU(2 * n_feat + 1, 2 * width, batch_first=True)
            dim += 2 * width
        if use_time:
            dim += 2
        self.fuse = nn.Sequential(nn.Linear(dim, 32), nn.GELU())
        self.head = nn.Linear(32, H if head == 'hazard' else 1)

    def forward(self, raw, seq, tf):
        parts = []
        if self.use_raw:
            parts.append(self.raw(raw))
        if self.use_traj:
            parts.append(self.gru(seq)[0][:, -1])
        if self.use_time:
            parts.append(tf)
        return self.head(self.fuse(torch.cat(parts, 1)))


def hazard_loss(z, K):
    log_h, log_1mh = F.logsigmoid(z), F.logsigmoid(-z)
    k = torch.arange(1, H + 1, device=z.device)[None, :]
    before = (k < K[:, None]).float()
    event = (k == K[:, None]).float()
    return -(before * log_1mh + event * log_h).sum(1).mean()


def p300_from(z, head):
    if head == 'hazard':
        return 1 - torch.exp(F.logsigmoid(-z).sum(1))
    return torch.sigmoid(z[:, 0])


class Tensors:
    """Fold-specific, training-only normalised tensors."""

    def __init__(self, d, tr, chans=(0, 1), hist_len=H):
        raw = d['raw'][:, list(chans)]
        rr = raw[tr]
        ac = np.sqrt(((rr - rr.mean(-1, keepdims=True)) ** 2).mean(-1))
        self.scale = float(np.median(ac))
        cols = [c for ch in chans for c in range(ch * 9, ch * 9 + 9)]
        R = d['rich'][:, cols]
        self.mu, self.sd = R[tr].mean(0), R[tr].std(0) + 1e-9
        Rz = (R - self.mu) / self.sd
        hist = d['hist'][:, H - hist_len:]
        valid = hist >= 0
        seq = np.where(valid[..., None], Rz[np.maximum(hist, 0)], 0.0)
        prev_valid = np.concatenate([np.zeros((len(hist), 1), bool), valid[:, :-1]], 1)
        delta = np.where((valid & prev_valid)[..., None], np.diff(seq, axis=1, prepend=seq[:, :1]), 0.0)
        S = np.concatenate([seq, delta, valid[..., None].astype(float)], -1)
        x = np.clip((raw - raw.mean(-1, keepdims=True)) / self.scale, -10, 10)
        self.raw = torch.tensor(x, dtype=torch.float32, device=DEV)
        self.seq = torch.tensor(S, dtype=torch.float32, device=DEV)
        T = time_feats(d['elapsed'])
        self.tmu, self.tsd = T[tr].mean(0), T[tr].std(0) + 1e-9
        self.tf = torch.tensor((T - self.tmu) / self.tsd, dtype=torch.float32, device=DEV)
        self.K = torch.tensor(event_interval(d['tau']), device=DEV)
        self.y = torch.tensor(d['y'], dtype=torch.float32, device=DEV)
        self.n_feat = len(cols)
        self.cin = len(chans)


def predict_net(model, T, idx, head):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            b = torch.tensor(idx[s:s + 1024], device=DEV)
            out.append(p300_from(model(T.raw[b], T.seq[b], T.tf[b]), head))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_net(T, tr, va, cfg, arch, seed, max_epochs=MAX_EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CTHNet(T.n_feat, cfg['width'], cin=T.cin, **arch).to(DEV)
    # Training-only base-rate initialisation (design note D1): zero logits would give hazards of 0.5 per interval,
    # i.e. an initial P(<=300 s) of ~0.999. Start the head at the training prevalence instead.
    prior = float(T.y[torch.tensor(tr, device=DEV)].mean().clamp(1e-3, 1 - 1e-3))
    h0 = 1 - (1 - prior) ** (1 / H)
    with torch.no_grad():
        model.head.bias.fill_(float(np.log(h0 / (1 - h0))) if arch['head'] == 'hazard' else float(np.log(prior / (1 - prior))))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    yv = T.y[torch.tensor(va, device=DEV)].cpu().numpy()
    best, best_state, bad, ep_run = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), BATCH):
            b = torch.tensor(perm[s:s + BATCH], device=DEV)
            z = model(T.raw[b], T.seq[b], T.tf[b])
            loss = hazard_loss(z, T.K[b]) if arch['head'] == 'hazard' else F.binary_cross_entropy_with_logits(z[:, 0], T.y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ep_run = ep + 1
        score = float(((predict_net(model, T, va, arch['head']) - yv) ** 2).mean())
        if score < best - 1e-7:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, dict(inner_val_brier_ww=best, epochs=ep_run, train_s=round(time.time() - t0, 1),
                       n_params=int(sum(p.numel() for p in model.parameters())))


ARCHS = {
    'M2_raw_cnn_binary': dict(use_raw=True, use_traj=False, head='binary', use_time=False),
    'M3_cthnet': dict(use_raw=True, use_traj=True, head='hazard', use_time=False),
    'A1_no_trajectory': dict(use_raw=True, use_traj=False, head='hazard', use_time=False),
    'A2_no_raw': dict(use_raw=False, use_traj=True, head='hazard', use_time=False),
    'A3_binary_head': dict(use_raw=True, use_traj=True, head='binary', use_time=False),
    'A4_history_30s': dict(use_raw=True, use_traj=True, head='hazard', use_time=False),
    'A5_plus_time': dict(use_raw=True, use_traj=True, head='hazard', use_time=True),
    'A6a_fpz_only': dict(use_raw=True, use_traj=True, head='hazard', use_time=False),
    'A6b_pz_only': dict(use_raw=True, use_traj=True, head='hazard', use_time=False),
}
SPECIAL = {'A4_history_30s': dict(hist_len=1), 'A6a_fpz_only': dict(chans=(0,)), 'A6b_pz_only': dict(chans=(1,))}


# ------------------------------------------------------------------------ cross-validation
def run_cv(d, folds_to_run=None, max_epochs=MAX_EPOCHS, seeds=SEEDS, grid=GRID, archs=None):
    folds = person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    archs = archs or list(ARCHS)
    names = ['T0_constant', 'T1_time_cont', 'TH_time_hazard', 'M0_e5_logistic', 'M1_dynamics_logistic'] + archs
    oof = {m: np.full(len(d['y']), np.nan) for m in names}
    seed_oof = {m: {s: np.full(len(d['y']), np.nan) for s in seeds} for m in archs}
    log = []
    m1_all = m1_features(d)
    for f in (folds_to_run or range(N_FOLDS)):
        te = np.where(pf == f)[0]
        tr = np.where(pf != f)[0]
        inner_va = np.where(pf == (f + 1) % N_FOLDS)[0]
        inner_tr = np.setdiff1d(tr, inner_va)
        oof['T0_constant'][te] = fit_constant(tr, d)(te)
        oof['T1_time_cont'][te] = fit_time_cont(tr, d)(te)
        oof['TH_time_hazard'][te] = fit_time_hazard(tr, d)(te)
        oof['M0_e5_logistic'][te] = fit_m0(tr, d)(te)
        m1, m1info = fit_m1(tr, inner_tr, inner_va, d, m1_all)
        oof['M1_dynamics_logistic'][te] = m1(te)
        log.append(dict(fold=f, model='M1_dynamics_logistic', **m1info))
        Tfull = Tensors(d, inner_tr)
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
            T = Tensors(d, inner_tr, **SPECIAL.get(name, {})) if name in SPECIAL else Tfull
            preds = []
            for s in seeds:
                model, info = train_net(T, inner_tr, inner_va, cfg, ARCHS[name], seed=s, max_epochs=max_epochs)
                p = predict_net(model, T, te, ARCHS[name]['head'])
                seed_oof[name][s][te] = p
                preds.append(p)
                log.append(dict(fold=f, model=name, stage='seed', seed=s, cfg=cfg, **info))
            oof[name][te] = np.mean(preds, 0)
        print(f'[CTH] fold {f} done', flush=True)
    return oof, seed_oof, log, folds


def summarise(oof, seed_oof, log, d, folds):
    res = dict(models={}, comparisons={}, seeds={})
    for m, p in oof.items():
        res['models'][m] = metrics(p, d)
    for m, ss in seed_oof.items():
        res['seeds'][m] = {str(s): metrics(p, d)['brier_ww'] for s, p in ss.items()}
    pairs = [('M0_e5_logistic', 'M3_cthnet'), ('TH_time_hazard', 'M3_cthnet'), ('T0_constant', 'M3_cthnet'),
             ('M0_e5_logistic', 'M1_dynamics_logistic'), ('TH_time_hazard', 'M1_dynamics_logistic'),
             ('M0_e5_logistic', 'M2_raw_cnn_binary'), ('TH_time_hazard', 'M2_raw_cnn_binary'),
             ('T0_constant', 'M0_e5_logistic'), ('T0_constant', 'TH_time_hazard'), ('T0_constant', 'T1_time_cont'),
             ('M2_raw_cnn_binary', 'M3_cthnet')]
    pairs += [(a, 'M3_cthnet') for a in ARCHS if a.startswith('A')]
    for a, b in pairs:
        if a in oof and b in oof:
            res['comparisons'][f'{a}__vs__{b}'] = paired(oof[a], oof[b], d)
    params = {}
    for r in log:
        if r.get('stage') == 'seed':
            params.setdefault(r['model'], dict(n_params=r['n_params'], train_s=[], epochs=[]))
            params[r['model']]['train_s'].append(r['train_s'])
            params[r['model']]['epochs'].append(r['epochs'])
    res['compute'] = {m: dict(n_params=v['n_params'], mean_train_s=float(np.mean(v['train_s'])),
                              mean_epochs=float(np.mean(v['epochs']))) for m, v in params.items()}
    res['selected_configs'] = [r for r in log if r.get('stage') == 'grid']
    res['m1_C'] = [r for r in log if r['model'] == 'M1_dynamics_logistic']
    # predefined selection rule (protocol §7)
    cred = lambda m: all(res['comparisons'].get(f'{ref}__vs__{m}', {}).get('credible', False)  # noqa: E731
                         for ref in ('M0_e5_logistic', 'TH_time_hazard'))
    if cred('M3_cthnet'):
        sel = 'M3_cthnet'
    elif cred('M1_dynamics_logistic'):
        sel = 'M1_dynamics_logistic'
    elif cred('M2_raw_cnn_binary'):
        sel = 'M2_raw_cnn_binary'
    else:
        sel = 'M0_e5_logistic (retain E7B recipe)'
    res['selection'] = dict(selected=sel, credible_vs_both={m: cred(m) for m in ('M3_cthnet', 'M1_dynamics_logistic', 'M2_raw_cnn_binary')})
    res['data'] = dict(n_windows=int(len(d['y'])), n_positive=int(d['y'].sum()), n_persons=len(set(d['person'])),
                       n_recordings=len(set(d['recording'])), folds={k: int(v) for k, v in folds.items()})
    return res


# ------------------------------------------------------------------------ freeze (TRAIN-only final fit)
def freeze(model_name):
    import collections
    import csv
    import io
    d = load()
    res = json.loads((RES / 'cth_cv_results_v1.json').read_text())
    picks = []
    for f in range(N_FOLDS):
        g = [x for x in res['selected_configs'] if x['fold'] == f and x['model'] == model_name]
        picks.append(json.dumps(min(g, key=lambda x: x['inner_val_brier_ww'])['cfg'], sort_keys=True))
    cfg = json.loads(sorted(collections.Counter(picks).items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
    folds = person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    inner_va = np.where(pf == 0)[0]
    inner_tr = np.where(pf != 0)[0]
    T = Tensors(d, inner_tr)
    out = RES / 'frozen_m2_v1'
    if out.exists():
        raise RuntimeError('frozen model exists; refusing overwrite')
    out.mkdir(parents=True)
    arch = ARCHS[model_name]
    weights, infos = {}, []
    cpu_preds = []
    for s in SEEDS:
        model, info = train_net(T, inner_tr, inner_va, cfg, arch, seed=s)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        path = out / f'{model_name}_seed{s}.pt'
        torch.save(sd, path)
        weights[path.name] = digest(path)
        infos.append(dict(seed=s, **info))
        m_cpu = CTHNet(T.n_feat, cfg['width'], cin=T.cin, **arch)
        m_cpu.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        m_cpu.eval()
        with torch.no_grad():
            raw = T.raw.cpu()
            z = torch.cat([m_cpu(raw[i:i + 512], T.seq.cpu()[i:i + 512], T.tf.cpu()[i:i + 512]) for i in range(0, len(raw), 512)])
            cpu_preds.append(p300_from(z, arch['head']).numpy().astype(np.float64))
    ens = np.mean(cpu_preds, 0)
    buf = io.StringIO(newline='')
    w = csv.writer(buf)
    w.writerow(['recording', 'elapsed_s', 'p_ensemble'] + [f'p_seed{s}' for s in SEEDS])
    for i in range(len(ens)):
        w.writerow([d['recording'][i], float(d['elapsed'][i]), float(ens[i])] + [float(p[i]) for p in cpu_preds])
    (out / 'train_reference_predictions_cpu.csv').write_text(buf.getvalue())
    prep = dict(model=model_name, arch=arch, cfg=cfg, channels=['EEG Fpz-Cz', 'EEG Pz-Oz'], fs_hz=100, window_s=30,
                raw_preprocessing='per-window mean removal; divide by raw_scale_uv; clip [-10, 10]',
                raw_scale_uv=T.scale, rich_mu=T.mu.tolist(), rich_sd=T.sd.tolist(), time_mu=T.tmu.tolist(), time_sd=T.tsd.tolist(),
                ensemble='mean of p300 over seeds ' + str(list(SEEDS)), output='P(first R&K Stage 2 within 300 s)',
                training=dict(persons=46, recordings=len(set(d['recording'])), windows=int(len(d['y'])),
                              early_stopping_persons='fold 0', fit_persons='folds 1-4'),
                train_infos=infos)
    (out / 'preprocessing_and_config.json').write_text(json.dumps(prep, indent=2, default=float) + '\n')
    man = dict(version='H-EVAL-CTH-v1-frozen-' + model_name, weights_sha256=weights,
               preprocessing_sha256=digest(out / 'preprocessing_and_config.json'),
               reference_predictions_sha256=digest(out / 'train_reference_predictions_cpu.csv'),
               code_sha256=digest(Path(__file__)), dataset_sha256=digest(DATA),
               cv_results_sha256=digest(RES / 'cth_cv_results_v1.json'),
               protocol_sha256=digest(REPO / 'docs/decision/h_eval/cth_v1/cth_v1_protocol.md'),
               selection_note_sha256=digest(REPO / 'docs/decision/h_eval/cth_v1/cth_v1_selection_note.md'),
               environment=dict(torch=torch.__version__, numpy=np.__version__), inference_device_for_reference='cpu')
    (out / 'frozen_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    print(json.dumps(dict(cfg=cfg, weights=weights, manifest_sha256=digest(out / 'frozen_manifest.json')), indent=2))


def verify_frozen(model_name):
    import csv
    d = load()
    out = RES / 'frozen_m2_v1'
    prep = json.loads((out / 'preprocessing_and_config.json').read_text())
    raw = np.clip((d['raw'] - d['raw'].mean(-1, keepdims=True)) / prep['raw_scale_uv'], -10, 10)
    raw = torch.tensor(raw, dtype=torch.float32)
    dummy_seq = torch.zeros((len(raw), H, 37))
    dummy_tf = torch.zeros((len(raw), 2))
    preds = []
    for s in SEEDS:
        m = CTHNet(18, prep['cfg']['width'], cin=2, **prep['arch'])
        m.load_state_dict(torch.load(out / f'{model_name}_seed{s}.pt', map_location='cpu', weights_only=True))
        m.eval()
        with torch.no_grad():
            z = torch.cat([m(raw[i:i + 512], dummy_seq[i:i + 512], dummy_tf[i:i + 512]) for i in range(0, len(raw), 512)])
        preds.append(p300_from(z, prep['arch']['head']).numpy().astype(np.float64))
    ens = np.mean(preds, 0)
    with (out / 'train_reference_predictions_cpu.csv').open() as f:
        ref = np.array([float(r['p_ensemble']) for r in csv.DictReader(f)])
    return dict(max_abs_diff=float(np.abs(ens - ref).max()), n=len(ref))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('feasibility', 'cv', 'freeze', 'verify-frozen'))
    ap.add_argument('--model', default='M2_raw_cnn_binary')
    a = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = False
    d = load()
    t0 = time.time()
    if a.mode == 'freeze':
        freeze(a.model)
        return
    if a.mode == 'verify-frozen':
        print(json.dumps(verify_frozen(a.model)))
        return
    if a.mode == 'feasibility':
        oof, seed_oof, log, folds = run_cv(d, folds_to_run=[0], max_epochs=3, seeds=(0,), grid=GRID[:1],
                                            archs=['M2_raw_cnn_binary', 'M3_cthnet', 'A4_history_30s', 'A6a_fpz_only'])
        te = ~np.isnan(oof['M3_cthnet'])
        sub = {k: v[te] for k, v in d.items()}
        rep = dict(mode='feasibility_fold0_3epochs_NOT_A_RESULT', runtime_s=round(time.time() - t0, 1),
                   models={m: metrics(p[te], sub)['brier_ww'] for m, p in oof.items()}, log=log, device=str(DEV))
        print(json.dumps(rep, indent=1, default=float)[:3000])
        (LOC / 'feasibility_v1.json').write_text(json.dumps(rep, indent=1, default=float))
        return
    oof, seed_oof, log, folds = run_cv(d)
    res = summarise(oof, seed_oof, log, d, folds)
    res.update(version='H-EVAL-CTH-v1-cv', runtime_s=round(time.time() - t0, 1), device=str(DEV),
               protocol_sha256=digest(REPO / 'docs/decision/h_eval/cth_v1/cth_v1_protocol.md'),
               code_sha256=digest(Path(__file__)), dataset_sha256=digest(DATA))
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / 'cth_cv_results_v1.json'
    if out.exists():
        raise RuntimeError('results exist; refusing overwrite')
    out.write_text(json.dumps(res, indent=2, default=float) + '\n')
    np.savez(LOC / 'cth_oof_predictions_v1.npz', **{m: p for m, p in oof.items()}, y=d['y'], person=d['person'])
    (LOC / 'cth_cv_log_v1.json').write_text(json.dumps(log, indent=1, default=float))
    print(json.dumps(dict(selection=res['selection'], models={m: round(v['brier_ww'], 5) for m, v in res['models'].items()}), indent=2))


if __name__ == '__main__':
    main()
