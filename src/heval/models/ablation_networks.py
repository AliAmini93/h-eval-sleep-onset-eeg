"""Predeclared task-specific network (CTH-Net) and its ablation variants (paper Fig. 7, development CV only).

CTH-Net fuses (i) the raw-waveform branch of RawCNN, (ii) a GRU over the last ten 30 s windows of the same recording,
each described by 9 spectral features per channel (z-scored with training statistics) plus their first differences
and a validity mask, and optionally (iii) standardised elapsed-time features. A Linear 32 + GELU fusion layer feeds a
10-interval discrete-time hazard head (P(first N2 within 300 s) = 1 - prod_k (1 - h_k)) or a binary head.
Variants: M2 raw branch + binary head (became RawCNN); A1 no trajectory; A2 no raw branch; A3 binary head;
A4 30 s history; A5 + elapsed time; A6a Fpz-Cz only; A6b Pz-Oz only.
Training: output bias at the training prevalence (per-interval hazard for the hazard head), AdamW (weight decay 1e-3),
batch 256, gradient-norm clip 1.0, early stopping on an inner person-disjoint validation fold (window-weighted Brier,
patience 6, at most 40 epochs) with the best state restored.

Lineage: CTH v1 experiment (CTHNet, Tensors, hazard_loss, p300_from, train_net, predict_net) and CTH v1 data builder
(rich_features, history index) of the research archive, tag v0.1.0-paper-review. Arithmetic unchanged; the device is
a parameter and train_net also accepts a model constructor (as the representation analysis did for its STFT-CNN).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .raw_cnn import RawBranch

H = 10
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 256
FS = 100
RICH_BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 15), 'beta': (15, 30)}
RICH_NAMES = [f'{ch}_{n}' for ch in ('fpz', 'pz') for n in ('logtot', 'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma',
                                                            'rel_beta', 'log_alpha_theta', 'log_delta_theta', 'log_rms')]
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


# ------------------------------------------------------------------------ inputs
def rich_features(x):
    """One channel's 30 s window (3000 uV samples, float64) -> 9 features: log total power, relative delta/theta/alpha/
    sigma/beta power, log alpha/theta and delta/theta ratios, log1p AC RMS (five 4 s symmetric-Hann frames, 512-pt FFT)."""
    x = np.asarray(x, float)
    if x.shape != (3000,) or not np.isfinite(x).all():
        raise ValueError('expected one finite 3000-sample channel window')
    w = np.hanning(400)
    freqs = np.fft.rfftfreq(512, 1 / FS)
    P = np.zeros(len(freqs))
    for s in (0, 600, 1200, 1800, 2400):
        fr = x[s:s + 400]
        fr = (fr - fr.mean()) * w
        P += np.abs(np.fft.rfft(fr, 512)) ** 2
    bp = {k: P[(freqs >= lo) & (freqs < hi)].sum() for k, (lo, hi) in RICH_BANDS.items()}
    tot = sum(bp.values()) + 1e-12
    rms = np.sqrt(((x - x.mean()) ** 2).mean())
    return [np.log(tot), *(bp[k] / tot for k in RICH_BANDS), np.log((bp['alpha'] + 1e-9) / (bp['theta'] + 1e-9)),
            np.log((bp['delta'] + 1e-9) / (bp['theta'] + 1e-9)), np.log1p(rms)]


def history_index(rec_idx, elapsed, h=H):
    """(N, h) indices of each window's preceding windows of the same recording on the 30 s grid, chronological
    (last column = the window itself), -1 where missing. Rows must be ordered by recording, then elapsed time."""
    rec_idx, elapsed = np.asarray(rec_idx), np.asarray(elapsed, float)
    hist = np.full((len(rec_idx), h), -1, dtype=np.int64)
    for i in range(len(rec_idx)):
        for k in range(h):
            j = i - k
            if j >= 0 and rec_idx[j] == rec_idx[i] and abs(elapsed[i] - elapsed[j] - 30 * k) < 1e-6:
                hist[i, h - 1 - k] = j
            else:
                break
    return hist


def event_interval(tau):
    """K in 1..10 if the event is within 300 s (30 s intervals), else 11 (censored at the horizon)."""
    k = np.ceil(np.asarray(tau) / 30.0 - 1e-9).astype(int)
    return np.where(np.asarray(tau) <= 300 + 1e-8, np.clip(k, 1, H), H + 1)


def time_feats(elapsed):
    e = np.asarray(elapsed, float)
    return np.stack([np.log1p(e / 60.0), e / 3600.0], 1)


class AblationInputs:
    """Training-fold-normalised network inputs. d: dict with raw (N, 2, 3000), rich (N, 18), hist (N, 10), tau,
    elapsed, y. tr: training rows (all normalisation statistics come from these rows only)."""

    def __init__(self, d, tr, chans=(0, 1), hist_len=H, device='cpu'):
        self.device = device
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
        self.raw = torch.tensor(x, dtype=torch.float32, device=device)
        self.seq = torch.tensor(S, dtype=torch.float32, device=device)
        T = time_feats(d['elapsed'])
        self.tmu, self.tsd = T[tr].mean(0), T[tr].std(0) + 1e-9
        self.tf = torch.tensor((T - self.tmu) / self.tsd, dtype=torch.float32, device=device)
        self.K = torch.tensor(event_interval(d['tau']), device=device)
        self.y = torch.tensor(d['y'], dtype=torch.float32, device=device)
        self.n_feat = len(cols)
        self.cin = len(chans)


# ------------------------------------------------------------------------ network, loss, training
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

    def forward(self, raw, seq=None, tf=None):
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


def predict_net(model, T, idx, head):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            b = torch.tensor(idx[s:s + 1024], device=T.device)
            out.append(p300_from(model(T.raw[b], T.seq[b], T.tf[b]), head))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_net(T, tr, va, cfg, arch, seed, max_epochs=MAX_EPOCHS, make_model=None):
    """One seed with early stopping on va. arch: an ARCHS entry; make_model(width) replaces the CTHNet constructor
    (binary head) for other input encoders such as the STFT-CNN."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = 'binary' if make_model is not None else arch['head']
    model = (make_model(cfg['width']) if make_model is not None else CTHNet(T.n_feat, cfg['width'], cin=T.cin, **arch)).to(T.device)
    prior = float(T.y[torch.tensor(tr, device=T.device)].mean().clamp(1e-3, 1 - 1e-3))
    h0 = 1 - (1 - prior) ** (1 / H)
    with torch.no_grad():
        model.head.bias.fill_(float(np.log(h0 / (1 - h0))) if head == 'hazard' else float(np.log(prior / (1 - prior))))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    yv = T.y[torch.tensor(va, device=T.device)].cpu().numpy()
    best, best_state, bad, ep_run = np.inf, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), BATCH):
            b = torch.tensor(perm[s:s + BATCH], device=T.device)
            z = model(T.raw[b], T.seq[b], T.tf[b])
            loss = hazard_loss(z, T.K[b]) if head == 'hazard' else F.binary_cross_entropy_with_logits(z[:, 0], T.y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ep_run = ep + 1
        score = float(((predict_net(model, T, va, head) - yv) ** 2).mean())
        if score < best - 1e-7:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, dict(inner_val_brier_ww=best, epochs=ep_run, n_params=int(sum(p.numel() for p in model.parameters())))
