"""RawCNN: the generic multiscale 1D CNN on the raw two-channel 30 s window (114,817 parameters, width 32).

Architecture: three parallel Conv1d stems 2->32 (kernels 7, 25, 101, stride 2, 'same'-style padding k//2), concatenated
(96) -> GELU -> GroupNorm(4) -> Conv1d 96->64 (k 7, stride 4) -> GELU -> GroupNorm(4) -> Conv1d 64->64 -> GELU ->
GroupNorm(4) -> Conv1d 64->64 -> GELU; per-channel mean and log(variance + 1e-6) over time (128) -> Linear 32 + GELU
-> Linear 32 + GELU -> Linear 1 (logit of P(first N2 within 300 s)).
Input normalisation: per-window, per-channel mean removal, division by one TRAIN scale (median per-window AC RMS
over all TRAIN windows and both channels; 10.141 uV for the 46-person fit), clipping to [-10, 10].
Training (final fit): output bias initialised at the TRAIN prevalence (clamped to [1e-3, 1 - 1e-3]), binary
cross-entropy with logits, AdamW (lr 1e-3, weight decay 1e-3), batch 256, gradient-norm clip 1.0, 4 epochs, seeds
0, 1, 2; the prediction is the mean of the three seeds' probabilities.

Lineage (research archive, tag v0.1.0-paper-review): RawBranch and the binary-head CTHNet configuration
'M2_raw_cnn_binary' from CTH v1; normalisation, fixed-epoch final fit and CPU ensemble inference from CTH v2
(M2_46_FINAL). Module and parameter names are kept so the original state dicts load unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARAMETERS = 114_817
SEEDS = (0, 1, 2)
WIDTH, LR, WEIGHT_DECAY, BATCH, EPOCHS = 32, 1e-3, 1e-3, 256, 4
MAX_EPOCHS, PATIENCE = 40, 6                     # early-stopping replay used to derive the fixed epoch count


class RawBranch(nn.Module):
    def __init__(self, cin=2, w=WIDTH):
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


class RawCNN(nn.Module):
    """x: (batch, 2, 3000) normalised [Fpz-Cz, Pz-Oz] -> (batch, 1) logits."""

    def __init__(self, width=WIDTH, cin=2):
        super().__init__()
        self.raw = RawBranch(cin, width)
        self.fuse = nn.Sequential(nn.Linear(32, 32), nn.GELU())
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.fuse(self.raw(x)))


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


# ------------------------------------------------------------------------ normalisation
def _check_raw(raw_uv):
    raw_uv = np.asarray(raw_uv, dtype=np.float32)
    if raw_uv.ndim != 3 or raw_uv.shape[1:] != (2, 3000):
        raise ValueError('expected (N, 2, 3000) [Fpz-Cz, Pz-Oz] at 100 Hz')
    if not np.isfinite(raw_uv).all():
        raise ValueError('raw EEG contains non-finite values')
    return raw_uv


def train_scale(raw_uv):
    """Median per-window, per-channel AC RMS of the TRAIN windows (float32 arithmetic, as in the original)."""
    rr = _check_raw(raw_uv)
    return float(np.median(np.sqrt(((rr - rr.mean(-1, keepdims=True)) ** 2).mean(-1))))


def normalise(raw_uv, scale):
    raw_uv = _check_raw(raw_uv)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be a positive finite number')
    return np.clip((raw_uv - raw_uv.mean(-1, keepdims=True)) / np.float32(scale), -10, 10)


# ------------------------------------------------------------------------ training
def _epoch(model, opt, x, y, rows, rng, device):
    model.train()
    perm = rng.permutation(rows)
    for s in range(0, len(perm), BATCH):
        b = torch.tensor(perm[s:s + BATCH], device=device)
        z = model(x[b])
        loss = F.binary_cross_entropy_with_logits(z[:, 0], y[b])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _predict(model, x, idx, device):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            b = torch.tensor(idx[s:s + 1024], device=device)
            out.append(torch.sigmoid(model(x[b])[:, 0]))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def fit_seed(x_norm, y, seed, epochs=EPOCHS, train_rows=None, val_rows=None, device='cpu'):
    """Train one seed. Without val_rows: the fixed-epoch final fit. With val_rows: the development replay
    (early stopping on the validation window-weighted Brier, patience 6, at most 40 epochs, best state restored)."""
    x = torch.tensor(np.asarray(x_norm, np.float32), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (2, 3000) or len(yt) != len(x):
        raise ValueError('x_norm must be (N, 2, 3000) and y must have length N')
    rows = np.arange(len(x)) if train_rows is None else np.asarray(train_rows)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RawCNN().to(device)
    prior = float(yt[torch.tensor(rows, device=device)].mean().clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        model.head.bias.fill_(float(np.log(prior / (1 - prior))))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    if val_rows is None:
        for _ in range(epochs):
            _epoch(model, opt, x, yt, rows, rng, device)
        return model.eval(), dict(seed=seed, epochs=epochs)
    yv = yt[torch.tensor(val_rows, device=device)].cpu().numpy()
    best, best_ep, best_state, bad, ep_run = np.inf, 0, None, 0, 0
    for ep in range(MAX_EPOCHS):
        _epoch(model, opt, x, yt, rows, rng, device)
        ep_run = ep + 1
        score = float(((_predict(model, x, val_rows, device) - yv) ** 2).mean())
        if score < best - 1e-7:
            best, best_ep, bad = score, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval(), dict(seed=seed, best_epoch=best_ep, epochs_run=ep_run, best_val_brier_ww=best)


def epoch_schedule(best_epochs):
    """Fixed epoch count of the final fit: round-half-up median of the replayed best epochs."""
    return max(1, int(np.floor(float(np.median(best_epochs)) + 0.5)))


def fit_ensemble(raw_uv, y, seeds=SEEDS, epochs=EPOCHS, device='cpu'):
    """Final fit on all given TRAIN windows: returns (scale, [model per seed])."""
    scale = train_scale(raw_uv)
    x = normalise(raw_uv, scale)
    return scale, [fit_seed(x, y, s, epochs=epochs, device=device)[0] for s in seeds]


def predict_ensemble(raw_uv, scale, models):
    """(N, 2, 3000) microvolts -> (ensemble mean probability float64, per-seed list); CPU, batches of 512."""
    x = torch.tensor(normalise(raw_uv, scale), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for m in models:
            m = m.cpu().eval()
            z = torch.cat([m(x[i:i + 512]) for i in range(0, len(x), 512)])
            out.append(torch.sigmoid(z[:, 0]).numpy().astype(np.float64))
    return np.mean(out, 0), out


def load_state_dicts(paths):
    """Load per-seed state dicts (e.g. a separately obtained M2_46_FINAL_seed{0,1,2}.pt) into RawCNN models on CPU."""
    models = []
    for p in paths:
        m = RawCNN()
        m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        models.append(m.eval())
    return models
