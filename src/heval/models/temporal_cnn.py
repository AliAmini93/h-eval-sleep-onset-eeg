"""Causal temporal CNN hazard models of the TRAIN-only wake-to-first-N1 analysis.

Families (fixed configuration, no search): TCN-30s (raw, current window), TCN-5m (raw, last 10 windows with a
missing-history mask channel), TCN-5m+stage (adds an auxiliary W-vs-Stage-1 head trained on TRAIN labels and a
state-conditional N2 hazard mixture; the stage is never an input) and Spec-5m+stage (log-STFT 0.5-25 Hz input).
Each network has a 10-interval discrete hazard head for first N1 (E1 windows) and one for first N2 (all windows).
Training: AdamW (lr 1e-3, weight decay 1e-3), batch 128, early stopping on a person-disjoint validation fold
(patience 5, at most 30 epochs); seeds 0, 1, 2 averaged. Personalised variants (A-E) add elapsed-time features,
a causal within-recording EEG baseline, a GRU over the recording's own embeddings and/or a calibration-aware loss;
the donor control replaces the history by another person's recording.

Lineage: W2S neural families, nested fixed-epoch refits (P0b) and personalisation (P1) of the research archive
(tag v0.1.0-paper-review). Arithmetic unchanged; the device is a parameter (the original ran on CUDA).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, SEEDS, MAX_EPOCHS, PATIENCE, BATCH, LR, WD, STAGE_W = 10, (0, 1, 2), 30, 5, 128, 1e-3, 1e-3, 0.5
MODELS = {'N1_tcn30s': dict(enc='raw', hist=1, state=False), 'N2_tcn5min': dict(enc='raw', hist=10, state=False),
          'N3_tcn5min_state': dict(enc='raw', hist=10, state=True), 'N4_spec5min_state': dict(enc='spec', hist=10, state=True)}
VARIANTS = {'A': dict(ref=False, gru=False, cal=False), 'B': dict(ref=True, gru=False, cal=False),
            'C': dict(ref=False, gru=True, cal=False), 'D': dict(ref=False, gru=False, cal=True),
            'E': dict(ref=True, gru=True, cal=True)}
LAMBDA_CAL = 2.0


class CausalBlock(nn.Module):
    def __init__(self, c, k, dil):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(c, c, k, dilation=dil)
        self.c2 = nn.Conv1d(c, c, k, dilation=dil)
        self.n1, self.n2 = nn.GroupNorm(4, c), nn.GroupNorm(4, c)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(F.pad(x, (self.pad, 0)))))
        h = self.n2(self.c2(F.pad(h, (self.pad, 0))))
        return F.gelu(x + h)


class RawTCN(nn.Module):
    """Strided causal stem and three stages of dilated causal blocks with pooling; input (B, 3, L*3000)."""

    def __init__(self, cin=3, c=32):
        super().__init__()
        self.stem = nn.Conv1d(cin, c, 11, stride=5)
        self.stages = nn.ModuleList([nn.Sequential(*[CausalBlock(c, 5, d) for d in (1, 2, 4, 8)]) for _ in range(3)])
        self.out = nn.Linear(2 * c, 64)

    def forward(self, x):
        h = self.stem(F.pad(x, (10, 0)))
        for i, s in enumerate(self.stages):
            h = s(h)
            if i < 2:
                h = F.avg_pool1d(h, 4)
        tail = h[..., -8:]                          # most recent ~25-30 s of the causal feature sequence
        return F.gelu(self.out(torch.cat([tail.mean(-1), h[..., -1]], 1)))


class SpecTCN(nn.Module):
    def __init__(self, cin=101, c=64):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.blocks = nn.Sequential(*[CausalBlock(c, 3, d) for d in (1, 2, 4, 8, 16, 32)])
        self.out = nn.Linear(2 * c, 64)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        return F.gelu(self.out(torch.cat([h[..., -30:].mean(-1), h[..., -1]], 1)))


class HazardNet(nn.Module):
    def __init__(self, enc, state):
        super().__init__()
        self.enc = RawTCN() if enc == 'raw' else SpecTCN()
        self.state = state
        self.n1 = nn.Linear(64, H)
        if state:
            self.s, self.hw, self.hs1 = nn.Linear(64, 1), nn.Linear(64, H), nn.Linear(64, H)
        else:
            self.n2 = nn.Linear(64, H)

    def forward(self, x):
        z = self.enc(x)
        log_h1, log_1mh1 = F.logsigmoid(self.n1(z)), F.logsigmoid(-self.n1(z))
        if self.state:
            s = self.s(z)[:, 0]
            pi = torch.sigmoid(s)[:, None]
            h = (pi * torch.sigmoid(self.hs1(z)) + (1 - pi) * torch.sigmoid(self.hw(z))).clamp(1e-6, 1 - 1e-6)
            log_h2, log_1mh2 = torch.log(h), torch.log1p(-h)
        else:
            s = None
            log_h2, log_1mh2 = F.logsigmoid(self.n2(z)), F.logsigmoid(-self.n2(z))
        return (log_h1, log_1mh1), (log_h2, log_1mh2), s


def hazard_nll(lh, l1mh, K, w):
    """Discrete-time hazard NLL; K in 1..10 = event interval, 11 = no event within 300 s; w = window weights."""
    k = torch.arange(1, H + 1, device=K.device)[None, :]
    nll = -(((k < K[:, None]).float() * l1mh) + ((k == K[:, None]).float() * lh)).sum(1)
    return (nll * w).sum() / w.sum().clamp(min=1)


def event_interval(tau):
    """Time to event (s) -> interval index 1..10 within 300 s, else 11."""
    tau = np.asarray(tau, float)
    return np.where(tau <= 300 + 1e-8, np.clip(np.ceil(tau / 30 - 1e-9), 1, H), H + 1).astype(int)


class HistoryInputs:
    """Network inputs built from per-window raw EEG and the causal history index of each window.

    d: dict with 'hist' (N, 10) chronological indices of the same recording's previous windows (-1 = missing,
    last column = current window), 'tau' (time to first N2, s), 'k1' (first-N1 interval 1..11), 'e1' (E1 mask),
    'stage' (scored input stage, used only as an auxiliary training target). raw: (N, 2, 3000) microvolts.
    tr: training rows (normalisation statistics are fitted on these only)."""

    def __init__(self, d, raw, tr, cfg, device='cpu'):
        self.device = device
        raw = np.asarray(raw)
        rr = raw[tr]
        self.scale = float(np.median(np.sqrt(((rr - rr.mean(-1, keepdims=True)) ** 2).mean(-1))))
        x = np.clip((raw - raw.mean(-1, keepdims=True)) / self.scale, -10, 10).astype(np.float32)
        self.raw = torch.tensor(x, device=device)
        self.hist = torch.tensor(np.asarray(d['hist'])[:, -cfg['hist']:], device=device)
        self.cfg = cfg
        self.K2 = torch.tensor(event_interval(d['tau']), device=device)
        self.K1 = torch.tensor(np.asarray(d['k1']), device=device)
        self.e1 = torch.tensor(np.asarray(d['e1']).astype(np.float32), device=device)
        self.st = torch.tensor((np.asarray(d['stage']) == '1').astype(np.float32), device=device)
        self.mu = self.sd = None
        if cfg['enc'] == 'spec':
            S = torch.cat([self.batch(torch.tensor(tr[i:i + 256], device=device), raw_only=True) for i in range(0, len(tr), 256)])
            valid = S[:, -1:, :] < 0.5                                   # frames without masked samples
            v = S[:, :-1, :].permute(0, 2, 1)[valid[:, 0, :]]
            self.mu, self.sd = v.mean(0)[None, :, None], v.std(0)[None, :, None] + 1e-6

    def batch(self, idx, raw_only=False):
        h = self.hist[idx]                                               # (B, L) chronological, -1 = masked
        m = (h >= 0)
        x = self.raw[h.clamp(min=0)] * m[..., None, None]                # (B, L, 2, 3000)
        B, L = h.shape
        x = x.permute(0, 2, 1, 3).reshape(B, 2, L * 3000)
        mask = (~m).float()[:, None, :, None].expand(B, 1, L, 3000).reshape(B, 1, L * 3000)
        if self.cfg['enc'] == 'raw':
            return torch.cat([x, mask], 1)
        win = torch.hann_window(200, device=self.device)
        spec = torch.stft(x.reshape(B * 2, -1), n_fft=200, hop_length=100, window=win, center=False,
                          return_complex=True).abs().pow(2)[:, 1:51]      # 0.5-25 Hz
        spec = torch.log(spec + 1e-6).reshape(B, 100, -1)
        mfr = mask.unfold(2, 200, 100).mean(-1)                           # (B, 1, frames)
        S = torch.cat([spec, mfr], 1)
        if raw_only:
            return S
        S[:, :-1] = ((S[:, :-1] - self.mu) / self.sd) * (S[:, -1:] < 0.5)
        return S


def _loss(net, D, idx, cfg, train):
    b = torch.tensor(idx, device=D.device)
    (l1, m1), (l2, m2), s = net(D.batch(b))
    loss = hazard_nll(l2, m2, D.K2[b], torch.ones(len(idx), device=D.device)) + hazard_nll(l1, m1, D.K1[b], D.e1[b])
    if cfg['state'] and train:
        loss = loss + STAGE_W * F.binary_cross_entropy_with_logits(s, D.st[b])
    return loss


def train_hazard_net(d, raw, cfg, tr, va, te, seed, fixed_epochs=None, device='cpu', max_epochs=MAX_EPOCHS):
    """One seed: early stopping on va (or exactly fixed_epochs with va unused). Returns (p_first_N1, p_first_N2, info)
    for the rows te, where p = 1 - prod(1 - hazard) over the ten 30 s intervals."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = HistoryInputs(d, raw, tr, cfg, device)
    net = HazardNet(cfg['enc'], cfg['state']).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep_run = np.inf, None, 0, 0
    for ep in range(fixed_epochs or max_epochs):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), BATCH):
            opt.zero_grad()
            _loss(net, D, order[i:i + BATCH], cfg, True).backward()
            opt.step()
        ep_run = ep + 1
        if fixed_epochs:
            continue
        net.eval()
        with torch.no_grad():
            vl = sum(float(_loss(net, D, va[i:i + 256], cfg, False)) * len(va[i:i + 256]) for i in range(0, len(va), 256)) / len(va)
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    p1, p2 = [], []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            (_, m1), (_, m2), _ = net(D.batch(torch.tensor(te[i:i + 256], device=device)))
            p1.append((1 - torch.exp(m1.sum(1))).cpu().numpy())
            p2.append((1 - torch.exp(m2.sum(1))).cpu().numpy())
    return np.concatenate(p1), np.concatenate(p2), dict(best_val=best, epochs=ep_run,
                                                        n_params=sum(p.numel() for p in net.parameters()))


# ------------------------------------------------------------------------ personalised variants (whole recordings)
class PersonalNet(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
        self.enc = RawTCN()
        dim = 64 + 2 + (64 if v['ref'] else 0) + (64 if v['gru'] else 0)
        if v['gru']:
            self.gru = nn.GRU(64, 64, batch_first=True)
        self.body = nn.Sequential(nn.Linear(dim, 64), nn.GELU())
        self.h1, self.h2 = nn.Linear(64, H), nn.Linear(64, H)

    def heads(self, e, hist_e, tf):
        """e: (n, 64) embeddings of one recording in elapsed order; hist_e: history sequence (own e or a donor's);
        tf: (n, 2) standardised time features. Position t uses hist_e[:t] and e[t] only."""
        parts = [e, tf]
        if self.v['ref']:
            prev = torch.cumsum(hist_e, 0) - hist_e                       # sum of hist_e[:t]
            cnt = torch.arange(len(e), device=e.device, dtype=e.dtype)[:, None]
            parts.append(e - (prev + e) / (cnt + 1))
        if self.v['gru']:
            h_seq, _ = self.gru(hist_e[None])
            before = torch.cat([torch.zeros(1, 64, device=e.device), h_seq[0, :-1]], 0)
            parts.append(self.gru(e[:, None, :], before[None].contiguous())[0][:, 0])
        z = self.body(torch.cat(parts, 1))
        return self.h1(z), self.h2(z)


class RecordingInputs:
    def __init__(self, d, raw, tr_idx, device='cpu'):
        self.W = HistoryInputs(d, raw, tr_idx, MODELS['N2_tcn5min'], device)
        el = np.asarray(d['elapsed'], float)
        T = np.column_stack([np.log1p(el / 60), el / 3600])
        mu, sd = T[tr_idx].mean(0), T[tr_idx].std(0) + 1e-9
        self.tf = torch.tensor((T - mu) / sd, dtype=torch.float32, device=device)
        self.y1 = torch.tensor(np.asarray(d['y1']), dtype=torch.float32, device=device)
        self.device = device


def recording_windows(d):
    out = {}
    for r in np.unique(d['recording']):
        idx = np.where(d['recording'] == r)[0]
        out[r] = idx[np.argsort(np.asarray(d['elapsed'])[idx])]
    return out


def p_from(logits):
    return 1 - torch.exp(F.logsigmoid(-logits).sum(1))


def _rec_loss(l1, l2, D, idx, v, train):
    b = torch.tensor(idx, device=D.device)
    loss = hazard_nll(F.logsigmoid(l2), F.logsigmoid(-l2), D.W.K2[b], torch.ones(len(b), device=D.device)) * len(b)
    w1 = D.W.e1[b]
    loss = loss + hazard_nll(F.logsigmoid(l1), F.logsigmoid(-l1), D.W.K1[b], w1) * w1.sum()
    cal = None
    if v['cal'] and train and w1.sum() > 0:
        p1 = p_from(l1)
        cal = (((p1 * w1).sum() / w1.sum()) - ((D.y1[b] * w1).sum() / w1.sum())) ** 2
    return loss, len(b), cal


def _group_losses(net, D, group, v, train):
    all_idx = np.concatenate(group)
    E = net.enc(D.W.batch(torch.tensor(all_idx, device=D.device)))
    out, start = [], 0
    for idx in group:
        e = E[start:start + len(idx)]
        start += len(idx)
        l1, l2 = net.heads(e, e, D.tf[torch.tensor(idx, device=D.device)])
        out.append(_rec_loss(l1, l2, D, idx, v, train))
    return out


def _batches(recs, rng, per_step=4):
    order = list(rng.permutation(len(recs)))
    for i in range(0, len(order), per_step):
        yield [recs[j] for j in order[i:i + per_step]]


def train_personal(d, raw, v, tr_recs, va_recs, seed, fixed_epochs=None, device='cpu', max_epochs=MAX_EPOCHS):
    """tr_recs / va_recs: lists of per-recording row-index arrays in elapsed order (4 recordings per step)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = RecordingInputs(d, raw, np.concatenate(tr_recs), device)
    net = PersonalNet(v).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep_run, best_ep = np.inf, None, 0, 0, 0
    for ep in range(fixed_epochs or max_epochs):
        net.train()
        for group in _batches(tr_recs, rng):
            opt.zero_grad()
            res = _group_losses(net, D, group, v, True)
            n = sum(k for _, k, _ in res)
            cals = [c for _, _, c in res if c is not None]
            loss = sum(l for l, _, _ in res) / n + (LAMBDA_CAL * torch.stack(cals).mean() if cals else 0.0)
            loss.backward()
            opt.step()
        ep_run = ep + 1
        if fixed_epochs:
            continue
        net.eval()
        with torch.no_grad():
            vl = sum(float(l) for i in range(0, len(va_recs), 4) for l, _, _ in _group_losses(net, D, va_recs[i:i + 4], v, False)) \
                / sum(len(i) for i in va_recs)
        if vl < best - 1e-5:
            best, bad, best_ep, best_state = vl, 0, ep_run, {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, D, dict(epochs_run=ep_run, best_epoch=(best_ep if not fixed_epochs else fixed_epochs), best_val=float(best))


def donor_for(r, recs_fold, d):
    """Different-person recording in the same fold with the smallest length >= own length (else the longest)."""
    me = d['person'][recs_fold[r][0]]
    cand = [(len(ix), q) for q, ix in recs_fold.items() if d['person'][ix[0]] != me]
    if not cand:
        raise ValueError('donor control needs at least two persons in the fold')
    ok = sorted(c for c in cand if c[0] >= len(recs_fold[r]))
    return (ok[0] if ok else max(cand))[1]


def predict_personal(net, D, recs_fold, d, donor):
    """{recording: (p_first_N1, p_first_N2)} for the recordings of one fold; donor=True swaps in a donor history."""
    out = {}
    with torch.no_grad():
        for r, idx in recs_fold.items():
            b = torch.tensor(idx, device=D.device)
            e = net.enc(D.W.batch(b))
            he = e
            if donor:
                q = recs_fold[donor_for(r, recs_fold, d)]
                hist = np.concatenate([q, np.repeat(q[-1:], max(0, len(idx) - len(q)))])[:len(idx)]
                he = net.enc(D.W.batch(torch.tensor(hist, device=D.device)))
            l1, l2 = net.heads(e, he, D.tf[b])
            out[r] = (p_from(l1).cpu().numpy(), p_from(l2).cpu().numpy())
    return out
