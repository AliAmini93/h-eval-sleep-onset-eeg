#!/usr/bin/env python3
"""H-EVAL W2S: neural families N1-N4 on TRAIN (design w2s_design_v1.md §4). Fixed config, no grid search.

N1 raw causal TCN, current 30 s only            | hazard heads N2 (all windows) + N1 (E1 windows)
N2 raw causal TCN, last 5 min (masked history)  | same heads
N3 = N2 encoder + state-aware head: auxiliary stage supervision (W vs S1, training labels only) and a
     state-conditional N2 hazard mixture h = pi*h_S1 + (1-pi)*h_W, pi = sigmoid(stage logit). Stage is never an input.
N4 = log-STFT (2 s frames, 1 s hop, 0.5-25 Hz) over the last 5 min + causal temporal conv + N3 head.
Folds: CTH partition A; early stopping on the CTH inner-validation fold. Seeds 0,1,2 averaged.
Writes OOF to .local/h_eval/discovery_w2s/w2s_neural_oof_<model>.npz and a training log.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w2s_eval as ev  # noqa: E402

REPO = ev.REPO
LOC = REPO / '.local/h_eval/discovery_w2s'
DATA = REPO / '.local/h_eval/cth_v1/dataset_v1.npz'
DEV = torch.device('cuda')
H, SEEDS, MAX_EPOCHS, PATIENCE, BATCH, LR, WD, STAGE_W = 10, (0, 1, 2), 30, 5, 128, 1e-3, 1e-3, 0.5
MODELS = {'N1_tcn30s': dict(enc='raw', hist=1, state=False), 'N2_tcn5min': dict(enc='raw', hist=10, state=False),
          'N3_tcn5min_state': dict(enc='raw', hist=10, state=True), 'N4_spec5min_state': dict(enc='spec', hist=10, state=True)}


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
    """Multiresolution causal TCN: strided causal stem, 3 stages of dilated causal blocks with pooling."""
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


class Net(nn.Module):
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
    k = torch.arange(1, H + 1, device=K.device)[None, :]
    nll = -(((k < K[:, None]).float() * l1mh) + ((k == K[:, None]).float() * lh)).sum(1)
    return (nll * w).sum() / w.sum().clamp(min=1)


class Data:
    def __init__(self, d, tr, cfg):
        z = np.load(DATA)
        raw = z['raw']
        rr = raw[tr]
        self.scale = float(np.median(np.sqrt(((rr - rr.mean(-1, keepdims=True)) ** 2).mean(-1))))
        x = np.clip((raw - raw.mean(-1, keepdims=True)) / self.scale, -10, 10).astype(np.float32)
        self.raw = torch.tensor(x, device=DEV)
        hist = d['hist'][:, -cfg['hist']:]
        self.hist = torch.tensor(hist, device=DEV)
        self.cfg = cfg
        tau = d['tau']
        K2 = np.where(tau <= 300 + 1e-8, np.clip(np.ceil(tau / 30 - 1e-9), 1, H), H + 1)
        self.K2 = torch.tensor(K2, device=DEV)
        self.K1 = torch.tensor(d['k1'], device=DEV)
        self.e1 = torch.tensor(d['e1'].astype(np.float32), device=DEV)
        self.st = torch.tensor((d['stage'] == '1').astype(np.float32), device=DEV)
        self.mu = self.sd = None
        if cfg['enc'] == 'spec':
            S = torch.cat([self.batch(torch.tensor(tr[i:i + 256], device=DEV), raw_only=True) for i in range(0, len(tr), 256)])
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
        win = torch.hann_window(200, device=DEV)
        spec = torch.stft(x.reshape(B * 2, -1), n_fft=200, hop_length=100, window=win, center=False,
                          return_complex=True).abs().pow(2)[:, 1:51]      # 0.5-25 Hz
        spec = torch.log(spec + 1e-6).reshape(B, 100, -1)
        mfr = mask.unfold(2, 200, 100).mean(-1)                           # (B, 1, frames)
        S = torch.cat([spec, mfr], 1)
        if raw_only:
            return S
        S[:, :-1] = ((S[:, :-1] - self.mu) / self.sd) * (S[:, -1:] < 0.5)
        return S


def run_fold(d, cfg, tr, va, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = Data(d, tr, cfg)
    net = Net(cfg['enc'], cfg['state']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)

    def loss_on(idx, train):
        b = torch.tensor(idx, device=DEV)
        (l1, m1), (l2, m2), s = net(D.batch(b))
        loss = hazard_nll(l2, m2, D.K2[b], torch.ones(len(idx), device=DEV)) + hazard_nll(l1, m1, D.K1[b], D.e1[b])
        if cfg['state'] and train:
            loss = loss + STAGE_W * F.binary_cross_entropy_with_logits(s, D.st[b])
        return loss

    best, best_state, bad, ep_run = np.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), BATCH):
            opt.zero_grad()
            loss_on(order[i:i + BATCH], True).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(float(loss_on(va[i:i + 256], False)) * len(va[i:i + 256]) for i in range(0, len(va), 256)) / len(va)
        ep_run = ep + 1
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    p1, p2 = [], []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            (l1, m1), (l2, m2), _ = net(D.batch(torch.tensor(te[i:i + 256], device=DEV)))
            p1.append((1 - torch.exp(m1.sum(1))).cpu().numpy())
            p2.append((1 - torch.exp(m2.sum(1))).cpu().numpy())
    return np.concatenate(p1), np.concatenate(p2), dict(best_val=best, epochs=ep_run,
                                                        n_params=sum(p.numel() for p in net.parameters()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+', choices=list(MODELS))
    a = ap.parse_args(argv)
    d = ev.load()
    pf = ev.folds(d['person'], ev.FOLD_SALTS['A'])
    torch.backends.cudnn.benchmark = False
    for name in a.models:
        out = LOC / f'w2s_neural_oof_{name}.npz'
        if out.exists():
            raise SystemExit(f'{out} exists (write-once)')
        cfg = MODELS[name]
        P1, P2 = np.full(len(d['y']), np.nan), np.full(len(d['y']), np.nan)
        S1, S2 = {s: np.full(len(d['y']), np.nan) for s in SEEDS}, {s: np.full(len(d['y']), np.nan) for s in SEEDS}
        log, t0 = [], time.time()
        for f in range(ev.N_FOLDS):
            te = np.where(pf == f)[0]
            va = np.where(pf == (f + 1) % ev.N_FOLDS)[0]
            tr = np.where((pf != f) & (pf != (f + 1) % ev.N_FOLDS))[0]
            for s in SEEDS:
                p1, p2, info = run_fold(d, cfg, tr, va, te, s)
                S1[s][te], S2[s][te] = p1, p2
                log.append(dict(model=name, fold=f, seed=s, **info))
                print(f'[W2S] {name} fold {f} seed {s}: epochs {info["epochs"]} val {info["best_val"]:.4f}', flush=True)
            P1[te] = np.mean([S1[s][te] for s in SEEDS], 0)
            P2[te] = np.mean([S2[s][te] for s in SEEDS], 0)
        np.savez(out, p_n1=P1, p_n2=P2, **{f'p_n1_seed{s}': S1[s] for s in SEEDS}, **{f'p_n2_seed{s}': S2[s] for s in SEEDS})
        (LOC / f'w2s_neural_log_{name}.json').write_text(json.dumps(dict(model=name, cfg=cfg, log=log,
                                                                          runtime_s=round(time.time() - t0, 1)), indent=1))


if __name__ == '__main__':
    main()
