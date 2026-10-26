#!/usr/bin/env python3
"""H-EVAL W2S P1: personalised E1 hazard ablations A-E (design w2s_p0p1_design_v1.md, P1). TRAIN-only.

All variants: raw 5-min causal TCN encoder (w2s_neural.RawTCN), explicit elapsed-time features, hazard heads N1 (E1 windows)
and N2 (all windows), whole-recording batches (each recording processed in elapsed order; no state shared across nights).
  A  head([e_t, time])
  B  + causal baseline reference: [e_t - mean(e_1..e_t)]          (own EEG since LightsOff)
  C  + GRU over own embedding sequence e_1..e_t                    (own EEG trajectory since LightsOff)
  D  A + calibration-aware loss: 2 * mean_r (mean p1_r - mean y1_r)^2 over E1 windows of recordings in the batch
  E  A + B + C + D
Donor control (B, C, E): at inference, the history (all e_j, j < t) is replaced by the embeddings of an elapsed-time-matched
donor recording of a different person in the same outer-test fold; e_t (own current window) is kept.

Commands: outer --partition A|B ; inner --partition A|B --variants A E (fixed epochs, for nested calibration)
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
import w2s_neural as wn  # noqa: E402

LOC = ev.REPO / '.local/h_eval/discovery_w2s/p1'
DEV = wn.DEV
VARIANTS = {'A': dict(ref=False, gru=False, cal=False), 'B': dict(ref=True, gru=False, cal=False),
            'C': dict(ref=False, gru=True, cal=False), 'D': dict(ref=False, gru=False, cal=True),
            'E': dict(ref=True, gru=True, cal=True)}
LAMBDA_CAL = 2.0


class PNet(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
        self.enc = wn.RawTCN()
        dim = 64 + 2 + (64 if v['ref'] else 0) + (64 if v['gru'] else 0)
        if v['gru']:
            self.gru = nn.GRU(64, 64, batch_first=True)
        self.body = nn.Sequential(nn.Linear(dim, 64), nn.GELU())
        self.h1, self.h2 = nn.Linear(64, wn.H), nn.Linear(64, wn.H)

    def heads(self, e, hist_e, tf):
        """e: (n, 64) current embeddings of one recording in elapsed order; hist_e: (n, 64) sequence used as history
        (own e, or donor); tf: (n, 2). Causal: position t uses hist_e[:t] and e[t]."""
        parts = [e, tf]
        if self.v['ref']:
            # running mean of history up to and including t, with the current window always own e[t]
            prev = torch.cumsum(hist_e, 0) - hist_e                       # sum of hist_e[:t]
            cnt = torch.arange(len(e), device=e.device, dtype=e.dtype)[:, None]
            parts.append(e - (prev + e) / (cnt + 1))
        if self.v['gru']:
            h_seq, _ = self.gru(hist_e[None])                              # h_seq[0, j] = state after hist_e[0..j]
            before = torch.cat([torch.zeros(1, 64, device=e.device), h_seq[0, :-1]], 0)   # state after hist_e[:t]
            parts.append(self.gru(e[:, None, :], before[None].contiguous())[0][:, 0])     # one step with own e[t]
        z = self.body(torch.cat(parts, 1))
        return self.h1(z), self.h2(z)


class RecData:
    def __init__(self, d, tr_idx):
        cfg = wn.MODELS['N2_tcn5min']
        self.W = wn.Data(d, tr_idx, cfg)
        T = ev.time_feats(d['elapsed'])
        mu, sd = T[tr_idx].mean(0), T[tr_idx].std(0) + 1e-9
        self.tf = torch.tensor((T - mu) / sd, dtype=torch.float32, device=DEV)
        self.y1 = torch.tensor(d['y1'], dtype=torch.float32, device=DEV)


def rec_windows(d):
    out = {}
    for r in np.unique(d['recording']):
        idx = np.where(d['recording'] == r)[0]
        out[r] = idx[np.argsort(d['elapsed'][idx])]
    return out


def p_from(logits):
    return 1 - torch.exp(F.logsigmoid(-logits).sum(1))


def forward_rec(net, D, idx, hist_idx=None):
    b = torch.tensor(idx, device=DEV)
    e = net.enc(D.W.batch(b))
    he = e if hist_idx is None else net.enc(D.W.batch(torch.tensor(hist_idx, device=DEV)))
    return net.heads(e, he, D.tf[b])


def loss_from(l1, l2, D, idx, v, train):
    b = torch.tensor(idx, device=DEV)
    loss = wn.hazard_nll(F.logsigmoid(l2), F.logsigmoid(-l2), D.W.K2[b], torch.ones(len(b), device=DEV)) * len(b)
    w1 = D.W.e1[b]
    loss = loss + wn.hazard_nll(F.logsigmoid(l1), F.logsigmoid(-l1), D.W.K1[b], w1) * w1.sum()
    cal = None
    if v['cal'] and train and w1.sum() > 0:
        p1 = p_from(l1)
        cal = (((p1 * w1).sum() / w1.sum()) - ((D.y1[b] * w1).sum() / w1.sum())) ** 2
    return loss, len(b), cal


def group_losses(net, D, group, v, train):
    """One encoder pass for all windows of the group's recordings (speed only; identical maths), then per-recording
    causal heads (baseline / GRU never cross recordings)."""
    all_idx = np.concatenate(group)
    E = net.enc(D.W.batch(torch.tensor(all_idx, device=DEV)))
    out, start = [], 0
    for idx in group:
        e = E[start:start + len(idx)]
        start += len(idx)
        l1, l2 = net.heads(e, e, D.tf[torch.tensor(idx, device=DEV)])
        out.append(loss_from(l1, l2, D, idx, v, train))
    return out


def batches(recs, rng, per_step=4):
    """Design P1: whole-recording batches, 4 recordings per optimisation step."""
    order = list(rng.permutation(len(recs)))
    for i in range(0, len(order), per_step):
        yield [recs[j] for j in order[i:i + per_step]]


def train(d, v, tr_recs, va_recs, seed, fixed_epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_idx = np.concatenate(tr_recs)
    D = RecData(d, tr_idx)
    net = PNet(v).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=wn.LR, weight_decay=wn.WD)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep_run, best_ep = np.inf, None, 0, 0, 0
    for ep in range(fixed_epochs or wn.MAX_EPOCHS):
        net.train()
        for group in batches(tr_recs, rng):
            opt.zero_grad()
            res = group_losses(net, D, group, v, True)
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
            vl = sum(float(l) for i in range(0, len(va_recs), 4) for l, _, _ in group_losses(net, D, va_recs[i:i + 4], v, False)) \
                / sum(len(i) for i in va_recs)
        if vl < best - 1e-5:
            best, bad, best_ep, best_state = vl, 0, ep_run, {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= wn.PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, D, dict(epochs_run=ep_run, best_epoch=(best_ep if not fixed_epochs else fixed_epochs), best_val=float(best),
                        n_params=sum(p.numel() for p in net.parameters()))


def donor_for(r, recs_fold, d):
    """Different-person recording in the same fold with the smallest length >= own length (else the longest)."""
    me = d['person'][recs_fold[r][0]]
    cand = [(len(ix), q) for q, ix in recs_fold.items() if d['person'][ix[0]] != me]
    ok = sorted(c for c in cand if c[0] >= len(recs_fold[r]))
    return (ok[0] if ok else max(cand))[1]


def predict(net, D, recs_fold, d, donor):
    p1, p2 = {}, {}
    with torch.no_grad():
        for r, idx in recs_fold.items():
            hist = None
            if donor:
                q = recs_fold[donor_for(r, recs_fold, d)]
                hist = np.concatenate([q, np.repeat(q[-1:], max(0, len(idx) - len(q)))])[:len(idx)]
            l1, l2 = forward_rec(net, D, idx, hist)
            p1[r], p2[r] = p_from(l1).cpu().numpy(), p_from(l2).cpu().numpy()
    return p1, p2


def cmd_outer(part, variants):
    d = ev.load()
    pf = ev.folds(d['person'], ev.FOLD_SALTS[part])
    RW = rec_windows(d)
    rf = {r: pf[ix[0]] for r, ix in RW.items()}
    LOC.mkdir(parents=True, exist_ok=True)
    for vn in variants:
        out = LOC / f'p1_outer_{part}_{vn}.npz'
        if out.exists():
            raise SystemExit(f'{out} exists (write-once)')
        v = VARIANTS[vn]
        P = {k: {s: np.full(len(d['y']), np.nan) for s in wn.SEEDS} for k in ('p1', 'p2', 'p1_donor')}
        ledger = []
        for f in range(ev.N_FOLDS):
            te = {r: ix for r, ix in RW.items() if rf[r] == f}
            va = [ix for r, ix in RW.items() if rf[r] == (f + 1) % ev.N_FOLDS]
            tr = [ix for r, ix in RW.items() if rf[r] not in (f, (f + 1) % ev.N_FOLDS)]
            for s in wn.SEEDS:
                t0 = time.time()
                try:
                    net, D, info = train(d, v, tr, va, s)
                    p1, p2 = predict(net, D, te, d, False)
                    for r in te:
                        P['p1'][s][te[r]], P['p2'][s][te[r]] = p1[r], p2[r]
                    if v['ref'] or v['gru']:
                        q1, _ = predict(net, D, te, d, True)
                        for r in te:
                            P['p1_donor'][s][te[r]] = q1[r]
                    ledger.append(dict(partition=part, variant=vn, fold=f, seed=s, status='ok', seconds=round(time.time() - t0, 1), **info))
                except Exception as exc:                   # recorded, never silently dropped
                    ledger.append(dict(partition=part, variant=vn, fold=f, seed=s, status='FAILED', error=repr(exc)[:300]))
                    raise
                print(f'[P1] {part} {vn} f{f} s{s} best_ep {info["best_epoch"]} val {info["best_val"]:.4f}', flush=True)
        arrs = {f'{k}_seed{s}': P[k][s] for k in P for s in wn.SEEDS}
        arrs.update({k: np.mean([P[k][s] for s in wn.SEEDS], 0) for k in P})
        np.savez(out, **arrs)
        (LOC / f'p1_outer_ledger_{part}_{vn}.json').write_text(json.dumps(ledger, indent=1))


def cmd_inner(part, variants):
    d = ev.load()
    pf = ev.folds(d['person'], ev.FOLD_SALTS[part])
    RW = rec_windows(d)
    rf = {r: pf[ix[0]] for r, ix in RW.items()}
    for vn in variants:
        out = LOC / f'p1_inner_{part}_{vn}.npz'
        if out.exists():
            raise SystemExit(f'{out} exists (write-once)')
        led = json.loads((LOC / f'p1_outer_ledger_{part}_{vn}.json').read_text())
        best_ep = {(r['fold'], r['seed']): r['best_epoch'] for r in led}
        inner = {f: np.full(len(d['y']), np.nan) for f in range(ev.N_FOLDS)}
        ledger = []
        for f in range(ev.N_FOLDS):
            for h in [g for g in range(ev.N_FOLDS) if g != f]:
                te = {r: ix for r, ix in RW.items() if rf[r] == h}
                tr = [ix for r, ix in RW.items() if rf[r] not in (f, h)]
                ps = []
                for s in wn.SEEDS:
                    t0 = time.time()
                    net, D, info = train(d, VARIANTS[vn], tr, [], s, fixed_epochs=max(1, best_ep[(f, s)]))
                    p1, _ = predict(net, D, te, d, False)
                    arr = np.full(len(d['y']), np.nan)
                    for r in te:
                        arr[te[r]] = p1[r]
                    ps.append(arr)
                    ledger.append(dict(partition=part, variant=vn, outer_fold=f, inner_fold=h, seed=s, status='ok',
                                       epochs=info['best_epoch'], seconds=round(time.time() - t0, 1)))
                    print(f'[P1-inner] {part} {vn} f{f} h{h} s{s}', flush=True)
                idx = np.concatenate(list(te.values()))
                inner[f][idx] = np.mean([p[idx] for p in ps], 0)
        np.savez(out, **{f'outer{f}': x for f, x in inner.items()})
        (LOC / f'p1_inner_ledger_{part}_{vn}.json').write_text(json.dumps(ledger, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('outer', 'inner'))
    ap.add_argument('--partition', required=True, choices=('A', 'B'))
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a = ap.parse_args()
    (cmd_outer if a.cmd == 'outer' else cmd_inner)(a.partition, a.variants)


if __name__ == '__main__':
    main()
