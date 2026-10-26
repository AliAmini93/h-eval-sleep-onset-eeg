#!/usr/bin/env python3
"""CTH v2: M2_46_FINAL — all-46-person final fit of the frozen M2 architecture (no new search).

Steps (each writes once, never overwrites; CTH v1 code and artefacts are imported/read unchanged):
  schedule   replay the existing CTH v1 inner-fold M2 training at the fixed config (lr 1e-3, width 32, seeds 0/1/2,
             5 outer folds, same inner splits, same early stopping) to recover the ACTUAL best-validation epochs;
             freeze schedule = round-half-up median of the 15 best epochs. TRAIN-only.
  fit        train seeds 0/1/2 on ALL 46 TRAIN persons for exactly the frozen number of epochs (no validation set,
             no early stopping), normalisation fitted on all 46 persons; save weights, preprocessing, CPU reference
             predictions and a manifest.
  verify     reload on CPU, check hashes, channel order, preprocessing and byte-identical reference predictions.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_s = importlib.util.spec_from_file_location('cth_v1_archived', REPO / 'scripts/decision/h_eval/cth_v1/cth_experiment.py')
cx = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cx)

OUT = REPO / 'results/decision/h_eval/cth_v2'
FINAL_DIR = OUT / 'frozen_m2_46_final'
CFG = dict(lr=1e-3, width=32)
ARCH = cx.ARCHS['M2_raw_cnn_binary']
SEEDS = (0, 1, 2)
V1_EXPECTED = {'cth_experiment.py': '42e689ef24fe948da3039535b3fbce3d50973af9ef3a6cdd58bb2a244b4c949a',
               'cth_cv_results_v1.json': 'a8e9b368f5f1677806759e22614dcb7c9bb0769736aecf826a2d7d8f743f9a96',
               'dataset_v1.npz': '6e05b27848c7da43009ed9835e1f3f165d1d8cde7ee7e26d55893d770c573ae1',
               'frozen_m2_v1_manifest': '7522fe9c50bc25a9f118e466323bd65653e1f0875b59079c094527efd6c3e42b'}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_v1():
    got = {'cth_experiment.py': digest(REPO / 'scripts/decision/h_eval/cth_v1/cth_experiment.py'),
           'cth_cv_results_v1.json': digest(REPO / 'results/decision/h_eval/cth_v1/cth_cv_results_v1.json'),
           'dataset_v1.npz': digest(cx.DATA),
           'frozen_m2_v1_manifest': digest(REPO / 'results/decision/h_eval/cth_v1/frozen_m2_v1/frozen_manifest.json')}
    if got != V1_EXPECTED:
        raise RuntimeError('CTH v1 evidence changed: ' + json.dumps(got))
    return got


def write_once(path, data: bytes):
    path = Path(path)
    if path.exists():
        raise RuntimeError('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def init_bias(model, T, rows):
    prior = float(T.y[torch.tensor(rows, device=cx.DEV)].mean().clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        model.head.bias.fill_(float(np.log(prior / (1 - prior))))


def replay_best_epoch(T, tr, va, seed):
    """Same optimisation as cx.train_net (D1 bias init, AdamW, clip 1.0, BATCH, PATIENCE), recording the epoch of the
    best inner-validation score (1-based) and the score."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = cx.CTHNet(T.n_feat, CFG['width'], cin=T.cin, **ARCH).to(cx.DEV)
    init_bias(model, T, tr)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    yv = T.y[torch.tensor(va, device=cx.DEV)].cpu().numpy()
    best, best_ep, bad, ep_run = np.inf, 0, 0, 0
    for ep in range(cx.MAX_EPOCHS):
        _epoch(model, opt, T, tr, rng)
        ep_run = ep + 1
        score = float(((cx.predict_net(model, T, va, 'binary') - yv) ** 2).mean())
        if score < best - 1e-7:
            best, best_ep, bad = score, ep + 1, 0
        else:
            bad += 1
            if bad >= cx.PATIENCE:
                break
    return dict(best_epoch=best_ep, epochs_run=ep_run, best_inner_val_brier_ww=best)


def _epoch(model, opt, T, rows, rng):
    model.train()
    perm = rng.permutation(rows)
    for s in range(0, len(perm), cx.BATCH):
        b = torch.tensor(perm[s:s + cx.BATCH], device=cx.DEV)
        z = model(T.raw[b], T.seq[b], T.tf[b])
        loss = F.binary_cross_entropy_with_logits(z[:, 0], T.y[b])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def schedule():
    v1 = check_v1()
    d = cx.load()
    folds = cx.person_folds(d['person'])
    pf = np.array([folds[p] for p in d['person']])
    logged = json.loads((REPO / 'results/decision/h_eval/cth_v1/cth_cv_results_v1.json').read_text())['selected_configs']
    runs = []
    for f in range(cx.N_FOLDS):
        tr = np.where(pf != f)[0]
        inner_va = np.where(pf == (f + 1) % cx.N_FOLDS)[0]
        inner_tr = np.setdiff1d(tr, inner_va)
        T = cx.Tensors(d, inner_tr)
        for s in SEEDS:
            r = replay_best_epoch(T, inner_tr, inner_va, s)
            ref = next((x['inner_val_brier_ww'] for x in logged if x['fold'] == f and x['model'] == 'M2_raw_cnn_binary'
                        and x['cfg'] == CFG), None) if s == 0 else None
            runs.append(dict(fold=f, seed=s, n_inner_train_persons=len(set(d['person'][inner_tr])), **r,
                             logged_cv_seed0_inner_val_brier=ref,
                             replay_abs_diff_vs_logged=None if ref is None else abs(ref - r['best_inner_val_brier_ww'])))
    best = sorted(r['best_epoch'] for r in runs)
    med = float(np.median(best))
    n_epochs = max(1, int(np.floor(med + 0.5)))
    sch = dict(version='H-EVAL-CTH-v2-M2-epoch-schedule', rule='round-half-up median of best inner-validation epochs, '
               'fixed config lr=1e-3 width=32, 5 CTH v1 outer folds x seeds 0/1/2, CTH v1 inner splits and early stopping',
               config=CFG, best_epochs=best, median=med, frozen_n_epochs=n_epochs, runs=runs, cth_v1_hashes=v1,
               data_used='SC_TRAIN only (46 persons); no SC_VALIDATION / protected / ST / Muse',
               note='GPU training may be non-bitwise-deterministic; replay fidelity is reported per fold for seed 0.')
    write_once(OUT / 'm2_epoch_schedule_v2.json', (json.dumps(sch, indent=2) + '\n').encode())
    print(json.dumps(dict(best_epochs=best, median=med, frozen_n_epochs=n_epochs,
                          replay_max_abs_diff=max(r['replay_abs_diff_vs_logged'] or 0 for r in runs)), indent=2))


def fit():
    check_v1()
    sch_path = OUT / 'm2_epoch_schedule_v2.json'
    sch = json.loads(sch_path.read_text())
    n_epochs = sch['frozen_n_epochs']
    d = cx.load()
    all_rows = np.arange(len(d['y']))
    T = cx.Tensors(d, all_rows)
    weights, infos, preds = {}, [], []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = cx.CTHNet(T.n_feat, CFG['width'], cin=T.cin, **ARCH).to(cx.DEV)
        init_bias(model, T, all_rows)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-3)
        rng = np.random.default_rng(s)
        t0 = time.time()
        for _ in range(n_epochs):
            _epoch(model, opt, T, all_rows, rng)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        path = FINAL_DIR / f'M2_46_FINAL_seed{s}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError('refusing to overwrite ' + str(path))
        torch.save(sd, path)
        weights[path.name] = digest(path)
        infos.append(dict(seed=s, epochs=n_epochs, train_s=round(time.time() - t0, 2)))
    prep = dict(model='M2_46_FINAL', arch=ARCH, cfg=CFG, channels=['EEG Fpz-Cz', 'EEG Pz-Oz'], channel_order='[Fpz-Cz, Pz-Oz]',
                fs_hz=100, window_s=30, window='[t-30 s, t) ending at the decision time',
                raw_preprocessing='per-window mean removal; divide by raw_scale_uv; clip [-10, 10]; float32',
                raw_scale_uv=T.scale, ensemble='mean of p300 over seeds [0, 1, 2]',
                output='P(first R&K Stage 2 within 300 s)', epochs=n_epochs,
                gradient_training_persons=sorted(set(d['person'].tolist())),
                n_gradient_training_persons=len(set(d['person'].tolist())), recordings=len(set(d['recording'].tolist())),
                windows=int(len(d['y'])), validation_or_early_stopping='none (fixed schedule from CTH v1 CV replay)',
                train_infos=infos)
    write_once(FINAL_DIR / 'preprocessing_and_config.json', (json.dumps(prep, indent=2, default=float) + '\n').encode())
    ref, ens = cpu_predictions(d, prep)
    write_once(FINAL_DIR / 'train_reference_predictions_cpu.csv', ref)
    man = dict(version='H-EVAL-CTH-v2-M2_46_FINAL', identity='M2_46_FINAL', weights_sha256=weights,
               preprocessing_sha256=digest(FINAL_DIR / 'preprocessing_and_config.json'),
               reference_predictions_sha256=digest(FINAL_DIR / 'train_reference_predictions_cpu.csv'),
               epoch_schedule_sha256=digest(sch_path), code_sha256=digest(Path(__file__)),
               cth_v1_code_sha256=V1_EXPECTED['cth_experiment.py'], dataset_sha256=V1_EXPECTED['dataset_v1.npz'],
               preserved_m2_v1_manifest_sha256=V1_EXPECTED['frozen_m2_v1_manifest'],
               environment=dict(torch=torch.__version__, numpy=np.__version__),
               note='In-sample predictions on TRAIN are a reproducibility reference only, not a performance estimate.')
    write_once(FINAL_DIR / 'frozen_manifest.json', (json.dumps(man, indent=2) + '\n').encode())
    print(json.dumps(dict(epochs=n_epochs, weights=weights, manifest_sha256=digest(FINAL_DIR / 'frozen_manifest.json')), indent=2))


def load_final(final_dir=FINAL_DIR):
    prep = json.loads((Path(final_dir) / 'preprocessing_and_config.json').read_text())
    models = []
    for s in SEEDS:
        m = cx.CTHNet(18, prep['cfg']['width'], cin=2, **prep['arch'])
        m.load_state_dict(torch.load(Path(final_dir) / f'M2_46_FINAL_seed{s}.pt', map_location='cpu', weights_only=True))
        m.eval()
        models.append(m)
    return prep, models


def predict_raw(raw_uv, prep, models):
    """raw_uv: (N, 2, 3000) microvolts in [Fpz-Cz, Pz-Oz] order. Returns ensemble p300 (float64), CPU."""
    raw_uv = np.asarray(raw_uv, dtype=np.float32)
    if raw_uv.ndim != 3 or raw_uv.shape[1:] != (2, 3000):
        raise ValueError('expected (N, 2, 3000) [Fpz-Cz, Pz-Oz] at 100 Hz')
    x = np.clip((raw_uv - raw_uv.mean(-1, keepdims=True)) / np.float32(prep['raw_scale_uv']), -10, 10)
    x = torch.tensor(x, dtype=torch.float32)
    seq = torch.zeros((len(x), cx.H, 37))
    tf = torch.zeros((len(x), 2))
    out = []
    with torch.no_grad():
        for m in models:
            z = torch.cat([m(x[i:i + 512], seq[i:i + 512], tf[i:i + 512]) for i in range(0, len(x), 512)])
            out.append(torch.sigmoid(z[:, 0]).numpy().astype(np.float64))
    return np.mean(out, 0), out


def cpu_predictions(d, prep):
    _, models = load_final() if (FINAL_DIR / 'preprocessing_and_config.json').exists() else (None, None)
    ens, per = predict_raw(d['raw'], prep, models)
    buf = io.StringIO(newline='')
    w = csv.writer(buf)
    w.writerow(['recording', 'elapsed_s', 'p_ensemble'] + [f'p_seed{s}' for s in SEEDS])
    for i in range(len(ens)):
        w.writerow([d['recording'][i], float(d['elapsed'][i]), float(ens[i])] + [float(p[i]) for p in per])
    return buf.getvalue().encode(), ens


def verify():
    check_v1()
    man = json.loads((FINAL_DIR / 'frozen_manifest.json').read_text())
    bad = [n for n, h in man['weights_sha256'].items() if digest(FINAL_DIR / n) != h]
    for n, key in (('preprocessing_and_config.json', 'preprocessing_sha256'),
                   ('train_reference_predictions_cpu.csv', 'reference_predictions_sha256')):
        if digest(FINAL_DIR / n) != man[key]:
            bad.append(n)
    d = cx.load()
    prep, _ = load_final()
    blob, _ = cpu_predictions(d, prep)
    blob2, _ = cpu_predictions(d, prep)
    return dict(hash_failures=bad, reference_byte_identical=blob == (FINAL_DIR / 'train_reference_predictions_cpu.csv').read_bytes(),
                repeat_identical=blob == blob2, n_gradient_training_persons=prep['n_gradient_training_persons'],
                channel_order=prep['channel_order'], epochs=prep['epochs'],
                manifest_sha256=digest(FINAL_DIR / 'frozen_manifest.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('schedule', 'fit', 'verify'))
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    {'schedule': schedule, 'fit': fit, 'verify': lambda: print(json.dumps(verify(), indent=2))}[a.step]()
