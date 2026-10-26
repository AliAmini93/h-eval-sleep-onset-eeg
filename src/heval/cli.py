"""Command-line interface: python -m heval <command> --help.

Data-free:   info, demo, split, score, fit-references
Data-based:  cohort, windows (need a local Sleep-EDF Expanded v1.0.0 copy), fit-speclr, fit-rawcnn, predict,
             ablation-cv, representation, attribution-checks (need window files written by 'windows'; the last three
             are TRAIN-only development analyses).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

DATA_ENV = 'HEVAL_SLEEP_EDF_ROOT'
WINDOW_KEYS = ('raw', 'spectral', 'elapsed_s', 'time_to_n2_s', 'y', 'anchor', 'person', 'recording')


def _dump(obj, out):
    text = json.dumps(obj, indent=1, default=float) + '\n'
    if out:
        Path(out).write_text(text)
        print(f'written {out}')
    else:
        sys.stdout.write(text)


def _data_root(arg, metadata=True):
    root = arg or os.environ.get(DATA_ENV)
    if not root:
        raise SystemExit(f'error: give --data-root or set {DATA_ENV} to the local sleep-edfx 1.0.0 directory')
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f'error: data root {root} is not a directory')
    for rel in ('metadata/SHA256SUMS.txt', 'metadata/RECORDS', 'metadata/SC-subjects.xls', 'metadata/ST-subjects.xls') if metadata else ():
        if not (root / rel).is_file():
            raise SystemExit(f'error: {root / rel} not found (expected the PhysioNet sleep-edfx 1.0.0 layout)')
    return root


def _read_csv(path, required):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'error: {path} not found')
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'error: {path} has no rows')
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise SystemExit(f'error: {path} lacks column(s) {missing}')
    return rows


def _load_windows(paths):
    parts = []
    for p in paths:
        if not Path(p).is_file():
            raise SystemExit(f'error: {p} not found')
        z = np.load(p, allow_pickle=False)
        missing = [k for k in WINDOW_KEYS if k not in z.files]
        if missing:
            raise SystemExit(f'error: {p} lacks arrays {missing} (write it with: python -m heval windows)')
        parts.append({k: z[k] for k in WINDOW_KEYS})
    return {k: np.concatenate([q[k] for q in parts]) for k in WINDOW_KEYS}


# ------------------------------------------------------------------------ data-free commands
def cmd_info(a):
    import torch
    from .models.raw_cnn import N_PARAMETERS, RawCNN, count_parameters
    from .models.spectral import FEATURE_NAMES
    n = count_parameters(RawCNN())
    _dump(dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
               rawcnn_parameters=n, rawcnn_parameters_expected=N_PARAMETERS, speclr_features=list(FEATURE_NAMES)), a.out)
    if n != N_PARAMETERS:
        raise SystemExit('error: RawCNN parameter count differs from the published architecture')


def cmd_demo(a):
    """Synthetic end-to-end run of the paper's pipeline on person-disjoint TRAIN/TEST synthetic nights."""
    from .evaluation.bootstrap import SEED_REFERENCE_AUDIT, boot_index, reference_contrast
    from .evaluation.metrics import calibration, within_recording_auroc
    from .evaluation.stage_analysis import contributions
    from .models import raw_cnn, references, spectral
    from .synthetic import synthetic_nights
    d = synthetic_nights(n_persons=a.persons, seed=a.seed)
    persons = sorted(set(d['person']))
    test_p = set(persons[::3])
    te = np.array([p in test_p for p in d['person']])
    tr = ~te
    F = np.array([spectral.window_features(w) for w in d['raw']])
    M = spectral.fit_speclr(F[tr], d['elapsed_s'][tr], d['y'][tr])
    P_spec = spectral.predict_speclr(M, F[te], d['elapsed_s'][te])
    scale, models = raw_cnn.fit_ensemble(d['raw'][tr], d['y'][tr], epochs=a.epochs)
    p_cnn, _ = raw_cnn.predict_ensemble(d['raw'][te], scale, models)
    p_cnn = np.clip(p_cnn, 1e-6, 1 - 1e-6)
    s1 = (d['stage'] == '1').astype(float)
    refs = references.fit_references(d['elapsed_s'][tr], s1[tr], d['y'][tr], d['person'][tr])
    P = {'CONSTANT': P_spec['CONSTANT'], 'SpecLR': P_spec['EEG_LOGIT'], 'RawCNN': p_cnn,
         'Stage-only': references.predict_A(refs['A'], s1[te]),
         'Stage+time': references.predict_B(refs['B'], d['elapsed_s'][te], s1[te])}
    y, person, stage = d['y'][te].astype(float), d['person'][te], d['stage'][te]
    I = boot_index(len(set(person)), seed=SEED_REFERENCE_AUDIT, n=a.boot)
    out = dict(note='SYNTHETIC DATA ONLY - not a result of the paper', n_train_windows=int(tr.sum()),
               n_test_windows=int(te.sum()), n_test_persons=len(set(person)), rawcnn_scale=scale,
               models={k: dict(brier=float(((p - y) ** 2).mean()), **calibration(p, y),
                               within_recording_auroc=within_recording_auroc(p, y, d['recording'][te])['mean'])
                       for k, p in P.items()},
               contrasts={f'{r}->{c}': reference_contrast(P[r], P[c], y, person, I if len(set(person)) >= 8 else None)
                          for r, c in (('CONSTANT', 'SpecLR'), ('CONSTANT', 'RawCNN'), ('SpecLR', 'RawCNN'),
                                       ('CONSTANT', 'Stage-only'), ('Stage+time', 'RawCNN'))},
               stage_contribution={f'CONSTANT->{c}': contributions((P['CONSTANT'] - y) ** 2, (P[c] - y) ** 2, stage)
                                   for c in ('SpecLR', 'RawCNN')})
    _dump(out, a.out)


def cmd_split(a):
    from .data.cohort import person_split
    path = Path(a.subjects_file)
    if not path.is_file():
        raise SystemExit(f'error: {path} not found')
    codes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    bad = [c for c in codes if not c.startswith(('SC', 'ST'))]
    if bad:
        raise SystemExit(f'error: subject codes must start with SC (cassette) or ST (telemetry): {bad[:3]}')
    rows = [dict(subject_key=c, study='sleep-cassette' if c.startswith('SC') else 'sleep-telemetry') for c in codes]
    try:
        split = person_split(rows)
    except ValueError as exc:
        raise SystemExit(f'error: {exc}')
    _dump(dict(counts={k: sum(v == k for v in split.values()) for k in sorted(set(split.values()))}, by_subject=split), a.out)


def cmd_score(a):
    from .evaluation.bootstrap import boot_index, reference_contrast
    from .evaluation.stage_analysis import contributions
    rows = _read_csv(a.predictions, ['person', 'y', a.ref, a.cand] + ([a.stage_col] if a.stage_col else []))
    try:
        y = np.array([float(r['y']) for r in rows])
        p_ref = np.array([float(r[a.ref]) for r in rows])
        p_cand = np.array([float(r[a.cand]) for r in rows])
    except ValueError as exc:
        raise SystemExit(f'error: non-numeric value in {a.predictions}: {exc}')
    person = np.array([r['person'] for r in rows])
    n_p = len(set(person))
    I = boot_index(n_p, seed=a.seed, n=a.boot) if n_p >= 8 else None
    try:
        out = dict(contrast=reference_contrast(p_ref, p_cand, y, person, I), bootstrap=dict(n=a.boot, seed=a.seed))
        if a.stage_col:
            stage = np.array([r[a.stage_col] for r in rows])
            out['stage_contribution'] = contributions((p_ref - y) ** 2, (p_cand - y) ** 2, stage)
            out['by_stage'] = {s: reference_contrast(p_ref[stage == s], p_cand[stage == s], y[stage == s], person[stage == s],
                                                     boot_index(len(set(person[stage == s])), seed=a.seed, n=a.boot)
                                                     if len(set(person[stage == s])) >= 8 else None)
                               for s in sorted(set(stage))}
    except ValueError as exc:
        raise SystemExit(f'error: {exc}')
    _dump(out, a.out)


def cmd_fit_references(a):
    from .models.references import fit_references
    rows = _read_csv(a.train_csv, ['person', 'elapsed_s', 'stage', 'y'])
    stage = np.array([r['stage'] for r in rows])
    if not set(stage) <= {'W', '1'}:
        raise SystemExit('error: stage must be W or 1 for every TRAIN window')
    try:
        refs = fit_references(np.array([float(r['elapsed_s']) for r in rows]), (stage == '1').astype(float),
                              np.array([float(r['y']) for r in rows]), np.array([r['person'] for r in rows]))
    except ValueError as exc:
        raise SystemExit(f'error: {exc}')
    refs['note'] = 'Oracle diagnostics using the retrospectively scored stage; not deployable forecasters.'
    _dump(refs, a.out)


# ------------------------------------------------------------------------ data-based commands
def cmd_cohort(a):
    from .data.cohort import ELIGIBLE, build_cohort, person_split
    root = _data_root(a.data_root)
    rows = build_cohort(root, recordings=a.recordings)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (out / 'cohort.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = dict(recordings=len(rows), status_counts={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})})
    if a.recordings is None:
        split = person_split([r for r in rows if r['status'] == ELIGIBLE])
        (out / 'split.json').write_text(json.dumps(split, indent=1, sort_keys=True) + '\n')
        summary['split_counts'] = {k: sum(v == k for v in split.values()) for k in sorted(set(split.values()))}
    else:
        summary['split'] = 'not computed for a subset of recordings'
    _dump(summary, None)


def cmd_windows(a):
    from .data.windows import extract_recording
    root = _data_root(a.data_root, metadata=False)
    rows = {r['recording']: r for r in _read_csv(a.cohort_csv, ['recording', 'subject_key', 'status', 'psg_path',
                                                                 'lights_off_rel_s', 'first_n2_after_off_s'])}
    r = rows.get(a.recording)
    if r is None:
        raise SystemExit(f'error: recording {a.recording} not in {a.cohort_csv}')
    if r['status'] != 'ELIGIBLE_ORIGIN_INDEPENDENT':
        raise SystemExit(f'error: recording {a.recording} is not eligible ({r["status"]})')
    w = extract_recording(root / r['psg_path'], float(r['lights_off_rel_s']), float(r['first_n2_after_off_s']))
    n = len(w['y'])
    np.savez(a.out, **w, person=np.array([r['subject_key']] * n), recording=np.array([a.recording] * n))
    print(f'written {a.out}: {n} windows')


def cmd_fit_speclr(a):
    from .models.spectral import fit_speclr
    d = _load_windows(a.windows)
    _dump(fit_speclr(d['spectral'], d['elapsed_s'], d['y']), a.out)


def cmd_fit_rawcnn(a):
    import torch
    from .models.raw_cnn import fit_ensemble
    d = _load_windows(a.windows)
    scale, models = fit_ensemble(d['raw'], d['y'], seeds=a.seeds, epochs=a.epochs, device=a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s, m in zip(a.seeds, models):
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, out / f'rawcnn_seed{s}.pt')
    (out / 'rawcnn_config.json').write_text(json.dumps(dict(raw_scale_uv=scale, seeds=a.seeds, epochs=a.epochs), indent=1) + '\n')
    print(f'written {len(models)} seed models to {out}')


def cmd_predict(a):
    from .models import raw_cnn, spectral
    d = _load_windows(a.windows)
    cols = dict(person=d['person'], recording=d['recording'], elapsed_s=d['elapsed_s'], y=d['y'])
    if a.speclr:
        P = spectral.predict_speclr(json.loads(Path(a.speclr).read_text()), d['spectral'], d['elapsed_s'])
        cols.update(p_constant=P['CONSTANT'], p_speclr=P['EEG_LOGIT'])
    if a.rawcnn_dir:
        cfg = json.loads((Path(a.rawcnn_dir) / 'rawcnn_config.json').read_text())
        models = raw_cnn.load_state_dicts([Path(a.rawcnn_dir) / f'rawcnn_seed{s}.pt' for s in cfg['seeds']])
        cols['p_rawcnn'] = np.clip(raw_cnn.predict_ensemble(d['raw'], cfg['raw_scale_uv'], models)[0], 1e-6, 1 - 1e-6)
    if not (a.speclr or a.rawcnn_dir):
        raise SystemExit('error: give --speclr and/or --rawcnn-dir')
    with Path(a.out).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        w.writerows(zip(*[np.asarray(v).tolist() for v in cols.values()]))
    print(f'written {a.out}: {len(d["y"])} rows')


def _development(paths):
    from .analysis.architecture_ablation import development_table
    d = _load_windows(paths)
    rich = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        if 'rich' not in z.files:
            raise SystemExit(f'error: {p} lacks the rich features (rewrite it with: python -m heval windows)')
        rich.append(z['rich'])
    try:
        return development_table(d['raw'], d['spectral'], np.concatenate(rich), d['elapsed_s'], d['time_to_n2_s'], d['y'],
                                 d['person'], d['recording'])
    except ValueError as exc:
        raise SystemExit(f'error: {exc}')


def _cv_args(a):
    from .analysis.architecture_ablation import GRID, N_FOLDS
    if not 1 <= a.grid <= len(GRID) or (a.folds and not set(a.folds) <= set(range(N_FOLDS))):
        raise SystemExit(f'error: --grid must be 1..{len(GRID)} and --folds within 0..{N_FOLDS - 1}')
    return dict(folds_to_run=a.folds, max_epochs=a.max_epochs, seeds=tuple(a.seeds), grid=GRID[:a.grid], device=a.device)


def cmd_ablation_cv(a):
    from .analysis.architecture_ablation import ablation_cv, summarise_ablation
    d = _development(a.windows)
    archs = a.archs or None
    if archs is not None and 'M3_cthnet' not in archs:
        raise SystemExit('error: --archs must include M3_cthnet (the comparator of every variant)')
    try:
        oof, _, _, _ = ablation_cv(d, archs=archs, **_cv_args(a))
    except ValueError as exc:
        raise SystemExit(f'error: {exc}')
    out = summarise_ablation(oof, d, n_boot=a.boot)
    out['note'] = 'TRAIN-only development cross-validation (paper Fig. 7); R > 0 means the full network is better'
    _dump(out, a.out)


def cmd_representation(a):
    from .analysis import perturbation as P
    from .analysis.architecture_ablation import person_folds
    from .analysis.representation import matched_cv, summarise_matched_cv
    d = _development(a.windows)
    try:
        oof, _, _, fold_store = matched_cv(d, **_cv_args(a))
    except ValueError as exc:
        raise SystemExit(f'error: {exc}')
    out = dict(note='TRAIN-only development analysis (paper Fig. 8)', matched_cv=summarise_matched_cv(oof, d, n_boot=a.boot))
    folds = person_folds(d['person'])
    pf_all = np.array([folds[p] for p in d['person']])
    m = np.isin(pf_all, list(fold_store))
    sub = dict(raw=d['raw'][m], y=d['y'][m], person=d['person'][m], rec_idx=d['rec_idx'][m])
    pf = pf_all[m]
    fold_models = {f: P.fold_predictors(st, a.device) for f, st in fold_store.items()}
    donor = P.fold_donors(sub['person'], d['elapsed'][m], pf)
    conds = P.conditions(sub['raw'].astype(np.float64), donor) + P.followup_conditions()
    if a.battery == 'reported':
        keep = set(P.FIGURE_CONDITIONS) | {'P5_delta_uniform_control', 'F2f_delta_phase_fpz_only'}
        conds = [c for c in conds if c[0] in keep]
    res, per_person, _ = P.run_battery(sub, fold_models, pf, oof['T0_constant'][m], conds, a.device)
    out['perturbation'] = {k: v for k, v in res.items() if not k.startswith('F2')}
    out['followup_F2'] = {k: v for k, v in res.items() if k.startswith('F2')}
    if a.battery == 'full':
        out['band_specificity'] = P.band_specificity(per_person, sub['person'])
    _dump(out, a.out)


def cmd_attribution_checks(a):
    from .analysis.representation import deletion_check, randomisation_check
    from .models import raw_cnn
    d = _load_windows(a.windows)
    cfg_path = Path(a.rawcnn_dir) / 'rawcnn_config.json'
    if not cfg_path.is_file():
        raise SystemExit(f'error: {cfg_path} not found (fit the models with: python -m heval fit-rawcnn)')
    cfg = json.loads(cfg_path.read_text())
    models = [m.to(a.device) for m in raw_cnn.load_state_dicts([Path(a.rawcnn_dir) / f'rawcnn_seed{s}.pt' for s in cfg['seeds']])]
    rnd, ig_map = randomisation_check(models, d['raw'], cfg['raw_scale_uv'], d['y'], d['person'], a.device)
    dele = deletion_check(models[0], d['raw'], cfg['raw_scale_uv'], d['person'], d['elapsed_s'], ig_map, device=a.device)
    _dump(dict(note='in-sample attribution sanity checks of a fitted RawCNN ensemble (development data only)',
               randomisation=rnd, deletion=dele), a.out)


def build_parser():
    ap = argparse.ArgumentParser(prog='python -m heval', description='H-EVAL methods (Sleep-EDF sleep-onset forecasting evaluation).')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('info', help='environment and RawCNN parameter check')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_info)
    p = sp.add_parser('demo', help='synthetic end-to-end run (no real data)')
    p.add_argument('--persons', type=int, default=24)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--out')
    p.set_defaults(fn=cmd_demo)
    p = sp.add_parser('split', help='salted person-disjoint split of eligible subject codes')
    p.add_argument('--subjects-file', required=True, help='one eligible subject code (SC4ss / ST7ss) per line')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_split)
    p = sp.add_parser('score', help='paired Brier skill with person-cluster bootstrap and stage attribution')
    p.add_argument('--predictions', required=True, help='CSV with columns person, y, <ref>, <cand> [, stage]')
    p.add_argument('--ref', required=True)
    p.add_argument('--cand', required=True)
    p.add_argument('--stage-col')
    p.add_argument('--boot', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--out')
    p.set_defaults(fn=cmd_score)
    p = sp.add_parser('fit-references', help='TRAIN fit of the Stage-only and Stage+time references')
    p.add_argument('--train-csv', required=True, help='CSV with columns person, elapsed_s, stage (W/1), y')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_fit_references)
    p = sp.add_parser('cohort', help='eligibility table and person split from a local Sleep-EDF copy')
    p.add_argument('--data-root', help=f'sleep-edfx 1.0.0 directory (or set {DATA_ENV})')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--recordings', nargs='+', help='process only these recording codes (no split)')
    p.set_defaults(fn=cmd_cohort)
    p = sp.add_parser('windows', help='decision windows and spectral features of one eligible recording')
    p.add_argument('--data-root', help=f'sleep-edfx 1.0.0 directory (or set {DATA_ENV})')
    p.add_argument('--cohort-csv', required=True)
    p.add_argument('--recording', required=True)
    p.add_argument('--out', required=True, help='output .npz')
    p.set_defaults(fn=cmd_windows)
    p = sp.add_parser('fit-speclr', help='fit SpecLR (and the frozen five-model family) on window files')
    p.add_argument('--windows', nargs='+', required=True)
    p.add_argument('--out')
    p.set_defaults(fn=cmd_fit_speclr)
    p = sp.add_parser('fit-rawcnn', help='fit the RawCNN seed ensemble on window files')
    p.add_argument('--windows', nargs='+', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.set_defaults(fn=cmd_fit_rawcnn)
    p = sp.add_parser('predict', help='apply fitted SpecLR / RawCNN to window files')
    p.add_argument('--windows', nargs='+', required=True)
    p.add_argument('--speclr')
    p.add_argument('--rawcnn-dir')
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_predict)
    for name, fn, hlp in (('ablation-cv', cmd_ablation_cv, 'architecture ablation CV of the predeclared network (Fig. 7)'),
                          ('representation', cmd_representation, 'matched RawCNN / STFT-CNN / SpecLR CV and perturbations (Fig. 8)')):
        p = sp.add_parser(name, help=hlp)
        p.add_argument('--windows', nargs='+', required=True, help='TRAIN window files written by the windows command')
        p.add_argument('--folds', type=int, nargs='+', help='outer folds to run (default: all 5)')
        p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
        p.add_argument('--max-epochs', type=int, default=40)
        p.add_argument('--grid', type=int, default=4, help='use the first N of the 4 learning-rate x width configurations')
        p.add_argument('--boot', type=int, default=10_000)
        p.add_argument('--device', default='cpu')
        p.add_argument('--out')
        if name == 'ablation-cv':
            p.add_argument('--archs', nargs='+', help='variants to run (must include M3_cthnet)')
        else:
            p.add_argument('--battery', choices=('reported', 'full'), default='full',
                           help='full: all perturbations and controls; reported: those in Fig. 8 and the text')
        p.set_defaults(fn=fn)
    p = sp.add_parser('attribution-checks', help='IG randomisation and deletion checks of a fitted RawCNN (in-sample)')
    p.add_argument('--windows', nargs='+', required=True)
    p.add_argument('--rawcnn-dir', required=True, help='directory written by fit-rawcnn')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_attribution_checks)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
