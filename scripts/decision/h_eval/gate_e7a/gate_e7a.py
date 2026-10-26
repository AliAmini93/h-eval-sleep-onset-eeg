#!/usr/bin/env python3
"""H-EVAL Gate E7A: frozen-prediction falsification audit of Gate E6 (POST-HOC, EXPLORATORY).

Audit-only, SC_TRAIN-only. Reads only frozen Gate E6 outputs: no EEG, no refit, no tuning, no GPU,
no SC_VALIDATION / SC_PROTECTED_MODEL_TEST / ST / Muse. Implements docs/decision/h_eval/gate_e7a/gate_e7a_protocol.md.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

VERSION = 'H-EVAL-E7A-frozen-prediction-audit-v1-2026-10-23'
PLAN_REL = 'docs/decision/h_eval/gate_e7a/gate_e7a_protocol.md'
PLAN_SHA256 = 'e7fdb53046c2a42332d9512ff9d7d3a2c7d3d0ab7acd28c8ad749a9d1739e0f4'
E6_PROTOCOL_SHA256 = 'f6de7f0bcb50bb234f47f81d7d47358f124bc3585301c56cc4084a130f7f51c0'
SEED = 20261023
N_BOOT = 10_000
N_PERM = 2_000
TOL = 1e-12
SHORT_N = 30
MODELS4 = ('TIME_CONT', 'CONSTANT', 'EEG_LOGIT', 'TIME_CONT_EEG_LOGIT')
CONTRASTS = {'C1': ('TIME_CONT', 'TIME_CONT_EEG_LOGIT'), 'C2': ('CONSTANT', 'EEG_LOGIT'),
             'C3': ('CONSTANT', 'TIME_CONT_EEG_LOGIT'), 'C4': ('TIME_CONT', 'EEG_LOGIT'),
             'C5': ('CONSTANT', 'TIME_CONT')}
CELLS = [(ts, es) for ts in ('FULL', 'ANCHOR') for es in ('FULL', 'ANCHOR')]
MAIN = ('FULL', 'FULL')


class AuditStop(RuntimeError):
    pass


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def import_py(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------ verification
def load_and_verify(repo: Path):
    e6dir, loc = repo / 'results/decision/h_eval/gate_e6', repo / '.local/h_eval/gate_e6'
    if digest(repo / PLAN_REL) != PLAN_SHA256:
        raise AuditStop('E7A plan differs from its frozen hash')
    if digest(repo / 'docs/decision/h_eval/gate_e6/gate_e6_protocol.md') != E6_PROTOCOL_SHA256:
        raise AuditStop('E6 protocol hash changed')
    prov = json.loads((e6dir / 'gate_e6_data_provenance.json').read_text())
    checks = {}
    for n, want in prov['outputs_sha256'].items():
        checks[f'results/{n}'] = digest(e6dir / n) == want
    for n, want in prov['local_outputs_sha256'].items():
        checks[f'.local/{n}'] = digest(loc / n) == want
    if not all(checks.values()):
        raise AuditStop('E6 output hash mismatch: ' + json.dumps(checks))
    e6 = import_py(repo / 'scripts/decision/h_eval/gate_e6/gate_e6.py', 'e6_for_e7a')
    if digest(repo / 'scripts/decision/h_eval/gate_e6/gate_e6.py') != prov['input_sha256']['e6_code']:
        raise AuditStop('E6 code changed since the E6 run')
    with (loc / 'gate_e6_window_predictions.csv').open(newline='') as f:
        win = list(csv.DictReader(f))
    with (e6dir / 'gate_e6_person_scores.csv').open(newline='') as f:
        ptab_saved = list(csv.DictReader(f))
    summary = json.loads((e6dir / 'gate_e6_summary.json').read_text())
    boot = json.loads((e6dir / 'gate_e6_cluster_bootstrap.json').read_text())
    return dict(e6=e6, prov=prov, win=win, ptab_saved=ptab_saved, summary=summary, boot=boot, hash_checks=checks,
                files={'e6_window_predictions_sha256': prov['local_outputs_sha256']['gate_e6_window_predictions.csv']})


def rows_and_pred(win, models):
    rows = [dict(recording=w['recording'], subject_key=w['person'], origin=w['origin'],
                 elapsed_s=float(w['elapsed_s']), time_to_n2_s=float(w['time_to_n2_s']),
                 anchor=w['anchor'] == '1', y=int(w['y'])) for w in win]
    pred = {ts: {m: np.array([float(w[f'p_{ts}_{m}']) for w in win]) for m in models} for ts in ('FULL', 'ANCHOR')}
    return rows, pred


def recompute_and_compare(X):
    e6 = X['e6']
    rows, pred = rows_and_pred(X['win'], e6.MODELS)
    ptab = e6.person_table(rows, pred)
    saved = {(r['person'], r['train_support'], r['eval_support'], r['model']): r for r in X['ptab_saved']}
    worst_person, n_person = 0.0, 0
    for r in ptab:
        s = saved[(r['person'], r['train_support'], r['eval_support'], r['model'])]
        for k in ('n_windows', 'n_positive'):
            if int(s[k]) != r[k]:
                raise AuditStop(f'person-score count mismatch {k}')
        for k in ('brier', 'logloss', 'brier_sum', 'logloss_sum'):
            worst_person = max(worst_person, abs(float(s[k]) - r[k]))
            n_person += 1
    if len(ptab) != len(X['ptab_saved']) or worst_person > TOL:
        raise AuditStop(f'per-person recomputation mismatch {worst_person}')
    agg = {(a['train_support'], a['eval_support'], a['model']): a for a in e6.aggregate(ptab)}
    worst_agg, n_agg = 0.0, 0
    for m in X['summary']['metrics']:
        a = agg[(m['train_support'], m['eval_support'], m['model'])]
        for k in ('brier_subject_equal', 'logloss_subject_equal', 'brier_window_weighted', 'logloss_window_weighted'):
            worst_agg = max(worst_agg, abs(a[k] - m[k]))
            n_agg += 1
    if worst_agg > TOL:
        raise AuditStop(f'aggregate recomputation mismatch {worst_agg}')
    persons = sorted({r['subject_key'] for r in rows})
    B = e6.Bootstrap(persons)
    if hashlib.sha256(B.I.tobytes()).hexdigest() != X['boot']['draw_matrix_sha256']:
        raise AuditStop('bootstrap draw matrix differs')
    worst_boot, n_boot = 0.0, 0
    for c in X['boot']['contrasts']:
        r = B.contrast(ptab, c['train_support'], c['eval_support'], c['reference'], c['candidate'], c['metric'])
        vals = [(r['subject_equal']['delta'], c['subject_equal']['delta']),
                (r['window_weighted']['delta'], c['window_weighted']['delta'])]
        vals += list(zip(r['subject_equal']['ci95'], c['subject_equal']['ci95']))
        vals += list(zip(r['window_weighted']['ci95'], c['window_weighted']['ci95']))
        vals += [(r['per_person_delta'][p], c['per_person_delta'][p]) for p in persons]
        for a, b in vals:
            worst_boot = max(worst_boot, abs(a - b))
        n_boot += 1
    if n_boot != 72 or worst_boot > TOL:
        raise AuditStop(f'bootstrap contrast recomputation mismatch {worst_boot} over {n_boot}')
    prim = X['summary']['q1']['primary_delta']
    pc = next(c for c in X['boot']['contrasts'] if (c['train_support'], c['eval_support'], c['reference'], c['candidate'],
                                                    c['metric']) == ('FULL', 'FULL', 'TIME_CONT', 'TIME_CONT_EEG_LOGIT', 'brier'))
    if abs(pc['subject_equal']['delta'] - prim['delta']) > TOL:
        raise AuditStop('summary primary differs from bootstrap file')
    return rows, pred, ptab, dict(per_person_values_checked=n_person, per_person_max_abs_diff=worst_person,
                                  aggregate_values_checked=n_agg, aggregate_max_abs_diff=worst_agg,
                                  bootstrap_contrasts_recomputed=n_boot, bootstrap_max_abs_diff=worst_boot,
                                  draw_matrix_sha256_matches=True, tolerance=TOL, passed=True)


# ------------------------------------------------------------------------ diagnostics
def level_within(p, y):
    """Exact decomposition Brier = L + W for one person (population moments)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    pb, yb = p.mean(), y.mean()
    L = (pb - yb) ** 2 + yb * (1 - yb)
    W = ((p - pb) ** 2).mean() - 2 * ((p - pb) * (y - yb)).mean()
    return L, W


def auroc(p, y):
    """Mann-Whitney AUROC with ties = 0.5; None if only one class."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def tie_ranks(p):
    order = np.argsort(p, kind='mergesort')
    ranks = np.empty(len(p))
    sp = np.asarray(p)[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def person_frames(rows, pred, ts, es):
    """Per person: indices of the evaluation windows in cell (ts, es)."""
    subj = np.array([r['subject_key'] for r in rows])
    anchor = np.array([r['anchor'] for r in rows])
    y = np.array([r['y'] for r in rows])
    out = {}
    for s in sorted(set(subj)):
        k = (subj == s) & (anchor if es == 'ANCHOR' else True)
        out[s] = np.where(k)[0]
    return out, y


def person_diagnostics(rows, pred):
    lat = {}
    for r in rows:
        lat.setdefault(r['subject_key'], r['elapsed_s'] + r['time_to_n2_s'])
    origin = {r['subject_key']: r['origin'] for r in rows}
    recs = []
    for ts, es in CELLS:
        frames, y = person_frames(rows, pred, ts, es)
        for s, ix in frames.items():
            yy = y[ix]
            rec = dict(train_support=ts, eval_support=es, person=s, origin=origin[s], latency_s=lat[s],
                       n_windows=len(ix), n_positive=int(yy.sum()), all_positive=bool(yy.min() == 1))
            for m in MODELS4:
                p = pred[ts][m][ix]
                b = float(((p - yy) ** 2).mean())
                L, W = level_within(p, yy)
                if abs(L + W - b) > 1e-12:
                    raise AuditStop('level/within decomposition does not sum to Brier')
                a = auroc(p, yy)
                rec.update({f'brier_{m}': b, f'level_{m}': L, f'within_{m}': W, f'mean_p_{m}': float(p.mean()),
                            f'auroc_{m}': ('' if a is None else a)})
            for cid, (ref, cand) in CONTRASTS.items():
                rec[f'dBrier_{cid}'] = rec[f'brier_{ref}'] - rec[f'brier_{cand}']
                rec[f'dLevel_{cid}'] = rec[f'level_{ref}'] - rec[f'level_{cand}']
                rec[f'dWithin_{cid}'] = rec[f'within_{ref}'] - rec[f'within_{cand}']
            recs.append(rec)
    return recs


def boot_matrix(S):
    return np.random.default_rng(SEED).integers(0, S, size=(N_BOOT, S))


def summarise(vals, n, I):
    vals, n = np.asarray(vals, float), np.asarray(n, float)
    se = vals[I].mean(1)
    ww = (vals * n)[I].sum(1) / n[I].sum(1)
    return dict(subject_equal=float(vals.mean()), subject_equal_ci95=[float(x) for x in np.percentile(se, [2.5, 97.5])],
                window_weighted=float((vals * n).sum() / n.sum()),
                window_weighted_ci95=[float(x) for x in np.percentile(ww, [2.5, 97.5])])


def cell_records(diag, ts, es):
    return sorted([d for d in diag if d['train_support'] == ts and d['eval_support'] == es], key=lambda d: d['person'])


def decomposition(diag):
    out = {}
    for ts, es in CELLS:
        R = cell_records(diag, ts, es)
        n = np.array([r['n_windows'] for r in R], float)
        S = len(R)
        I = boot_matrix(S)
        cell = {}
        for cid in CONTRASTS:
            d = np.array([r[f'dBrier_{cid}'] for r in R])
            w = n / n.sum()
            gap_contrib = (1 / S - w) * d
            cell[cid] = dict(
                contrast=CONTRASTS[cid], brier=summarise(d, n, I),
                level=summarise([r[f'dLevel_{cid}'] for r in R], n, I),
                within=summarise([r[f'dWithin_{cid}'] for r in R], n, I),
                se_minus_ww_gap=float(gap_contrib.sum()),
                covariance_identity_ww_minus_se=float(S * np.mean((w - w.mean()) * (d - d.mean()))),
                gap_contributions={r['person']: float(g) for r, g in zip(R, gap_contrib)})
            if abs(cell[cid]['brier']['window_weighted'] - cell[cid]['brier']['subject_equal']
                   - cell[cid]['covariance_identity_ww_minus_se']) > 1e-12:
                raise AuditStop('covariance identity failed')
        out[f'train_{ts}__eval_{es}'] = cell
    return out


def influence(diag):
    out = {}
    for ts, es in CELLS:
        R = cell_records(diag, ts, es)
        res = {}
        for cid in CONTRASTS:
            d = np.array([r[f'dBrier_{cid}'] for r in R])
            n = np.array([r['n_windows'] for r in R], float)
            allpos = np.array([r['all_positive'] for r in R])
            short = n < SHORT_N

            def agg(mask):
                if mask.sum() == 0:
                    return dict(n_persons=0, subject_equal=None, window_weighted=None)
                return dict(n_persons=int(mask.sum()), subject_equal=float(d[mask].mean()),
                            window_weighted=float((d[mask] * n[mask]).sum() / n[mask].sum()))
            loo_se = [float(np.delete(d, i).mean()) for i in range(len(d))]
            loo_ww = [float(np.delete(d * n, i).sum() / np.delete(n, i).sum()) for i in range(len(d))]
            res[cid] = dict(all=agg(np.ones(len(d), bool)), excl_all_positive=agg(~allpos),
                            excl_short_lt30=agg(~short), only_short_lt30=agg(short),
                            all_positive_persons=[r['person'] for r, a in zip(R, allpos) if a],
                            short_persons=[r['person'] for r, s in zip(R, short) if s],
                            loo_subject_equal=dict(min=min(loo_se), max=max(loo_se)),
                            loo_window_weighted=dict(min=min(loo_ww), max=max(loo_ww)),
                            max_single_person_share_of_se=dict(person=R[int(np.argmax(np.abs(d)))]['person'],
                                                                share=float(d[int(np.argmax(np.abs(d)))] / len(d) / d.mean())) if d.mean() else None)
        out[f'train_{ts}__eval_{es}'] = res
    return out


def strata(diag):
    out = {}
    for ts, es in CELLS:
        R = cell_records(diag, ts, es)
        order = sorted(R, key=lambda r: (r['n_windows'], r['person']))
        k = len(order) // 3
        groups = {'short': order[:k], 'middle': order[k:2 * k], 'long': order[2 * k:]}
        cell = {}
        for g, rr in groups.items():
            n = np.array([r['n_windows'] for r in rr], float)
            cell[g] = dict(persons=[r['person'] for r in rr], n_windows_range=[int(n.min()), int(n.max())],
                           latency_s_range=[min(r['latency_s'] for r in rr), max(r['latency_s'] for r in rr)])
            for cid in CONTRASTS:
                for comp in ('dBrier', 'dLevel', 'dWithin'):
                    v = np.array([r[f'{comp}_{cid}'] for r in rr])
                    cell[g][f'{comp}_{cid}'] = dict(subject_equal=float(v.mean()),
                                                     window_weighted=float((v * n).sum() / n.sum()))
        out[f'train_{ts}__eval_{es}'] = cell
    return out


def within_discrimination(rows, pred):
    table, summary = [], {}
    rng = np.random.default_rng(SEED)
    for ts, es in CELLS:
        frames, y = person_frames(rows, pred, ts, es)
        cs = {}
        for m in MODELS4:
            vals = []
            for s, ix in frames.items():
                yy, p = y[ix], pred[ts][m][ix]
                a = auroc(p, yy)
                table.append(dict(train_support=ts, eval_support=es, model=m, person=s, n_windows=len(ix),
                                  n_positive=int(yy.sum()), n_negative=int(len(yy) - yy.sum()),
                                  auroc='' if a is None else a,
                                  status='defined' if a is not None else 'undefined: single-class recording'))
                if a is not None:
                    vals.append((s, a, tie_ranks(p), int(yy.sum()), len(ix)))
            obs = float(np.mean([v[1] for v in vals]))
            null = np.zeros(N_PERM)
            for k in range(N_PERM):
                tot = 0.0
                for s, a, rk, npos, nn in vals:
                    sel = rng.permutation(nn)[:npos]
                    tot += (rk[sel].sum() - npos * (npos + 1) / 2) / (npos * (nn - npos))
                null[k] = tot / len(vals)
            cs[m] = dict(n_persons_defined=len(vals), n_persons_undefined=len(frames) - len(vals),
                         undefined_persons=sorted(set(frames) - {v[0] for v in vals}),
                         mean_within_auroc=obs, median_within_auroc=float(np.median([v[1] for v in vals])),
                         n_persons_auroc_gt_half=int(sum(v[1] > .5 for v in vals)),
                         permutation_null_mean=float(null.mean()),
                         permutation_null_95th=float(np.percentile(null, 95)),
                         permutation_p_one_sided=float((1 + (null >= obs).sum()) / (1 + N_PERM)))
        for cid, (ref, cand) in CONTRASTS.items():
            ra = {t['person']: t['auroc'] for t in table if (t['train_support'], t['eval_support'], t['model']) == (ts, es, ref)}
            ca = {t['person']: t['auroc'] for t in table if (t['train_support'], t['eval_support'], t['model']) == (ts, es, cand)}
            d = [ca[p] - ra[p] for p in ra if ra[p] != '' and ca[p] != '']
            cs[f'paired_auroc_diff_{cid}'] = dict(candidate_minus_reference_mean=float(np.mean(d)) if d else None,
                                                 n_persons=len(d), n_candidate_higher=int(sum(x > 0 for x in d)))
        summary[f'train_{ts}__eval_{es}'] = cs
    return table, summary


def correlations(diag):
    R = cell_records(diag, *MAIN)
    lat = [r['latency_s'] for r in R]
    out = {}
    for cid in CONTRASTS:
        for comp in ('dBrier', 'dLevel', 'dWithin'):
            rho, p = spearmanr(lat, [r[f'{comp}_{cid}'] for r in R])
            out[f'{comp}_{cid}'] = dict(spearman_rho=float(rho), p_value=float(p))
    return out


def classify_source(dec_main):
    out = {}
    for cid in ('C1', 'C2'):
        lv, wi = dec_main[cid]['level']['subject_equal_ci95'], dec_main[cid]['within']['subject_equal_ci95']
        lpos, wpos = lv[0] > 0, wi[0] > 0
        lab = ('both' if lpos and wpos else 'primarily between-person level' if lpos and wi[0] <= 0 <= wi[1]
               else 'primarily within-person' if wpos and lv[0] <= 0 <= lv[1] else 'neither established')
        out[cid] = dict(label=lab, level_se_ci95=lv, within_se_ci95=wi,
                        note='association observed in frozen predictions; not a causal explanation')
    return out


def erratum_record(repo):
    dev = repo / 'docs/decision/h_eval/gate_e6/gate_e6_deviation_01_download_mirror.md'
    mv = repo / '.local/h_eval/gate_e6/mirror_verification.json'
    return dict(
        erratum='gate_e6_erratum_01', applies_to='Gate E6 report and summary messages (E6 artefacts left unchanged)',
        items=[
            dict(id=1, issue='three longest recordings misdescribed',
                 original='E6 report §5.2: "The three longest recordings (95, 220 and 233 windows) all favour time-only"',
                 correction='[per-person identifiers and per-person results redacted in the paper-only release]'),
            dict(id=2, issue='candidate count',
                 original='E6 report §2: "41 candidate SC_TRAIN subjects"',
                 correction=('47 SC_TRAIN - 6 retained = 41 remaining subjects, of which 40 had an eligible recording '
                             '(one person excluded: its only E2-eligible recording had stage 1 at LightsOff and its other '
                             'recording had N2 active at LightsOff; identifiers redacted in the paper-only release). '
                             'gate_e6_data_provenance.json correctly records n_candidates = 40.')),
            dict(id=3, issue='machine-readable mirror deviation reference missing from E6 provenance',
                 correction='This record supplies it; gate_e6_data_provenance.json is immutable and was not edited.',
                 deviation_document=str(dev.relative_to(repo)), deviation_document_sha256=digest(dev),
                 mirror_base_url='https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/',
                 mirror_verification_file=str(mv.relative_to(repo)),
                 mirror_verification_sha256=digest(mv) if mv.exists() else None,
                 mirror_files=json.loads(mv.read_text()) if mv.exists() else None,
                 run1_log='.local/h_eval/gate_e6/console_run1.log',
                 run1_log_sha256=digest(repo / '.local/h_eval/gate_e6/console_run1.log'),
                 preserved_partial='data/raw/sleep_edfx_v1_0_0/sleep-cassette/[recording redacted in the paper-only release].e4-part')],
        effect_on_results='None: all three are documentation errors; no number, file or decision in E6 changes.')


# ------------------------------------------------------------------------ main
def to_csv(rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    X = load_and_verify(repo)
    rows, pred, ptab, recompute = recompute_and_compare(X)
    diag = person_diagnostics(rows, pred)
    dec = decomposition(diag)
    infl = influence(diag)
    strat = strata(diag)
    wtab, wsum = within_discrimination(rows, pred)
    corr = correlations(diag)
    src = classify_source(dec['train_FULL__eval_FULL'])
    verification = dict(version=VERSION, plan_sha256=PLAN_SHA256, e6_protocol_sha256=E6_PROTOCOL_SHA256,
                        e6_output_hash_checks=X['hash_checks'], recomputation=recompute,
                        n_windows=len(rows), n_persons=len({r['subject_key'] for r in rows}))
    summary = dict(version=VERSION, status='POST_HOC_EXPLORATORY_AUDIT_TRAIN_ONLY_NOT_CONFIRMATORY',
                   main_cell='train FULL / eval FULL', contrasts={k: list(v) for k, v in CONTRASTS.items()},
                   decomposition=dec, influence=infl, latency_strata=strat,
                   within_person_discrimination=wsum, latency_correlations_main_cell=corr, source_classification=src,
                   structural_note=('All-event cohort: positives are always the final windows of a recording, so any predictor '
                                    'monotonically increasing in elapsed time has within-recording AUROC = 1 by construction.'),
                   guards=dict(eeg_read=False, refit=False, gpu=False, protected_splits_used=False, downloads=False))
    blobs = {'gate_e7a_verification.json': (json.dumps(verification, indent=2) + '\n').encode(),
             'gate_e7a_summary.json': (json.dumps(summary, indent=2) + '\n').encode(),
             'gate_e7a_person_diagnostics.csv': to_csv(diag),
             'gate_e7a_within_person_discrimination.csv': to_csv(wtab),
             'gate_e6_erratum_01.json': (json.dumps(erratum_record(repo), indent=2) + '\n').encode()}
    e6 = X['e6']
    for n, b in blobs.items():
        if (a.out / n).exists() and (a.out / n).read_bytes() != b:
            raise AuditStop('output conflict ' + n)
    for n, b in blobs.items():
        e6.immut(a.out / n, b)
    print(json.dumps(dict(verification=recompute, source=src,
                          outputs={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}), indent=2))


if __name__ == '__main__':
    main()
