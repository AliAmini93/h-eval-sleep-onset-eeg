#!/usr/bin/env python3
"""Gate E7A amendment 01: elapsed-time-matched cross-person discrimination on frozen E6 predictions.

POST-HOC EXPLORATORY (see docs/decision/h_eval/gate_e7a/gate_e7a_amendment_01.md). Reads only frozen E6 outputs,
re-verified through gate_e7a.load_and_verify. No EEG, no refit, no protected data.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AMEND_REL = 'docs/decision/h_eval/gate_e7a/gate_e7a_amendment_01.md'
AMEND_SHA256 = 'f9b3e54f34b36a2e74b2f4670cad4de3bf6bd711b5bd80da3f372d73fb66017a'
STRATUM_S = 300.
N_BOOT = 2_000
SEED = 20261023
MODELS = ('CONSTANT', 'TIME_CONT', 'EEG_LOGIT', 'TIME_CONT_EEG_LOGIT')
CELLS = (('FULL', 'FULL'), ('FULL', 'ANCHOR'))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pair_matrices(p, y, person, stratum, persons):
    """WIN[a, b], PAIRS[a, b]: concordance of positives of person a against negatives of person b (a != b),
    summed over elapsed-time strata."""
    idx = {s: i for i, s in enumerate(persons)}
    S = len(persons)
    win, pairs = np.zeros((S, S)), np.zeros((S, S))
    n_strata_used = 0
    for st in np.unique(stratum):
        k = stratum == st
        pos = np.where(k & (y == 1))[0]
        neg = np.where(k & (y == 0))[0]
        used = False
        for i in pos:
            for j in neg:
                a, b = idx[person[i]], idx[person[j]]
                if a == b:
                    continue
                pairs[a, b] += 1
                win[a, b] += 1. if p[i] > p[j] else .5 if p[i] == p[j] else 0.
                used = True
        n_strata_used += used
    return win, pairs, n_strata_used


def weighted_auc(win, pairs, c):
    w = np.outer(c, c)
    np.fill_diagonal(w, 0)
    den = (w * pairs).sum()
    return float((w * win).sum() / den) if den > 0 else float('nan')


def run(repo: Path):
    e7 = _load('e7a_for_amend', 'gate_e7a.py')
    if hashlib.sha256((repo / AMEND_REL).read_bytes()).hexdigest() != AMEND_SHA256:
        raise e7.AuditStop('Amendment document differs from its frozen hash')
    X = e7.load_and_verify(repo)
    rows, pred, _, rep = e7.recompute_and_compare(X)
    persons = sorted({r['subject_key'] for r in rows})
    person = np.array([r['subject_key'] for r in rows])
    y = np.array([r['y'] for r in rows])
    anchor = np.array([r['anchor'] for r in rows])
    stratum = np.floor(np.array([r['elapsed_s'] for r in rows]) / STRATUM_S).astype(int)
    I = np.random.default_rng(SEED).integers(0, len(persons), size=(N_BOOT, len(persons)))
    C = np.stack([np.bincount(r, minlength=len(persons)) for r in I]).astype(float)
    out = dict(version='H-EVAL-E7A-amendment01-time-matched-v1', status='POST_HOC_EXPLORATORY',
               amendment_sha256=AMEND_SHA256, verification_passed=rep['passed'], stratum_width_s=STRATUM_S,
               n_boot=N_BOOT, seed=SEED, cells={})
    for ts, es in CELLS:
        k = np.ones(len(rows), bool) if es == 'FULL' else anchor
        mats, cell = {}, {}
        for m in MODELS:
            win, pairs, used = pair_matrices(pred[ts][m][k], y[k], person[k], stratum[k], persons)
            mats[m] = (win, pairs)
            boots = np.array([weighted_auc(win, pairs, c) for c in C])
            cell[m] = dict(auroc=weighted_auc(win, pairs, np.ones(len(persons))),
                           ci95=[float(x) for x in np.nanpercentile(boots, [2.5, 97.5])],
                           n_cross_person_pairs=int(pairs.sum()), n_strata_contributing=int(used),
                           _boots=boots)
        for cand in ('EEG_LOGIT', 'TIME_CONT_EEG_LOGIT'):
            d = cell[cand]['_boots'] - cell['TIME_CONT']['_boots']
            cell[f'{cand}_minus_TIME_CONT'] = dict(delta=cell[cand]['auroc'] - cell['TIME_CONT']['auroc'],
                                                   ci95=[float(x) for x in np.nanpercentile(d, [2.5, 97.5])])
        for m in MODELS:
            cell[m].pop('_boots')
        n_strata_total = int(len(np.unique(stratum[k])))
        cell['n_strata_total'] = n_strata_total
        cell['note_constant'] = ('CONSTANT varies across persons only through the LOPO fold prior; deviation from 0.5 '
                                 'is a cross-validation artefact, not information.')
        out['cells'][f'train_{ts}__eval_{es}'] = cell
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    res = run(repo)
    blob = (json.dumps(res, indent=2) + '\n').encode()
    e6 = _load('e6_for_amend', '../gate_e6/gate_e6.py')
    e6.immut(a.out / 'gate_e7a_time_matched_discrimination.json', blob)
    print(json.dumps(res, indent=2))
    print('sha256', hashlib.sha256(blob).hexdigest())


if __name__ == '__main__':
    main()
