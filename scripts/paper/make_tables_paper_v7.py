#!/usr/bin/env python3
"""Manuscript v7 appendix tables generated directly from hashed aggregate results (no hand transcription).
Writes docs/paper/h_eval_ed_v7/latex/tables_generated.tex."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RD = REPO / 'results/aggregates'
OUT = REPO / 'build/tables_generated_v7.tex'


def ld(p):
    return json.loads((RD / p).read_text())


def pc(x):
    return f'{100 * x:.1f}'.replace('-', '$-$')


def rci(c):
    if not isinstance(c, dict) or 'R_ci95' not in c:
        return pc(c['R']) + ' (n.r.)' if isinstance(c, dict) and c.get('R') is not None else 'n/a'
    return f"{pc(c['R'])} ({pc(c['R_ci95'][0])}, {pc(c['R_ci95'][1])})"


def num(x, d=4):
    return f'{x:.{d}f}'.replace('-', '$-$')


def r2_table():
    r2 = ld('repair_r2/r2_stage_results.json')['populations']
    models = {'VAL': [('SpecLR-18', 'CONSTANT->SpecLR-18 [EEG_LOGIT]')],
              'TEST': [('SpecLR-18', 'CONSTANT->SpecLR-18 [E7B_EEG_LOGIT]'), ('SpecLR-46', 'CONSTANT->SpecLR-46 [M0_46]'),
                       ('RawCNN-46', 'CONSTANT->RawCNN-46 [M2_46]')],
              'EXT': [('SpecLR-18', 'CONSTANT->SpecLR-18 [E7B_EEG_LOGIT]'), ('SpecLR-46', 'CONSTANT->SpecLR-46 [M0_46]'),
                      ('RawCNN-46', 'CONSTANT->RawCNN-46 [M2_46]')]}
    subs = [('ALL', 'All'), ('STAGE_1', 'Stage 1'), ('STAGE_W', 'W'), ('PRE_FIRST_SLEEP', 'Pre-sleep'),
            ('W_POST_FIRST_SLEEP', 'Returning W')]
    L = [r'{\footnotesize\setlength{\tabcolsep}{2.5pt}', r'\begin{longtable}{@{}lllrrrrl@{}}',
         r'\caption{Held-out stage-conditioned audit (R2; post-hoc, descriptive; evidence class C4). Brier reduction of each frozen '
         r'model relative to the frozen Constant on the same windows. $\Delta_{\mathrm{WW}}$: absolute window-weighted Brier '
         r'difference; $R$: relative reduction (\%) with 95\% person-cluster CI (n.r.: fewer than 8 persons, CI not reported). '
         r'Subsets are scorer-defined and were not available to the models.}\label{tab:r2full}\\', r'\toprule',
         r'Set & Model & Subset & Windows (pos.) & Persons & Recs & $\Delta_{\mathrm{WW}}$ & $R$ (\%) \\', r'\midrule', r'\endhead']
    for pop, ms in models.items():
        for name, key in ms:
            for s, lab in subs:
                S = r2[pop]['subsets'][s]
                c = S['contrasts'][key]
                L.append(f"{pop} & {name} & {lab} & {S['n_windows']} ({S['n_positive']}) & {S['n_persons']} & {S['n_recordings']} & "
                         f"{num(c['delta_ww'])} & {rci(c)} \\\\")
            L.append(r'\addlinespace[1pt]')
    L += [r'\bottomrule', r'\end{longtable}}']
    return '\n'.join(L)


def w2s_table():
    c = ld('discovery_w2s/w2s_classical_results_v1.json')['results']['A']
    n = ld('discovery_w2s/w2s_neural_results_v1.json')['results']
    rows = [('C1 (features, logistic)', 'C1_logistic', 'c'), ('C2 (features, boosting)', 'C2_hgb', 'c'),
            ('TCN-30s', 'N1_tcn30s', 'n'), ('TCN-5m', 'N2_tcn5min', 'n'),
            ('TCN-5m+stage', 'N3_tcn5min_state', 'n'), ('Spec-5m+stage', 'N4_spec5min_state', 'n'),
            ('ORACLE state (diagnostic)', 'ORACLE_state(diagnostic)', 'c')]
    cols = [('E1', 0, 'E1: W$\\to$N1'), ('E2', 1, 'E2-pre: W$\\to$N2'), ('E2', 2, 'E2-ret'), ('E3', 0, 'E3: N1$\\to$N2')]
    L = [r'\begin{table}[h]', r'\centering',
         r'\caption{W2S screen (exploratory, TRAIN-only, partition A; evidence class C5). $R$ (\%) relative to a within-population '
         r'time reference fitted in the training folds (95\% person-cluster CI). Population sizes (windows/positives): E1 '
         r'3058/763; E2-pre 3058/132; E2-ret 515/117; E3 1219/582. C1/C2: aperiodic, entropy, DFA and spectral features with L2 logistic regression or gradient boosting; TCN-30s/TCN-5m: causal temporal CNN on 30\,s or 5\,min of raw EEG; +stage: auxiliary stage head with a state-mixture hazard; Spec-5m: 5\,min log-spectrogram input. The oracle uses scorer-derived time since first sleep and is '
         r'not a usable forecaster.}', r'\label{tab:w2s}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{@{}lllll@{}}', r'\toprule', 'Model & ' + ' & '.join(c[2] for c in cols) + r' \\', r'\midrule']
    for lab, key, src in rows:
        cells = []
        for E, i, _ in cols:
            res = (c if src == 'c' else n)[E][i]['vs_time']
            cells.append(rci(res[key]) if key in res else '---')
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def p0b_p1_table():
    p0b = ld('discovery_w2s/w2s_p0b_calibration_v1.json')
    p1 = ld('discovery_w2s/w2s_p1_results_v1.json')['partitions']
    L = [r'\begin{table}[h]', r'\centering',
         r'\caption{E1 calibration and personalisation (exploratory, TRAIN-only; C5). Calibrators were fitted by nested '
         r'person-disjoint cross-fitting (no evaluated person\textquotesingle s labels used by base model, hyperparameters or '
         r'calibrator). K1: intercept; K2: intercept+slope; K3: score+elapsed-time. P1 variants: A base 5-min encoder + time; '
         r'B +own-EEG baseline reference; C +GRU over own EEG history; D +calibration-aware loss; E all. Donor: own history '
         r'replaced by another person\textquotesingle s. $R$ relative to the within-E1 time reference.}', r'\label{tab:p0p1}',
         r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
         r'Model & Partition & $R$ (\%) & CIL & Slope & Log loss \\', r'\midrule']
    for k in ('N2_tcn5min|K0', 'N2_tcn5min|K3', 'N3_tcn5min_state|K0', 'N3_tcn5min_state|K1', 'N3_tcn5min_state|K2',
              'N3_tcn5min_state|K3'):
        m = p0b['results']['models'][k]
        nm = {'N2_tcn5min': 'TCN-5m', 'N3_tcn5min_state': 'TCN-5m+stage'}[k.split('|')[0]]
        L.append(f"{nm} {k.split('|')[1].replace('K0', 'uncalibrated')} & A & {rci(p0b['results']['vs_time'][k])} & "
                 f"{num(m['cal_in_large'], 3)} & {m['cal_slope']:.2f} & {m['logloss_ww']:.3f} \\\\")
    L.append(r'\midrule')
    for part in 'AB':
        P = p1[part]['results']
        for k in ('A', 'B', 'B|donor_history', 'C', 'C|donor_history', 'D', 'E', 'E|donor_history', 'A|K3', 'E|K3'):
            m = P['models'][k]
            L.append(f"P1-{k.replace('|donor_history', ' (donor)').replace('|K3', ' + K3')} & {part} & {rci(P['vs_time'][k])} & "
                     f"{num(m['cal_in_large'], 3)} & {m['cal_slope']:.2f} & {m['logloss_ww']:.3f} \\\\")
        if part == 'A':
            L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def cohort_table():
    f = ld('repair_r1/r1_cohort_flow.json')['flow']
    sc, st = f['SC'], f['ST']
    ex = lambda d, k: d['excluded_by_reason'].get(k, {}).get('recordings', 0)  # noqa: E731
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Cohort flow (recordings; persons in parentheses). Reconstructed from the frozen eligibility inventories '
         r'(metadata only). Primary night: lowest-numbered eligible night per person (VAL, TEST); EXT used all eligible '
         r'nights with the person as the bootstrap cluster; TRAIN used all eligible nights.}', r'\label{tab:cohort}',
         r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrr@{}}', r'\toprule',
         r' & Sleep Cassette (SC) & Sleep Telemetry (ST) \\', r'\midrule',
         f"Source & {sc['source']['recordings']} ({sc['source']['persons']}) & {st['source']['recordings']} ({st['source']['persons']}) \\\\",
         f"Lights-off outside signal & $-${ex(sc, 'E2: OFF_CLOCK_OUTSIDE_PSG')} & $-${ex(st, 'E2: OFF_CLOCK_OUTSIDE_PSG')} \\\\",
         f"Scoring gap/overlap before first N2 & $-${ex(sc, 'E2: SCORING_GAP_OR_OVERLAP')} & $-${ex(st, 'E2: SCORING_GAP_OR_OVERLAP')} \\\\",
         f"N2 active at lights-off & $-${ex(sc, 'E2: N2_ACTIVE_AT_OFF')} & 0 \\\\",
         f"Not awake (W) at lights-off & $-${ex(sc, 'E3/E6 rule: stage at LightsOff = 1 (not W)') + ex(sc, 'E3/E6 rule: stage at LightsOff = R (not W)')} & 0 \\\\",
         f"Analysable & {sc['analysable_recordings']} ({sc['analysable_persons']}) & {st['analysable_recordings']} ({st['analysable_persons']}) \\\\",
         r'\midrule']
    b = sc['by_split']
    L += [f"TRAIN (development) & {b['SC_TRAIN']['analysed_recordings']} ({b['SC_TRAIN']['analysed_persons']}) & --- \\\\",
          f"VAL (one-shot; primary + sensitivity) & {b['SC_VALIDATION']['primary_recordings']} + {b['SC_VALIDATION']['sensitivity_recordings']} ({b['SC_VALIDATION']['analysed_persons']}) & --- \\\\",
          f"TEST (one-shot; primary + sensitivity) & {b['SC_PROTECTED_MODEL_TEST']['primary_recordings']} + {b['SC_PROTECTED_MODEL_TEST']['sensitivity_recordings']} ({b['SC_PROTECTED_MODEL_TEST']['analysed_persons']}) & --- \\\\",
          f"EXT (one-shot; all nights) & --- & {st['by_split']['ST_EXTERNAL_RESERVED']['analysed_recordings']} ({st['by_split']['ST_EXTERNAL_RESERVED']['analysed_persons']}) \\\\",
          r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def main():
    OUT.write_text('% GENERATED by scripts/paper/make_tables_paper_v7.py -- do not edit by hand\n'
                   + '\n\n'.join(['\\newcommand{\\TableCohort}{%\n' + cohort_table() + '}',
                                  '\\newcommand{\\TableRtwoFull}{%\n' + r2_table() + '}',
                                  '\\newcommand{\\TableWtwoS}{%\n' + w2s_table() + '}',
                                  '\\newcommand{\\TablePzeroPone}{%\n' + p0b_p1_table() + '}']) + '\n')
    print('written', OUT)


if __name__ == '__main__':
    main()
