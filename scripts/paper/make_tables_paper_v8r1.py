#!/usr/bin/env python3
"""Manuscript v8r1 tables generated directly from hashed result files (no hand transcription).
Reuses the v7 generators for cohort flow, held-out stage audit, W2S and calibration tables, and adds:
  TableFrozen   frozen one-shot results incl. sensitivity nights (envelope values and frozen intervals)
  TableCentral  central state-aware comparison (rows A-G) per population and model (R3B re-bootstrap intervals)
  TableRthreeB  full held-out state-aware contrasts (appendix)
Writes docs/paper/h_eval_scope_locked_v8r1/latex/tables_generated.tex."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RD = REPO / 'results/aggregates'
OUT = REPO / 'build/tables_generated_v8r1.tex'
spec = importlib.util.spec_from_file_location('t7', REPO / 'scripts/paper/make_tables_paper_v7.py')
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)
pc, rci, num, ld = t7.pc, t7.rci, t7.num, t7.ld


def ci_frozen(R, c):
    return f"{pc(R)} ({pc(c[0])}, {pc(c[1])})"


def frozen_table():
    v = ld('gate_e7b/execution_v1_2/gate_e7b_primary_result_v1_2.json')['content']
    vs = ld('gate_e7b/execution_v1_2/gate_e7b_sensitivity_result_v1_2.json')['content']['scores']
    t = ld('gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json')['content']['results']
    ts = ld('gate_t1/execution_v1/gate_t1_sensitivity_result_v1.json')['content']['scores']
    x = ld('gate_x1/execution_v1/gate_x1_result_v1.json')['content']['results']
    vp, tp, xp = v['scores']['primary'], t['H1']['scores']['primary'], x['H1']['scores']['primary']
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Frozen one-shot results on the pooled first-N2 target (E0). Window-weighted Brier; $R$ in \%, with the '
         r'\emph{frozen} 95\% person-cluster intervals produced by each test\textquotesingle s predeclared bootstrap. '
         r'$\Delta_{\mathrm{WW}}$: absolute Brier difference. Sensitivity rows pool all eligible nights (VAL 27, TEST 30) and '
         r'are decision-neutral. EXT H2 was not tested because H1 was not GO (descriptive value shown).}',
         r'\label{tab:frozen}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
         r' & VAL & TEST & EXT \\', r'\midrule',
         f"Persons / nights; windows (positive) & 15 / 15; {v['scores']['n_windows']} ({v['scores']['n_positive']}) & 15 / 15; {t['n_windows']} ({t['n_positive']}) & {x['n_persons']} / {x['n_recordings']}; {x['n_windows']} ({x['n_positive']}) \\\\",
         f"Brier: Constant / SpecLR-18 & {vp['ref_ww']:.3f} / {vp['cand_ww']:.3f} & {t['models']['E7B_CONSTANT']['brier_ww']:.3f} / {t['models']['E7B_EEG']['brier_ww']:.3f} & {x['models']['E7B_CONSTANT']['brier_ww']:.3f} / {x['models']['E7B_EEG']['brier_ww']:.3f} \\\\",
         f"Brier: SpecLR-46 / RawCNN-46 & --- & {t['models']['M0_46']['brier_ww']:.3f} / {t['models']['M2_46']['brier_ww']:.3f} & {x['models']['M0_46']['brier_ww']:.3f} / {x['models']['M2_46']['brier_ww']:.3f} \\\\",
         r'\midrule',
         f"H1: SpecLR-18 vs Constant, $R$ & {ci_frozen(vp['R'], vp['R_ci95'])} & {ci_frozen(tp['R'], tp['R_ci95'])} & {ci_frozen(xp['R'], xp['R_ci95'])} \\\\",
         f"\\quad absolute $\\Delta_{{\\mathrm{{WW}}}}$ & {num(vp['delta_ww'])} & {num(tp['delta_ww'])} & {num(xp['delta_ww'])} \\\\",
         f"\\quad decision & {v['decision']['outcome']} & {t['H1']['outcome']} & {x['H1']['outcome']} \\\\",
         f"H2: RawCNN-46 vs SpecLR-18, $R$ & --- & {ci_frozen(t['H2']['R'], t['H2']['R_ci95'])}, SUPPORTED & {ci_frozen(x['H2']['R'], x['H2']['R_ci95'])}, not tested \\\\",
         f"H3: RawCNN-46 vs SpecLR-46 (descriptive) & --- & {ci_frozen(t['H3']['R'], t['H3']['R_ci95'])} & {ci_frozen(x['H3']['R'], x['H3']['R_ci95'])} \\\\",
         r'\midrule',
         f"Sensitivity, all nights: H1 & {ci_frozen(vs['primary']['R'], vs['primary']['R_ci95'])} & {ci_frozen(ts['H1']['scores']['primary']['R'], ts['H1']['scores']['primary']['R_ci95'])} & (all nights primary) \\\\",
         f"Sensitivity, all nights: H2 / H3 & --- & {ci_frozen(ts['H2']['R'], ts['H2']['R_ci95'])} / {ci_frozen(ts['H3']['R'], ts['H3']['R_ci95'])} & --- \\\\",
         r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(L)


def models_table():
    t = ld('gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json')['content']['results']
    x = ld('gate_x1/execution_v1/gate_x1_result_v1.json')['content']['results']
    names = [('E7B_CONSTANT', 'Constant'), ('E7B_EEG', 'SpecLR-18'), ('M0_46', 'SpecLR-46'), ('M2_46', 'RawCNN-46')]
    L = [r'\begin{table}[H]', r'\centering',
         r'\caption{Per-model metrics of the frozen models on TEST and EXT \cls{C1, C2}, generated from the frozen result files. '
         r'WW: window-weighted; SE: subject-equal; CIL: calibration-in-the-large. Calibration slope is undefined for a constant '
         r'predictor (n/a; the TEST result file records a numerical value of 0.626 for the constant, which is not meaningful). '
         r'Time-matched AUROC as defined in the frozen protocols: cross-person pairs within elapsed-time strata.}',
         r'\label{tab:models}', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}', r'\resizebox{\linewidth}{!}{%',
         r'\begin{tabular}{@{}llccccccc@{}}', r'\toprule',
         r'Set & Model & Brier WW & Brier SE & Log loss WW & CIL & Slope & Within-rec.\ AUROC & Time-matched AUROC \\', r'\midrule']
    for k, (lab, R) in enumerate((('TEST', t), ('EXT', x))):
        if k:
            L.append(r'\midrule')
        for i, (key, nm) in enumerate(names):
            m = R['models'][key]
            slope = 'n/a' if key == 'E7B_CONSTANT' else f"{m['cal_slope']:.2f}"
            cil = f"{m['cal_in_large']:+.3f}".replace('-', '$-$').replace('+', '$+$')
            first = rf'\multirow{{4}}{{*}}{{{lab}}}' if i == 0 else ''
            L.append(f"{first} & {nm} & {m['brier_ww']:.3f} & {m['brier_se']:.3f} & {m['logloss_ww']:.3f} & {cil} & {slope} & "
                     f"{m['within_recording_auroc']['mean']:.2f} & {m['time_matched_auroc']['auroc']:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(L)


def fix_longtable(s):
    """Caption and label only on the first page; a continuation header on later pages (avoids duplicate labels)."""
    s = s.replace('\\addlinespace[1pt]\n\\bottomrule', '\\bottomrule')
    head, rest = s.split(r'\toprule', 1)
    cols, rest = rest.split(r'\endhead', 1)
    cap = head.split(r'\caption', 1)[1].rsplit(r'\\', 1)[0]
    pre = head.split(r'\caption', 1)[0]
    return (pre + r'\caption' + cap + r'\\' + '\n' + r'\toprule' + cols + r'\endfirsthead' + '\n'
            + r'\multicolumn{' + str(cols.split(r'\midrule')[0].count('&') + 1) + r'}{@{}l}{\emph{(continued)}}\\' + '\n' + r'\toprule' + cols + r'\endhead' + rest)


def r3b():
    return ld('repair_r3b/r3b_results_v1.json')['populations']


def central_table():
    R = r3b()
    rows = [('A: EEG vs Constant', 'ALL', 'E'), ('B: Stage-only vs Constant', 'ALL', '1:CONSTANT->A'),
            ('C: Stage+time vs Constant', 'ALL', '2:CONSTANT->B'), ('D: EEG vs Stage-only', 'ALL', 'A'),
            ('E: EEG vs Stage+time', 'ALL', 'B'), ('F: EEG vs Stage+time, Stage 1 inputs', 'STAGE_1', 'B'),
            ('G: EEG vs Stage+time, W inputs', 'W', 'B')]
    cols = [('VAL', 'SpecLR-18'), ('TEST', 'SpecLR-18'), ('TEST', 'RawCNN-46'), ('EXT', 'SpecLR-18'), ('EXT', 'RawCNN-46')]
    key = {'SpecLR-18': {'E': '0a:CONSTANT->SpecLR-18', 'A': '3:A->SpecLR-18', 'B': '4:B->SpecLR-18'},
           'RawCNN-46': {'E': '0c:CONSTANT->RawCNN-46', 'A': '7:A->RawCNN-46', 'B': '8:B->RawCNN-46'}}
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Held-out comparison with TRAIN-fitted oracle-stage references (post-hoc; frozen procedure). $R$ in \% '
         r'(95\% person-cluster re-bootstrap interval) for the second model relative to the first. Stage-only and Stage+time '
         r'use the retrospectively scored stage of the input epoch and are diagnostic references, not deployable models. '
         r'Rows B and C do not depend on the EEG model. Row A repeats the frozen comparisons with re-bootstrapped intervals (frozen intervals in '
         r'Table~\ref{tab:frozen}). Bold: interval excludes zero.}', r'\label{tab:central}',
         r'\footnotesize', r'\setlength{\tabcolsep}{2.6pt}', r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
         r' & VAL & \multicolumn{2}{c}{TEST} & \multicolumn{2}{c}{EXT} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         ' & ' + ' & '.join(m for _, m in cols) + r' \\', r'\midrule']
    for lab, sn, k in rows:
        cells = []
        for pop, mdl in cols:
            kk = key[mdl].get(k, k)
            c = R[pop]['strata'][sn]['contrasts'][kk]
            if c == 'not_available':
                cells.append('---')
                continue
            s = rci(c)
            if isinstance(c.get('R_ci95'), list) and (c['R_ci95'][0] > 0 or c['R_ci95'][1] < 0):
                s = r'\textbf{' + s + '}'
            if k in ('1:CONSTANT->A', '2:CONSTANT->B') and mdl == 'RawCNN-46':
                s = r'$\leftarrow$'
            cells.append(s)
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(L)


def full_table():
    R = r3b()
    contrasts = ['0a:CONSTANT->SpecLR-18', '0b:CONSTANT->SpecLR-46', '0c:CONSTANT->RawCNN-46', '1:CONSTANT->A', '2:CONSTANT->B',
                 '3:A->SpecLR-18', '4:B->SpecLR-18', '3s:A18->SpecLR-18', '4s:B18->SpecLR-18', '5:A->SpecLR-46',
                 '6:B->SpecLR-46', '7:A->RawCNN-46', '8:B->RawCNN-46', '9:SpecLR-46->RawCNN-46']
    name = {'A': 'Stage-only', 'B': 'Stage+time', 'A18': 'Stage-only$_{18}$', 'B18': 'Stage+time$_{18}$', 'CONSTANT': 'Constant'}
    strata = [('ALL', 'All'), ('STAGE_1', 'Stage 1'), ('W', 'W'), ('PRE_FIRST_SLEEP', 'Pre-sleep'), ('RETURNING_W', 'Returning W')]
    L = [r'{\footnotesize\setlength{\tabcolsep}{2.5pt}', r'\begin{longtable}{@{}lllrrl@{}}',
         r'\caption{All held-out state-aware contrasts (post-hoc; frozen procedure; re-bootstrap intervals, 10{,}000 person-cluster '
         r'replicates, seed 20261026; n.r.: fewer than 8 persons). Subscript 18: reference refitted on the 18-person SpecLR-18 '
         r'training roster.}\label{tab:r3bfull}\\', r'\toprule',
         r'Set & Stratum & Contrast & Persons & $\Delta_{\mathrm{WW}}$ & $R$ (\%) \\', r'\midrule', r'\endhead']
    for pop in ('VAL', 'TEST', 'EXT'):
        for sn, slab in strata:
            S = R[pop]['strata'][sn]
            for k in contrasts:
                c = S['contrasts'][k]
                if c == 'not_available' or (k.startswith(('3s', '4s')) and sn != 'ALL'):
                    continue
                a, b = k.split(':')[1].split('->')
                L.append(f"{pop} & {slab} & {name.get(a, a)} $\\to$ {name.get(b, b)} & {c['n_persons']} & {num(c['delta_ww'])} & {rci(c)} \\\\")
            L.append(r'\addlinespace[1pt]')
    L += [r'\bottomrule', r'\end{longtable}}']
    return '\n'.join(L)


def main():
    parts = {'TableCohort': t7.cohort_table().replace(r'\begin{table}[t]', r'\begin{table}[H]'),  # appendix in v8r1
              'TableFrozen': frozen_table(), 'TableModels': models_table(), 'TableCentral': central_table(),
             'TableRtwoFull': fix_longtable(t7.r2_table()), 'TableRthreeB': fix_longtable(full_table()), 'TableWtwoS': t7.w2s_table(),
             'TablePzeroPone': t7.p0b_p1_table()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('% GENERATED by scripts/paper/make_tables_paper_v8r1.py -- do not edit by hand\n'
                   + '\n\n'.join(f'\\newcommand{{\\{k}}}{{%\n{v}}}' for k, v in parts.items()) + '\n')
    print('written', OUT)


if __name__ == '__main__':
    main()
