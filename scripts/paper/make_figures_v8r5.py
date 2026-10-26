#!/usr/bin/env python3
"""H-EVAL v8r5 redesigned figures 1-8, drawn ONLY from existing, git-tracked, hash-recorded aggregate result JSONs.

No model is fitted, no prediction is recomputed, no interval is re-estimated. Derived quantities are limited to
exact arithmetic on stored values (stage shares C_1/Delta_WW; C_s = total x stored share for the TRAIN-CV rows),
and every plotted value is written to a ledger.

Outputs
  paper/figures/v8r5_fig{1..8}_*.{pdf,png}
  paper/figures/provenance/{figure_provenance_v8r5.json, plotted_values_v8r5.csv}
Paper-only release: inputs are the aggregate-only snapshots in results/aggregates (see MANIFEST.json).
Shared style: figstyle_v8r5.py.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle_v8r5 as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RD = 'results/aggregates/'
OUT = REPO / 'paper/figures'
PROV = REPO / 'paper/figures/provenance'
INPUTS = {
    'r1': RD + 'repair_r1/r1_stage_construct.json',
    'r2': RD + 'repair_r2/r2_stage_results.json',
    'r3b': RD + 'repair_r3b/r3b_results_v1.json',
    't1': RD + 'gate_t1/execution_v1/gate_t1_confirmatory_result_v1.json',
    'x1': RD + 'gate_x1/execution_v1/gate_x1_result_v1.json',
    'cth': RD + 'cth_v1/cth_cv_results_v1.json',
    'p0a': RD + 'discovery_w2s/w2s_p0a_diagnostic_v1.json',
    'p0b': RD + 'discovery_w2s/w2s_p0b_calibration_v1.json',
    'p1': RD + 'discovery_w2s/w2s_p1_results_v1.json',
    'ci': RD + 'discovery_w2s/w2s_v7r1_descriptive_ci.json',
    'cv': RD + 'interp_v1/interp_partB_cv_results_v1.json',
    'pert': RD + 'interp_v1/interp_partA_perturbation_v1.json',
    'fu': RD + 'interp_v1/interp_followup_F1_F2_v1.json',
}
# Every input snapshot must match its SHA-256 in results/aggregates/MANIFEST.json (which also records the SHA-256
# of the frozen source file it was exported from).
MANIFEST = REPO / 'results/aggregates/MANIFEST.json'

LEDGER, CLIPS = [], S.ClipLog()


def sha(rel):
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def load():
    man = {f['snapshot']: f['snapshot_sha256'] for f in json.loads(MANIFEST.read_text())['files']}
    J = {}
    for k, rel in INPUTS.items():
        if sha(rel) != man[rel[len(RD):]]:
            raise SystemExit(f'STOP: {rel} differs from results/aggregates/MANIFEST.json')
        d = json.loads((REPO / rel).read_text())
        J[k] = d.get('content', d)
    return J


def rec(fig, panel, row, src, path, point, lo=None, hi=None, unit='R (%)', interval='', status=''):
    LEDGER.append(dict(figure=fig, panel=panel, row=row, source=INPUTS.get(src, src), json_path=path, point=point,
                       lo='' if lo is None else lo, hi='' if hi is None else hi, unit=unit, interval=interval, status=status))


def pct(c, k='R', ci='R_ci95'):
    return 100 * c[k], 100 * c[ci][0], 100 * c[ci][1]


def cohort_marker(ax, x, y, cohort, color, filled=True, size=None, z=4, transform=None, clip_on=True):
    ms = size or S.MSIZE[cohort]
    kw = dict(marker=S.MARK[cohort], ms=ms, ls='', color=color, mec=color, mew=0.9, zorder=z, clip_on=clip_on)
    kw['mfc'] = color if filled else 'white'
    if transform is not None:
        kw['transform'] = transform
    ax.plot([x], [y], **kw)


def forest_row(ax, y, v, lo, hi, color, cohort, xlim, filled=True, ctx='', lw=1.3):
    S.draw_interval(ax, y, lo, hi, xlim, color, lw=lw, clip_log=CLIPS, label_ctx=ctx)
    cohort_marker(ax, v, y, cohort, color, filled=filled)


def vline0(ax, x=0, orient='v'):
    (ax.axvline if orient == 'v' else ax.axhline)(x, color=S.INK2, lw=0.7, zorder=1)


def light_grid(ax, axis='x'):
    ax.grid(axis=axis, color=S.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------------------------- Fig 1
def fig1(_):
    """Illustrative timeline (no data). Epoch k covers [30k, 30k+30) s; decision at t = 30k uses epoch k-1.
    Colour version (author-approved): states use S.STATE_FILL (lightness-ordered tints) with thin borders."""
    F = S.STATE_FILL
    fs, fm = 7.5, 8.0                                     # regular text / labels carrying mathematical notation
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 2.12))
    ax = fig.add_axes([0.005, 0.02, 0.99, 0.96])
    n_ep, T1, T2, ret_w, ex = 32, 14, 22, {17}, 19        # example decision k=19 -> input epoch 18 (Stage 1)
    state = ['W' if (k < T1 or k in ret_w) else ('1' if k < T2 else '2') for k in range(n_ep)]
    y0, h = 0.0, 0.55
    k = 0
    while k < n_ep:                                       # contiguous scored-state blocks, thin border (greyscale cue)
        j = k
        while j < n_ep and state[j] == state[k]:
            j += 1
        ax.add_patch(Rectangle((k, y0), j - k, h, facecolor=F[state[k]], edgecolor=S.INK2, lw=0.5, zorder=2))
        k = j
    for k in range(1, n_ep):                              # epoch boundaries as short ticks (text stays clean)
        for yy in ((y0, y0 + 0.10), (y0 + h - 0.10, y0 + h)):
            ax.plot([k, k], yy, color='white', lw=0.8, zorder=3)
    for x, txt, col in ((7, 'W (scored wake)', S.INK), (15.5, 'N1', S.INK), (17.5, 'W', S.INK), (20.5, 'N1', S.INK),
                        (27, 'N2 (first Stage 2 from $T_2$)', 'white')):
        ax.text(x, y0 + h / 2, txt, ha='center', va='center', fontsize=fs, color=col, zorder=4)
    ax.add_patch(Rectangle((ex - 1, y0 - 0.035), 1, h + 0.07, fill=False, edgecolor=S.INK, lw=1.5, zorder=5))
    for x, txt, ha in ((0, 'lights-off', 'left'), (T1, '$T_1$: first N1', 'center'), (T2, '$T_2$: first N2', 'center')):
        ax.plot([x, x], [y0 - 0.02, y0 - 0.12], color=S.INK2, lw=0.7)
        ax.text(x, y0 - 0.16, txt, ha=ha, va='top', fontsize=fm if '$' in txt else fs)
    ax.text(n_ep, y0 - 0.16, 'time (30-s epochs)', ha='right', va='top', fontsize=fs, color=S.INK2)
    yd = 0.86
    for k in range(1, T2):
        s_in = state[k - 1]
        big = k == ex
        ax.plot([k], [yd], marker='v', ms=6.6 if big else 5.0, ls='', mfc=F[s_in],
                mec=S.INK if big else S.INK2, mew=1.2 if big else 0.6, zorder=5)

    def bracket(x0, x1, y, txt):
        ax.plot([x0, x0, x1, x1], [y - 0.07, y, y, y - 0.07], color=S.INK, lw=0.8, zorder=3, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, y + 0.05, txt, ha='center', va='bottom', fontsize=fm)
    bracket(1, T2 - 1, 1.62, r'risk set $\mathcal{A}_0$: decision points every 30 s while $t < T_2$')
    bracket(1, T2 - 11, 1.13, r'$Y_t=0$')
    bracket(T2 - 10, T2 - 1, 1.13, r'$Y_t=1$: first N2 within 300 s')
    ye = -0.82
    ax.plot([ex, ex], [ye, yd - 0.12], color=S.INK, lw=0.8, ls=(0, (2, 1.5)), zorder=4)
    ax.annotate('', xy=(ex + 10, ye), xytext=(ex, ye),
                arrowprops=dict(arrowstyle='-|>', color=S.INK, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.plot([T2], [ye], marker='*', ms=11, color=F['2'], mec='white', mew=0.7, zorder=6)
    ax.text(ex + 0.35, ye - 0.13, 'horizon $(t,\\, t+300$ s$]$\ncontains $T_2$, so $Y_t=1$', ha='left', va='top',
            fontsize=fm, linespacing=1.2)
    ax.text(ex - 0.25, ye, 'decision time $t$; input = preceding 30 s\n(outlined epoch, scored N1) ',
            ha='right', va='center', fontsize=fm, linespacing=1.15)
    lx = 23.4
    for i, (s_in, txt) in enumerate((('W', 'input scored W'), ('1', 'input scored Stage 1'))):
        yy = 1.70 - i * 0.31
        ax.plot([lx], [yy], marker='v', ms=5.0, ls='', mfc=F[s_in], mec=S.INK2, mew=0.6)
        ax.text(lx + 0.45, yy, txt, va='center', fontsize=fs)
    ax.text(lx - 0.25, 2.02, 'decision point', va='bottom', fontsize=fs, color=S.INK2)
    ax.text(0, 2.12, 'Illustrative timeline (not data)', fontsize=S.FS['small'], style='italic', color=S.INK2, va='bottom')
    ax.set_xlim(-0.3, n_ep + 0.3)
    ax.set_ylim(-1.52, 2.42)
    ax.axis('off')
    rec('Fig1', '-', 'schematic', 'illustrative', 'n/a', 'h=300 s; 30-s epochs; T1=14, T2=22 epochs; example k=19',
        unit='illustrative', status='not data')
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 2
def fig2(J):
    r1, r2 = J['r1']['share_of_total_gain_vs_T0_by_stage'], J['r2']['populations']
    rows = []
    for lab, k, colk in (('SpecLR recipe', 'M0', 'SpecLR-46'), ('RawCNN recipe', 'M2', 'RawCNN-46')):
        tot = r1[k]['total_delta_ww']
        cw, c1 = tot * r1[k]['W'], tot * r1[k]['1']
        rows.append(('TRAIN', lab, colk, cw, c1, tot, f'share_of_total_gain_vs_T0_by_stage.{k}', 'r1'))
    keys = {'VAL': [('SpecLR-18', 'CONSTANT->SpecLR-18 [EEG_LOGIT]')],
            'TEST': [('SpecLR-18', 'CONSTANT->SpecLR-18 [E7B_EEG_LOGIT]'), ('SpecLR-46', 'CONSTANT->SpecLR-46 [M0_46]'),
                     ('RawCNN-46', 'CONSTANT->RawCNN-46 [M2_46]')],
            'EXT': [('SpecLR-18', 'CONSTANT->SpecLR-18 [E7B_EEG_LOGIT]'), ('SpecLR-46', 'CONSTANT->SpecLR-46 [M0_46]'),
                    ('RawCNN-46', 'CONSTANT->RawCNN-46 [M2_46]')]}
    for pop, lst in keys.items():
        for lab, key in lst:
            sc = r2[pop]['stage_contribution'][key]
            cw, c1, tot = sc['absolute']['W'], sc['absolute']['1'], sc['delta_ww_all']
            assert abs(cw + c1 - tot) < 1e-12, (pop, key)
            assert abs(sc['share']['1'] - c1 / tot) < 1e-12, (pop, key)
            rows.append((pop, lab, lab, cw, c1, tot, f'populations.{pop}.stage_contribution["{key}"]', 'r2'))
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 2.75))
    ax = fig.add_axes([0.29, 0.16, 0.555, 0.74])
    ypos, y, prev = [], 0, None
    for r in rows:
        if prev is not None and r[0] != prev:
            y -= 0.5
        ypos.append(y)
        prev = r[0]
        y -= 1
    xlim = (-0.0045, 0.064)
    hb = 0.30                                              # half-row bar height (two-level waterfall)
    for (pop, lab, colk, cw, c1, tot, path, src), y in zip(rows, ypos):
        ax.add_patch(Rectangle((0, y + 0.02), cw, hb, facecolor=S.STATE['W'], edgecolor=S.INK2, lw=0.5, zorder=3))
        ax.add_patch(Rectangle((cw, y - 0.02 - hb), c1, hb, facecolor=S.STATE['1'], edgecolor=S.INK2, lw=0.5, zorder=3))
        ax.plot([cw, cw], [y - 0.02, y + 0.02], color=S.INK2, lw=0.5, zorder=3)
        ax.plot([tot, tot], [y - 0.42, y + 0.42], color=S.INK, lw=1.6, zorder=4, solid_capstyle='butt')
        share = 100 * c1 / tot
        ax.text(1.02, y, f'{share:.0f}%', transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=S.FS['tick'])
        ax.text(1.15, y, f'{tot:.4f}', transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=S.FS['tick'], color=S.INK2)
        cohort_marker(ax, -0.03, y, pop, S.COL[colk], transform=ax.get_yaxis_transform(), clip_on=False)
        status = 'development 5-fold CV' if pop == 'TRAIN' else 'retrospective attribution of original recorded predictions'
        rec('Fig2', '-', f'{pop} {lab} total', src, path, tot, unit='Delta_WW (Brier)', interval='point only', status=status)
        rec('Fig2', '-', f'{pop} {lab} C_W', src, path, cw, unit='Brier', interval='point only', status=status)
        rec('Fig2', '-', f'{pop} {lab} C_1', src, path, c1, unit='Brier', interval='point only', status=status)
        rec('Fig2', '-', f'{pop} {lab} Stage-1 share', src, path, share, unit='% of Delta_WW',
            interval='derived C_1/Delta_WW', status=status)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis='y', length=0, pad=13)
    for pop in ('TRAIN', 'VAL', 'TEST', 'EXT'):
        ys = [y for r, y in zip(rows, ypos) if r[0] == pop]
        lab = {'TRAIN': 'TRAIN\n5-fold CV', 'VAL': 'VAL', 'TEST': 'TEST', 'EXT': 'EXT'}[pop]
        ax.text(-0.505, sum(ys) / len(ys), lab, transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=S.FS['tick'], fontweight='bold', linespacing=1.05)
    for x, txt in ((1.02, 'Stage 1\nshare'), (1.15, 'total\n$\\Delta_{\\mathrm{WW}}$')):
        ax.text(x, ypos[0] + 0.6, txt, transform=ax.get_yaxis_transform(), ha='left', va='bottom',
                fontsize=S.FS['small'], color=S.INK2, linespacing=1.05)
    vline0(ax)
    light_grid(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(ypos[-1] - 0.6, ypos[0] + 0.6)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    ax.set_xticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06'])
    ax.set_xlabel(r'Contribution $C_s$ to the pooled Brier improvement over the constant (absolute Brier units)')
    handles = [Patch(facecolor=S.STATE['W'], edgecolor=S.INK2, lw=0.5, label=r'W-scored, $C_{\mathrm{W}}$'),
               Patch(facecolor=S.STATE['1'], edgecolor=S.INK2, lw=0.5, label=r'Stage 1-scored, $C_1$'),
               Line2D([], [], color=S.INK, lw=1.6, label=r'total $\Delta_{\mathrm{WW}}$')]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(xlim[1], ypos[2] + 0.35),
              bbox_transform=ax.transData, fontsize=S.FS['small'], handlelength=1.3, borderaxespad=0.0)
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 3
def r3(J, pop, sn, key):
    return J['r3b']['populations'][pop]['strata'][sn]['contrasts'][key]


def fig3(J):
    """Stacked cohort blocks on one shared axis. Each block lists only results that exist for that cohort: VAL scored
    SpecLR-18 only, so its block has no SpecLR-46 / RawCNN-46 rows (no placeholders, never zero)."""
    items = [('SpecLR-18', '0a:CONSTANT->SpecLR-18', False), ('SpecLR-46', '0b:CONSTANT->SpecLR-46', False),
             ('RawCNN-46', '0c:CONSTANT->RawCNN-46', False), ('Stage-only', '1:CONSTANT->A', True),
             ('Stage+time', '2:CONSTANT->B', True)]
    npers = {'VAL': 15, 'TEST': 15, 'EXT': 17}
    rows, y = [], 0.0                                      # (pop, label, oracle, contrast, y)
    bands, blocks = [], []
    for pop in ('VAL', 'TEST', 'EXT'):
        y0, prev_oracle, band_y = y, None, []
        for lab, key, oracle in items:
            c = r3(J, pop, 'ALL', key)
            path = f'populations.{pop}.strata.ALL.contrasts["{key}"]'
            if c == 'not_available':
                rec('Fig3', pop, lab, 'r3b', path, 'not_available',
                    status='not evaluated on this cohort; row omitted from the plot (no placeholder, not zero)')
                continue
            if prev_oracle is False and oracle:
                y -= 0.3                                   # small gap between EEG rows and reference rows
            rows.append((pop, lab, oracle, c, path, y))
            if oracle:
                band_y.append(y)
            prev_oracle = oracle
            y -= 1
        bands.append((min(band_y) - 0.5, max(band_y) + 0.5))
        blocks.append((pop, y0, y + 1))
        y -= 0.65                                          # gap between cohort blocks
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 2.5))
    ax = fig.add_axes([0.255, 0.155, 0.715, 0.83])
    xlim = (-12, 62)
    for lo_b, hi_b in bands:
        ax.axhspan(lo_b, hi_b, color=S.ORACLE_BAND, zorder=0, lw=0)
    light_grid(ax)
    vline0(ax)
    for pop, lab, oracle, c, path, yy in rows:
        v, lo, hi = pct(c)
        forest_row(ax, yy, v, lo, hi, S.COL[lab], pop, xlim, filled=not oracle, ctx=f'Fig3 {pop} {lab}')
        rec('Fig3', pop, lab, 'r3b', path, v, lo, hi, interval=c.get('interval_label', 'R3B re-bootstrap'),
            status='retrospective (post-hoc) re-bootstrap of original predictions' + ('; oracle reference' if oracle else ''))
    ax.set_yticks([r[5] for r in rows])
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    for pop, ytop, ybot in blocks:
        ax.text(-0.345, (ytop + ybot) / 2, f'{pop}\n{npers[pop]} persons', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=S.FS['tick'], linespacing=1.15, fontweight='bold')
    ax.set_xlim(*xlim)
    ax.set_ylim(rows[-1][5] - 0.6, 0.6)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel('Brier skill relative to the constant, $R$ (%); 0 = constant, > 0 = better')
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 4
def fig4(J):
    """RawCNN-46 was not evaluated on VAL: its VAL rows are omitted (recorded in the ledger), never drawn as empty or zero."""
    cols = [('vs Stage-only', 'all inputs', 'ALL', 'A'), ('vs Stage+time', 'all inputs', 'ALL', 'B'),
            ('vs Stage+time', 'Stage 1-scored inputs', 'STAGE_1', 'B'), ('vs Stage+time', 'W-scored inputs', 'W', 'B')]
    keys = {'SpecLR-18': {'A': '3:A->SpecLR-18', 'B': '4:B->SpecLR-18'},
            'RawCNN-46': {'A': '7:A->RawCNN-46', 'B': '8:B->RawCNN-46'}}
    rows = [('SpecLR-18', 'VAL', 4.9), ('SpecLR-18', 'TEST', 3.9), ('SpecLR-18', 'EXT', 2.9),
            ('RawCNN-46', 'TEST', 1.2), ('RawCNN-46', 'EXT', 0.2)]
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 2.05))
    xlim = (-100, 62)
    left, w, gap, extra = 0.195, 0.176, 0.016, 0.03
    axes = [fig.add_axes([left + i * (w + gap) + (extra if i >= 2 else 0), 0.245, w, 0.535]) for i in range(4)]
    for ax, (t1, t2, sn, ref) in zip(axes, cols):
        light_grid(ax)
        vline0(ax)
        key = keys['RawCNN-46'][ref]
        assert r3(J, 'VAL', sn, key) == 'not_available', 'RawCNN-46 VAL unexpectedly available'
        rec('Fig4', f'{t1} {t2}', 'RawCNN-46 VAL', 'r3b', f'populations.VAL.strata.{sn}.contrasts["{key}"]', 'not_available',
            status='not evaluated on VAL; row omitted from the plot (no placeholder, not zero)')
        for mdl, pop, y in rows:
            key = keys[mdl][ref]
            c = r3(J, pop, sn, key)
            path = f'populations.{pop}.strata.{sn}.contrasts["{key}"]'
            v, lo, hi = pct(c)
            S.draw_interval(ax, y, lo, hi, xlim, S.COL[mdl], clip_log=CLIPS, label_ctx=f'Fig4 {t1} {t2} {mdl} {pop}',
                            label_side='below', label_offset=0.30)
            cohort_marker(ax, v, y, pop, S.COL[mdl])
            rec('Fig4', f'{t1} {t2}', f'{mdl} {pop}', 'r3b', path, v, lo, hi,
                interval=c.get('interval_label', 'R3B re-bootstrap'),
                status='retrospective (post-hoc), descriptive; no multiplicity adjustment')
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.45, 5.45)
        ax.set_xticks([-80, -40, 0, 40])
        ax.set_title(f'{t1}\n{t2}', fontsize=S.FS['tick'], linespacing=1.1)
        ax.set_yticks([y for _, _, y in rows])
        ax.set_yticklabels([f'{m}  {p}' for m, p, _ in rows] if ax is axes[0] else [])
        ax.tick_params(axis='y', length=0)
        if ax is not axes[0]:
            ax.spines['left'].set_visible(False)
    for i0, i1, txt in ((0, 1, 'All scored inputs'), (2, 3, 'Within the scored input state')):
        x0, x1 = axes[i0].get_position().x0, axes[i1].get_position().x1
        fig.add_artist(Line2D([x0 + 0.004, x1 - 0.004], [0.92, 0.92], color=S.INK2, lw=0.6))
        fig.text((x0 + x1) / 2, 0.93, txt, ha='center', va='bottom', fontsize=S.FS['tick'], fontweight='bold')
    xc = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.text(xc, 0.083, 'Incremental Brier skill of the EEG model over the scored-stage reference, $R$ (%)',
             ha='center', va='bottom', fontsize=S.FS['base'])
    fig.text(xc, 0.012, '< 0: reference better  ·  > 0: EEG better', ha='center', va='bottom',
             fontsize=S.FS['small'], color=S.INK2)
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 5
def fig5(J):
    x1, t1 = J['x1']['results'], J['t1']['results']
    st = x1['condition_strata_descriptive']
    rows = [('SpecLR-18 vs constant, 28 nights', x1['H1']['scores']['primary'], 'SpecLR-18', True,
             'primary', 'prespecified primary', 'results.H1.scores.primary'),
            ('placebo nights (15)', st['placebo']['E7B_EEG_vs_CONSTANT'], 'SpecLR-18', True, 'subgroup',
             'descriptive subgroup', 'results.condition_strata_descriptive.placebo.E7B_EEG_vs_CONSTANT'),
            ('temazepam nights (13)', st['temazepam']['E7B_EEG_vs_CONSTANT'], 'SpecLR-18', True, 'subgroup',
             'descriptive subgroup', 'results.condition_strata_descriptive.temazepam.E7B_EEG_vs_CONSTANT'),
            ('RawCNN-46 vs SpecLR-18', x1['H2'], 'RawCNN-46', True, 'descriptive†',
             'descriptive (conditional test not reached)', 'results.H2'),
            ('RawCNN-46 vs SpecLR-46', x1['H3'], 'RawCNN-46', True, 'descriptive', 'descriptive', 'results.H3'),
            ('donor-EEG SpecLR-18 vs constant', x1['H1']['scores']['NC2_donor'], 'Constant', False, 'control',
             'negative control', 'results.H1.scores.NC2_donor'),
            ('donor-EEG RawCNN-46 vs constant', x1['m2_negative_controls']['donor_raw_vs_constant'], 'Constant', False,
             'control', 'negative control', 'results.m2_negative_controls.donor_raw_vs_constant')]
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 3.45))
    a = fig.add_axes([0.335, 0.50, 0.32, 0.43])
    xlim = (-32, 36)
    light_grid(a)
    vline0(a)
    a.axvline(5, color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    a.text(6.0, 0.05, '5% criterion\n(primary only)', fontsize=S.FS['small'], color=S.INK2, va='center', ha='left',
           linespacing=1.0)
    for i, (lab, c, colk, filled, tag, status, path) in enumerate(rows):
        y = len(rows) - 1 - i
        v, lo, hi = pct(c)
        forest_row(a, y, v, lo, hi, S.COL[colk], 'EXT', xlim, filled=filled, ctx=f'Fig5a {lab}')
        a.text(1.03, y, f'{v:.1f} ({lo:.1f}, {hi:.1f})'.replace('-', '−'), transform=a.get_yaxis_transform(),
               va='center', ha='left', fontsize=S.FS['small'])
        a.text(1.58, y, tag, transform=a.get_yaxis_transform(), va='center', ha='left', fontsize=S.FS['small'],
               color=S.INK2, style='italic')
        rec('Fig5', 'a', lab, 'x1', path, v, lo, hi, interval='original X1 procedure, 95% person-cluster bootstrap',
            status=status)
    a.text(1.03, len(rows) - 0.35, '$R$ (95% CI)', transform=a.get_yaxis_transform(), va='bottom', ha='left',
           fontsize=S.FS['small'], color=S.INK2)
    a.text(1.58, len(rows) - 0.35, 'status', transform=a.get_yaxis_transform(), va='bottom', ha='left',
           fontsize=S.FS['small'], color=S.INK2)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[0] for r in reversed(rows)])
    a.tick_params(axis='y', length=0)
    a.set_xlim(*xlim)
    a.set_ylim(-0.6, len(rows) - 0.4)
    a.set_xlabel('$R$ (%) of first-named vs comparator; > 0 favours first-named', fontsize=S.FS['tick'])
    fig.text(0.012, 0.945, 'a', fontsize=S.FS['panel'], fontweight='bold', va='bottom')
    fig.text(0.045, 0.945, 'External cohort EXT (17 persons, 28 nights): Brier skill', fontsize=S.FS['base'],
             va='bottom')
    b = fig.add_axes([0.335, 0.075, 0.43, 0.245])
    names = [('SpecLR-18', 'E7B_EEG'), ('SpecLR-46', 'M0_46'), ('RawCNN-46', 'M2_46')]
    light_grid(b, 'y')
    vline0(b, 0, orient='h')
    for j, (lab, k) in enumerate(names):
        vt, vx = t1['models'][k]['cal_in_large'], x1['models'][k]['cal_in_large']
        b.annotate('', xy=(j + 0.17, vx), xytext=(j - 0.17, vt),
                   arrowprops=dict(arrowstyle='-|>', color=S.INK2, lw=0.6, mutation_scale=6, shrinkA=3.5, shrinkB=3.5))
        cohort_marker(b, j - 0.17, vt, 'TEST', S.COL[lab])
        cohort_marker(b, j + 0.17, vx, 'EXT', S.COL[lab])
        for xx, vv, ha in ((j - 0.26, vt, 'right'), (j + 0.26, vx, 'left')):
            b.text(xx, vv, f'{vv:+.3f}'.replace('-', '−'), ha=ha, va='center', fontsize=S.FS['small'],
                   color=S.INK2, bbox=dict(fc='white', ec='none', pad=0.3), zorder=5)
        rec('Fig5', 'b', f'{lab} TEST', 't1', f'results.models.{k}.cal_in_large', vt, unit='mean p - event rate',
            interval='point only', status='prespecified secondary, descriptive')
        rec('Fig5', 'b', f'{lab} EXT', 'x1', f'results.models.{k}.cal_in_large', vx, unit='mean p - event rate',
            interval='point only', status='prespecified secondary, descriptive')
    b.set_xticks(range(3))
    b.set_xticklabels([n for n, _ in names])
    b.set_xlim(-0.7, 2.95)
    b.set_ylim(-0.16, 0.06)
    b.set_yticks([-0.15, -0.10, -0.05, 0, 0.05])
    b.set_yticklabels(['−0.15', '−0.10', '−0.05', '0', '0.05'])
    b.set_ylabel('mean prediction\n− event rate', fontsize=S.FS['tick'], linespacing=1.05)
    fig.text(0.012, 0.345, 'b', fontsize=S.FS['panel'], fontweight='bold', va='bottom')
    fig.text(0.045, 0.345, 'Calibration-in-the-large, TEST → EXT', fontsize=S.FS['base'], va='bottom')
    h = [Line2D([], [], marker=S.MARK['TEST'], ls='', color=S.INK2, ms=4.6, label='TEST'),
         Line2D([], [], marker=S.MARK['EXT'], ls='', color=S.INK2, ms=3.9, label='EXT'),
         Line2D([], [], marker=S.MARK['EXT'], ls='', color=S.COL['Constant'], mfc='white', ms=3.9,
                label='donor-EEG control')]
    h += [Line2D([], [], color=S.COL[n], lw=2.2, label=n) for n in ('SpecLR-18', 'SpecLR-46', 'RawCNN-46')]
    fig.legend(handles=h, loc='lower left', bbox_to_anchor=(0.80, 0.05), fontsize=S.FS['small'], ncol=1,
               handlelength=1.2, borderaxespad=0.2)
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 6
def fig6(J):
    p0a, p0b, p1, lead = J['p0a'], J['p0b'], J['p1'], J['ci']['lead_time']
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 4.05))
    a = fig.add_axes([0.10, 0.625, 0.40, 0.285])
    b = fig.add_axes([0.665, 0.625, 0.32, 0.285])
    c = fig.add_axes([0.49, 0.14, 0.47, 0.325])
    vnames = [('V1_300s_pooled', 'pooled'), ('V2_300s_within_fold', 'within fold'),
              ('V3_30s_exact_pooled', 'pooled'), ('V4_30s_exact_within_fold_PRIMARY', 'within fold')]
    xs = [0, 1, 2.4, 3.4]
    light_grid(a, 'y')
    a.axhline(0.5, color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for x, (v, _) in zip(xs, vnames):
        for m, col, dx, filled in (('TIME_pop', S.COL['Time-only'], -0.13, False), ('N3_tcn5min_state', S.COL['TCN'], 0.13, True)):
            e = p0a['variants'][v]['models'][m]
            S.draw_interval(a, x + dx, e['ci95'][0], e['ci95'][1], (0.2, 1.0), col, orient='v', clip_log=CLIPS,
                            label_ctx=f'Fig6a {v} {m}', fmt=lambda z: f'{z:.2f}')
            a.plot([x + dx], [e['auroc']], marker='o', ms=4.6, ls='', color=col, mfc=col if filled else 'white',
                   mew=1.0, zorder=4)
            rec('Fig6', 'a', f'{v} {m}', 'p0a', f'variants.{v}.models.{m}', e['auroc'], e['ci95'][0], e['ci95'][1],
                unit='AUROC', interval='95% person-cluster bootstrap (P0a)', status='TRAIN only, exploratory')
    a.text(3.4 - 0.13, 0.455, '0.5 by\nconstruction', ha='center', va='top', fontsize=S.FS['small'], color=S.INK2,
           linespacing=1.0)
    a.set_xticks(xs)
    a.set_xticklabels([n for _, n in vnames])
    for x0, x1, t in ((0, 1, '300 s time bins'), (2.4, 3.4, 'exact 30 s landmark')):
        a.text((x0 + x1) / 2, -0.25, t, transform=a.get_xaxis_transform(), ha='center', va='top', fontsize=S.FS['tick'])
        a.plot([x0 - 0.3, x1 + 0.3], [-0.215, -0.215], transform=a.get_xaxis_transform(), color=S.INK2, lw=0.6,
               clip_on=False)
    a.text(3.4, 0.83, 'primary', ha='center', va='bottom', fontsize=S.FS['small'], color=S.INK2)
    a.set_xlim(-0.5, 3.9)
    a.set_ylim(0.2, 1.0)
    a.set_ylabel('cross-person AUROC')
    a.legend(handles=[Line2D([], [], marker='o', ls='', color=S.COL['TCN'], ms=4.6, label='TCN-5m+stage (EEG)'),
                      Line2D([], [], marker='o', ls='', color=S.COL['Time-only'], mfc='white', ms=4.6, mew=1.0,
                             label='time-only reference')],
             loc='upper left', fontsize=S.FS['small'], handletextpad=0.2, borderaxespad=0.1)
    S.panel_label(a, 'a', x=-0.215, y=1.04, ha='left')
    a.text(-0.12, 1.04, 'Matching scheme and fold pooling', transform=a.transAxes, fontsize=S.FS['base'], va='bottom')
    keys = list(lead)
    labs = ['0–60', '60–120', '120–180', '180–300']
    light_grid(b, 'y')
    b.axhline(0.5, color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i, k in enumerate(keys):
        e = lead[k]
        S.draw_interval(b, i, e['ci95'][0], e['ci95'][1], (0.4, 1.0), S.COL['TCN'], orient='v', clip_log=CLIPS,
                        label_ctx=f'Fig6b {k}', fmt=lambda z: f'{z:.2f}')
        b.plot([i], [e['auroc']], marker='o', ms=4.6, ls='', color=S.COL['TCN'], zorder=4)
        b.text(i, 0.415, f'n={e["n_positive_windows"]}', ha='center', va='bottom', fontsize=S.FS['small'], color=S.INK2)
        rec('Fig6', 'b', k, 'ci', f'lead_time["{k}"]', e['auroc'], e['ci95'][0], e['ci95'][1], unit='AUROC',
            interval='95% person-cluster bootstrap (descriptive follow-up)', status='TRAIN only, exploratory')
    b.set_xticks(range(len(keys)))
    b.set_xticklabels(labs)
    b.set_xlabel('time before first N1 (s)')
    b.set_xlim(-0.5, len(keys) - 0.5)
    b.set_ylim(0.4, 1.0)
    b.set_ylabel('AUROC vs far negatives\n(> 600 s before N1)', linespacing=1.05)
    S.panel_label(b, 'b', x=-0.36, y=1.04, ha='left')
    b.text(-0.25, 1.04, 'Lead time (TCN-5m+stage)', transform=b.transAxes, fontsize=S.FS['base'], va='bottom')
    rows = [('TCN-5m+stage, uncalibrated', p0b['results']['vs_time']['N3_tcn5min_state|K0'], 'TCN', 'A', 'K0'),
            ('TCN + intercept recalibration', p0b['results']['vs_time']['N3_tcn5min_state|K1'], 'TCN', 'A', 'K1'),
            ('TCN + slope recalibration', p0b['results']['vs_time']['N3_tcn5min_state|K2'], 'TCN', 'A', 'K2'),
            ('TCN (5 min, stage head) + time recalibration', p0b['results']['vs_time']['N3_tcn5min_state|K3'], 'TCN', 'A', 'K3')]
    for part in ('A', 'B'):
        v = p1['partitions'][part]['results']['vs_time']
        rows += [(f'Baseline + time recalibration ({part})', v['A|K3'], 'Personalised', part, 'A|K3'),
                 (f'Combined + time recalibration ({part})', v['E|K3'], 'Personalised', part, 'E|K3')]
    xlim = (-25, 22)
    light_grid(c)
    vline0(c)
    ys = [8.0, 7.0, 6.0, 5.0, 3.0, 2.0, 1.0, 0.0]
    for (lab, v, colk, part, key), y in zip(rows, ys):
        val, lo, hi = pct(v)
        S.draw_interval(c, y, lo, hi, xlim, S.COL[colk], clip_log=CLIPS, label_ctx=f'Fig6c {lab}')
        mk = 'o' if (colk == 'TCN' or part == 'A') else 's'   # personalised rows: circle = partition A, square = B
        c.plot([val], [y], marker=mk, ms=4.6 if mk == 'o' else 4.2, ls='', color=S.COL[colk],
               mfc=S.COL[colk] if colk == 'TCN' else 'white', mew=1.0, zorder=4)
        src = 'p0b' if colk == 'TCN' else 'p1'
        path = f'results.vs_time["N3_tcn5min_state|{key}"]' if colk == 'TCN' else f'partitions.{part}.results.vs_time["{key}"]'
        rec('Fig6', 'c', lab, src, path, val, lo, hi, interval='95% person-cluster bootstrap (nested cross-fitting)',
            status='TRAIN only, exploratory')
    c.set_yticks(ys)
    c.set_yticklabels([r[0] for r in rows])
    c.tick_params(axis='y', length=0)
    c.set_xlim(*xlim)
    c.set_ylim(-0.6, 9.55)
    for yy, t, colk in ((8.85, 'Recalibration of TCN-5m+stage (partition A)', 'TCN'),
                        (3.85, 'Personalisation, time-recalibrated (\u25cb A, \u25a1 B)', 'Personalised')):
        c.text(-1.02, yy, t, transform=c.get_yaxis_transform(), ha='left', va='center', fontsize=S.FS['small'],
               fontweight='bold', color=S.COL[colk])
    c.set_xlabel('Brier skill relative to the within-E1 time reference, $R$ (%)')
    c.text(0.0, -0.215, '← time reference better', transform=c.transAxes, ha='left', va='top',
           fontsize=S.FS['small'], color=S.INK2)
    c.text(1.0, -0.215, 'model better →', transform=c.transAxes, ha='right', va='top', fontsize=S.FS['small'],
           color=S.INK2)
    c.xaxis.set_label_coords(0.5, -0.30)
    S.panel_label(c, 'c', x=-1.02, y=1.0, ha='left')
    c.text(-0.97, 1.0, 'Brier skill after nested, leakage-safe recalibration', transform=c.transAxes,
           fontsize=S.FS['base'], va='bottom')
    fig.text(0.985, 0.985, 'TRAIN only (46 persons) · exploratory', ha='right', va='top', fontsize=S.FS['small'],
             style='italic', color=S.INK2, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=S.INK2, lw=0.5))
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 7
def fig7(J):
    C = J['cth']['comparisons']
    groups = [('component', [
        ('Remove raw-waveform branch', 'A2_no_raw__vs__M3_cthnet'),
        ('Remove 5-min trajectory', 'A1_no_trajectory__vs__M3_cthnet'),
        ('30-s instead of 5-min history', 'A4_history_30s__vs__M3_cthnet'),
        ('Binary instead of hazard head', 'A3_binary_head__vs__M3_cthnet'),
        ('Add elapsed-time fusion', 'A5_plus_time__vs__M3_cthnet')]),
        ('channel', [('Fpz-Cz only', 'A6a_fpz_only__vs__M3_cthnet'), ('Pz-Oz only', 'A6b_pz_only__vs__M3_cthnet')]),
        ('simplified', [('Plain multiscale CNN (RawCNN)', 'M2_raw_cnn_binary__vs__M3_cthnet')])]
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 1.68))
    ax = fig.add_axes([0.345, 0.225, 0.43, 0.69])
    xlim = (-20, 25)
    light_grid(ax)
    vline0(ax)
    y, ys, labs, seps = 0.0, [], [], []
    for gi, (g, items) in enumerate(groups):
        if gi:
            seps.append(-y + 0.55)
        for lab, k in items:
            v, lo, hi = pct(C[k])
            col = S.COL['RawCNN-46'] if k.startswith('M2_') else S.INK2
            S.draw_interval(ax, -y, lo, hi, xlim, col, clip_log=CLIPS, label_ctx=f'Fig7 {lab}')
            ax.plot([v], [-y], marker='o', ms=4.0, ls='', color=col, zorder=4)
            ax.text(1.02, -y, f'{v:.1f} ({lo:.1f}, {hi:.1f})'.replace('-', '\u2212'), transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=S.FS['small'])
            rec('Fig7', '-', lab, 'cth', f'comparisons.{k}', v, lo, hi,
                interval='95% person-cluster bootstrap, 5-fold TRAIN CV', status='development CV (TRAIN only)')
            ys.append(-y)
            labs.append(lab)
            y += 1
        y += 0.1
    for sy in seps:                                         # thin rules between groups (component | channel | simplified)
        ax.plot([-0.78, 1.32], [sy, sy], transform=ax.get_yaxis_transform(), color=S.GRID, lw=0.8, clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels(labs)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(min(ys) - 0.6, 0.6)
    ax.spines['left'].set_visible(False)
    ax.text(1.02, 1.0, '$R$ (95% CI)', transform=ax.transAxes, ha='left', va='bottom', fontsize=S.FS['small'],
            color=S.INK2)
    ax.set_xlabel('\u2190 variant better      $R$ (%) of the full predeclared network vs each variant      full network better \u2192',
                  fontsize=S.FS['small'])
    return fig


# ---------------------------------------------------------------------------------------------------------------- Fig 8
def fig8(J):
    Cc, M = J['cv']['comparisons'], J['cv']['models']
    P, F2 = J['pert']['conditions'], J['fu']['F2']
    col = {'M2': S.COL['RawCNN-46'], 'STFT': S.COL['STFT-CNN'], 'M0': S.COL['SpecLR-46']}
    lab = {'M2': 'RawCNN (CV)', 'STFT': 'STFT-CNN (CV)', 'M0': 'SpecLR recipe (CV)'}
    # single-cohort figure: marker shape is a redundant model cue (greyscale: STFT orange ~ SpecLR sky-blue luminance)
    mk = {'M2': 'o', 'STFT': 's', 'M0': '^'}
    ms = {'M2': 4.2, 'STFT': 3.8, 'M0': 4.4}
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 3.85))
    a = fig.add_axes([0.355, 0.665, 0.30, 0.285])
    b = fig.add_axes([0.80, 0.665, 0.185, 0.285])
    c = fig.add_axes([0.105, 0.175, 0.47, 0.31])
    d = fig.add_axes([0.66, 0.175, 0.325, 0.31])
    rows = [('STFT-CNN vs RawCNN', 'M2_raw_cnn__vs__STFT_cnn', 'STFT'),
            ('STFT-CNN vs SpecLR', 'M0_outer_train__vs__STFT_cnn', 'STFT'),
            ('RawCNN vs SpecLR', 'M0_outer_train__vs__M2_raw_cnn', 'M2'),
            ('STFT-CNN vs SpecLR (inner-train fit)', 'M0_inner_train__vs__STFT_cnn', 'STFT'),
            ('RawCNN vs SpecLR (inner-train fit)', 'M0_inner_train__vs__M2_raw_cnn', 'M2')]
    xlim = (-25, 27)
    light_grid(a)
    vline0(a)
    for i, (lb, k, m) in enumerate(rows):
        ref, cand = k.split('__vs__')
        bref, bcand = M[ref]['brier_ww'], M[cand]['brier_ww']
        assert abs((bref - bcand) / bref - Cc[k]['R']) < 1e-9, k          # proves reference/candidate orientation
        v, lo, hi = pct(Cc[k])
        y = len(rows) - 1 - i
        S.draw_interval(a, y, lo, hi, xlim, col[m], clip_log=CLIPS, label_ctx=f'Fig8a {lb}')
        a.plot([v], [y], marker=mk[m], ms=ms[m], ls='', color=col[m], zorder=4)
        rec('Fig8', 'a', lb, 'cv', f'comparisons.{k}', v, lo, hi, interval='95% person-cluster bootstrap, 5-fold TRAIN CV',
            status='post-hoc, development data; label = candidate vs reference')
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[0] for r in reversed(rows)])
    a.tick_params(axis='y', length=0)
    a.set_xlim(*xlim)
    a.set_ylim(-0.6, len(rows) - 0.4)
    a.set_xlabel('$R$ (%) of candidate (first-named)\nvs reference; > 0: candidate better', linespacing=1.05)
    S.panel_label(a, 'a', x=-0.96, y=1.03, ha='left')
    a.text(-0.89, 1.03, 'Matched 5-fold TRAIN CV (46 persons)', transform=a.transAxes, fontsize=S.FS['base'], va='bottom')
    names = [('M0', 'M0_outer_train', 'SpecLR'), ('STFT', 'STFT_cnn', 'STFT'), ('M2', 'M2_raw_cnn', 'RawCNN')]
    light_grid(b, 'y')
    for j, (m, k, _) in enumerate(names):
        b.plot([j], [M[k]['brier_ww']], marker=mk[m], ms=ms[m] + 0.4, ls='', color=col[m], zorder=4)
        b.plot([j], [M[k]['brier_se']], marker=mk[m], ms=ms[m] + 0.4, ls='', color=col[m], mfc='white', mew=1.0, zorder=4)
        rec('Fig8', 'b', f'{k} window-weighted', 'cv', f'models.{k}.brier_ww', M[k]['brier_ww'], unit='Brier',
            interval='point only', status='post-hoc, development data')
        rec('Fig8', 'b', f'{k} subject-equal', 'cv', f'models.{k}.brier_se', M[k]['brier_se'], unit='Brier',
            interval='point only', status='post-hoc, development data')
    b.set_xticks(range(3))
    b.set_xticklabels([n for _, _, n in names], rotation=30, ha='right', rotation_mode='anchor')
    b.set_xlim(-0.5, 2.5)
    b.set_ylim(0.09, 0.19)
    b.set_yticks([0.10, 0.12, 0.14, 0.16, 0.18])
    b.set_ylabel('Brier (lower is better)')
    k0 = names[0][1]                                        # direct labels instead of a legend
    b.annotate('subject-equal', xy=(0, M[k0]['brier_se']), xytext=(0.22, M[k0]['brier_se'] + 0.006),
               fontsize=S.FS['small'], color=S.INK2, va='bottom', ha='left')
    b.annotate('window-\nweighted', xy=(0, M[k0]['brier_ww']), xytext=(0.16, M[k0]['brier_ww'] + 0.008),
               fontsize=S.FS['small'], color=S.INK2, va='bottom', ha='left', linespacing=0.95)
    S.panel_label(b, 'b', x=-0.55, y=1.03, ha='left')
    b.text(-0.40, 1.03, 'Brier by estimand', transform=b.transAxes, fontsize=S.FS['base'], va='bottom')

    def grouped(ax, groups, getter, ylim, tag, side):
        w = 0.25
        for j, m in enumerate(('M2', 'STFT', 'M0')):
            for i, g in enumerate(groups):
                v, lo, hi, path, src = getter(g, m)
                x = i + (j - 1) * w
                S.draw_interval(ax, x, lo, hi, ylim, col[m], orient='v', clip_log=CLIPS, label_ctx=f'Fig8{tag} {g} {m}',
                                fmt=lambda z: f'{z:.2f}'.replace('-', '−'), label_side=side.get((g, m), 'right'),
                                label_offset=0.10)
                if v < ylim[0] or v > ylim[1]:
                    raise SystemExit(f'point outside axis: {tag} {g} {m}')
                ax.plot([x], [v], marker=mk[m], ms=ms[m] - 0.3, ls='', color=col[m], zorder=4)
                rec('Fig8', tag, f'{g} {lab[m]}', src, path, v, lo, hi, unit='skill retention rho',
                    interval='95% person-cluster bootstrap', status='post-hoc, development data (out-of-fold)')
        ax.axhline(1, color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0, color=S.INK2, lw=0.7, zorder=1)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(groups)))
        ax.set_xlim(-0.55, len(groups) - 0.45)

    def rho(cnd, m):
        return cnd[m]['retention'], cnd[m]['retention_ci95'][0], cnd[m]['retention_ci95'][1]

    bands = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma']
    light_grid(c, 'y')
    grouped(c, bands, lambda g, m: (*rho(P[f'P3_{g}_s1.0'], m), f'conditions.P3_{g}_s1.0.{m}', 'pert'), (-1.6, 1.25), 'c',
            side={('beta', 'STFT'): 'left', ('beta', 'M0'): 'right'})
    c.set_xticklabels(['δ\n0.5–4', 'θ\n4–8', 'α\n8–12', 'σ\n12–15',
                       'β\n15–30', 'γ\n30–50 Hz'], linespacing=1.0)
    c.set_ylabel('skill retention ρ')
    fig.text(0.012, 0.497, 'c', fontsize=S.FS['panel'], fontweight='bold', va='bottom')
    fig.text(0.045, 0.497, 'Band removed (out-of-fold)', fontsize=S.FS['base'], va='bottom')
    cond = {'P1': ('conditions.P1_fourier_surrogate', P['P1_fourier_surrogate'], 'pert'),
            'P2': ('conditions.P2_local_surrogate_2s', P['P2_local_surrogate_2s'], 'pert'),
            'P6': ('conditions.P6_delta_phase', P['P6_delta_phase'], 'pert'),
            'F2f': ('F2.F2f_delta_phase_fpz_only', F2['F2f_delta_phase_fpz_only'], 'fu')}
    light_grid(d, 'y')
    grouped(d, list(cond), lambda g, m: (*rho(cond[g][1], m), f'{cond[g][0]}.{m}', cond[g][2]), (-0.2, 1.25), 'd', side={})
    d.set_xticklabels(['30-s\nFourier', '2-s\nlocal', 'δ phase\n(both)', 'δ phase\n(Fpz)†'],
                      linespacing=1.0, fontsize=S.FS['small'])
    fig.text(0.60, 0.497, 'd', fontsize=S.FS['panel'], fontweight='bold', va='bottom')
    fig.text(0.633, 0.497, 'Power kept, structure destroyed', fontsize=S.FS['base'], va='bottom')
    h = [Line2D([], [], marker=mk[m], color=col[m], lw=1.3, ms=ms[m] - 0.3, label=lab[m]) for m in ('M2', 'STFT', 'M0')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.45, 0.005), ncol=3, fontsize=S.FS['small'],
               handlelength=1.6)
    fig.text(0.985, 0.012, '† post-hoc follow-up', ha='right', va='bottom', fontsize=S.FS['small'], color=S.INK2)
    return fig


FIGS = [('v8r5_fig1_task_schematic', fig1), ('v8r5_fig2_stage_attribution', fig2),
        ('v8r5_fig3_reference_comparison', fig3), ('v8r5_fig4_incremental_skill', fig4),
        ('v8r5_fig5_external_calibration', fig5), ('v8r5_fig6_w2s_exploratory', fig6),
        ('v8r5_fig7_architecture_ablations', fig7), ('v8r5_fig8_representation_perturbation', fig8)]


def main():
    S.setup()
    J = load()
    OUT.mkdir(parents=True, exist_ok=True)
    PROV.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, fn in FIGS:
        fig = fn(J)
        paths = S.save(fig, OUT, name)
        made[name] = {ext: dict(path=str(p.relative_to(REPO)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for ext, p in paths.items()}
    prov = dict(
        script='scripts/paper/make_figures_v8r5.py',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        style_sha256=hashlib.sha256((Path(__file__).parent / 'figstyle_v8r5.py').read_bytes()).hexdigest(),
        matplotlib=plt.matplotlib.__version__,
        inputs={k: dict(path=rel, sha256=sha(rel), git_tracked=True) for k, rel in INPUTS.items()},
        outputs=made, clipped_interval_ends=list(CLIPS), n_plotted_values=len(LEDGER),
        note='All values read from the listed frozen JSONs; no model fitted, no interval re-estimated.')
    (PROV / 'figure_provenance_v8r5.json').write_text(json.dumps(prov, indent=1) + '\n')
    with open(PROV / 'plotted_values_v8r5.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(LEDGER[0]))
        w.writeheader()
        w.writerows(LEDGER)
    print(json.dumps({k: v['pdf']['sha256'][:12] for k, v in made.items()}, indent=1))
    print('clipped ends:', json.dumps(list(CLIPS)))


if __name__ == '__main__':
    main()
