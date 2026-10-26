"""H-EVAL v8r5 figure design system (shared by make_figures_v8r5.py and qa_figures_v8r5.py).

Principles (see docs/paper/h_eval_editorial_v8r5/figure_redesign/DESIGN_SPEC.md):
- every figure is drawn at its final printed size (NeurIPS text width 5.5 in), so font sizes in this file are printed sizes;
- one colour = one model or reference in every figure; cohort is carried by marker shape; oracle (scored-stage)
  references are open markers on a shaded band; scored sleep state is carried by grey lightness;
- no interval is ever clipped silently: draw_interval() annotates every clipped end with an arrowhead and its value.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TEXTWIDTH_IN = 5.5          # NeurIPS 2026 \linewidth
FONT = 'Liberation Sans'    # metric-compatible Helvetica/Arial clone, TrueType (embedded as Type 42)
FS = dict(base=8.0, tick=7.5, small=7.0, panel=9.0)
MIN_FONT_PT = 7.0

# Okabe-Ito based semantic palette (Okabe & Ito 2008); identity is also given by labels/markers.
COL = {
    'SpecLR-18': '#0072B2',     # blue
    'SpecLR-46': '#56B4E9',     # sky blue (also: SpecLR recipe in 46-person TRAIN CV)
    'RawCNN-46': '#D55E00',     # vermillion (also: RawCNN recipe in TRAIN CV)
    'Stage-only': '#009E73',    # bluish green, oracle reference
    'Stage+time': '#CC79A7',    # reddish purple, oracle reference
    'Constant': '#7A7A7A',      # grey, original reference
    'Time-only': '#7A7A7A',     # grey, time reference (exploratory analyses)
    'STFT-CNN': '#E69F00',      # orange (development interpretation analysis only)
    'TCN': '#882255',           # wine (Tol muted), exploratory wake-to-N1 temporal EEG model (Fig 6 only)
    'Personalised': '#332288',  # indigo (Tol muted), exploratory personalised variants (Fig 6 only)
}
INK, INK2, GRID = '#1A1A1A', '#4D4D4D', '#E6E6E6'
ORACLE_BAND = '#F2F2F2'
# Scored sleep state: ordinal grey lightness (CVD- and greyscale-safe, distinct from model hues); used in Fig 2.
STATE = {'W': '#E3E3E3', '1': '#9A9A9A', '2': '#3A3A3A'}
# Fig 1 schematic (author-approved colour version): low-chroma tints, lightness-ordered W > N1 > N2 (L* 92/71/27), so the
# order survives greyscale and CVD; none of these hues is a model colour.
STATE_FILL = {'W': '#F6E7B4', '1': '#9AB0D0', '2': '#2C3E66'}
# Cohort encoding by marker shape, used in every figure that shows several cohorts.
MARK = {'TRAIN': '^', 'VAL': 's', 'TEST': 'o', 'EXT': 'D'}
MSIZE = {'TRAIN': 4.6, 'VAL': 4.0, 'TEST': 4.6, 'EXT': 3.9}


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic', 'mathtext.bf': f'{FONT}:bold',
        'mathtext.cal': 'cmsy10',
        'font.size': FS['base'], 'axes.titlesize': FS['base'], 'axes.labelsize': FS['base'],
        'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['tick'],
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlepad': 3, 'axes.labelpad': 2,
        'legend.frameon': False, 'legend.handletextpad': 0.4, 'legend.borderaxespad': 0.2, 'legend.columnspacing': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.unicode_minus': True,
    })


def panel_label(ax, letter, x=-0.02, y=1.0, transform=None, ha='right'):
    ax.text(x, y, letter, transform=transform or ax.transAxes, fontsize=FS['panel'], fontweight='bold',
            ha=ha, va='bottom', color=INK)


def fmt_pct(v):
    """Signed percentage with a typographic minus."""
    return f'{v:.0f}'.replace('-', '−')


class ClipLog(list):
    """Records every clipped interval end so the provenance ledger can prove nothing was clipped silently."""


def draw_interval(ax, y, lo, hi, xlim, color, lw=1.3, ls='-', clip_log=None, label_ctx='', orient='h', z=2,
                  fmt=None, label_side=None, label_offset=0.32):
    """Draw an interval [lo, hi]; ends outside xlim are cut at the limit, marked with an arrowhead and labelled with
    their true value ('to -133'). orient='v' draws along y (then `y` is the x position).
    label_side: 'above'/'below' (orient h) or 'left'/'right' (orient v). Returns list of clipped ends."""
    fmt = fmt or fmt_pct
    a, b = xlim
    clipped = []
    lo_d, hi_d = max(lo, a), min(hi, b)
    if orient == 'h':
        ax.plot([lo_d, hi_d], [y, y], color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=z)
    else:
        ax.plot([y, y], [lo_d, hi_d], color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=z)
    span = b - a
    for end, val, lim, sgn in (('lower', lo, a, -1), ('upper', hi, b, 1)):
        if (sgn < 0 and val < lim) or (sgn > 0 and val > lim):
            clipped.append((end, val))
            arrow = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=7, shrinkA=0, shrinkB=0)
            if orient == 'h':
                ax.annotate('', xy=(lim, y), xytext=(lim - sgn * 0.06 * span, y), arrowprops=arrow, zorder=z + 1,
                            annotation_clip=False)
                side = label_side or 'above'
                ax.text(lim - sgn * 0.01 * span, y + (label_offset if side == 'above' else -label_offset),
                        f'to {fmt(val)}', ha='left' if sgn < 0 else 'right', va='bottom' if side == 'above' else 'top',
                        fontsize=FS['small'], color=color, zorder=z + 1)
            else:
                ax.annotate('', xy=(y, lim), xytext=(y, lim - sgn * 0.08 * span), arrowprops=arrow, zorder=z + 1,
                            annotation_clip=False)
                side = label_side or 'right'
                ax.text(y + (label_offset if side == 'right' else -label_offset), lim - sgn * 0.03 * span,
                        f'to {fmt(val)}', ha='left' if side == 'right' else 'right', va='center',
                        fontsize=FS['small'], color=color, zorder=z + 1)
            if clip_log is not None:
                clip_log.append(dict(context=label_ctx, end=end, true_value=val, axis_limit=lim))
    return clipped


def save(fig, out_dir, name, dpi_png=300):
    paths = {}
    for ext in ('pdf', 'png'):
        p = out_dir / f'{name}.{ext}'
        meta = {'CreationDate': None, 'ModDate': None, 'Producer': None} if ext == 'pdf' else None
        fig.savefig(p, dpi=dpi_png, metadata=meta)
        paths[ext] = p
    plt.close(fig)
    return paths
