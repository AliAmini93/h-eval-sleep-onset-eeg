"""Decision windows: labels, scored input stage S(t) and EEG extraction for one recording.

A decision at time t uses only the EEG window [t-30 s, t). The event is the first scored Stage 2 epoch (T2);
Y_t = 1{T2 - t <= 300 s}. The scored stage of the input epoch is used only to define post-hoc strata, never as input.

Lineage (research archive, tag v0.1.0-paper-review): stage_events / attribute / first_sleep / check_alignment / labels
from the post-hoc stage audit R2; input_stages_and_first_sleep from the TRAIN stage construction R1 (used by the
wake-to-N1 table); extract_recording from the CTH v1 data builder (incl. rich features) and the T1/X1 extractors.
Logic unchanged.
"""
from pathlib import Path

import numpy as np

from . import edf
from .cohort import HORIZON_S, LOOKBACK_S, decision_times, stage_at
from ..models.ablation_networks import rich_features
from ..models.spectral import channel_features

EPS = 1e-6
SLEEP = {'1', '2', '3', '4', 'R'}


def labels(off, n2, elapsed):
    """Elapsed decision times (s after lights-off) -> (labels Y_t, absolute times t)."""
    t = off + np.asarray(elapsed, float)
    return ((n2 - t) <= HORIZON_S + 1e-8).astype(int), t


def stage_events(events):
    """(onset, end, label) for 'Sleep stage X' annotations; label = X upper-cased."""
    out = []
    for onset, dur, lab in events:
        if lab.lower().startswith('sleep stage '):
            out.append((float(onset), float(onset) + float(dur), lab[len('sleep stage '):].strip().upper()))
    return sorted(out)


def attribute(stages, a, b):
    """Scored stage of the input epoch [a, b): a single label, 'MIXED:x|y', 'UNSCORED' or 'AMBIGUOUS'."""
    ov = [(max(s, a), min(e, b), lab) for s, e, lab in stages if min(e, b) - max(s, a) > EPS]
    if not ov:
        return 'UNSCORED'
    ov.sort()
    for i in range(len(ov)):
        for j in range(i + 1, len(ov)):
            if ov[j][0] < ov[i][1] - EPS and ov[i][2] != ov[j][2]:
                return 'AMBIGUOUS'
    cur = a
    for s, e, _ in ov:                       # coverage check (same-label overlaps allowed)
        if s > cur + EPS:
            return 'UNSCORED'
        cur = max(cur, e)
    if cur < b - EPS:
        return 'UNSCORED'
    labs = []
    for _, _, lab in ov:
        if lab not in labs:
            labs.append(lab)
    return labs[0] if len(labs) == 1 else 'MIXED:' + '|'.join(labs)


def first_sleep(stages, off):
    """Onset of the first scored sleep epoch (1, 2, 3, 4 or R) at or after lights-off, or None."""
    c = [max(s, off) for s, e, lab in stages if lab in SLEEP and e > off + EPS]
    return min(c) if c else None


def check_alignment(stages, off, n2):
    """A Stage 2 annotation must start exactly at T2 and none may overlap [off, T2)."""
    if not any(lab == '2' and abs(s - n2) <= EPS for s, e, lab in stages):
        raise ValueError(f'no stage-2 annotation starts at n2={n2}')
    for s, e, lab in stages:
        if lab == '2' and min(e, n2) - max(s, off) > EPS:
            raise ValueError(f'stage 2 inside [off, n2) at {s}')


def input_stages_and_first_sleep(events, off, n2, elapsed):
    """TRAIN construction used by the wake-to-N1 table: stage of each input epoch (must be a single scored stage over
    the window) and the first-any-sleep and first-Stage-1 times after lights-off (30 s stepping)."""
    stage = []
    for el in np.asarray(elapsed, float):
        t = off + el
        a, b = stage_at(events, t - 30), stage_at(events, t - 0.5)
        if a != b:
            raise ValueError(f'input window spans two stages at t={t}')
        stage.append(a)
    tt, first_any, first_s1 = off, None, None
    while tt < n2 + 1e-6:
        s = stage_at(events, tt)
        if s in ('1', '2', '3', '4', 'R') and first_any is None:
            first_any = tt - off
        if s == '1' and first_s1 is None:
            first_s1 = tt - off
        tt += 30.0
    if stage_at(events, n2) != '2':
        raise ValueError('first N2 time is not scored Stage 2')
    return np.array(stage, dtype=str), first_any, first_s1


def extract_recording(psg_path, off, n2):
    """Two-channel [Fpz-Cz, Pz-Oz] windows for every decision point of one recording.

    Returns dict with raw (N, 2, 3000) float32 microvolts, spectral (N, 10) and rich (N, 18) features (both from the
    float64 samples), elapsed_s, time_to_n2_s, y (300 s horizon) and anchor (T2 - t <= 1200 s)."""
    path = Path(psg_path)
    landmarks = decision_times(off, n2)
    raw, feats, rich = [], [], []
    with path.open('rb') as f:
        h = edf.signal_header(f, path.stat().st_size)
        if [ch[1]['label'] for ch in h['eeg_channels']] != list(edf.EEG_CHANNELS):
            raise ValueError('Unexpected channel order')
        if n2 > h['duration_s'] or min(landmarks) - 30 < 0:
            raise ValueError('Decision window outside the EEG signal')
        for t in landmarks:
            chans, x = [], []
            for ch in h['eeg_channels']:
                samples, _ = edf.read_window(f, h, ch, t - 30, t)
                chans.append(samples)
                x.extend(channel_features(samples))
            raw.append(np.asarray(chans, dtype=np.float32))
            feats.append(x)
            rich.append([v for c in chans for v in rich_features(np.array(c))])
    t = np.array(landmarks)
    return dict(raw=np.stack(raw), spectral=np.array(feats, float), rich=np.array(rich), elapsed_s=t - off, time_to_n2_s=n2 - t,
                y=(n2 - t <= HORIZON_S + 1e-8).astype(int), anchor=(n2 - t <= LOOKBACK_S + 1e-8).astype(int))
