"""Sleep-EDF cohort construction: lights-off anchors, eligibility, person-disjoint split and analysis rosters.

Lineage (research archive, tag v0.1.0-paper-review), arithmetic and rules unchanged:
  recording pairing ............ E metadata audit (pairs_create)
  lights-off parsing, risk profile, eligibility status ... E2
  stage at lights-off, salted person split ................ E3
  SpecLR-18 training nights ..... E4 v3 latency-spread pilot (6 persons) + E6 hash-ranked expansion (12 persons)
  TRAIN / VAL / TEST / EXT rosters ... CTH v1 data builder, E7B, T1 and X1 roster builders
The original pipeline read PSG headers over HTTP ranges; here they are read from a local copy of the dataset.
"""
import collections
import hashlib
import math
import re
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path

from . import edf
from .xls import read_sheets

ELIGIBLE = 'ELIGIBLE_ORIGIN_INDEPENDENT'
SPLIT_SALT = 'H-EVAL-SC-subject-groups-20261022-v1'
HORIZON_S, LOOKBACK_S = 300., 1200.
MILESTONES = (120, 300, 600)
SPLIT_LABELS = ('SC_TRAIN', 'SC_VALIDATION', 'SC_PROTECTED_MODEL_TEST', 'ST_EXTERNAL_RESERVED')
PSG_RE = re.compile(r'^(sleep-cassette/SC|sleep-telemetry/ST)([0-9A-Za-z]{6})-PSG\.edf$')
HYPNO_RE = re.compile(r'^(sleep-cassette/SC|sleep-telemetry/ST)([0-9A-Za-z]{6})-Hypnogram\.edf$')
NIGHTS = {'sleep-cassette': 153, 'sleep-telemetry': 44}
# SpecLR-18 training-night selection (E4 pilot + E6 expansion)
PILOT_COUNT, PILOT_EACH_CAP = 6, 70_000_000
EXPANSION_SALT, EXPANSION_N, EXPANSION_BUDGET = 'H-EVAL-E6-SC-TRAIN-expansion-20261023-v1', 12, 850_000_000


# ------------------------------------------------------------------------ official file inventory
def records_parse(raw):
    paths = [line.strip() for line in raw.decode('ascii').splitlines() if line.strip()]
    if len(paths) != 197 or len(set(paths)) != 197:
        raise ValueError('Expected 197 unique official PSG recordings')
    if not all(PSG_RE.fullmatch(p) for p in paths):
        raise ValueError('Unexpected PSG path in RECORDS')
    if collections.Counter(p.split('/')[0] for p in paths) != NIGHTS:
        raise ValueError('Unexpected study counts')
    return paths


def pair_recordings(paths, sums):
    """PSG paths + official checksums -> one row per recording with its unique hypnogram (person = first 5 chars)."""
    candidates = defaultdict(list)
    for name in sums:
        if HYPNO_RE.fullmatch(name):
            candidates[(name.split('/')[0], Path(name).name[:6])].append(name)
    out = []
    for p in paths:
        study, key = p.split('/')[0], Path(p).name[:6]
        hyp = candidates.get((study, key), [])
        if len(hyp) != 1:
            raise ValueError('Missing/ambiguous hypnogram for ' + p + ': ' + str(hyp))
        if p not in sums:
            raise ValueError('No official PSG SHA ' + p)
        out.append({'recording': key, 'subject_key': key[:5], 'study': study, 'night': key[5], 'psg_path': p,
                    'hypnogram_path': hyp[0]})
    if len({x['recording'] for x in out}) != len(out):
        raise ValueError('Duplicate recording identity')
    return sorted(out, key=lambda x: x['recording'])


# ------------------------------------------------------------------------ lights-off metadata (subject spreadsheets)
def norm(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).strip().lower()).strip('_')


def nint(x):
    if isinstance(x, bool):
        return None
    try:
        f = float(str(x).strip())
        if math.isfinite(f) and f == int(f) and f >= 0:
            return int(f)
    except (TypeError, ValueError):
        pass
    return None


def parse_clock(x):
    """Clock value (text such as 00:38, 22:15:00, 23:05h, or an Excel day fraction) -> seconds after midnight."""
    if isinstance(x, (dtime, datetime)):
        return x.hour * 3600 + x.minute * 60 + x.second
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        if not math.isfinite(x) or x < 0:
            return None
        frac = x % 1
        if frac == 0 and 0 <= x < 24 and isinstance(x, int):
            return None                                     # an integer is ambiguous as a clock time
        return round(frac * 86400, 3)
    s = str(x).strip()
    if not s or s.lower() in ('none', 'nan', 'missing', '?', '-'):
        return None
    m = re.fullmatch(r'\s*(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*(?:h|hr|hrs)?\s*', s, re.I)
    if m:
        hh, mm, ss = int(m[1]), int(m[2]), float(m[3]) if m[3] else 0
        if hh > 24 or (hh == 24 and (mm or ss)) or mm >= 60 or ss >= 60:
            return None
        return (hh % 24) * 3600 + mm * 60 + ss
    try:
        x = float(s)
        if not math.isfinite(x) or x < 0:
            return None
        return round((x % 1) * 86400, 3)
    except ValueError:
        return None


def identify_sc(sheets):
    """SC-subjects.xls grids -> {SC4ssN: lights-off record}."""
    possible = []
    for sh, grid in sheets.items():
        for ir, row in enumerate(grid[:18]):
            cols = {norm(x): idx for idx, x in enumerate(row)}
            sub = next((v for k, v in cols.items() if k in ('subject', 'subject_number', 'subject_nr')), None)
            night = next((v for k, v in cols.items() if k in ('night', 'night_nr', 'night_number')), None)
            off = next((v for k, v in cols.items() if k in ('lightsoff', 'lights_off', 'lights_out')), None)
            if all(v is not None for v in (sub, night, off)):
                possible.append((sh, grid, ir, sub, night, off))
    if len(possible) != 1:
        raise ValueError('Ambiguous/missing SC columns: subject, night, LightsOff: ' + str([x[0] for x in possible]))
    sh, grid, header, subcol, nightcol, offcol = possible[0]
    rows, seen = {}, 0
    for row in grid[header + 1:]:
        def col(i):
            return row[i] if i < len(row) else ''
        subject, night = nint(col(subcol)), nint(col(nightcol))
        if subject is None or night is None or not 0 <= subject <= 99 or night not in (1, 2):
            continue
        seen += 1
        key = f'SC4{subject:02}{night}'
        if key in rows:
            raise ValueError('Duplicate subject-night key in SC metadata: ' + key)
        rows[key] = {'anchor_clock_seconds': parse_clock(col(offcol)), 'anchor_source': 'SC-subjects.xls:LightsOff',
                     'condition': 'home_cassette'}
    if seen < 150:
        raise ValueError('SC metadata parsed too few subject/night rows: ' + str(seen))
    return rows, {'sheet': sh, 'column_headers_at_row': header + 1, 'metadata_rows': seen}


def identify_st(sheets):
    """ST-subjects.xls grids (two-level Placebo/Temazepam header) -> {ST7ssN: lights-off record}."""
    candidates = []
    for sh, grid in sheets.items():
        for r in range(min(16, len(grid) - 1)):
            first, second = [norm(x) for x in grid[r]], [norm(x) for x in grid[r + 1]]
            group, last = [], ''
            for v in first:
                if v:
                    last = v
                group.append(last)
            cols = [(group[i] if i < len(group) else '', v) for i, v in enumerate(second)]

            def match(group_part, sub_parts):
                return [i for i, (g, v) in enumerate(cols) if group_part in g and all(p in v for p in sub_parts)]
            sub = match('subject', ['nr'])
            p_night = match('placebo', ['night', 'nr']) or match('placebo', ['nr'])
            p_off = match('placebo', ['lights', 'off'])
            t_night = match('temazepam', ['night', 'nr']) or match('temazepam', ['nr'])
            t_off = match('temazepam', ['lights', 'off'])
            if all(len(x) == 1 for x in (sub, p_night, p_off, t_night, t_off)):
                candidates.append((sh, grid, r, [x[0] for x in (sub, p_night, p_off, t_night, t_off)]))
    if len(candidates) != 1:
        raise ValueError('Missing/ambiguous ST two-level Placebo/Temazepam lights-off columns: ' + str([x[0] for x in candidates]))
    sh, grid, h, cols = candidates[0]
    rows, seen = {}, 0
    for row in grid[h + 2:]:
        def v(i):
            return row[i] if i < len(row) else ''
        subject = nint(v(cols[0]))
        if subject is None or subject < 0 or subject > 99:
            continue
        seen += 1
        for night_i, off_i, condition in [(cols[1], cols[2], 'placebo'), (cols[3], cols[4], 'temazepam')]:
            night = nint(v(night_i))
            if night not in (1, 2):
                continue
            key = f'ST7{subject:02}{night}'
            if key in rows:
                raise ValueError('Duplicate ST subject/night key: ' + key)
            rows[key] = {'anchor_clock_seconds': parse_clock(v(off_i)), 'anchor_source': 'ST-subjects.xls:' + condition +
                         '_lights_off', 'condition': condition}
    if seen < 20:
        raise ValueError('ST metadata parsed too few subject rows: ' + str(seen))
    return rows, {'sheet': sh, 'column_headers_at_row': h + 1, 'metadata_rows': seen}


def match_time(header, anchor):
    """Lights-off clock (s after midnight) -> seconds after PSG start, or None if not uniquely inside the PSG."""
    if anchor is None or header['psg_duration_s'] >= 86400:
        return None
    t = (anchor - header['psg_clock_start_s']) % 86400
    return None if t > header['psg_duration_s'] - 30 else t


# ------------------------------------------------------------------------ eligibility and decision times
def stage_risk_profile(events, off, recording_duration_s, horizons=MILESTONES):
    """Hypnogram events + lights-off -> (status, counts, first N2 onset). ELIGIBLE needs an unbroken W/N1/... scoring
    from lights-off to the first N2 epoch and at least one 30 s window before it."""
    stages = sorted([(t, t + d, label) for t, d, label in events if label.lower().startswith('sleep stage ')], key=lambda z: z[0])
    if not stages:
        return 'NO_STAGES', {}, None
    if any(not math.isfinite(t) or not math.isfinite(b) or b <= t for t, b, _ in stages):
        return 'INVALID_STAGE_EPOCH', {}, None
    if off < 0 or off >= recording_duration_s:
        return 'OFF_OUT_OF_PSG', {}, None
    if any(t <= off < b and label.lower() == 'sleep stage 2' for t, b, label in stages):
        return 'N2_ACTIVE_AT_OFF', {}, None
    n2 = [t for t, b, label in stages if label.lower() == 'sleep stage 2' and off < t < recording_duration_s]
    if not n2:
        return 'NO_STAGE2_AFTER_OFF', {}, None
    first = min(n2)
    cur = off
    for t, b, label in stages:
        left, right = max(t, off), min(b, first)
        if right <= left:
            continue
        if label.lower() in ('sleep stage ?', 'sleep stage m', 'sleep stage unknown'):
            return 'UNKNOWN_STAGE_IN_RISK_SET', {}, first
        if left > cur + 0.05 or left < cur - 0.05:
            return 'SCORING_GAP_OR_OVERLAP', {}, first
        cur = max(cur, right)
    if cur < first - 0.05:
        return 'SCORING_GAP_OR_OVERLAP', {}, first
    if first - off < 30:
        return 'NO_30S_HISTORY_BEFORE_EVENT', {}, first
    landmarks = decision_times(off, first, strict=False)
    if not landmarks:
        return 'NO_LANDMARKS', {}, first
    summary = {'landmark_n': len(landmarks), 'anchored_lb1200_n': sum(first - t <= 1200 for t in landmarks),
               'latency_s': first - off}
    for h in horizons:
        summary[f'positive_h{h}_n'] = sum(first <= t + h + 1e-8 for t in landmarks)
        summary[f'negative_h{h}_n'] = len(landmarks) - summary[f'positive_h{h}_n']
    return ELIGIBLE, summary, first


def decision_times(off, n2, strict=True):
    """Decision points t (s after PSG start) on the absolute 30 s grid with [t-30, t) after lights-off and t < first N2."""
    if strict and (not math.isfinite(off) or not math.isfinite(n2) or not off + 30 < n2):
        raise ValueError('Unexpected LightsOff/N2 timing')
    first = math.ceil((off + 30) / 30 - 1e-9)
    last = math.floor(n2 / 30 + 1e-9)
    return [k * 30. for k in range(first, last + 1) if k * 30 < n2 - 1e-8]


def checked_decision_times(row):
    """decision_times() for a cohort row, cross-checked against the stored eligibility counts."""
    off, n2 = float(row['lights_off_rel_s']), float(row['first_n2_after_off_s'])
    tt = decision_times(off, n2)
    if len(tt) != int(row['landmark_n']):
        raise ValueError('Landmark count disagreement: ' + row['recording'])
    if sum(n2 - t <= LOOKBACK_S + 1e-8 for t in tt) != int(row['anchored_lb1200_n']) or \
            sum(n2 - t <= HORIZON_S + 1e-8 for t in tt) != int(row['positive_h300_n']):
        raise ValueError('Anchor or label reconstruction mismatch: ' + row['recording'])
    return tt


def stage_at(events, t):
    """Scored stage active at time t ('W', '1', ...), 'UNSCORED', 'AMBIGUOUS' or 'NO_ORIGIN'."""
    if t is None or not math.isfinite(t):
        return 'NO_ORIGIN'
    active = [lab.lower().removeprefix('sleep stage ').strip() for x, d, lab in events
              if lab.lower().startswith('sleep stage ') and x - 1.e-7 <= t < x + d - 1.e-7]
    return active[0].upper() if len(active) == 1 else ('AMBIGUOUS' if active else 'UNSCORED')


# ------------------------------------------------------------------------ person-disjoint split
def person_split(eligible_rows, salt=SPLIT_SALT):
    """Eligible rows (study, subject_key) -> {subject_key: split}. SC persons are ordered by sha256(salt|key): the first
    20% form the protected TEST split, the next 20% VAL, the rest TRAIN; every ST person is the external cohort."""
    by_study = defaultdict(set)
    for r in eligible_rows:
        by_study[r['study']].add(r['subject_key'])
    if len(by_study['sleep-cassette']) < 60 or len(by_study['sleep-telemetry']) < 10:
        raise ValueError('Too few groups for predeclared exploratory allocation')
    ordered = sorted(by_study['sleep-cassette'], key=lambda s: hashlib.sha256((salt + '|' + s).encode()).hexdigest())
    n = len(ordered)
    n_test, n_val = round(n * .20), round(n * .20)
    if not n > n_test + n_val + 25:
        raise ValueError('Insufficient development SC groups')
    split = {s: 'SC_PROTECTED_MODEL_TEST' for s in ordered[:n_test]}
    split.update({s: 'SC_VALIDATION' for s in ordered[n_test:n_test + n_val]})
    split.update({s: 'SC_TRAIN' for s in ordered[n_test + n_val:]})
    for s in by_study['sleep-telemetry']:
        if s in split:
            raise ValueError('SC-ST subject ID collision')
        split[s] = 'ST_EXTERNAL_RESERVED'
    return split


# ------------------------------------------------------------------------ building the cohort table from local files
def build_cohort(root, recordings=None):
    """Local Sleep-EDF v1.0.0 root -> list of per-recording eligibility rows (no EEG samples are read).

    recordings: optional iterable of recording codes to process (the default processes all 197). Every processed
    recording needs its hypnogram and PSG file locally; their official SHA-256 values are verified.
    """
    root = Path(root)
    sums = edf.parse_sha256sums((root / 'metadata/SHA256SUMS.txt').read_bytes())
    if edf.sha256_file(root / 'metadata/RECORDS') != sums.get('RECORDS'):
        raise ValueError('Official RECORDS checksum mismatch')
    pairs = pair_recordings(records_parse((root / 'metadata/RECORDS').read_bytes()), sums)
    if recordings is not None:
        wanted = set(recordings)
        pairs = [p for p in pairs if p['recording'] in wanted]
        if {p['recording'] for p in pairs} != wanted:
            raise ValueError('Unknown recording code(s): ' + ', '.join(sorted(wanted - {p['recording'] for p in pairs})))
    meta = {}
    for fn, reader in (('SC-subjects.xls', identify_sc), ('ST-subjects.xls', identify_st)):
        p = root / 'metadata' / fn
        if edf.sha256_file(p) != sums.get(fn):
            raise ValueError('Official metadata checksum mismatch ' + fn)
        meta.update(reader(read_sheets(p))[0])
    rows = []
    for row in pairs:
        out = dict(row)
        mp = meta.get(row['recording'][:6])
        if not mp:
            rows.append(out | {'status': 'NO_MATCHED_SUBJECT_NIGHT'})
            continue
        out.update(condition=mp['condition'], metadata_source=mp['anchor_source'], lights_off_clock_s=mp['anchor_clock_seconds'])
        if mp['anchor_clock_seconds'] is None:
            rows.append(out | {'status': 'LIGHTS_OFF_MISSING_OR_INVALID'})
            continue
        for key in ('psg_path', 'hypnogram_path'):
            if edf.sha256_file(root / row[key]) != sums[row[key]]:
                raise ValueError('Official checksum mismatch ' + row[key])
        hdr = edf.read_psg_clock_header(root / row['psg_path'])
        off = match_time(hdr, mp['anchor_clock_seconds'])
        out.update(psg_duration_s=hdr['psg_duration_s'], psg_size_bytes=hdr['psg_size_bytes'], lights_off_rel_s=off)
        if off is None:
            rows.append(out | {'status': 'OFF_CLOCK_OUTSIDE_PSG'})
            continue
        events, _ = edf.decode_edfplus_annotations((root / row['hypnogram_path']).read_bytes())
        status, extra, first = stage_risk_profile(events, off, hdr['psg_duration_s'])
        out.update(status=status, first_n2_after_off_s=first, stage_at_lights_off=stage_at(events, off), **extra)
        rows.append(out)
    return rows


# ------------------------------------------------------------------------ analysis rosters
def _eligible_w(row, split, label, study):
    return (split.get(row['subject_key']) == label and row['status'] == ELIGIBLE and row['study'] == study
            and row.get('stage_at_lights_off') == 'W')


def roster(rows, split, cohort):
    """Recordings analysed per cohort (all require ELIGIBLE status and stage W at lights-off).
    TRAIN: all nights of SC_TRAIN persons. VAL/TEST: role PRIMARY = lowest night per person, SENSITIVITY = others.
    EXT: all ST nights (placebo and temazepam)."""
    label, study = {'TRAIN': ('SC_TRAIN', 'sleep-cassette'), 'VAL': ('SC_VALIDATION', 'sleep-cassette'),
                    'TEST': ('SC_PROTECTED_MODEL_TEST', 'sleep-cassette'),
                    'EXT': ('ST_EXTERNAL_RESERVED', 'sleep-telemetry')}[cohort]
    by = defaultdict(list)
    for r in rows:
        if _eligible_w(r, split, label, study):
            by[r['subject_key']].append(r)
    out = []
    for s in sorted(by):
        for i, r in enumerate(sorted(by[s], key=lambda r: (int(r['night']), r['recording']))):
            role = 'PRIMARY' if cohort in ('TRAIN', 'EXT') or i == 0 else 'SENSITIVITY'
            out.append(dict(r, role=role))
    return out


def speclr18_training_nights(rows, split):
    """The 18 one-night TRAIN persons used to fit SpecLR-18: six persons spread over the observed TRAIN latency
    distribution (outcome-dependent pilot selection) plus up to twelve persons ranked by a salted hash of the person
    code within a download budget. Requires psg_size_bytes in the rows."""
    eligible = []
    for r in rows:
        if not _eligible_w(r, split, 'SC_TRAIN', 'sleep-cassette'):
            continue
        lat = float(r['latency_s'])
        if not math.isfinite(lat) or lat < 360 or int(r['landmark_n']) < 10 or not 0 < int(r['psg_size_bytes']) <= PILOT_EACH_CAP:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: (float(r['latency_s']), r['recording']))
    if len({r['subject_key'] for r in eligible}) < PILOT_COUNT:
        raise ValueError('Insufficient distinct training subjects')
    pilot, used = [], set()
    for k in range(PILOT_COUNT):
        target = round((len(eligible) - 1) * (k + 0.5) / PILOT_COUNT)
        cand = sorted(range(len(eligible)), key=lambda i: (abs(i - target), i))
        row = next(eligible[i] for i in cand if eligible[i]['subject_key'] not in used)
        pilot.append(row)
        used.add(row['subject_key'])
    by_subject = defaultdict(list)
    for r in rows:
        if r['subject_key'] not in used and _eligible_w(r, split, 'SC_TRAIN', 'sleep-cassette'):
            by_subject[r['subject_key']].append(r)
    cands = sorted((sorted(rr, key=lambda r: (int(r['night']), r['recording']))[0] for rr in by_subject.values()),
                   key=lambda r: hashlib.sha256((EXPANSION_SALT + '|' + r['subject_key']).encode()).hexdigest())
    expansion, total, stopped = [], 0, False
    for c in cands:
        if len(expansion) == EXPANSION_N or stopped:
            continue
        b = int(c['psg_size_bytes'])
        if b > PILOT_EACH_CAP or total + b > EXPANSION_BUDGET:
            stopped = True                                   # budget prefix stop: later ranks are never used
            continue
        expansion.append(c)
        total += b
    return pilot + expansion
