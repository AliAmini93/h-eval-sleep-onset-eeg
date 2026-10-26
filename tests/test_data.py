"""Data layer: decision windows, labels, scored input stage, eligibility, person split, EDF/EDF+ reading."""
import numpy as np
import pytest

from heval.data import cohort, edf, windows
from heval.data.xls import read_sheets

EV = [(0.0, 1300.0, 'Sleep stage W'), (1300.0, 100.0, 'Sleep stage 1'), (1400.0, 60.0, 'Sleep stage W'),
      (1460.0, 140.0, 'Sleep stage 1'), (1600.0, 600.0, 'Sleep stage 2'), (500.0, 0.0, 'Lights off')]


# ------------------------------------------------------------------------ risk set, horizon, input stage
def test_decision_times_grid_and_risk_set():
    t = cohort.decision_times(1015.0, 1600.0)
    assert t[0] == 1050.0                   # first full 30 s window after lights-off on the absolute 30 s grid
    assert t[-1] == 1590.0 and all(x < 1600.0 for x in t)    # risk set t < T2
    assert np.all(np.diff(t) == 30.0)
    assert cohort.decision_times(0.0, 95.0) == [30.0, 60.0, 90.0]


def test_first_n2_horizon_labels():
    y, t = windows.labels(1000.0, 1600.0, [30, 300, 330, 570])
    assert list(y) == [0, 1, 1, 1]          # T2 - t = 570, 300 (boundary included), 270, 30
    assert list(t) == [1030.0, 1300.0, 1330.0, 1570.0]


def test_checked_decision_times_rejects_count_mismatch():
    row = dict(recording='R', lights_off_rel_s=1000.0, first_n2_after_off_s=1600.0, landmark_n=19, anchored_lb1200_n=19,
               positive_h300_n=10)
    assert len(cohort.checked_decision_times(row)) == 19
    with pytest.raises(ValueError, match='Landmark count'):
        cohort.checked_decision_times(dict(row, landmark_n=20))
    with pytest.raises(ValueError, match='LightsOff/N2'):
        cohort.decision_times(1000.0, 1020.0)


def test_scored_input_stage_of_window():
    st = windows.stage_events(EV)
    assert windows.attribute(st, 1200, 1230) == 'W'
    assert windows.attribute(st, 1300, 1330) == '1'
    assert windows.attribute(st, 1290, 1320) == 'MIXED:W|1'          # boundary windows are not relabelled
    assert windows.attribute(st, 490, 520) == 'W'                    # non-stage annotations ignored
    assert windows.attribute(windows.stage_events([(0.0, 100.0, 'Sleep stage W'), (110.0, 100.0, 'Sleep stage W')]), 90, 120) == 'UNSCORED'
    assert windows.attribute(windows.stage_events([(0.0, 100.0, 'Sleep stage W'), (80.0, 100.0, 'Sleep stage 1')]), 70, 100) == 'AMBIGUOUS'
    assert windows.first_sleep(st, 1000.0) == 1300.0
    windows.check_alignment(st, 1000.0, 1600.0)
    with pytest.raises(ValueError, match='no stage-2'):
        windows.check_alignment(st, 1000.0, 1630.0)


def test_input_stages_and_first_sleep():
    stage, first_any, first_s1 = windows.input_stages_and_first_sleep(EV, 1000.0, 1600.0, [30, 300, 330, 460, 570])
    assert list(stage) == ['W', 'W', '1', 'W', '1'] and first_any == first_s1 == 300.0
    with pytest.raises(ValueError, match='spans two stages'):
        windows.input_stages_and_first_sleep(EV, 1000.0, 1600.0, [315])


# ------------------------------------------------------------------------ eligibility
def test_risk_profile_eligible_and_counts():
    status, s, first = cohort.stage_risk_profile(EV, 1000.0, 5000.0)
    assert status == cohort.ELIGIBLE and first == 1600.0
    assert s['landmark_n'] == 19 and s['positive_h300_n'] == 10 and s['latency_s'] == 600.0 and s['anchored_lb1200_n'] == 19


@pytest.mark.parametrize('events,off,status', [
    (EV, 1700.0, 'N2_ACTIVE_AT_OFF'),
    (EV, 6000.0, 'OFF_OUT_OF_PSG'),
    (EV[:4], 1000.0, 'NO_STAGE2_AFTER_OFF'),
    (EV, 1590.0, 'NO_30S_HISTORY_BEFORE_EVENT'),
    ([(0.0, 1000.0, 'Sleep stage W'), (1100.0, 500.0, 'Sleep stage 1'), (1600.0, 60.0, 'Sleep stage 2')], 0.0, 'SCORING_GAP_OR_OVERLAP'),
    ([(0.0, 1000.0, 'Sleep stage W'), (1000.0, 300.0, 'Sleep stage ?'), (1300.0, 60.0, 'Sleep stage 2')], 0.0, 'UNKNOWN_STAGE_IN_RISK_SET'),
    ([(0.0, 10.0, 'Lights off')], 0.0, 'NO_STAGES')])
def test_risk_profile_exclusions(events, off, status):
    assert cohort.stage_risk_profile(events, off, 5000.0)[0] == status


def test_lights_off_clock_parsing_and_alignment():
    assert cohort.parse_clock('00:38') == 2280 and cohort.parse_clock('23:05h') == 83100
    assert cohort.parse_clock(0.25) == 21600.0 and cohort.parse_clock('nan') is None and cohort.parse_clock(3) is None
    hdr = dict(psg_clock_start_s=80000, psg_duration_s=30000.)
    assert cohort.match_time(hdr, 81000) == 1000 and cohort.match_time(hdr, 1000) == 7400   # crosses midnight
    assert cohort.match_time(hdr, 80000 + 29990) is None and cohort.match_time(hdr, None) is None


def test_metadata_grid_parsers():
    grid = [['title'], ['Subject', 'Night', 'LightsOff']] + [[s, n, f'{22 + n}:{s:02d}'] for s in range(80) for n in (1, 2)]
    rows, about = cohort.identify_sc({'Sheet1': grid})
    assert about['metadata_rows'] == 160 and rows[f'SC4{7:02}{1}']['anchor_clock_seconds'] == 23 * 3600 + 7 * 60
    with pytest.raises(ValueError, match='Ambiguous/missing SC columns'):
        cohort.identify_sc({'Sheet1': [['a', 'b']]})


# ------------------------------------------------------------------------ person-disjoint split
def _subjects(n_sc=78, n_st=17):
    return ([dict(study='sleep-cassette', subject_key=f'SCsyn{i:02d}') for i in range(n_sc)]
            + [dict(study='sleep-telemetry', subject_key=f'STsyn{i:02d}') for i in range(n_st)])


def test_person_split_isolation_and_sizes():
    split = cohort.person_split(_subjects())
    counts = {k: sum(v == k for v in split.values()) for k in cohort.SPLIT_LABELS}
    assert counts == {'SC_TRAIN': 46, 'SC_VALIDATION': 16, 'SC_PROTECTED_MODEL_TEST': 16, 'ST_EXTERNAL_RESERVED': 17}
    assert split == cohort.person_split(list(reversed(_subjects())))          # order-independent, deterministic
    assert all(v == 'ST_EXTERNAL_RESERVED' for k, v in split.items() if k.startswith('ST'))
    # every night of a person lands in the same split: rosters are built from person keys only
    rows = [dict(r, recording=r['subject_key'] + str(n), night=str(n), status=cohort.ELIGIBLE, stage_at_lights_off='W')
            for r in _subjects() for n in (1, 2)]
    groups = {c: {r['subject_key'] for r in cohort.roster(rows, split, c)} for c in ('TRAIN', 'VAL', 'TEST', 'EXT')}
    assert all(not (groups[a] & groups[b]) for a in groups for b in groups if a < b)
    assert len(cohort.roster(rows, split, 'TRAIN')) == 92                       # all nights of 46 TRAIN persons
    val = cohort.roster(rows, split, 'VAL')
    assert sum(r['role'] == 'PRIMARY' for r in val) == 16 and all(r['night'] == '1' for r in val if r['role'] == 'PRIMARY')
    salted = cohort.person_split(_subjects(), salt='another-salt')
    assert salted != split


def test_person_split_rejects_too_few_groups():
    with pytest.raises(ValueError, match='Too few groups'):
        cohort.person_split(_subjects(n_sc=40))


# ------------------------------------------------------------------------ EDF / EDF+ reading
def test_hypnogram_decoding(synthetic_hypnogram):
    path, stages = synthetic_hypnogram
    events, duration = edf.decode_edfplus_annotations(path.read_bytes())
    assert duration is None and events == sorted((float(o), float(d), lab) for o, d, lab in stages)
    status, s, first = cohort.stage_risk_profile(events, 300.0, 3600.0)
    assert status == cohort.ELIGIBLE and first == 1800.0 and s['landmark_n'] == 49


def test_psg_header_and_windows(synthetic_psg):
    path, fpz, pz = synthetic_psg
    hdr = edf.read_psg_clock_header(path)
    assert hdr['psg_clock_start_s'] == 22 * 3600 and hdr['psg_duration_s'] == 3600.0
    with path.open('rb') as f:
        h = edf.signal_header(f, path.stat().st_size)
        phys, dig = edf.read_window(f, h, h['eeg_channels'][1], 60.0, 90.0)
    assert dig == pz[6000:9000].tolist()
    assert np.allclose(phys, -200 + (np.array(dig) + 2000) * 0.1)               # EDF digital -> physical calibration
    w = windows.extract_recording(path, 300.0, 1800.0)
    assert w['raw'].shape == (49, 2, 3000) and w['raw'].dtype == np.float32 and w['spectral'].shape == (49, 10)
    assert np.isfinite(w['spectral']).all() and list(w['y']).count(1) == 10 and w['elapsed_s'][0] == 30.0
    assert np.allclose(w['raw'][0, 0], (-200 + (fpz[30000:33000] + 2000) * 0.1).astype(np.float32))


def test_edf_rejections(tmp_path, synthetic_psg):
    bad = tmp_path / 'bad.edf'
    bad.write_bytes(b'not an edf' * 40)
    with pytest.raises(ValueError, match='Not an EDF'):
        edf.decode_edfplus_annotations(bad.read_bytes())
    with pytest.raises(ValueError):
        edf.read_psg_clock_header(bad)
    path, _, _ = synthetic_psg
    truncated = tmp_path / 'trunc.edf'
    truncated.write_bytes(path.read_bytes()[:-2])
    with truncated.open('rb') as f, pytest.raises(ValueError, match='physical size mismatch'):
        edf.signal_header(f, truncated.stat().st_size)
    with path.open('rb') as f:
        h = edf.signal_header(f, path.stat().st_size)
        with pytest.raises(ValueError, match='outside EDF'):
            edf.read_window(f, h, h['eeg_channels'][0], 3590.0, 3620.0)
    with pytest.raises(ValueError, match='Not OLE2'):
        read_sheets(bad)
    with pytest.raises(ValueError, match='Malformed'):
        edf.parse_sha256sums(b'zz  file\n')
