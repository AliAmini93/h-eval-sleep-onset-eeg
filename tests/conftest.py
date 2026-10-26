"""Shared synthetic fixtures. Makes `src/` importable so `python -m pytest` works without installing the package."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))


def _field(v, width):
    s = str(v).encode('ascii')
    assert len(s) <= width, (v, width)
    return s.ljust(width)


def write_edf(path, signals, record_s=30, start='22.00.00', edf_plus=False, n_records=None):
    """signals: list of dict(label, unit, phys_min, phys_max, dig_min, dig_max, data=int16 array or bytes, spr)."""
    n = len(signals)
    nrec = n_records if n_records is not None else len(signals[0]['data']) // signals[0]['spr']
    head = b''.join([_field('0', 8), _field('X', 80), _field('Startdate X', 80), _field('01.01.89', 8), _field(start, 8),
                     _field(256 * (n + 1), 8), _field('EDF+C' if edf_plus else '', 44), _field(nrec, 8), _field(record_s, 8),
                     _field(n, 4)])
    for key, width in (('label', 16), ('transducer', 80), ('unit', 8), ('phys_min', 8), ('phys_max', 8), ('dig_min', 8),
                       ('dig_max', 8), ('prefilter', 80), ('spr', 8), ('reserved', 32)):
        head += b''.join(_field(s.get(key, ''), width) for s in signals)
    body = b''
    for r in range(nrec):
        for s in signals:
            if isinstance(s['data'], bytes):
                body += s['data']
            else:
                body += np.asarray(s['data'][r * s['spr']:(r + 1) * s['spr']], '<i2').tobytes()
    Path(path).write_bytes(head + body)
    return path


def eeg_signal(label, data):
    return dict(label=label, unit='uV', phys_min=-200, phys_max=200, dig_min=-2000, dig_max=2000, data=data, spr=3000)


def hypnogram_bytes(stages):
    """stages: list of (onset_s, duration_s, label) -> EDF+ annotation-only file bytes (single record, duration 0)."""
    tal = b'+0\x14\x14\x00' + b''.join(f'+{o:g}\x15{d:g}\x14{lab}\x14\x00'.encode() for o, d, lab in stages)
    tal += b'\x00' * ((-len(tal)) % 2)
    return tal


@pytest.fixture
def synthetic_psg(tmp_path):
    """60 min PSG at 100 Hz: two EEG channels (digital ramp patterns) plus one EOG channel; start clock 22:00:00."""
    nrec = 120
    rng = np.random.default_rng(0)
    fpz = rng.integers(-1500, 1500, size=nrec * 3000)
    pz = rng.integers(-1000, 1000, size=nrec * 3000)
    eog = np.zeros(nrec * 3000, int)
    sig = [eeg_signal('EEG Fpz-Cz', fpz), eeg_signal('EEG Pz-Oz', pz), eeg_signal('EOG horizontal', eog)]
    path = write_edf(tmp_path / 'SYN01-PSG.edf', sig)
    return path, fpz, pz


@pytest.fixture
def synthetic_hypnogram(tmp_path):
    stages = [(0, 1200, 'Sleep stage W'), (1200, 300, 'Sleep stage 1'), (1500, 60, 'Sleep stage W'), (1560, 240, 'Sleep stage 1'),
              (1800, 1800, 'Sleep stage 2'), (300, 0, 'Lights off')]
    tal = hypnogram_bytes(stages)
    sig = [dict(label='EDF Annotations', unit='', phys_min=-1, phys_max=1, dig_min=-32768, dig_max=32767, data=tal,
                spr=len(tal) // 2)]
    return write_edf(tmp_path / 'SYN01-Hypnogram.edf', sig, record_s=0, edf_plus=True, n_records=1), stages


@pytest.fixture(scope='session')
def nights():
    from heval.synthetic import synthetic_nights
    return synthetic_nights(n_persons=10, seed=1)


@pytest.fixture
def cli_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(SRC) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env
