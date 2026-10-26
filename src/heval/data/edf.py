"""EDF / EDF+ readers for Sleep-EDF Expanded: hypnogram annotations, PSG headers and calibrated EEG windows.

Lineage (research archive, tag v0.1.0-paper-review): annotation decoder and checksum parser from the E metadata
audit; PSG clock header from E2; signal header and window reader from E4 v3. Logic unchanged; only the inputs
are local files instead of network ranges, and the original exception messages are kept.
"""
import hashlib
import math
import re
import struct
from pathlib import Path

EEG_CHANNELS = ('EEG Fpz-Cz', 'EEG Pz-Oz')
HEADER_SIGNALS_MAX = 100


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(raw):
    """Official SHA256SUMS.txt bytes -> {relative path: sha256}."""
    result = {}
    for line in raw.decode('ascii').splitlines():
        m = re.fullmatch(r'([a-fA-F0-9]{64})\s+\*?(.+)', line.strip())
        if not m:
            raise ValueError('Malformed official SHA256SUMS line')
        name = m[2].removeprefix('./')
        if name in result:
            raise ValueError('Duplicate official hash ' + name)
        result[name] = m[1].lower()
    return result


# ------------------------------------------------------------------------ EDF+ annotations (hypnograms)
def _annotation_header(data):
    def field(a, b):
        return data[a:b].decode('ascii', 'replace').strip()
    if len(data) < 256 or field(0, 8) != '0':
        raise ValueError('Not an EDF+ file')
    n = int(field(252, 256))
    header = int(field(184, 192))
    nrec = int(field(236, 244))
    dt = float(field(244, 252))
    if not 1 <= n <= 32 or header != 256 * (n + 1) or nrec < 1 or not math.isfinite(dt) or dt < 0:
        raise ValueError('EDF+ header structure invalid')
    if len(data) < header:
        raise ValueError('Truncated EDF+ header')
    labels = [data[256 + 16 * i:256 + 16 * (i + 1)].decode('ascii', 'replace').strip() for i in range(n)]
    pos = 256 + n * 216
    samples = [int(data[pos + 8 * i:pos + 8 * (i + 1)].decode('ascii', 'replace').strip()) for i in range(n)]
    if any(v <= 0 for v in samples):
        raise ValueError('Invalid sample count')
    full = header + 2 * sum(samples) * nrec
    if full != len(data):
        raise ValueError('EDF+ byte size mismatch: ' + str((full, len(data))))
    anns = [i for i, lab in enumerate(labels) if lab.lower() == 'edf annotations']
    if not anns:
        raise ValueError('EDF+ has no EDF Annotations channel')
    # One official hypnogram is a single-record annotation-only file with record duration 0; that is legal here.
    if dt == 0 and not (nrec == 1 and n == 1 and len(anns) == 1 and data[192:236].strip().startswith(b'EDF+')):
        raise ValueError('Zero EDF data-record duration invalid outside single-record EDF+ annotation-only file')
    return header, nrec, dt, samples, anns


def decode_edfplus_annotations(data):
    """EDF+ bytes -> (sorted [(onset_s, duration_s, label)], file duration or None)."""
    header, nrec, dt, samples, anns = _annotation_header(data)
    record_size = 2 * sum(samples)
    offsets = [2 * sum(samples[:i]) for i in range(len(samples))]
    result = []
    for j in range(nrec):
        start = header + j * record_size
        for i in anns:
            blob = data[start + offsets[i]:start + offsets[i] + 2 * samples[i]]
            for item in blob.split(b'\0'):
                if not item.strip(b'\x00\x14\x15'):
                    continue
                pieces = item.split(b'\x14')
                if not pieces:
                    continue
                parts = pieces[0].decode('ascii', 'replace').split('\x15')
                try:
                    onset = float(parts[0])
                    dur = float(parts[1]) if len(parts) > 1 and parts[1] else 0.
                except ValueError:
                    raise ValueError('Bad TAL onset/duration')
                if not math.isfinite(onset) or not math.isfinite(dur) or dur < 0:
                    raise ValueError('Invalid TAL time')
                for title in pieces[1:]:
                    label = title.decode('latin-1').strip()
                    if label:
                        result.append((onset, dur, label))
    if not result:
        raise ValueError('No EDF+ annotations')
    return sorted(result), nrec * dt if dt > 0 else None


# ------------------------------------------------------------------------ PSG clock header (lights-off alignment)
def psg_clock_header(raw, size):
    """First header bytes of a PSG EDF and its total byte size -> start clock, duration and layout checks."""
    def field(a, b):
        return raw[a:b].decode('ascii', 'replace').strip()
    if len(raw) < 256 or field(0, 8) != '0':
        raise ValueError('Invalid PSG EDF first 256 bytes')
    n = int(field(252, 256))
    hb = int(field(184, 192))
    nr = int(field(236, 244))
    duration = float(field(244, 252))
    if n < 1 or n > HEADER_SIGNALS_MAX or hb != 256 * (1 + n) or len(raw) != hb or nr <= 0 or not math.isfinite(duration) \
            or duration <= 0:
        raise ValueError('Invalid PSG EDF header layout')
    labels = [raw[256 + i * 16:256 + (i + 1) * 16].decode('latin1').strip() for i in range(n)]
    start = 256 + 216 * n
    ns = [int(raw[start + 8 * i:start + 8 * (i + 1)].decode('ascii', 'replace').strip()) for i in range(n)]
    if any(x <= 0 for x in ns):
        raise ValueError('Bad EDF sample count')
    if hb + nr * sum(ns) * 2 != size:
        raise ValueError('EDF byte size disagrees with parsed signal layout')
    if not any('EEG' in v.upper() for v in labels):
        raise ValueError('PSG lacks EEG channel label')
    hm = re.fullmatch(r'(\d{2})\.(\d{2})\.(\d{2})', field(176, 184))
    if not hm or int(hm[1]) > 23 or int(hm[2]) > 59 or int(hm[3]) > 59:
        raise ValueError('Invalid PSG EDF clock start time')
    return {'psg_clock_start': field(176, 184), 'psg_clock_start_s': int(hm[1]) * 3600 + int(hm[2]) * 60 + int(hm[3]),
            'psg_duration_s': nr * duration, 'psg_channels': labels, 'psg_sample_rates_hz': [x / duration for x in ns],
            'psg_size_bytes': size}


def read_psg_clock_header(path):
    path = Path(path)
    with path.open('rb') as f:
        first = f.read(256)
        if len(first) < 256:
            raise ValueError('Invalid PSG EDF first 256 bytes')
        n = int(first[252:256].decode('ascii').strip())
        if not 1 <= n <= HEADER_SIGNALS_MAX:
            raise ValueError('PSG signal count out of bound')
        raw = first + f.read(256 * n)
    return psg_clock_header(raw, path.stat().st_size)


# ------------------------------------------------------------------------ calibrated EEG windows
def signal_header(f, physical_bytes):
    """Open binary EDF file object -> layout plus the two EEG channels (exactly one each, uV, 100 Hz)."""
    fixed = f.read(256)
    if len(fixed) != 256 or fixed[:8].strip() != b'0':
        raise ValueError('Not EDF fixed header')

    def num(a, b, type_=int):
        return type_(fixed[a:b].decode('ascii').strip())
    n, hdr, nr, dr = num(252, 256), num(184, 192), num(236, 244), num(244, 252, float)
    if not 1 <= n <= 32 or hdr != 256 * (n + 1) or nr <= 0 or dr <= 0 or not math.isfinite(dr):
        raise ValueError('Unexpected EDF record structure')
    extra = f.read(hdr - 256)
    if len(extra) != hdr - 256:
        raise ValueError('Truncated EDF signal header')
    pos, blocks = 0, {}
    for name, width in [('label', 16), ('transducer', 80), ('unit', 8), ('phys_min', 8), ('phys_max', 8), ('dig_min', 8),
                        ('dig_max', 8), ('prefilter', 80), ('samples', 8), ('reserved', 32)]:
        blocks[name] = [extra[pos + i * width:pos + (i + 1) * width].decode('latin1').strip() for i in range(n)]
        pos += n * width
    if pos != len(extra):
        raise ValueError('Signal field lengths incorrect')
    info = []
    for i in range(n):
        a = {name: blocks[name][i] for name in blocks}
        for k in ('phys_min', 'phys_max'):
            a[k] = float(a[k])
        for k in ('dig_min', 'dig_max', 'samples'):
            a[k] = int(a[k])
        if not (a['phys_max'] > a['phys_min'] and a['dig_max'] > a['dig_min'] and a['samples'] > 0):
            raise ValueError('Invalid EDF signal calibration')
        a.update(rate_hz=a['samples'] / dr, byte_offset=2 * sum(int(x) for x in blocks['samples'][:i]))
        info.append(a)
    record_bytes = 2 * sum(int(x) for x in blocks['samples'])
    if hdr + nr * record_bytes != physical_bytes:
        raise ValueError('EDF physical size mismatch: ' + str((hdr + nr * record_bytes, physical_bytes)))
    selected = []
    for lbl in EEG_CHANNELS:
        match = [(i, s) for i, s in enumerate(info) if s['label'].lower() == lbl.lower()]
        if len(match) != 1:
            raise ValueError('Expected exactly one ' + lbl + ' EEG channel')
        i, s = match[0]
        if s['unit'].lower() not in ('uv', 'µv', 'μv'):
            raise ValueError('Unexpected EEG physical unit ' + s['unit'])
        if abs(s['rate_hz'] - 100) > 1e-7:
            raise ValueError('Unexpected EEG sampling rate ' + str(s['rate_hz']))
        selected.append((i, s))
    return {'n_signals': n, 'header_bytes': hdr, 'n_records': nr, 'data_record_seconds': dr, 'record_bytes': record_bytes,
            'duration_s': dr * nr, 'eeg_channels': selected, 'info': info}


def read_window(f, header, channel, t1, t2):
    """Samples of one channel in [t1, t2) seconds -> (physical uV list, digital list)."""
    _, ch = channel
    samp = ch['samples']
    if t1 < 0 or t2 <= t1 or t2 > header['duration_s'] + 1e-8:
        raise ValueError('EEG window outside EDF')
    start, end = int(round(t1 * ch['rate_hz'])), int(round(t2 * ch['rate_hz']))
    out, idx = [], start
    while idx < end:                  # read by EDF data record to respect per-signal interleaving
        rec, ofs = idx // samp, idx % samp
        take = min(end - idx, samp - ofs)
        if rec >= header['n_records']:
            raise ValueError('Requested samples exceed EDF data records')
        f.seek(header['header_bytes'] + rec * header['record_bytes'] + ch['byte_offset'] + ofs * 2)
        raw = f.read(take * 2)
        if len(raw) != take * 2:
            raise ValueError('Truncated raw signal at record ' + str(rec))
        out.extend(struct.unpack('<' + str(take) + 'h', raw))
        idx += take
    gain = (ch['phys_max'] - ch['phys_min']) / (ch['dig_max'] - ch['dig_min'])
    return [ch['phys_min'] + (x - ch['dig_min']) * gain for x in out], out
