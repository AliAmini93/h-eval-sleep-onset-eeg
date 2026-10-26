"""Deterministic synthetic nights for tests and the runnable demo (no real EEG).

Each synthetic person has one night: lights-off at 0 s, scored W until a first Stage 1 time T1 and Stage 1 until the
first Stage 2 time T2; decision points every 30 s before T2. The two 'EEG' channels are noise plus a 10 Hz (alpha)
component that fades and a 6 Hz (theta) component that grows as T2 approaches, so the methods have a signal to find.
"""
import numpy as np

FS, WIN = 100, 3000


def synthetic_nights(n_persons=10, seed=0, min_latency=600, max_latency=1500, with_raw=True):
    rng = np.random.default_rng(seed)
    t_axis = np.arange(WIN) / FS
    cols = {k: [] for k in ('person', 'recording', 'elapsed_s', 'time_to_n2_s', 'y', 'stage', 'first_sleep_s', 'raw')}
    for i in range(n_persons):
        person = f'P{i:02d}'
        t2 = 30.0 * rng.integers(min_latency // 30, max_latency // 30 + 1)
        t1 = t2 - 30.0 * rng.integers(4, 12)
        for t in np.arange(30.0, t2, 30.0):
            tau = t2 - t
            cols['person'].append(person)
            cols['recording'].append(person + 'N1')
            cols['elapsed_s'].append(t)
            cols['time_to_n2_s'].append(tau)
            cols['y'].append(int(tau <= 300 + 1e-8))
            cols['stage'].append('W' if t - 30 < t1 - 1e-9 else '1')   # input epoch [t-30, t)
            cols['first_sleep_s'].append(t1)
            if with_raw:
                closeness = float(np.exp(-tau / 600.0))
                x = rng.normal(0, 8, size=(2, WIN))
                x += (12 * (1 - closeness)) * np.sin(2 * np.pi * 10 * t_axis + rng.uniform(0, 6.28))
                x += (14 * closeness) * np.sin(2 * np.pi * 6 * t_axis + rng.uniform(0, 6.28))
                cols['raw'].append(x.astype(np.float32))
    out = {k: np.array(v) for k, v in cols.items() if k != 'raw'}
    out['y'] = out['y'].astype(int)
    if with_raw:
        out['raw'] = np.stack(cols['raw'])
    return out
