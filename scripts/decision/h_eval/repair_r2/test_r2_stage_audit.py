"""Synthetic tests for R2 (no held-out data). Run before the freeze record is written."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('r2', Path(__file__).resolve().parent / 'r2_stage_audit.py')
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)

# synthetic hypnogram: off = 1000; W until 1300, S1 1300-1400, W 1400-1460 (returning wake), S1 1460-1600, N2 at 1600
EV = [(0.0, 1300.0, 'Sleep stage W'), (1300.0, 100.0, 'Sleep stage 1'), (1400.0, 60.0, 'Sleep stage W'),
      (1460.0, 140.0, 'Sleep stage 1'), (1600.0, 600.0, 'Sleep stage 2'), (500.0, 0.0, 'Lights off')]
ST = r2.stage_events(EV)


class TestAttribution(unittest.TestCase):
    def test_pure_stages(self):
        self.assertEqual(r2.attribute(ST, 1200, 1230), 'W')
        self.assertEqual(r2.attribute(ST, 1300, 1330), '1')

    def test_mixed_boundary_not_relabelled(self):
        self.assertEqual(r2.attribute(ST, 1290, 1320), 'MIXED:W|1')
        self.assertEqual(r2.attribute(ST, 1380, 1410), 'MIXED:1|W')

    def test_unscored_gap(self):
        st = r2.stage_events([(0.0, 100.0, 'Sleep stage W'), (110.0, 100.0, 'Sleep stage W')])
        self.assertEqual(r2.attribute(st, 90, 120), 'UNSCORED')
        self.assertEqual(r2.attribute(st, 500, 530), 'UNSCORED')

    def test_ambiguous_overlap(self):
        st = r2.stage_events([(0.0, 100.0, 'Sleep stage W'), (80.0, 100.0, 'Sleep stage 1')])
        self.assertEqual(r2.attribute(st, 70, 100), 'AMBIGUOUS')

    def test_same_label_split_annotations_are_pure(self):
        st = r2.stage_events([(0.0, 100.0, 'Sleep stage W'), (100.0, 100.0, 'Sleep stage W')])
        self.assertEqual(r2.attribute(st, 90, 120), 'W')

    def test_non_stage_annotations_ignored(self):
        self.assertEqual(r2.attribute(ST, 490, 520), 'W')


class TestAlignmentAndFirstSleep(unittest.TestCase):
    def test_first_sleep_and_pre_subset(self):
        self.assertEqual(r2.first_sleep(ST, 1000.0), 1300.0)
        st = r2.stage_events([(0.0, 900.0, 'Sleep stage 1'), (900.0, 5000.0, 'Sleep stage W')])
        self.assertEqual(r2.first_sleep(st, 1000.0), None)

    def test_alignment_ok(self):
        r2.check_alignment(ST, 1000.0, 1600.0)

    def test_alignment_wrong_n2_aborts(self):
        with self.assertRaises(r2.Abort):
            r2.check_alignment(ST, 1000.0, 1630.0)

    def test_stage2_before_n2_aborts(self):
        st = r2.stage_events(EV + [(1100.0, 30.0, 'Sleep stage 2')])
        with self.assertRaises(r2.Abort):
            r2.check_alignment(st, 1000.0, 1600.0)

    def test_labels_horizon(self):
        y, t = r2.labels(1000.0, 1600.0, [30, 300, 330, 570])
        self.assertEqual(list(y), [0, 1, 1, 1])     # n2 - t = 570, 300, 270, 30
        self.assertEqual(list(t), [1030.0, 1300.0, 1330.0, 1570.0])


class TestStatistics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.person = np.repeat([f'P{i:02d}' for i in range(10)], 20)
        self.y = (rng.random(200) < .2).astype(float)
        self.ref = np.full(200, .2)
        self.cand = np.clip(self.ref + .3 * (self.y - .2) + rng.normal(0, .02, 200), .01, .99)

    def test_identical_predictions_zero(self):
        out = r2.contrast(self.ref, self.ref, self.y, self.person, r2.boot_index(10, n=500))
        self.assertEqual(out['delta_ww'], 0.0)
        self.assertEqual(out['R'], 0.0)
        self.assertEqual(out['delta_ww_ci95'], [0.0, 0.0])

    def test_known_values(self):
        out = r2.contrast(self.ref, self.cand, self.y, self.person, r2.boot_index(10, n=500))
        bref, bc = ((self.ref - self.y) ** 2).mean(), ((self.cand - self.y) ** 2).mean()
        self.assertAlmostEqual(out['delta_ww'], bref - bc, places=14)
        self.assertAlmostEqual(out['R'], (bref - bc) / bref, places=14)
        self.assertLessEqual(out['delta_ww_ci95'][0], out['delta_ww'])
        self.assertGreaterEqual(out['delta_ww_ci95'][1], out['delta_ww'])

    def test_ci_suppressed_below_8_persons(self):
        m = np.isin(self.person, ['P00', 'P01', 'P02'])
        out = r2.contrast(self.ref[m], self.cand[m], self.y[m], self.person[m], None)
        self.assertEqual(out['ci'], 'not_reported_lt_8_persons')

    def test_subject_equal_unequal_sizes(self):
        person = np.array(['A'] * 3 + ['B'])
        y = np.array([0., 0, 0, 1])
        ref, cand = np.full(4, .5), np.array([.5, .5, .5, .9])
        out = r2.contrast(ref, cand, y, person)
        self.assertAlmostEqual(out['delta_se'], (0 + (.25 - .01)) / 2, places=14)
        self.assertAlmostEqual(out['delta_ww'], (.25 - .01) / 4, places=14)

    def test_contributions_sum(self):
        cats = np.array(['W'] * 150 + ['1'] * 50)
        c = r2.contributions((self.ref - self.y) ** 2, (self.cand - self.y) ** 2, cats)
        self.assertAlmostEqual(sum(c['absolute'].values()), c['delta_ww_all'], places=14)
        self.assertAlmostEqual(sum(c['share'].values()), 1.0, places=12)

    def test_auroc(self):
        self.assertEqual(r2.auroc(np.array([.1, .2, .9]), np.array([0, 0, 1])), 1.0)
        self.assertIsNone(r2.auroc(np.array([.1, .2]), np.array([0, 0])))


class TestGuards(unittest.TestCase):
    def test_canonical_envelope_hash(self):
        import hashlib
        import json
        c = dict(b=1, a=[0.5, None])
        self.assertEqual(r2.canon_sha(c), hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest())

    def test_write_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.json'
            r2.write_new(p, b'1')
            with self.assertRaises(FileExistsError):
                r2.write_new(p, b'2')

    def test_no_psg_paths_in_inputs(self):
        src = (Path(__file__).resolve().parent / 'r2_stage_audit.py').read_text()
        self.assertNotIn('psg_path', src.replace("'psg_path'", ''))
        self.assertNotIn('read_window', src)
        self.assertNotIn('torch', src)


if __name__ == '__main__':
    unittest.main()
