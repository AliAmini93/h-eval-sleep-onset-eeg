import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

P=Path(__file__).with_name('gate_e5.py')
s=importlib.util.spec_from_file_location('gate_e5_test',P)
e5=importlib.util.module_from_spec(s);s.loader.exec_module(e5)

class TestE5(unittest.TestCase):
    def test_fft_impulse(self):
        z=e5.fft_real([1,0,0,0,0,0,0,0])
        self.assertTrue(all(abs(x-1)<1e-10 for x in z))
    def test_fft_tone(self):
        z=e5.fft_real([math.sin(2*math.pi*i/16) for i in range(16)])
        self.assertAlmostEqual(abs(z[1]),8,places=6)
        self.assertLess(abs(z[2]),1e-8)
    def test_fft_reject_shape(self):
        with self.assertRaises(ValueError):e5.fft_real([1,2,3])
    def test_two_tones_features(self):
        d=e5.channel_features([10*math.sin(2*math.pi*2*i/100) for i in range(3000)])
        t=e5.channel_features([10*math.sin(2*math.pi*6*i/100) for i in range(3000)])
        self.assertGreater(d[0],d[1]);self.assertGreater(t[1],t[0])
    def test_reject_wrong_length(self):
        with self.assertRaises(ValueError):e5.channel_features([0]*2999)
    def test_landmark_e2_match(self):
        x=e5.make_landmarks(0,540,{'landmark_n':'17','anchored_lb1200_n':'17','positive_h300_n':'10','recording':'SC0001'})
        self.assertEqual(len(x),17);self.assertEqual(x[-1],510)
    def test_landmark_strict(self):
        with self.assertRaises(ValueError):e5.make_landmarks(0,540,{'landmark_n':'18','anchored_lb1200_n':'17','positive_h300_n':'10','recording':'SC0001'})
    def test_groups_strict(self):
        with self.assertRaises(ValueError):e5.score_cv([{'subject_key':'S1'}])
    def test_preprocessing_train_only(self):
        train=[{'features':[2.]*10},{'features':[4.]*10}]
        held=[{'features':[10.]*10}]
        t,h=e5.preprocess(train,held)
        self.assertEqual(h[0][1],[5.]*10)
        self.assertAlmostEqual(t[0][1][0],-1)
    def test_fixed_time_bins(self):
        self.assertEqual(e5.bin_idx(300),1);self.assertEqual(e5.bin_idx(299.9),0)
    def test_logistic_predict_bounded(self):
        w=e5.logistic_fit([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],[0,1,1,1],steps=300)
        self.assertGreater(e5.predict_logit(w,[1,1]),e5.predict_logit(w,[0,0]))
    def test_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s';e5.immut(p,b'a');e5.immut(p,b'a')
            with self.assertRaises(ValueError):e5.immut(p,b'b')
    def test_cv_loo_and_paired(self):
        rows=[]
        for j in range(6):
            for t in range(30,631,30):
                rows.append({'recording':f'SC{j}', 'subject_key':f'S{j}', 'elapsed_s':t,
                    'anchor':t>240,'y':int(t>330),
                    'features':[math.log(1+(t/50)+(j%3))]*10})
        per,agg=e5.score_cv(rows)
        self.assertEqual(len(per),6*2*2*5)
        self.assertEqual(len(agg),20)
        self.assertTrue(all(x['n_subjects']==6 for x in agg))
        self.assertEqual(len({r['held_subject'] for r in per}),6)

if __name__=='__main__':unittest.main()
