import json
import tempfile
import unittest
from pathlib import Path
import gate_e3 as g

class E3Tests(unittest.TestCase):
    def test_stage_at_origin(self):
        self.assertEqual(g.stage_at([(0,30,'Sleep stage W'),(30,30,'Sleep stage 1')],15),'W')
        self.assertEqual(g.stage_at([(0,30,'Sleep stage W'),(30,30,'Sleep stage 1')],30),'1')
    def test_ambiguous_at_origin(self):
        self.assertEqual(g.stage_at([(0,30,'Sleep stage W'),(0,30,'Sleep stage 1')],15),'AMBIGUOUS')
    def test_no_origin(self):
        self.assertEqual(g.stage_at([],None),'NO_ORIGIN')
    def test_gap(self):
        z=g.gap_details([(0,30,'Sleep stage W'),(60,30,'Sleep stage 1')],0,90)
        self.assertEqual(z['issue'],'GAP')
        self.assertEqual(z['problem_start_s'],30)
    def test_overlap(self):
        z=g.gap_details([(0,50,'Sleep stage W'),(40,50,'Sleep stage 1')],0,90)
        self.assertEqual(z['issue'],'OVERLAP')
    def test_movement_annotation_in_gap(self):
        z=g.gap_details([(0,30,'Sleep stage W'),(30,30,'Movement time'),(60,30,'Sleep stage 1')],0,90)
        self.assertEqual(z['preceding_nonstage_annotation'],'Movement time')
    def test_continuous(self):
        self.assertEqual(g.gap_details([(0,30,'Sleep stage W'),(30,30,'Sleep stage 1')],5,55)['issue'],'CONTIGUOUS')
    def test_degenerate(self):
        self.assertEqual(g.gap_details([],None,None)['issue'],'NO_VALID_ORIGIN_OR_TARGET')
    def test_split_grouping(self):
        e=[]
        for i in range(77):
            for night in (1,2):e.append({'study':'sleep-cassette','subject_key':f'SC4{i:02}','recording':f'SC4{i:02}{night}'})
        for i in range(17):e.append({'study':'sleep-telemetry','subject_key':f'ST7{i:02}','recording':f'ST7{i:02}1'})
        sp=g.group_split(e)
        self.assertEqual(len(sp),94)
        self.assertEqual(sum(v=='SC_PROTECTED_MODEL_TEST' for v in sp.values()),15)
        self.assertEqual(sum(v=='SC_VALIDATION' for v in sp.values()),15)
        self.assertEqual(sum(v=='ST_EXTERNAL_RESERVED' for v in sp.values()),17)
        self.assertEqual(sp,g.group_split(list(reversed(e))))
    def test_single_subject_insufficient(self):
        with self.assertRaises(ValueError):g.group_split([{'study':'sleep-cassette','subject_key':'SC401'}])
    def test_anchor_grid(self):
        r={'recording':'SC4001','lights_off_rel_s':'33.43','first_n2_after_off_s':'480',
           'landmark_n':'13','anchored_lb1200_n':'13','positive_h300_n':'10'}
        a=g.make_landmarks(r)
        self.assertEqual(len(a),13)
        self.assertEqual(sum(x[1] for x in a),10)
        self.assertTrue(all(x[2] for x in a))
    def test_anchor_grid_mismatch(self):
        r={'recording':'SC4001','lights_off_rel_s':'33.43','first_n2_after_off_s':'480',
           'landmark_n':'13','anchored_lb1200_n':'13','positive_h300_n':'11'}
        with self.assertRaises(ValueError):g.make_landmarks(r)
    def test_bins_and_losses(self):
        m=g.fit([(10,0,True),(400,1,True),(700,1,False)],'FULL')
        self.assertGreater(m['CONSTANT'],0)
        self.assertEqual(len(m['TIME_BIN']),7)
        self.assertLess(g.losses(.9,1)[0],g.losses(.1,1)[0])
    def test_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.json'
            g.immutable_write(path,b'1');g.immutable_write(path,b'1')
            with self.assertRaises(ValueError):g.immutable_write(path,b'2')
    def test_sc_dev_only(self):
        # SC TEST records are intentionally inconsistent and must never be scored.
        rs=[];sp={}
        for i in range(77):
            sub=f'SC4{i:02}';sp[sub]='SC_TRAIN' if i<45 else ('SC_VALIDATION' if i<60 else 'SC_PROTECTED_MODEL_TEST')
            r={'subject_key':sub,'study':'sleep-cassette','status':g.ELIGIBLE,
               'recording':sub+'1','lights_off_rel_s':'0','first_n2_after_off_s':'330',
               'landmark_n':'10','anchored_lb1200_n':'10','positive_h300_n':'10'}
            if sp[sub]=='SC_PROTECTED_MODEL_TEST':r['landmark_n']='INVALID_NO_TEST_LABELS_READ'
            rs.append(r)
        per,m=g.score_sc_development(rs,sp)
        self.assertEqual(len(m),8)
        self.assertTrue(per)
        self.assertTrue(all(r['subject_key'] not in {k for k in sp if sp[k]=='SC_PROTECTED_MODEL_TEST'} for r in per))

if __name__=='__main__':unittest.main()
