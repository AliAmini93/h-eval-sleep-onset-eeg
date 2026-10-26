import unittest
import tempfile
from pathlib import Path
import gate_e as a


def sample_file(annotations, record_duration=3600, reserve=b'EDF+C'):
    # Single-channel EDF+ containing TAL records and padded annotation samples.
    tal=b''.join((('+'+str(t)+'\x15'+str(d)+'\x14'+lab+'\x14\x00').encode('ascii') for t,d,lab in annotations))
    if len(tal)>400:raise ValueError('fixture too long')
    n=1;h=bytearray(b' '*512)
    def put(pos,width,x):h[pos:pos+width]=str(x).encode().ljust(width,b' ')
    put(0,8,'0');put(184,8,512);put(236,8,1);put(244,8,record_duration);put(252,4,1)
    h[192:192+len(reserve)]=reserve
    put(256,16,'EDF Annotations')
    put(256+216,8,200)
    return bytes(h)+tal.ljust(400,b'\0')

class GateETests(unittest.TestCase):
    def test_real_sleep_edf_annotation_only_zero_duration(self):
        # Reproduces the layout observed in the publicly available
        # SC4001EC-Hypnogram.edf: nrec=1, record-duration=0, samples=2054,
        # EDF+C annotation-only and TAL timestamps relative to PSG recording.
        b=sample_file([(0,30630,'Sleep stage W'),
                       (30630,120,'Sleep stage 1'),(30750,390,'Sleep stage 2')],record_duration=0)
        v=a.annotation_profile(b)
        self.assertTrue(v['n2_present'])
        self.assertEqual(v['first_n2_from_psg_origin_s'],30750)
        self.assertIsNone(v['edf_annotation_record_duration_total_s'])
        self.assertFalse(v['strict_independent_origin_candidate'])
    def test_non_annotation_edf_zero_duration_rejected(self):
        b=bytearray(sample_file([(1,30,'Sleep stage W')],record_duration=0))
        b[192:236]=b' '*44
        with self.assertRaisesRegex(ValueError,'Zero EDF data-record duration'):
            a.decode_edfplus_annotations(bytes(b))
    def test_manifest_gap_refusal(self):
        with self.assertRaises(ValueError):a.records_parse(b'sleep-cassette/SC4001E0-PSG.edf\n')
    def test_sums(self):
        got=a.sums_parse(('a'*64+'  RECORDS\n'+'b'*64+' *sleep-cassette/SC4001EC-Hypnogram.edf\n').encode())
        self.assertEqual(got['RECORDS'],'a'*64)
        self.assertEqual(len(got),2)
    def test_edf_annotations(self):
        b=sample_file([(0,300,'Sleep stage W'),(300,300,'Sleep stage 1'),(600,300,'Sleep stage 2')])
        p=a.annotation_profile(b)
        self.assertEqual(p['first_n2_from_psg_origin_s'],600)
        self.assertFalse(p['has_explicit_lights_off'])
        self.assertFalse(p['strict_independent_origin_candidate'])
    def test_explicit_anchor(self):
        b=sample_file([(10,0,'Lights off'),(600,0,'Lights on'),(100,50,'Sleep stage 2')])
        p=a.annotation_profile(b)
        self.assertTrue(p['strict_independent_origin_candidate'])
        self.assertEqual(p['first_n2_after_lights_off_s'],100)
    def test_only_n2_without_off_not_enough(self):
        b=sample_file([(0,0,'Sleep stage 2')])
        p=a.annotation_profile(b)
        self.assertFalse(p['strict_independent_origin_candidate'])
    def test_truncated_edf(self):
        b=sample_file([(0,30,'Sleep stage W')])
        with self.assertRaises(ValueError):a.decode_edfplus_annotations(b[:-2])
    def test_summary_no_false_start(self):
        rows=[{'recording':'SC4001','subject_key':'SC400','study':'sleep-cassette',
               'status':'PARSED_SHA_VERIFIED','n2_present':True,'has_explicit_lights_off':False,
               'has_explicit_lights_on':False,'strict_independent_origin_candidate':False,'annotation_labels':{'Sleep stage 2':1}}]
        s=a.summarize(rows)
        self.assertEqual(s['strict_independent_origin_candidates_n'],0)
        self.assertEqual(s['decision'],'REVIEW_ANCHOR_FEASIBILITY_MANUALLY')
    def test_pair_validation(self):
        with self.assertRaises(ValueError):a.pairs_create(['sleep-cassette/SC4001E0-PSG.edf'],{})
    def test_sha_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data.bin';p.write_bytes(b'incorrect')
            with self.assertRaises(ValueError):a.safe_store_bytes(p,b'correct',a.sha_bytes(b'correct'))
            self.assertEqual(p.read_bytes(),b'incorrect')

if __name__=='__main__':unittest.main()
