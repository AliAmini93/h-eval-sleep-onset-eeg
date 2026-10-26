import io
import math
import tempfile
import unittest
from pathlib import Path
import gate_e2 as g
from xls_stdlib import read_sheets

HERE=Path(__file__).parent

class GateE2(unittest.TestCase):
    def test_sc_workbook_real_biff_structure(self):
        rows,meta=g.identify_sc(read_sheets(HERE/'fixtures/SC-mock.xls'))
        self.assertEqual(len(rows),153)
        self.assertEqual(rows['SC4001']['anchor_clock_seconds'],38*60)
        self.assertIn('SC4762',rows)
    def test_st_workbook_real_biff_structure(self):
        rows,meta=g.identify_st(read_sheets(HERE/'fixtures/ST-mock.xls'))
        self.assertEqual(len(rows),44)
        self.assertEqual(rows['ST7001']['condition'],'placebo')
        self.assertEqual(rows['ST7002']['condition'],'temazepam')
        self.assertEqual(rows['ST7002']['anchor_clock_seconds'],22*3600+55*60)
    def test_clock_text_formats(self):
        self.assertEqual(g.parse_clock('00:38h'),2280)
        self.assertEqual(g.parse_clock('22:15:12'),80112)
        self.assertEqual(g.parse_clock(0.5),43200)
        self.assertEqual(g.parse_clock('24:01'),None)
        self.assertIsNone(g.parse_clock('n/a'))
    def test_clock_midnight_wrap_and_bounds(self):
        h={'psg_clock_start_s':23*3600+30*60,'psg_duration_s':20*3600}
        self.assertEqual(g.match_time(h,38*60),68*60)
        self.assertIsNone(g.match_time({'psg_clock_start_s':0,'psg_duration_s':3600},7200))
    def test_stage_risk_after_explicit_off(self):
        ev=[(0,600,'Sleep stage W'),(600,300,'Sleep stage 1'),(900,30,'Sleep stage 2')]
        status,stats,n2=g.stage_risk_profile(ev,300,1200)
        self.assertEqual(status,'ELIGIBLE_ORIGIN_INDEPENDENT')
        self.assertEqual(n2,900)
        self.assertEqual(stats['latency_s'],600)
        self.assertEqual(stats['positive_h300_n'],10)
    def test_stage_risk_reject_active_n2(self):
        ev=[(0,900,'Sleep stage 2')]
        self.assertEqual(g.stage_risk_profile(ev,300,1200)[0],'N2_ACTIVE_AT_OFF')
    def test_stage_risk_gap_rejection(self):
        ev=[(0,360,'Sleep stage W'),(390,510,'Sleep stage 1'),(900,30,'Sleep stage 2')]
        self.assertEqual(g.stage_risk_profile(ev,300,1200)[0],'SCORING_GAP_OR_OVERLAP')
    def test_stage_risk_unknown_rejection(self):
        ev=[(0,360,'Sleep stage W'),(360,540,'Sleep stage ?'),(900,30,'Sleep stage 2')]
        self.assertEqual(g.stage_risk_profile(ev,300,1200)[0],'UNKNOWN_STAGE_IN_RISK_SET')
    def test_stage_risk_n2_prior_does_not_define_anchor(self):
        ev=[(0,30,'Sleep stage 2'),(30,870,'Sleep stage W'),(900,30,'Sleep stage 2')]
        status,stats,n2=g.stage_risk_profile(ev,300,1200)
        self.assertEqual(status,'ELIGIBLE_ORIGIN_INDEPENDENT')
        self.assertEqual(n2,900)
    def test_no_event_post_off(self):
        ev=[(0,30,'Sleep stage 2'),(30,870,'Sleep stage W')]
        self.assertEqual(g.stage_risk_profile(ev,300,1200)[0],'NO_STAGE2_AFTER_OFF')
    def test_header_200_and_truncated_rejected(self):
        # A ranged EDF read must be HTTP 206, even if server replies with data.
        class FakeResponse:
            status=200
            headers={'Content-Range':'bytes 0-255/1000'}
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self,*a):return b' '*256
        saved=g.urllib.request.urlopen
        g.urllib.request.urlopen=lambda *a,**kw:FakeResponse()
        try:
            with self.assertRaises(RuntimeError):g.range_read('https://example.test/a.edf',0,255,retries=1)
        finally:g.urllib.request.urlopen=saved
    def test_no_rewrites(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'test.txt'
            g.write_immutable(p,b'original')
            g.write_immutable(p,b'original')
            with self.assertRaises(ValueError):g.write_immutable(p,b'altered')
    def test_does_not_infer_anchor_from_first_N2(self):
        x={'psg_clock_start_s':14*3600,'psg_duration_s':90000}
        self.assertIsNone(g.match_time(x,None))
    def test_off_not_after_first_n2(self):
        ev=[(0,1200,'Sleep stage W'),(1200,30,'Sleep stage 2')]
        status,stat,n2=g.stage_risk_profile(ev,1500,2000)
        self.assertEqual(status,'NO_STAGE2_AFTER_OFF')
    def test_sample_time_reference_synthetic_header(self):
        b=bytearray(b' '*768)
        def put(a,length,x):b[a:a+length]=str(x).encode('ascii').ljust(length,b' ')
        put(0,8,'0');put(176,8,'23.30.00');put(168,8,'24.04.89')
        put(184,8,'768');put(236,8,'72000');put(244,8,'1');put(252,4,'2')
        put(256,16,'EEG Fpz-Cz');put(272,16,'Event marker')
        off=256+216*2;put(off,8,'100');put(off+8,8,'1')
        hdr=g.edf_header(bytes(b),768+72000*101*2)
        self.assertEqual(hdr['psg_clock_start_s'],84600)
        self.assertEqual(g.match_time(hdr,2280),4080)
        self.assertEqual(hdr['psg_duration_s'],72000)

if __name__=='__main__':unittest.main()
