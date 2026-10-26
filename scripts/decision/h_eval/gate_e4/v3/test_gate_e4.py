import hashlib
import importlib
import io
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import gate_e4 as m


def new_edf(path, nrec=20, dr=30, rate=100, digmin=-32768,digmax=32767):
    labels=['EEG Fpz-Cz','EEG Pz-Oz']
    n=2
    dat=bytearray(b' '* (256*(n+1)))
    def put(start,width,value):
        b=str(value).encode('latin1')
        assert len(b)<=width
        dat[start:start+width]=b.ljust(width,b' ')
    put(0,8,'0');put(184,8,256*(n+1));put(236,8,nrec);put(244,8,dr);put(252,4,n)
    pos=256
    info=[('label',16,labels),('trans',80,['','']),('unit',8,['uV','uV']),
      ('pmin',8,[-100.,-100.]),('pmax',8,[100.,100.]),
      ('dmin',8,[digmin,digmin]),('dmax',8,[digmax,digmax]),
      ('pref',80,['','']),('samples',8,[dr*rate,dr*rate]),('res',32,['',''])]
    for name,width,vals in info:
        for v in vals:
            b=str(v).encode('ascii')
            assert len(b)<=width
            dat[pos:pos+width]=b.ljust(width,b' ');pos+=width
    assert pos==len(dat)
    with path.open('wb') as f:
        f.write(dat)
        for i in range(nrec):
            for ch in range(2):
                v=[int((3000 if ch==0 else -5000)+(2000 if j%2 else -2000)) for j in range(dr*rate)]
                f.write(struct.pack('<'+str(len(v))+'h',*v))


class TestE4(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory();self.addCleanup(self.d.cleanup)
        self.root=Path(self.d.name)
    def test_parser_two_channels_calibration(self):
        p=self.root/'a.edf';new_edf(p)
        with p.open('rb') as f:
            h=m.edf_header(f,p.stat().st_size)
            self.assertEqual([z[1]['rate_hz'] for z in h['eeg_channels']],[100,100])
            self.assertEqual(h['duration_s'],600)
            a,d=m.read_window(f,h,h['eeg_channels'][0],30,60)
            self.assertEqual(len(a),3000)
            self.assertEqual(d[0],1000)
            self.assertEqual(d[1],5000)
            result=m.qc_window(f,h,h['eeg_channels'][1],150,180)
            self.assertGreater(result['rms_ac_uV'],0)
            self.assertEqual(result['non_finite_samples'],0)
    def test_shallow_header_rejected(self):
        p=self.root/'a.edf';p.write_bytes(b'0'+b' ' * 255)
        with p.open('rb') as f:
            with self.assertRaises(ValueError):m.edf_header(f,p.stat().st_size)
    def test_wrong_size_rejected(self):
        p=self.root/'a.edf';new_edf(p);p.write_bytes(p.read_bytes()[:-2])
        with p.open('rb') as f:
            with self.assertRaisesRegex(ValueError,'physical size'):m.edf_header(f,p.stat().st_size)
    def test_bad_montage_rejected(self):
        p=self.root/'a.edf';new_edf(p)
        with p.open('r+b') as f:f.seek(256);f.write(b'ECG'.ljust(16,b' '))
        with p.open('rb') as f:
            with self.assertRaisesRegex(ValueError,'Fpz-Cz'):m.edf_header(f,p.stat().st_size)
    def test_no_overwrite_invalid_local(self):
        p=self.root/'a.edf';p.write_bytes(b'INVALID')
        with self.assertRaisesRegex(ValueError,'EXISTING PSG FAILED SHA'):
            m.resume_download('https://example.invalid/',p,'0'*64,10)
    def test_existing_sha_verified_idempotent(self):
        p=self.root/'a.edf';new_edf(p)
        self.assertEqual('already_verified',m.resume_download('https://example.invalid/',p,m.sha(p),p.stat().st_size))
    def test_landmarks_strictly_before_event(self):
        ts=m.choose_landmarks({'lights_off_rel_s':30,'first_n2_after_off_s':900,'landmark_n':28})
        self.assertEqual(len(ts),5)
        self.assertTrue(all(60<=x<900 for x in ts))
    def test_does_not_touch_test_or_ST(self):
        source=Path(m.__file__).read_text()
        self.assertIn("split!='SC_TRAIN'",source)
        self.assertIn("row['study']!='sleep-cassette'",source)
    def test_manifest_rejects_duplicates(self):
        p=self.root/'SHA256SUMS.txt';p.write_text('0'*64+'  one\n'+'1'*64+'  one\n')
        with self.assertRaises(ValueError):m.manifest_hashes(p)


if __name__=='__main__':unittest.main()
