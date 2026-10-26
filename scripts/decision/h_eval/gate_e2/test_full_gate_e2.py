"""Offline end-to-end validation: 197 synthetic, SHA-bound hypnograms; no real dataset claims."""
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from xls_stdlib import read_sheets
import gate_e2 as g

FIX=pathlib.Path(__file__).parent/'fixtures'
HERE=pathlib.Path(__file__).resolve()
DEFAULT_OLD=(HERE.parents[4]/'scripts/decision/h_eval/gate_e/gate_e.py') if len(HERE.parents)>4 else pathlib.Path('/missing-old-gate-e')
OLD=pathlib.Path(os.environ.get('HEVAL_ORIGINAL_GATE_E',str(DEFAULT_OLD)))

def sample_file(annotations,record_duration=0):
    tal=b''.join((('+'+str(t)+'\x15'+str(d)+'\x14'+lab+'\x14\x00').encode('ascii') for t,d,lab in annotations))
    if len(tal)>400:raise ValueError('Fixture TAL too long')
    h=bytearray(b' '*512)
    def put(a,n,x):h[a:a+n]=str(x).encode().ljust(n,b' ')
    put(0,8,'0');put(184,8,512);put(236,8,1);put(244,8,record_duration);put(252,4,1)
    h[192:197]=b'EDF+C';put(256,16,'EDF Annotations');put(472,8,200)
    return bytes(h)+tal.ljust(400,b'\x00')


class FullDryRun(unittest.TestCase):
 def test_all_197_synthetic_replay_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   repo=pathlib.Path(d)/'eeg-sleep-onset-forecasting';(repo/'.git').mkdir(parents=True)
   path=repo/'scripts/decision/h_eval/gate_e/gate_e.py';path.parent.mkdir(parents=True)
   shutil.copyfile(OLD,path)
   inputs=repo/'results/decision/h_eval/gate_e';inputs.mkdir(parents=True)
   up=repo/'results/decision/h_eval/gate_d/gate_d_preflight.json';up.parent.mkdir(parents=True);up.write_text('{}')
   root=repo/'data/raw/sleep_edfx_v1_0_0';(root/'metadata').mkdir(parents=True)
   a,_=g.identify_sc(read_sheets(FIX/'SC-mock.xls'))
   b,_=g.identify_st(read_sheets(FIX/'ST-mock.xls'))
   assert len(a)==153 and len(b)==44
   rec=[];dig=[];inv=[]
   fake=sample_file([(0,12000,'Sleep stage W'),(12000,2400,'Sleep stage 1'),(14400,300,'Sleep stage 2')],record_duration=0)
   for prefix,study,entries in [('SC','sleep-cassette',a),('ST','sleep-telemetry',b)]:
    for key in sorted(entries):
     rid=key+'E0' if prefix=='SC' else key+'J0'
     hyp=key+('EC' if prefix=='SC' else 'JH')
     psg=f'{study}/{rid}-PSG.edf';hp=f'{study}/{hyp}-Hypnogram.edf'
     rec.append(psg);dig.append('0'*64+'  '+psg)
     dig.append(hashlib.sha256(fake).hexdigest()+'  '+hp)
     p=root/hp;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(fake)
     inv.append({'recording':key,'subject_key':key[:5],'study':study,'night':key[5],
       'psg_path':psg,'hypnogram_path':hp,'status':'PARSED_SHA_VERIFIED'})
   raw=('\n'.join(rec)+'\n').encode();(root/'metadata/RECORDS').write_bytes(raw)
   dig.append(hashlib.sha256(raw).hexdigest()+'  RECORDS')
   for filename,fx in [('SC-subjects.xls','SC-mock.xls'),('ST-subjects.xls','ST-mock.xls')]:
    z=(FIX/fx).read_bytes();(root/'metadata'/filename).write_bytes(z)
    dig.append(hashlib.sha256(z).hexdigest()+'  '+filename)
   (root/'metadata/SHA256SUMS.txt').write_text('\n'.join(dig)+'\n')
   (inputs/'sleep_edf_eligibility.json').write_text(json.dumps({'hypnograms_parsed_sha_verified_n':197,'hypnograms_with_explicit_lights_off_n':0}))
   import csv
   with (inputs/'sleep_edf_recording_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)
   orig=g.probe_header
   def fakeprobe(psg):
    return {'psg_clock_start_s':21*3600,'psg_clock_start':'21.00.00',
      'psg_duration_s':36000,'header_sha256':'f'*64,'psg_size_bytes':123456}
   g.probe_header=fakeprobe
   try:
    o=repo/'results/decision/h_eval/gate_e2'
    first=g.execute(repo,o,workers=1)
    self.assertEqual(first['recording_inventory_n'],197)
    self.assertEqual(first['matched_metadata_n'],197)
    self.assertGreaterEqual(first['origin_independent_eligible_recordings_n'],150)
    self.assertTrue(first['flags']['no_full_PSG_download'])
    second=g.execute(repo,o,workers=1)
    self.assertEqual(first,second)
    self.assertTrue((o/'sleep_edf_origin_inventory.csv').exists())
   finally:g.probe_header=orig

if __name__=='__main__':unittest.main()
