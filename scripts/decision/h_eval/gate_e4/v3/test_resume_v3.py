import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gate_e4 as m
from test_gate_e4 import new_edf


class Response(io.BytesIO):
    def __init__(self, payload, status=200, content_range=None, fail_after=None):
        super().__init__(payload)
        self.status = status
        self.headers = {}
        if content_range:
            self.headers['Content-Range'] = content_range
        self.fail_after = fail_after
        self.failed = False

    def read(self, n=-1):
        if self.fail_after is not None and not self.failed and self.tell() >= self.fail_after:
            self.failed = True
            raise TimeoutError('simulated slow PhysioNet connection')
        if self.fail_after is not None and not self.failed:
            n = min(n, self.fail_after - self.tell())
        return super().read(n)


class LiveResumeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.target = Path(self.t.name) / 'SC4022E0-PSG.edf'
        self.part = self.target.with_suffix('.edf.e4-part')
        self.payload = bytes((i * 41 + 19) % 256 for i in range(9001))
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def call(self, serve, **kwargs):
        with patch.object(m.urllib.request,'urlopen',side_effect=serve), patch.object(m.time,'sleep'):
            return m.resume_download('https://example.invalid/file.edf', self.target, self.digest,
                                     len(self.payload), chunk=101, **kwargs)

    def test_repeated_timeouts_with_progress_recover(self):
        sizes = [1300, 2900, 5000, 7000, len(self.payload)]
        calls=[]
        def serve(req, timeout):
            header = req.headers.get('Range')
            start = int(header[6:].split('-')[0]) if header else 0
            calls.append((start, timeout))
            cap = sizes[len(calls)-1]
            stop = min(cap, len(self.payload))
            return Response(self.payload[start:stop], 206 if start else 200,
                 f'bytes {start}-{len(self.payload)-1}/{len(self.payload)}' if start else None,
                 fail_after=stop-start if stop < len(self.payload) else None)
        self.assertEqual(self.call(serve,max_consecutive_stalls=2),'downloaded_sha_verified')
        self.assertEqual([x[0] for x in calls],[0,1300,2900,5000,7000])
        self.assertEqual(self.target.read_bytes(),self.payload)
        self.assertFalse(self.part.exists())

    def test_stall_budget_preserves_nonzero_partial(self):
        self.part.write_bytes(self.payload[:300])
        def bad(req,timeout):
            raise TimeoutError('no bytes delivered')
        with self.assertRaisesRegex(RuntimeError,'No download progress'):
            self.call(bad,max_consecutive_stalls=3,max_reconnects=7)
        self.assertEqual(self.part.read_bytes(),self.payload[:300])

    def test_consecutive_stall_counter_resets_after_progress(self):
        self.part.write_bytes(self.payload[:123])
        attempts=0
        def serve(req,timeout):
            nonlocal attempts
            attempts+=1
            start=int(req.headers['Range'][6:].split('-')[0])
            if attempts in (1,2,4,5):raise TimeoutError('network silent')
            if attempts==3:
                return Response(self.payload[start:start+700],206,
                                f'bytes {start}-{len(self.payload)-1}/{len(self.payload)}',
                                fail_after=700)
            return Response(self.payload[start:],206,f'bytes {start}-{len(self.payload)-1}/{len(self.payload)}')
        self.assertEqual(self.call(serve,max_consecutive_stalls=3),'downloaded_sha_verified')
        self.assertEqual(attempts,6)

    def test_bad_range_does_not_append_or_retry(self):
        self.part.write_bytes(self.payload[:100])
        calls=[]
        def bad(req,timeout):
            calls.append(1)
            return Response(self.payload,200)
        with self.assertRaisesRegex(ValueError,'Range resume refused'):
            self.call(bad)
        self.assertEqual(len(calls),1)
        self.assertEqual(self.part.read_bytes(),self.payload[:100])

    def test_exceed_max_reconnects_preserves_partial(self):
        def bad(req,timeout):raise TimeoutError('socket timed out')
        with self.assertRaisesRegex(RuntimeError,'No download progress|safety budget'):
            self.call(bad,max_consecutive_stalls=8,max_reconnects=2)
        if self.part.exists():
            self.assertEqual(self.part.stat().st_size,0)

    def test_refuse_symlink_final(self):
        other=Path(self.t.name)/'other';other.write_bytes(self.payload)
        self.target.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'symlink completed'):
            self.call(lambda req,timeout:self.fail('unexpected network'))

    def test_full_synthetic_qc_train_only_idempotent(self):
        repo=Path(self.t.name)/'repo'; (repo/'.git').mkdir(parents=True)
        base=repo/m.SOURCE_PATH/'sleep-cassette';base.mkdir(parents=True)
        rows=[]
        for i in range(6):
            rec=f'SC40{i:02}1';sub=f'SC40{i:02}'
            relpath=f'sleep-cassette/{rec}E0-PSG.edf'
            p=repo/m.SOURCE_PATH/relpath
            new_edf(p,nrec=20)
            rows.append({'recording':rec,'subject_key':sub,'psg_path':relpath,
                         'source_sha256':m.sha(p),'expected_bytes':p.stat().st_size,
                         'lights_off_rel_s':0.0,'first_n2_after_off_s':390.0,
                         'latency_s':390.0,'landmark_n':12,'study':'sleep-cassette'})
        out=repo/'.local/h_eval/gate_e4/outputs-v3'
        with patch.object(m,'entries',return_value=(rows,{'source':'synthetic_test'})):
            res,details=m.run(repo,out,offline=True)
            self.assertEqual(len(res['sampled_train_recordings']),6)
            self.assertEqual(len(details),60)
            orig1=(out/'gate_e4_signal_qc.json').read_bytes()
            orig2=(out/'gate_e4_signal_windows.csv').read_bytes()
            m.run(repo,out,offline=True)
        self.assertEqual((out/'gate_e4_signal_qc.json').read_bytes(),orig1)
        self.assertEqual((out/'gate_e4_signal_windows.csv').read_bytes(),orig2)

if __name__=='__main__':unittest.main()
