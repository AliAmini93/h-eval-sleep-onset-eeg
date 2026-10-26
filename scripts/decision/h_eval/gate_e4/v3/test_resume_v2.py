import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gate_e4 as m

class Response(io.BytesIO):
    def __init__(self, payload, status=200, content_range=None):
        super().__init__(payload)
        self.status=status
        self.headers={}
        if content_range:self.headers['Content-Range']=content_range

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dest=Path(self.temp.name)/'SC4012E0-PSG.edf'
        self.part=self.dest.with_suffix('.edf.e4-part')
        self.data=bytes((i*53+17)%256 for i in range(3193))
        self.digest=hashlib.sha256(self.data).hexdigest()

    def download(self, mocked):
        with patch.object(m.urllib.request,'urlopen',side_effect=mocked):
            return m.resume_download('https://example.invalid/SC4012E0-PSG.edf',self.dest,self.digest,len(self.data),chunk=89)

    def test_existing_empty_part_can_resume_at_zero(self):
        self.part.touch()
        calls=[]
        def serve(req,timeout):
            calls.append(req.headers.get('Range'))
            return Response(self.data)
        self.assertEqual(self.download(serve),'downloaded_sha_verified')
        self.assertEqual(calls,[None])
        self.assertEqual(self.dest.read_bytes(),self.data)
        self.assertFalse(self.part.exists())

    def test_existing_nonzero_part_resumes_from_exact_byte(self):
        n=1123
        self.part.write_bytes(self.data[:n])
        def serve(req,timeout):
            self.assertEqual(req.headers.get('Range'),f'bytes={n}-{len(self.data)-1}')
            return Response(self.data[n:],206,f'bytes {n}-{len(self.data)-1}/{len(self.data)}')
        self.assertEqual(self.download(serve),'downloaded_sha_verified')
        self.assertEqual(self.dest.read_bytes(),self.data)

    def test_bad_resume_http_200_keeps_partial_intact(self):
        self.part.write_bytes(self.data[:51])
        with self.assertRaisesRegex(ValueError,'Range resume refused'):
            self.download(lambda req,timeout:Response(self.data))
        self.assertEqual(self.part.read_bytes(),self.data[:51])
        self.assertFalse(self.dest.exists())

    def test_complete_part_sha_is_checked_before_publish(self):
        self.part.write_bytes(self.data)
        self.assertEqual(self.download(lambda req,timeout:self.fail('unexpected HTTP')),'downloaded_sha_verified')
        self.assertEqual(self.dest.read_bytes(),self.data)

    def test_bad_sha_keeps_full_partial(self):
        self.part.write_bytes(b'X'*len(self.data))
        with self.assertRaisesRegex(ValueError,'SHA FAILED'):
            self.download(lambda req,timeout:self.fail('unexpected HTTP'))
        self.assertFalse(self.dest.exists())
        self.assertTrue(self.part.exists())

    def test_verified_target_is_reused_without_network(self):
        self.dest.write_bytes(self.data)
        self.assertEqual(self.download(lambda req,timeout:self.fail('unexpected HTTP')),'already_verified')

    def test_no_overwrite_of_existing_invalid_target(self):
        self.dest.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'EXISTING PSG FAILED SHA'):
            self.download(lambda req,timeout:self.fail('unexpected HTTP'))
        self.assertEqual(self.dest.read_bytes(),b'bad')

    def test_reject_oversized_partial(self):
        self.part.write_bytes(self.data+b'X')
        with self.assertRaisesRegex(ValueError,'Part file exceeds'):
            self.download(lambda req,timeout:self.fail('unexpected HTTP'))
        self.assertFalse(self.dest.exists())

    def test_reject_partial_symlink(self):
        another=Path(self.temp.name)/'other'
        another.write_bytes(self.data)
        self.part.symlink_to(another)
        with self.assertRaisesRegex(ValueError,'symlink partial'):
            self.download(lambda req,timeout:self.fail('unexpected HTTP'))
        self.assertEqual(another.read_bytes(),self.data)

    def test_transfer_short_then_resumes_in_same_invocation(self):
        count=0
        def serve(req,timeout):
            nonlocal count
            count+=1
            if count==1:
                self.assertIsNone(req.headers.get('Range'))
                return Response(self.data[:1500])
            self.assertEqual(req.headers.get('Range'),f'bytes=1500-{len(self.data)-1}')
            return Response(self.data[1500:],206,f'bytes 1500-{len(self.data)-1}/{len(self.data)}')
        self.assertEqual(self.download(serve),'downloaded_sha_verified')
        self.assertEqual(count,2)
        self.assertEqual(self.dest.read_bytes(),self.data)

if __name__=='__main__': unittest.main()
