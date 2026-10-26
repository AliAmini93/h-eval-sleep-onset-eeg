"""No EEG, network or model access; check fail-closed release behavior."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from release_preflight import inspect,REQUIRED_SOURCE,NEEDED_TO_REPLAY

class ReleasePreflightTests(unittest.TestCase):
    def seed(self,root):
        for name in REQUIRED_SOURCE:
            p=root/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text("x=1\n")
    def test_source_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root)
            self.assertEqual(inspect(root,'source')['verdict'],'SOURCE_LAYOUT_AND_SYNTAX_PASS_ONLY')
    def test_missing_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root);(root/REQUIRED_SOURCE[0]).unlink()
            self.assertEqual(inspect(root,'source')['verdict'],'SOURCE_BLOCKED')
    def test_bad_syntax(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root);(root/REQUIRED_SOURCE[0]).write_text("def x(:")
            self.assertIn(REQUIRED_SOURCE[0],inspect(root,'source')['python_parse_failures'])
    def test_private_inputs_block(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root)
            out=inspect(root,'replay')
            self.assertEqual(out['verdict'],'REPLAY_BLOCKED_PRIVATE_ARTIFACTS')
            self.assertEqual(out['missing_replay_inputs'],list(NEEDED_TO_REPLAY))
    def test_presence_does_not_prove_reproduction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root)
            for name in NEEDED_TO_REPLAY:
                f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.touch()
            self.assertEqual(inspect(root,'replay')['verdict'],'REPLAY_INPUTS_PRESENT_NOT_VALIDATED')
if __name__=='__main__':
    unittest.main()
