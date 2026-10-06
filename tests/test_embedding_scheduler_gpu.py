import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class TestEmbeddingSchedulerGPU(unittest.TestCase):
    def test_help_exit_zero(self):
        """`--help` should show usage and exit with 0."""
        p = subprocess.run([PY, str(REPO_ROOT / 'scripts' / 'embedding_scheduler_gpu.py'), '--help'], capture_output=True, text=True)
        # exit should be 0, and output should contain usage text
        self.assertEqual(p.returncode, 0)
        out = (p.stdout or p.stderr or "").lower()
        self.assertTrue('onnx model to run' in out or 'usage' in out)

    def test_missing_onnx_file_exit_nonzero(self):
        """Passing a non-existent --onnx path should return a non-zero code and a clear message."""
        fake = REPO_ROOT / 'tests' / 'no_such_file_hopefully.onnx'
        p = subprocess.run([PY, str(REPO_ROOT / 'scripts' / 'embedding_scheduler_gpu.py'), '--onnx', str(fake)], capture_output=True, text=True)
        # The script is designed to return 2 when the ONNX file doesn't exist
        self.assertNotEqual(p.returncode, 0)
        out = (p.stdout or p.stderr or "").lower()
        self.assertTrue('does not exist' in out)


if __name__ == '__main__':
    unittest.main()
