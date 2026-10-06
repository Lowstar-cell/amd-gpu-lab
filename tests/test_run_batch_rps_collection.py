import os
import subprocess
import sys
import tempfile
import shutil


def test_run_batch_rps_collection_dry():
    rps_root = os.path.abspath('data/RenderPipelineShaders-main')
    assert os.path.exists(rps_root)
    with tempfile.TemporaryDirectory() as tmpdir:
        # run a small batch with dry-run; it should iterate over multiple candidate files
        cmd = [sys.executable, 'scripts/run_batch_rps_collection.py', '--rps-root', rps_root, '--dry-run', '--iterations', '2']
        subprocess.check_call(cmd)
