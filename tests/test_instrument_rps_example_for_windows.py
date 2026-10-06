import os
import shutil
import subprocess
import tempfile
import sys


def test_instrument_rps_example_for_windows():
    tmpdir = tempfile.mkdtemp()
    try:
        src = os.path.abspath('data/RenderPipelineShaders-main/examples/hello_triangle.cpp')
        work = os.path.join(tmpdir, 'hello_triangle.cpp')
        shutil.copy(src, work)
        # run patch script on copy
        subprocess.check_call([sys.executable, 'scripts/instrument_rps_example_for_windows.py', '--example', work])
        # check backup exist
        assert os.path.exists(work + '.bak')
        # read patched file and ensure includes present
        with open(work, 'r', encoding='utf-8') as f:
            s = f.read()
        assert 'gpu_perf_api.h' in s
        assert 'GpaInitialize' in s
    finally:
        shutil.rmtree(tmpdir)
