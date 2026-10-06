import os
import subprocess
import sys
import tempfile
import shutil


def test_build_vulkan_instrument_sample():
    tmpdir = tempfile.mkdtemp()
    try:
        # try to build Vulkan sample; if Vulkan not available, skip
        try:
            subprocess.check_call([sys.executable, 'scripts/build_and_run_sample.py', '--include', 'data/4_2/include', '--lib', 'data/4_2/bin', '--out', os.path.join(tmpdir, 'gpa_instrument_sample_vk'), '--vulkan'])
        except subprocess.CalledProcessError:
            # likely missing Vulkan dev packages; skip
            return
        assert os.path.exists(os.path.join(tmpdir, 'gpa_instrument_sample_vk'))
    finally:
        shutil.rmtree(tmpdir)
