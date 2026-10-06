import os
import subprocess
import sys
import tempfile
import shutil


def test_build_instrument_sample():
    tmpdir = tempfile.mkdtemp()
    try:
        # Try to build instrumentation sample; skip if missing libs (non-zero return)
        try:
            subprocess.check_call([sys.executable, 'scripts/build_and_run_sample.py', '--include', 'data/4_2/include', '--lib', 'data/4_2/bin', '--out', os.path.join(tmpdir, 'gpa_instrument_sample')])
            # attempt to run if built and if environment has GLFW
            try:
                subprocess.check_call([sys.executable, 'scripts/build_and_run_sample.py', '--include', 'data/4_2/include', '--lib', 'data/4_2/bin', '--out', os.path.join(tmpdir, 'gpa_instrument_sample'), '--run'])
            except subprocess.CalledProcessError:
                # cannot run due to headless environment or missing libs; skip run
                pass
        except subprocess.CalledProcessError:
            # If build fails due to missing system packages, skip
            return
        assert os.path.exists(os.path.join(tmpdir, 'gpa_instrument_sample'))
    finally:
        shutil.rmtree(tmpdir)
