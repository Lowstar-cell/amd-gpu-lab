import os
import subprocess
import tempfile
import shutil
import sys


def test_instrument_gpa_fallback():
    tmpdir = tempfile.mkdtemp()
    try:
        # run the instrument script which will either produce a CSV by probing or fallback to HTML parser
        subprocess.check_call([sys.executable, 'scripts/instrument_gpa_counters.py', '--libdir', 'data/4_2/bin', '--include', 'data/4_2/include', '--out', os.path.join(tmpdir, 'gpa_counters.csv'), '--build'])
        # Verify output exists either CSV or JSON
        assert os.path.exists('data/gpa_functions.json') or os.path.exists(os.path.join(tmpdir, 'gpa_counters.csv'))
    finally:
        shutil.rmtree(tmpdir)
