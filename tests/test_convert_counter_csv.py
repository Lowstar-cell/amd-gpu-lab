import os
import tempfile
import shutil
import csv
import subprocess
import sys


def test_convert_counter_csv_to_dataset():
    tmpdir = tempfile.mkdtemp()
    try:
        input_csv = os.path.join(tmpdir, 'gpa_sample.csv')
        with open(input_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter','counter_0','counter_1'])
            writer.writerow(['0','10','200'])
        out_csv = os.path.join(tmpdir, 'gpa_dataset.csv')
        subprocess.check_call([sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', input_csv, '--out', out_csv])
        assert os.path.exists(out_csv)
        with open(out_csv,'r') as f:
            rows = f.read().strip().splitlines()
            assert len(rows) > 1
    finally:
        shutil.rmtree(tmpdir)
