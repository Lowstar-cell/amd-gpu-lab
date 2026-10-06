import os
import tempfile
import csv
import subprocess
import sys


def test_label_single_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        numeric = os.path.join(tmpdir, 'data.numeric.csv')
        with open(numeric, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'counter_0', 'label'])
            w.writerow(['0', '0.1', '0'])
            w.writerow(['1', '0.2', '0'])

        out = os.path.join(tmpdir, 'out.csv')
        cmd = [sys.executable, 'scripts/label_traces.py', '--in', numeric, '--out', out, '--set-label', '2', '--sample-ids', '1']
        subprocess.check_call(cmd)

        with open(out, 'r', newline='') as f:
            r = list(csv.DictReader(f))
            assert r[0]['sample_id'] == '0'
            assert r[0]['label'] == '0'
            assert r[1]['sample_id'] == '1'
            assert r[1]['label'] == '2'


def test_label_range_and_inplace():
    with tempfile.TemporaryDirectory() as tmpdir:
        numeric = os.path.join(tmpdir, 'data.numeric.csv')
        with open(numeric, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'counter_0', 'label'])
            w.writerow(['0', '0.1', '0'])
            w.writerow(['1', '0.2', '0'])
            w.writerow(['2', '0.3', '0'])

        cmd = [sys.executable, 'scripts/label_traces.py', '--in', numeric, '--inplace', '--set-label', '1', '--sample-range', '1-2']
        subprocess.check_call(cmd)

        # file should be updated in place
        with open(numeric, 'r', newline='') as f:
            r = list(csv.DictReader(f))
            assert r[0]['label'] == '0'
            assert r[1]['label'] == '1'
            assert r[2]['label'] == '1'
