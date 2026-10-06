import os
import json
import tempfile
import subprocess
import sys


def test_convert_numeric_robust():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'gpa_rps_output_0.csv')
        with open(csv_path, 'w') as f:
            f.write('iter,counter_0,counter_1\n')
            f.write('0,10,100\n')
            f.write('1,20,120\n')

        # meta with device id 42
        meta_path = csv_path + '.meta.json'
        with open(meta_path, 'w') as m:
            json.dump({'device_id': 42, 'counters': []}, m)

        # stats.json with device 42 counter stats: median and iqr
        stats = {'collected_dir': tmpdir, 'devices': {'42': {'0': {'median': 15.0, 'iqr': 10.0}, '1': {'median': 110.0, 'iqr': 20.0}}}, 'global': {}}
        stats_path = os.path.join(tmpdir, 'stats.json')
        with open(stats_path, 'w') as s:
            json.dump(stats, s)

        out = os.path.join(tmpdir, 'out_dataset.csv')
        cmd = [sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_path, '--out', out, '--as-numeric', '--stats', stats_path, '--norm-method', 'robust']
        subprocess.check_call(cmd)
        numeric_out = out + '.numeric.csv'
        assert os.path.exists(numeric_out)
        # read numeric values and check normalization: for counter_0 values 10 and 20 with median 15 and iqr 10 -> (-0.5, 0.5)
        with open(numeric_out, 'r') as nf:
            rows = [r.strip().split(',') for r in nf.readlines()]
        # header + 2 rows
        assert len(rows) == 3
        # check values in rows: header columns sample_id, counter_0, counter_1, label
        assert abs(float(rows[1][1]) - ((10 - 15.0) / 10.0)) < 1e-6
        assert abs(float(rows[2][1]) - ((20 - 15.0) / 10.0)) < 1e-6
