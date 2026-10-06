import os
import json
import tempfile
import subprocess
import sys


def test_convert_applies_usage_mapping_before_normalization():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'gpa_rps_output_0.csv')
        with open(csv_path, 'w') as f:
            f.write('iter,counter_0,counter_1\n')
            # percentage (50 -> 0.5), bytes (1048576 -> 1.0 MB)
            f.write('0,50,1048576\n')
            # percentage (100 -> 1.0), bytes (2097152 -> 2.0 MB)
            f.write('1,100,2097152\n')

        # meta with device id 42 and usage mappings
        meta_path = csv_path + '.meta.json'
        meta = {
            'device_id': 42,
            'counters': [
                {'index': 0, 'name': 'counter_0', 'usage': 'kGpaUsageTypePercentage'},
                {'index': 1, 'name': 'counter_1', 'usage': 'kGpaUsageTypeBytes'},
            ]
        }
        with open(meta_path, 'w') as m:
            json.dump(meta, m)

        # stats.json for device 42 using converted units (percentage->fraction, bytes->MB)
        # counter_0 values map -> 0.5 and 1.0, choose median 0.75 iqr 0.5
        # counter_1 values map -> 1.0 and 2.0, choose median 1.5 iqr 1.0
        stats = {
            'collected_dir': tmpdir,
            'devices': {
                '42': {
                    '0': {'median': 0.75, 'iqr': 0.5},
                    '1': {'median': 1.5, 'iqr': 1.0}
                }
            },
            'global': {}
        }
        stats_path = os.path.join(tmpdir, 'stats.json')
        with open(stats_path, 'w') as s:
            json.dump(stats, s)

        out = os.path.join(tmpdir, 'out_dataset.csv')
        cmd = [sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_path, '--out', out, '--as-numeric', '--stats', stats_path, '--norm-method', 'robust']
        subprocess.check_call(cmd)
        numeric_out = out + '.numeric.csv'
        assert os.path.exists(numeric_out)
        with open(numeric_out, 'r') as nf:
            rows = [r.strip().split(',') for r in nf.readlines()]

        # header + 2 rows
        assert len(rows) == 3

        # After mapping and robust normalization we expect -0.5 and 0.5 for both counters
        assert abs(float(rows[1][1]) - (-0.5)) < 1e-6
        assert abs(float(rows[2][1]) - (0.5)) < 1e-6

        assert abs(float(rows[1][2]) - (-0.5)) < 1e-6
        assert abs(float(rows[2][2]) - (0.5)) < 1e-6
