import os
import json
import tempfile
import subprocess
import sys


def test_convert_applies_derived_formula_and_normalizes():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'gpa_rps_output_0.csv')
        with open(csv_path, 'w') as f:
            f.write('iter,counter_0,counter_1\n')
            f.write('0,10,2\n')
            f.write('1,30,3\n')

        # meta with device id 42
        meta_path = csv_path + '.meta.json'
        with open(meta_path, 'w') as m:
            json.dump({'device_id': 42, 'counters': []}, m)

        # derived config: compute ratio counter_0/counter_1 -> name: c_ratio
        derived_cfg_path = os.path.join(tmpdir, 'derived.json')
        derived_cfg = {'derived': [{'name': 'c_ratio', 'expr': 'counter_0 / counter_1'}]}
        with open(derived_cfg_path, 'w') as d:
            json.dump(derived_cfg, d)

        # stats.json for device 42 derived c_ratio values -> 5 and 10 -> median 7.5 iqr 5
        stats = {
            'collected_dir': tmpdir,
            'devices': {
                '42': {
                    'c_ratio': {'median': 7.5, 'iqr': 5.0}
                }
            },
            'global': {}
        }
        stats_path = os.path.join(tmpdir, 'stats.json')
        with open(stats_path, 'w') as s:
            json.dump(stats, s)

        out = os.path.join(tmpdir, 'out_dataset.csv')
        cmd = [sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_path, '--out', out, '--as-numeric', '--stats', stats_path, '--norm-method', 'robust', '--derived', derived_cfg_path]
        subprocess.check_call(cmd)
        numeric_out = out + '.numeric.csv'
        assert os.path.exists(numeric_out)
        with open(numeric_out, 'r') as nf:
            rows = [r.strip().split(',') for r in nf.readlines()]

        # header + 2 rows
        assert len(rows) == 3
        # find column index for c_ratio
        header = rows[0]
        idx = header.index('c_ratio')
        assert abs(float(rows[1][idx]) - (-0.5)) < 1e-6
        assert abs(float(rows[2][idx]) - (0.5)) < 1e-6
