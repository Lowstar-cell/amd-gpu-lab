import os
import json
import tempfile
import shutil
import sys


def test_compute_counter_stats_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two sample CSVs
        for i in range(2):
            csv_path = os.path.join(tmpdir, f'gpa_rps_output_{i}.csv')
            with open(csv_path, 'w') as f:
                f.write('iter,counter_0,counter_1\n')
                f.write('0,10,100\n')
                f.write('1,20,120\n')
            # add minimal meta JSON
            meta_path = csv_path + '.meta.json'
            with open(meta_path, 'w') as m:
                m.write('{"device_id": 0, "counters": [{"index":0, "name":"counter_0", "usage":"kGpaUsageTypeItems"}, {"index":1, "name":"counter_1", "usage":"kGpaUsageTypeItems"}]}')

        out_stats = os.path.join(tmpdir, 'stats.json')
        ret = os.system(f"{sys.executable} scripts/compute_counter_stats.py {tmpdir} --out {out_stats}")
        assert ret == 0
        assert os.path.exists(out_stats)
        with open(out_stats, 'r') as f:
            s = json.load(f)
        assert 'devices' in s
        assert 'global' in s
        # device 0 entries should exist
        assert '0' in s['devices']
        assert '0' in s['global']
