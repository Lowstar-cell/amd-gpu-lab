import os
import json
import tempfile
import subprocess
import sys


def test_discover_sdk_derived_and_convert():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create an RST-like source file (help/_sources) with counters
        srcdir = os.path.join(tmpdir, 'help', '_sources')
        os.makedirs(srcdir, exist_ok=True)
        rst = os.path.join(srcdir, 'graphics_counter_tables_sample.rst.txt')
        with open(rst, 'w') as f:
            f.write('"CacheHitPercent", "Discrete", "Percentage", "Cache hit percentage computed from hits and requests"\n')
            f.write('"CacheHits", "Discrete", "Items", "Number of cache hits"\n')
            f.write('"CacheRequests", "Discrete", "Items", "Number of cache requests"\n')

        derived_out = os.path.join(tmpdir, 'derived.json')
        subprocess.check_call([sys.executable, 'scripts/discover_derived_counters.py', tmpdir, '--out', derived_out])

        with open(derived_out, 'r') as df:
            dj = json.load(df)
        # expect CacheHitPercent derived expression to be created
        names = [d.get('name') for d in dj.get('derived', [])]
        assert 'CacheHitPercent' in names
        # check that expr exists and references CacheHits and CacheRequests
        exprs = {d['name']: d['expr'] for d in dj.get('derived', [])}
        assert 'CacheHits' in exprs['CacheHitPercent'] and 'CacheRequests' in exprs['CacheHitPercent']

        # now produce a sample CSV and meta.json that maps index 0->CacheHits, 1->CacheRequests
        csv_path = os.path.join(tmpdir, 'gpa_rps_output_0.csv')
        with open(csv_path, 'w') as cf:
            cf.write('iter,counter_0,counter_1\n')
            cf.write('0,10,20\n')
            cf.write('1,5,10\n')

        meta = {'device_id': 42, 'counters': [{'index': 0, 'name': 'CacheHits', 'usage': 'kGpaUsageTypeItems'}, {'index': 1, 'name': 'CacheRequests', 'usage': 'kGpaUsageTypeItems'}]}
        with open(csv_path + '.meta.json', 'w') as m:
            json.dump(meta, m)

        out = os.path.join(tmpdir, 'out.csv')
        cmd = [sys.executable, 'scripts/convert_counter_csv_to_dataset.py', '--in', csv_path, '--out', out, '--as-numeric', '--derived', derived_out]
        subprocess.check_call(cmd)

        numeric_out = out + '.numeric.csv'
        assert os.path.exists(numeric_out)
        with open(numeric_out, 'r') as nf:
            rows = [r.strip().split(',') for r in nf.readlines()]
        assert rows[0][3] == 'CacheHitPercent'
        # numeric values: 10/20 = 0.5 and 5/10 = 0.5
        assert abs(float(rows[1][3]) - 0.5) < 1e-6
        assert abs(float(rows[2][3]) - 0.5) < 1e-6
