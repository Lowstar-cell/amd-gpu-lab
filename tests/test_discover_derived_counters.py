import os
import json
import tempfile
import subprocess
import sys


def test_discover_from_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        jf = os.path.join(tmpdir, 'counters.json')
        data = {'derived': [{'name': 'c_ratio', 'expr': 'counter_0 / counter_1'}]}
        with open(jf, 'w') as f:
            json.dump(data, f)

        out = os.path.join(tmpdir, 'derived_out.json')
        cmd = [sys.executable, 'scripts/discover_derived_counters.py', tmpdir, '--out', out]
        subprocess.check_call(cmd)
        assert os.path.exists(out)
        with open(out, 'r') as of:
            j = json.load(of)
        assert any(d.get('name') == 'c_ratio' for d in j.get('derived', []))


def test_discover_from_text_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        hf = os.path.join(tmpdir, 'sample.h')
        with open(hf, 'w') as f:
            f.write('// Derived formula: my_ratio = counter_0 / counter_1\n')
            f.write('/* another = 1 + 2 */\n')

        out = os.path.join(tmpdir, 'derived_out.json')
        cmd = [sys.executable, 'scripts/discover_derived_counters.py', tmpdir, '--out', out]
        subprocess.check_call(cmd)
        with open(out, 'r') as of:
            j = json.load(of)
        assert any(d.get('name') == 'my_ratio' for d in j.get('derived', []))
