import subprocess
import sys
from pathlib import Path


def test_generate_raster_files(tmp_path):
    out = tmp_path / 'raster_out'
    cmd = [sys.executable, 'scripts/generate_rasterization_data.py', '--out-dir', str(out), '--num-files', '2', '--frames', '50', '--format', 'csv']
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0
    # expect run_ directory
    run_dirs = list(out.glob('run_*'))
    assert run_dirs, 'No run directory created'
    run = run_dirs[0]
    assert (run / 'manifest.json').exists()
    # two csvs
    files = list(run.glob('*.csv'))
    assert len(files) == 2
