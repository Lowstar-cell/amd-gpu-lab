import os
import shutil
import subprocess
import sys
import tempfile
import csv


def test_run_instrumented_rps_pipeline_multi_iter_dry():
    src = os.path.abspath('data/RenderPipelineShaders-main/tests/gui/test_triangle_vk.cpp')
    assert os.path.exists(src)
    with tempfile.TemporaryDirectory() as tmpdir:
        work = os.path.join(tmpdir, 'test_triangle_vk.cpp')
        shutil.copy2(src, work)
        dataset_out = os.path.join(tmpdir, 'rps_dataset', 'gpa_dataset.csv')
        cmd = [sys.executable, 'scripts/run_instrumented_rps_pipeline.py', '--example', work, '--no-build', '--no-run', '--dry-run', '--dataset-out', dataset_out, '--no-train', '--iterations', '3']
        subprocess.check_call(cmd)
        assert os.path.exists(dataset_out)
        # Count non-header lines
        with open(dataset_out, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) - 1 >= 3
