import os
import shutil
import subprocess
import sys
import tempfile


def test_run_instrumented_rps_pipeline_dry_run():
    src = os.path.abspath('data/RenderPipelineShaders-main/tests/gui/test_triangle_vk.cpp')
    assert os.path.exists(src)
    with tempfile.TemporaryDirectory() as tmpdir:
        work = os.path.join(tmpdir, 'test_triangle_vk.cpp')
        shutil.copy2(src, work)
        dataset_out = os.path.join(tmpdir, 'rps_dataset', 'gpa_dataset.csv')
        cmd = [sys.executable, 'scripts/run_instrumented_rps_pipeline.py', '--example', work, '--no-build', '--no-run', '--dry-run', '--dataset-out', dataset_out, '--no-train']
        subprocess.check_call(cmd)
        assert os.path.exists(dataset_out)
