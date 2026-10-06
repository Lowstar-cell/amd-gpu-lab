import os
import shutil
import subprocess
import sys
import tempfile


def test_run_instrumented_rps_pipeline_restore():
    src = os.path.abspath('data/RenderPipelineShaders-main/tests/gui/test_triangle_vk.cpp')
    assert os.path.exists(src)
    with tempfile.TemporaryDirectory() as tmpdir:
        work = os.path.join(tmpdir, 'test_triangle_vk.cpp')
        shutil.copy2(src, work)
        # Save original content
        with open(work, 'r', encoding='utf-8') as f:
            original_content = f.read()
        # Run pipeline with patch but dry-run and restore true
        cmd = [sys.executable, 'scripts/run_instrumented_rps_pipeline.py', '--example', work, '--no-build', '--no-run', '--dry-run', '--dataset-out', os.path.join(tmpdir, 'rps_dataset', 'gpa_dataset.csv'), '--no-train', '--restore']
        subprocess.check_call(cmd)
        # After restore, file should be restored to original
        with open(work, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == original_content
