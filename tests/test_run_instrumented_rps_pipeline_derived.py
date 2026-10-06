import os
import shutil
import subprocess
import sys
import tempfile


def test_pipeline_generates_derived_json_dry_run():
    src = os.path.abspath('data/RenderPipelineShaders-main/tests/gui/test_triangle_vk.cpp')
    assert os.path.exists(src)
    with tempfile.TemporaryDirectory() as tmpdir:
        work = os.path.join(tmpdir, 'test_triangle_vk.cpp')
        shutil.copy2(src, work)
        collected_dir = os.path.join(tmpdir, 'collected')
        dataset_out = os.path.join(tmpdir, 'rps_dataset', 'gpa_dataset.csv')
        cmd = [sys.executable, 'scripts/run_instrumented_rps_pipeline.py', '--example', work, '--no-build', '--no-run', '--dry-run', '--dataset-out', dataset_out, '--no-train', '--collected-dir', collected_dir]
        subprocess.check_call(cmd)
        # pipeline should have produced derived.json in collected_dir because autodiscovery is enabled by default
        derived_path = os.path.join(collected_dir, 'derived.json')
        assert os.path.exists(derived_path)
        # dataset should exist
        assert os.path.exists(dataset_out)
