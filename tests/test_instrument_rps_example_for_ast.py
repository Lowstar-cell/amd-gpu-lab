import os
import shutil
import subprocess
import sys
import tempfile

import pytest


def test_instrument_rps_example_for_ast():
    try:
        from clang import cindex  # noqa: F401
    except Exception:
        pytest.skip('clang Python bindings not available; skipping AST patcher test')

    src = os.path.abspath('data/RenderPipelineShaders-main/tests/gui/test_triangle_vk.cpp')
    assert os.path.exists(src)
    with tempfile.TemporaryDirectory() as tmpdir:
        work = os.path.join(tmpdir, 'test_triangle_vk.cpp')
        shutil.copy2(src, work)
        # run the AST instrument script
        subprocess.check_call([sys.executable, 'scripts/instrument_rps_example_ast.py', '--example', work, '--include', 'data/RenderPipelineShaders-main/include'])
        with open(work, 'r', encoding='utf-8') as f:
            s = f.read()
        assert 'GpaInitialize' in s
        assert 'GpaBeginSample' in s
        assert 'GpaEndSession' in s
