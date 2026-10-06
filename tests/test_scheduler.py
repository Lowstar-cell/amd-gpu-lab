import os
import tempfile
import shutil
import subprocess
import sys
import time


def test_scheduler_smoke_run():
    tmpdir = tempfile.mkdtemp()
    try:
        # generate small dataset
        script = os.path.join(os.path.dirname(__file__), '..', 'sample_data_gen.py')
        script = os.path.abspath(script)
        outdir = os.path.join(tmpdir, 'data')
        os.makedirs(outdir, exist_ok=True)
        subprocess.check_call([sys.executable, script, '--out-dir', outdir, '--num-files', '1', '--samples-per-file', '30'])

        # precompute embeddings
        pre_script = os.path.join(os.path.dirname(__file__), '..', 'precompute_embeddings.py')
        pre_script = os.path.abspath(pre_script)
        emb_out = os.path.join(tmpdir, 'embeddings.npz')
        subprocess.check_call([sys.executable, pre_script, '--data-dir', outdir, '--out', emb_out])

        # run scheduler demo
        demo_script = os.path.join(os.path.dirname(__file__), '..', 'scheduler_demo.py')
        demo_script = os.path.abspath(demo_script)
        subprocess.check_call([sys.executable, demo_script, '--data-dir', outdir, '--out', emb_out, '--n', '2'])
    finally:
        shutil.rmtree(tmpdir)
