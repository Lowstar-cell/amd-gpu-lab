import os
import tempfile
import shutil
import subprocess
import sys


def test_contrastive_smoke_run():
    tmpdir = tempfile.mkdtemp()
    try:
        # generate dataset
        script = os.path.join(os.path.dirname(__file__), '..', 'sample_data_gen.py')
        script = os.path.abspath(script)
        outdir = os.path.join(tmpdir, 'data')
        os.makedirs(outdir, exist_ok=True)
        subprocess.check_call([sys.executable, script, '--out-dir', outdir, '--num-files', '1', '--samples-per-file', '60'])

        # run contrastive training for 1 epoch with small batch
        train_script = os.path.join(os.path.dirname(__file__), '..', 'train_contrastive.py')
        train_script = os.path.abspath(train_script)
        model_out = os.path.join(tmpdir, 'contrastive_encoder.pt')
        subprocess.check_call([sys.executable, train_script, '--data-dir', outdir, '--epochs', '1', '--batch-size', '8', '--device', 'cpu', '--lr', '1e-3', '--save-out', model_out])
        # ensure saved encoder exists
        assert os.path.exists(model_out)
    finally:
        # cleanup model file if exists
        model_out = os.path.join(tmpdir, 'contrastive_encoder.pt')
        if os.path.exists(model_out):
            os.remove(model_out)
        shutil.rmtree(tmpdir)
