import os
import tempfile
import shutil
import subprocess
import sys


def test_train_smoke_run():
    tmpdir = tempfile.mkdtemp()
    try:
        # generate small dataset
        script = os.path.join(os.path.dirname(__file__), '..', 'sample_data_gen.py')
        script = os.path.abspath(script)
        outdir = os.path.join(tmpdir, 'data')
        os.makedirs(outdir, exist_ok=True)
        subprocess.check_call([sys.executable, script, '--out-dir', outdir, '--num-files', '1', '--samples-per-file', '20'])

        # run train for 1 epoch
        train_script = os.path.join(os.path.dirname(__file__), '..', 'train.py')
        train_script = os.path.abspath(train_script)
        model_out = os.path.join(tmpdir, 'model.pt')
        subprocess.check_call([sys.executable, train_script, '--data-dir', outdir, '--epochs', '1', '--batch-size', '8', '--device', 'cpu', '--model-out', model_out])

        assert os.path.exists(model_out)
    finally:
        shutil.rmtree(tmpdir)
