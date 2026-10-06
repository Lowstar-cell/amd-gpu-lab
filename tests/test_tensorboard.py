import os
import tempfile
import shutil
import subprocess
import sys


def test_tensorboard_event_files_created():
    tmpdir = tempfile.mkdtemp()
    try:
        # generate dataset
        script = os.path.join(os.path.dirname(__file__), '..', 'sample_data_gen.py')
        script = os.path.abspath(script)
        outdir = os.path.join(tmpdir, 'data')
        os.makedirs(outdir, exist_ok=True)
        subprocess.check_call([sys.executable, script, '--out-dir', outdir, '--num-files', '1', '--samples-per-file', '40'])

        # run contrastive training with a logdir
        train_script = os.path.join(os.path.dirname(__file__), '..', 'train_contrastive.py')
        train_script = os.path.abspath(train_script)
        logdir = os.path.join(tmpdir, 'logs')
        os.makedirs(logdir, exist_ok=True)
        subprocess.check_call([sys.executable, train_script, '--data-dir', outdir, '--epochs', '1', '--batch-size', '8', '--device', 'cpu', '--logdir', logdir, '--save-out', os.path.join(tmpdir, 'ce.pt')])
        # assert a tfevents file is present
        files = os.listdir(logdir)
        assert any('events.out.tfevents' in f for f in files)
    finally:
        shutil.rmtree(tmpdir)

