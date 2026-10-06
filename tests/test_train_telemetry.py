import subprocess
import sys
from pathlib import Path


def test_train_telemetry_short(tmp_path):
    # create tiny Parquet input for a quick training check
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=40, freq='S'),
        'gpu_use_pct': np.linspace(0, 100, 40),
        'temp_edge_C': 40 + np.random.rand(40) * 5,
        'vram_used_B': 1000 + (np.arange(40) * 10)
    })

    out = tmp_path / 'sample.parquet'
    df.to_parquet(out)

    # run train_telemetry with a short time budget (0.2 minutes = 12s) so test completes quickly
    cmd = [sys.executable, 'scripts/train_telemetry.py', '--data-dir', str(tmp_path), '--minutes', '0.2', '--device', 'cpu', '--logdir', str(tmp_path / 'logs')]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0
