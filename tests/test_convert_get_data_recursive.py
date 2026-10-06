from pathlib import Path
import pandas as pd

from scripts.convert_get_data_to_parquet import convert_dir


def test_recursive_conversion(tmp_path):
    # create nested search structure
    input_dir = tmp_path / 'get_data'
    run = input_dir / 'run_abc'
    run.mkdir(parents=True)

    df = pd.DataFrame({'Timestamp': ['2025-01-01 00:00:00'], 'device': ['card0'], 'GPU use (%)': [0]})
    csv = run / 'rocm_smi.csv'
    df.to_csv(csv, index=False)

    out = tmp_path / 'processed'
    n = convert_dir(str(input_dir), str(out), overwrite=True)

    assert n == 1
    parq = out / 'run_abc' / 'rocm_smi.parquet'
    assert parq.exists()
