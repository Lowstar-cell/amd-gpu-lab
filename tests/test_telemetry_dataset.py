import pandas as pd
import numpy as np
from pathlib import Path

from dataset_utils import TelemetryDataset


def test_telemetry_dataset_basic(tmp_path):
    # create a small sample dataframe and write to parquet
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='S'),
        'gpu_use_pct': np.arange(10) * 10,
        'temp_edge_C': 40 + np.arange(10) * 0.5,
        'vram_used_B': 1000 + np.arange(10) * 100,
    })

    out = tmp_path / 'sample.parquet'
    df.to_parquet(out)

    ds = TelemetryDataset(str(tmp_path), window_size=4, step=1)
    # 10 rows -> windows = 10 - 4 + 1 = 7
    assert len(ds) == 7

    w, label, length = ds[0]
    assert w.shape == (4, 3)
    assert length == 4
    # check numeric ranges
    assert float(w[0, 0]) == 0.0
