import pandas as pd
import numpy as np
from pathlib import Path

from dataset_utils import RasterDataset


def test_raster_dataset(tmp_path):
    # create a small csv with frame rows
    out = tmp_path / 'run'
    out.mkdir()
    df = pd.DataFrame({
        'frame_id': list(range(10)),
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='S'),
        'primitives': np.arange(10) + 10,
        'vertices': (np.arange(10) + 10) * 3,
        'shader_invocations': (np.arange(10) + 10) * 20,
        'rasterized_fragments': (np.arange(10) + 10) * 2,
        'label': [0] * 9 + [1]
    })
    path = out / 'raster.csv'
    df.to_csv(path, index=False)

    ds = RasterDataset(str(out))
    assert len(ds) == 1
    frames, label, n = ds[0]
    assert n == 10
    assert frames.shape[1] >= 3
    assert label.item() in (0.0, 1.0)
