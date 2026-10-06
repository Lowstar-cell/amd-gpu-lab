import shutil
from pathlib import Path
import pandas as pd

from scripts.convert_get_data_to_parquet import convert_dir


def test_convert_get_data(tmp_path):
    # Setup temporary output directory
    out = tmp_path / "processed"
    out.mkdir()

    # Use repository's get_data directory as input (contains CSVs)
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = repo_root / "get_data"

    # Run conversion
    n = convert_dir(str(input_dir), str(out), overwrite=True)
    assert n > 0, "No files converted; expected at least one CSV in get_data"

    # Check that parquet files have the canonical columns
    parquet_files = list(out.glob("*.parquet"))
    assert parquet_files, "No parquet files written"

    df = pd.read_parquet(parquet_files[0])
    # We expect at least the timestamp + device + file_source columns
    assert "timestamp" in df.columns
    assert "device" in df.columns
    assert "file_source" in df.columns
