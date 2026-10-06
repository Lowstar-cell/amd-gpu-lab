import os
import tempfile
try:
    import pandas as pd
except Exception:
    pd = None


def test_load_and_save_csv_backend():
    if pd is None:
        # pandas not installed in test environment; skip
        print('SKIP: pandas not present, skipping UI backend load/save test')
        return
    from scripts.label_ui import load_csv, save_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'sample.csv')
        df = pd.DataFrame([{'sample_id': 0, 'counter_0': 1.0, 'label': 0}, {'sample_id': 1, 'counter_0': 2.0, 'label': 0}])
        df.to_csv(path, index=False)

        loaded = load_csv(path)
        assert 'sample_id' in loaded.columns

        # change and save
        loaded.loc[loaded['sample_id'] == 1, 'label'] = 2
        out = os.path.join(tmpdir, 'out.csv')
        save_csv(loaded, out)
        assert os.path.exists(out)
        reloaded = load_csv(out)
        assert int(reloaded[reloaded['sample_id'] == 1]['label'].iloc[0]) == 2
