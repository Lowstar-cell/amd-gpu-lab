import os
import tempfile
try:
    import pandas as pd
except Exception:
    pd = None


def test_build_uncertainty_queue_with_scores():
    if pd is None:
        print('SKIP: pandas missing')
        return
    from scripts.label_ui import build_uncertainty_queue
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame([{'sample_id': '0', 'counter_0': 1.0, 'label': 0}, {'sample_id': '1', 'counter_0': 2.0, 'label': 0}])
        score_file = os.path.join(tmpdir, 'scores.csv')
        pd.DataFrame([{'sample_id': '0', 'score': 0.51}, {'sample_id': '1', 'score': 0.9}]).to_csv(score_file, index=False)
        q = build_uncertainty_queue(df, score_file)
        # sample 0 is closer to 0.5 -> more uncertain -> should be first
        assert q.iloc[0]['sample_id'] == '0'


def test_build_uncertainty_queue_fallback():
    if pd is None:
        print('SKIP: pandas missing')
        return
    from scripts.label_ui import build_uncertainty_queue
    import numpy as np
    df = pd.DataFrame([{'sample_id': '0', 'counter_0': 1.0, 'counter_1': 2.0, 'label': 0}, {'sample_id': '1', 'counter_0': 100.0, 'counter_1': 2.0, 'label': 0}])
    q = build_uncertainty_queue(df, score_csv=None)
    # expect the outlier with 100.0 in counter_0 to be first
    assert q.iloc[0]['sample_id'] == '1'
