def test_train_telemetry_windows_exists():
    from pathlib import Path
    assert Path('scripts/train_telemetry_windows.ps1').exists(), 'Windows training launcher missing'
