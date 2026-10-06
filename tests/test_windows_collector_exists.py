def test_windows_collector_exists():
    from pathlib import Path
    p = Path('get_data/collect_windows.ps1')
    assert p.exists(), 'Windows collector script missing'
