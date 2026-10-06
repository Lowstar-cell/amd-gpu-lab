def test_windows_collector_sanity():
    """Check that the Windows collector has common helpers and writes a pid file when run.
    (We only validate content here — runtime checks are environment-specific.)
    """
    from pathlib import Path
    p = Path('get_data/collect_windows.ps1')
    content = p.read_text(encoding='utf8')
    assert 'Get-UniqueRunDir' in content, 'collector missing Get-UniqueRunDir helper'
    assert 'collector.pid' in content or 'collector.pid' in content.lower(), 'collector should write a collector.pid marker'
