def test_train_windows_sanity():
    """Sanity checks for the windows trainer launcher script contents.
    We validate presence of python-finder and run metadata writing.
    """
    from pathlib import Path
    p = Path('scripts/train_telemetry_windows.ps1')
    content = p.read_text(encoding='utf8')
    assert 'Find-PythonExecutable' in content, 'train launcher missing Find-PythonExecutable helper'
    assert 'telemetry_smoke_run.json' in content or 'telemetry_smoke.pid' in content, 'train launcher should write run metadata or pid'
