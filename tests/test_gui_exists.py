def test_gui_exists():
    from pathlib import Path
    p = Path('scripts/launch_gui.py')
    assert p.exists(), 'GUI launcher missing'
    content = p.read_text(encoding='utf8')
    assert 'Dark Mode' in content and 'start_collector' in content and 'start_trainer' in content, 'GUI should include controls and dark mode'
