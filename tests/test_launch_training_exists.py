def test_launch_training_exists():
    from pathlib import Path
    p = Path('scripts/launch_training.py')
    assert p.exists(), 'Cross-platform launcher missing'
    content = p.read_text(encoding='utf8')
    assert 'start_trainer' in content and 'start_collector' in content, 'Launcher missing start helpers'
