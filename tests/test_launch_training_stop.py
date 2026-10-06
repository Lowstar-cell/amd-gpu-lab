def test_launch_training_stop_support():
    from pathlib import Path
    p = Path('scripts/launch_training.py')
    content = p.read_text(encoding='utf8')
    assert 'telemetry_collector.pid' in content, 'Launcher should write collector pid file'
    assert "if args.action == 'stop'" in content, 'Launcher should implement stop action'
