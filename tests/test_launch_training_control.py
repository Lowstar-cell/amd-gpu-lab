def test_launch_training_control_flags():
    from pathlib import Path
    p = Path('scripts/launch_training.py')
    content = p.read_text(encoding='utf8')
    assert "--action" in content and "status" in content and "stop" in content, 'launcher should support --action stop/status'
    assert 'DETACHED_PROCESS' in content or 'creationflags' in content, 'launcher should try to create detached processes on Windows'
