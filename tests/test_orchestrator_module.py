def test_orchestrator_manager_exists():
    from pathlib import Path
    p = Path('scripts/orchestrator/manager.py')
    assert p.exists(), 'orchestrator manager module missing'
    content = p.read_text(encoding='utf8')
    assert 'start_collector' in content and 'start_trainer' in content and 'stop_pid' in content, 'manager missing expected functions'
