def test_model_injection_exists():
    from pathlib import Path
    p = Path('scripts/model_injection.py')
    assert p.exists(), 'model_injection.py missing'
