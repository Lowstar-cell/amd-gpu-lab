def test_benchmark_runner_exists():
    from pathlib import Path
    p = Path('scripts/benchmark_runner.py')
    assert p.exists(), 'benchmark_runner.py missing'
