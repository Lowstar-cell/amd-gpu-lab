#!/usr/bin/env python3
"""
VOLLSTÄNDIGER ROCm TEST - Auf Linux mit AMD GPU ausführen!

Nutzung:
    python test_rocm_full.py

Testet:
1. GPU-Erkennung
2. GPU Embedding Service
3. Selbstlernendes System
4. Optimierter Rasterizer
5. Performance-Vergleich
"""

import sys
import time
import numpy as np

def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_gpu():
    """Test 1: GPU Erkennung"""
    divider("TEST 1: GPU ERKENNUNG")

    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Quick tensor test
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print(f"Matrix multiply test: OK")
        return True
    else:
        print("[FEHLER] Keine GPU gefunden!")
        print("Stelle sicher dass ROCm korrekt installiert ist:")
        print("  - rocm-smi sollte die GPU zeigen")
        print("  - PyTorch muss mit ROCm Support installiert sein")
        return False

def test_embedding_service():
    """Test 2: GPU Embedding Service"""
    divider("TEST 2: GPU EMBEDDING SERVICE")

    sys.path.insert(0, '.')
    from gpu_service.embedding_service import GPUEmbeddingService

    service = GPUEmbeddingService()
    print(f"Service Device: {service.device}")
    service.start()

    # Batch test
    latencies = []
    for i in range(20):
        features = np.random.randint(0, 100, size=(64,)).astype(np.float32)

        start = time.perf_counter()
        embedding, strategy = service.get_embedding_and_strategy(features)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        # Feedback
        if embedding is not None:
            service.report_performance(
                features, embedding, strategy,
                performance_ms=np.random.uniform(5, 15),
                baseline_ms=10.0
            )

    avg_latency = np.mean(latencies)
    print(f"Avg Latency: {avg_latency:.2f}ms")
    print(f"Stats: {service.get_stats()}")

    service.stop()
    print("[OK] Embedding Service funktioniert!")
    return True

def test_learning():
    """Test 3: Selbstlernendes System"""
    divider("TEST 3: SELBSTLERNENDES SYSTEM")

    sys.path.insert(0, '.')
    from gpu_service.embedding_service import GPUEmbeddingService

    service = GPUEmbeddingService(learn_interval=20)  # Schneller lernen für Test
    service.start()

    print("Simuliere 100 Workloads mit Performance-Feedback...")

    # Simuliere dass manche Strategien besser sind
    strategy_performance = {
        "default": 12.0,
        "memory_coalesce": 8.0,  # Diese ist am besten
        "wavefront_tune": 9.0,
        "occupancy_boost": 10.0,
        "cache_optimize": 9.5,
        "async_copy": 11.0,
    }

    for i in range(100):
        features = np.random.randint(0, 100, size=(64,)).astype(np.float32)
        embedding, strategy = service.get_embedding_and_strategy(features)

        if embedding is not None:
            # Simuliere Performance basierend auf Strategie
            base_perf = strategy_performance.get(strategy, 12.0)
            actual_perf = base_perf + np.random.uniform(-2, 2)

            service.report_performance(
                features, embedding, strategy,
                performance_ms=actual_perf,
                baseline_ms=10.0
            )

    time.sleep(3)  # Zeit zum Lernen geben

    stats = service.get_stats()
    print(f"Learn Steps: {stats['learn_steps']}")
    print(f"Buffer Size: {stats['buffer_size']}")
    print(f"Avg Reward: {stats['avg_reward']:.3f}")

    service.stop()

    if stats['learn_steps'] > 0:
        print("[OK] System lernt!")
        return True
    else:
        print("[WARNUNG] Keine Lernschritte ausgeführt")
        return False

def test_rasterizer():
    """Test 4: Optimierter Rasterizer"""
    divider("TEST 4: OPTIMIERTER RASTERIZER")

    sys.path.insert(0, '.')
    from gpu_rasterization.rasterizer_optimized import OptimizedRasterizer

    rasterizer = OptimizedRasterizer(
        width=512,
        height=512,
        enable_learning=True,
        baseline_ms=8.0
    )

    print("Rendere 50 Frames mit variablen Workloads...")

    for i in range(50):
        num_triangles = np.random.randint(20, 150)

        vertices = np.random.rand(num_triangles, 3, 3).astype(np.float32)
        vertices[:, :, :2] = vertices[:, :, :2] * 2 - 1
        colors = np.random.rand(num_triangles, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float32)

        color_img, _ = rasterizer.rasterize_triangles(
            vertices, colors, mvp, collect_data=False
        )

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/50")

    rasterizer.print_report()
    return True

def test_performance_comparison():
    """Test 5: Performance-Vergleich CPU vs GPU"""
    divider("TEST 5: CPU vs GPU PERFORMANCE")

    import torch

    if not torch.cuda.is_available():
        print("[SKIP] Keine GPU für Vergleich")
        return True

    sizes = [1000, 2000, 4000]

    for size in sizes:
        # CPU
        x_cpu = torch.randn(size, size)
        start = time.perf_counter()
        for _ in range(5):
            _ = torch.matmul(x_cpu, x_cpu)
        cpu_time = (time.perf_counter() - start) / 5 * 1000

        # GPU
        x_gpu = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            _ = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / 5 * 1000

        speedup = cpu_time / gpu_time
        print(f"Matrix {size}x{size}: CPU={cpu_time:.1f}ms, GPU={gpu_time:.1f}ms, Speedup={speedup:.1f}x")

    print("[OK] Performance-Test abgeschlossen!")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         AMD GPU SELBSTLERNENDES SYSTEM - VOLLTEST                ║
║                                                                  ║
║  Dieses Script testet alle Komponenten auf deiner AMD GPU        ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # Test 1
    try:
        results['gpu'] = test_gpu()
    except Exception as e:
        print(f"[FEHLER] {e}")
        results['gpu'] = False

    if not results['gpu']:
        print("\n[ABBRUCH] GPU nicht verfügbar - weitere Tests nicht möglich")
        return

    # Test 2
    try:
        results['service'] = test_embedding_service()
    except Exception as e:
        print(f"[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        results['service'] = False

    # Test 3
    try:
        results['learning'] = test_learning()
    except Exception as e:
        print(f"[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        results['learning'] = False

    # Test 4
    try:
        results['rasterizer'] = test_rasterizer()
    except Exception as e:
        print(f"[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        results['rasterizer'] = False

    # Test 5
    try:
        results['performance'] = test_performance_comparison()
    except Exception as e:
        print(f"[FEHLER] {e}")
        results['performance'] = False

    # Summary
    divider("ZUSAMMENFASSUNG")

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✓ OK" if passed else "✗ FEHLER"
        print(f"  {name:<15} {status}")

    print()
    if all_passed:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  ALLE TESTS BESTANDEN - System ist einsatzbereit!           ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    else:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  EINIGE TESTS FEHLGESCHLAGEN - Siehe Details oben           ║")
        print("╚══════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
