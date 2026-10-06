#!/usr/bin/env python3
"""
GPU Workload Runner + Data Collector

Führt verschiedene GPU Workloads aus und sammelt gleichzeitig:
- ROCm GPU Metriken (rocm-smi)
- Workload Performance (timing)
- GPU Rasterization Features

Perfekt für GPU Embedding Training!
"""

import torch
import time
import subprocess
import json
import threading
from pathlib import Path
from datetime import datetime


class GPUDataCollector:
    """Sammelt GPU Daten während Workloads laufen."""

    def __init__(self, output_dir="gpu_workload_data", interval=0.5):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.collecting = False
        self.metrics = []

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Data collection directory: {self.run_dir}")

    def collect_gpu_metrics(self):
        """Collect GPU metrics via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--csv"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    return {
                        'timestamp': time.time(),
                        'raw_output': result.stdout.strip()
                    }
        except Exception as e:
            print(f"Warning: Failed to collect GPU metrics: {e}")

        return None

    def start_collection(self):
        """Start background collection thread."""
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        print("GPU metrics collection started")

    def _collection_loop(self):
        """Background loop collecting metrics."""
        while self.collecting:
            metric = self.collect_gpu_metrics()
            if metric:
                self.metrics.append(metric)
            time.sleep(self.interval)

    def stop_collection(self):
        """Stop collection and save data."""
        self.collecting = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=2)

        # Save metrics
        if self.metrics:
            metrics_file = self.run_dir / "gpu_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"Saved {len(self.metrics)} metric snapshots to {metrics_file}")


class GPUWorkloadRunner:
    """Runs various GPU workloads for benchmarking."""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def matrix_multiply(self, size=4096, iterations=10):
        """Matrix multiplication workload."""
        print(f"\n[MATMUL] {size}x{size} matrices, {iterations} iterations")

        A = torch.randn(size, size, device=self.device)
        B = torch.randn(size, size, device=self.device)

        # Warmup
        _ = torch.matmul(A, B)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        timings = []
        for i in range(iterations):
            start = time.time()

            C = torch.matmul(A, B)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            timings.append(elapsed * 1000)  # ms

            print(f"  Iteration {i+1}/{iterations}: {elapsed*1000:.2f} ms")

        avg_time = sum(timings) / len(timings)
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  GFLOPS: {2 * size**3 / avg_time / 1e6:.2f}")

        return {'workload': 'matmul', 'size': size, 'avg_time_ms': avg_time, 'timings': timings}

    def convolution(self, batch=32, channels=64, size=256, iterations=10):
        """Convolution workload."""
        print(f"\n[CONV2D] batch={batch}, channels={channels}, size={size}x{size}, {iterations} iterations")

        input_tensor = torch.randn(batch, channels, size, size, device=self.device)
        conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1).to(self.device)

        # Warmup
        _ = conv(input_tensor)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        timings = []
        for i in range(iterations):
            start = time.time()

            output = conv(input_tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            timings.append(elapsed * 1000)

            print(f"  Iteration {i+1}/{iterations}: {elapsed*1000:.2f} ms")

        avg_time = sum(timings) / len(timings)
        print(f"  Average: {avg_time:.2f} ms")

        return {'workload': 'conv2d', 'batch': batch, 'channels': channels, 'size': size, 'avg_time_ms': avg_time, 'timings': timings}

    def memory_bandwidth(self, size_mb=1024, iterations=10):
        """Memory bandwidth test."""
        print(f"\n[MEMORY] {size_mb} MB transfer, {iterations} iterations")

        size_bytes = size_mb * 1024 * 1024
        elements = size_bytes // 4  # float32

        # Warmup
        data = torch.randn(elements, device=self.device)
        copy = data.clone()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        timings = []
        for i in range(iterations):
            start = time.time()

            copy = data.clone()

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            timings.append(elapsed * 1000)

            bandwidth_gbps = size_mb / 1024 / elapsed
            print(f"  Iteration {i+1}/{iterations}: {elapsed*1000:.2f} ms ({bandwidth_gbps:.2f} GB/s)")

        avg_time = sum(timings) / len(timings)
        avg_bandwidth = size_mb / 1024 / (avg_time / 1000)
        print(f"  Average bandwidth: {avg_bandwidth:.2f} GB/s")

        return {'workload': 'memory', 'size_mb': size_mb, 'avg_time_ms': avg_time, 'bandwidth_gbps': avg_bandwidth, 'timings': timings}


def main():
    import argparse

    parser = argparse.ArgumentParser(description='GPU Workload Runner + Data Collector')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--output', default='gpu_workload_data', help='Output directory')
    parser.add_argument('--interval', type=float, default=0.1, help='Metric collection interval (seconds)')
    parser.add_argument('--duration', type=int, default=60, help='Total duration (seconds)')
    args = parser.parse_args()

    print("=" * 70)
    print("GPU WORKLOAD RUNNER + DATA COLLECTOR")
    print("=" * 70)

    # Initialize
    collector = GPUDataCollector(output_dir=args.output, interval=args.interval)
    runner = GPUWorkloadRunner(device=args.device)

    # Start data collection
    collector.start_collection()

    # Run workloads
    results = []

    try:
        # Matrix multiplication (verschiedene Größen)
        for size in [2048, 4096]:
            result = runner.matrix_multiply(size=size, iterations=5)
            results.append(result)

        # Convolution
        result = runner.convolution(batch=16, channels=64, size=224, iterations=5)
        results.append(result)

        # Memory bandwidth
        result = runner.memory_bandwidth(size_mb=512, iterations=5)
        results.append(result)

        print("\n" + "=" * 70)
        print("✓ All workloads completed!")
        print("=" * 70)

    finally:
        # Stop collection
        collector.stop_collection()

        # Save results
        results_file = collector.run_dir / "workload_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved workload results to {results_file}")

        # Summary
        print(f"\nData saved to: {collector.run_dir}")
        print(f"- GPU metrics: {len(collector.metrics)} snapshots")
        print(f"- Workload results: {len(results)} benchmarks")


if __name__ == '__main__':
    main()
