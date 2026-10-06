"""
GPU Service Client - Einfache API für Workloads

Nutzung:
    from gpu_service.client import optimize_workload, report_result

    # Vor der Ausführung
    strategy = optimize_workload(workload_features)

    # Workload ausführen mit strategy
    result, elapsed_ms = run_workload(strategy)

    # Performance reporten (für Lernen)
    report_result(workload_features, elapsed_ms, expected_ms)
"""

import os
import sys
import time
import atexit
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager
import numpy as np

# Lazy import des Services
_service = None
_service_started = False


def _get_service():
    """Lazy initialization des Services"""
    global _service, _service_started
    if _service is None:
        from gpu_service.embedding_service import GPUEmbeddingService
        _service = GPUEmbeddingService()

    if not _service_started:
        _service.start()
        _service_started = True
        atexit.register(_shutdown)

    return _service


def _shutdown():
    """Cleanup bei Programmende"""
    global _service, _service_started
    if _service is not None and _service_started:
        _service.stop()
        _service_started = False


class OptimizationContext:
    """Context für einen Workload - trackt automatisch Performance"""

    def __init__(self, features: np.ndarray, baseline_ms: float = 10.0):
        self.features = features
        self.baseline_ms = baseline_ms
        self.embedding: Optional[np.ndarray] = None
        self.strategy: str = "default"
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        service = _get_service()
        self.embedding, self.strategy = service.get_embedding_and_strategy(self.features)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        # Report performance für Lernen
        if self.embedding is not None:
            service = _get_service()
            service.report_performance(
                self.features,
                self.embedding,
                self.strategy,
                self.elapsed_ms,
                self.baseline_ms
            )

        return False  # Don't suppress exceptions


def optimize_workload(
    features: np.ndarray,
    baseline_ms: float = 10.0
) -> OptimizationContext:
    """
    Holt Optimierungsstrategie und trackt Performance automatisch.

    Nutzung:
        with optimize_workload(features, baseline_ms=10.0) as opt:
            # opt.strategy enthält die empfohlene Strategie
            run_my_kernel(opt.strategy)
        # Performance wird automatisch reported

    Args:
        features: Workload-Features als numpy array
        baseline_ms: Erwartete Baseline-Performance (z.B. CUDA Zeit)

    Returns:
        OptimizationContext mit .strategy und .embedding
    """
    return OptimizationContext(features, baseline_ms)


def get_strategy(features: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """
    Holt nur die Strategie ohne automatisches Tracking.

    Args:
        features: Workload-Features

    Returns:
        (strategy_name, embedding)
    """
    service = _get_service()
    embedding, strategy = service.get_embedding_and_strategy(features)
    return strategy, embedding


def report_result(
    features: np.ndarray,
    elapsed_ms: float,
    baseline_ms: float,
    strategy: str = "default",
    embedding: Optional[np.ndarray] = None
):
    """
    Reportet manuell Performance-Ergebnis für Lernen.

    Args:
        features: Workload-Features
        elapsed_ms: Tatsächliche Ausführungszeit
        baseline_ms: Erwartete/Baseline Zeit
        strategy: Verwendete Strategie
        embedding: Optional - Embedding falls schon berechnet
    """
    service = _get_service()

    if embedding is None:
        embedding, _ = service.get_embedding_and_strategy(features)

    if embedding is not None:
        service.report_performance(
            features,
            embedding,
            strategy,
            elapsed_ms,
            baseline_ms
        )


def get_stats() -> Dict[str, Any]:
    """Holt Service-Statistiken"""
    service = _get_service()
    return service.get_stats()


# === Strategy Implementations ===
# Diese Funktionen können von Workloads genutzt werden

STRATEGY_PARAMS = {
    "default": {
        "block_size": 256,
        "grid_multiplier": 1,
        "use_shared_mem": False,
        "prefetch": False,
    },
    "memory_coalesce": {
        "block_size": 128,  # Kleiner für besseres coalescing
        "grid_multiplier": 2,
        "use_shared_mem": True,
        "prefetch": True,
    },
    "wavefront_tune": {
        "block_size": 64,  # AMD Wavefront = 64 (vs CUDA Warp = 32)
        "grid_multiplier": 1,
        "use_shared_mem": False,
        "prefetch": False,
    },
    "occupancy_boost": {
        "block_size": 512,
        "grid_multiplier": 4,
        "use_shared_mem": False,
        "prefetch": True,
    },
    "cache_optimize": {
        "block_size": 256,
        "grid_multiplier": 1,
        "use_shared_mem": True,
        "prefetch": True,
        "cache_hint": "L2",
    },
    "async_copy": {
        "block_size": 256,
        "grid_multiplier": 2,
        "use_shared_mem": True,
        "prefetch": True,
        "async_memcpy": True,
    },
}


def get_strategy_params(strategy: str) -> Dict[str, Any]:
    """
    Holt konkrete Parameter für eine Strategie.

    Diese können direkt beim Kernel-Launch verwendet werden.
    """
    return STRATEGY_PARAMS.get(strategy, STRATEGY_PARAMS["default"]).copy()


# === Convenience für PyTorch/ROCm ===

def apply_strategy_to_tensor_op(strategy: str, tensor: 'torch.Tensor') -> Dict[str, Any]:
    """
    Gibt Hinweise für PyTorch Operationen basierend auf Strategie.

    Returns dict mit:
        - device_sync: ob vorher synchronisiert werden soll
        - use_cuda_graphs: ob CUDA Graphs genutzt werden sollen
        - memory_format: empfohlenes Memory Format
    """
    import torch

    hints = {
        "device_sync": False,
        "use_cuda_graphs": False,
        "memory_format": torch.contiguous_format,
    }

    if strategy == "memory_coalesce":
        hints["memory_format"] = torch.channels_last if tensor.dim() == 4 else torch.contiguous_format

    elif strategy == "async_copy":
        hints["device_sync"] = False
        hints["use_cuda_graphs"] = True

    elif strategy == "cache_optimize":
        hints["device_sync"] = True  # Ensure cache is ready

    return hints


# === Feature Extraction Helpers ===

def extract_tensor_features(tensor: 'torch.Tensor') -> np.ndarray:
    """
    Extrahiert Features aus einem PyTorch Tensor für Strategie-Auswahl.

    Features:
        - Shape statistics
        - Memory layout
        - Data type
        - Size in bytes
    """
    import torch

    features = []

    # Shape features
    shape = list(tensor.shape)
    features.extend(shape[:4] if len(shape) >= 4 else shape + [0] * (4 - len(shape)))

    # Total elements (log scale)
    features.append(np.log10(tensor.numel() + 1))

    # Memory features
    features.append(tensor.element_size())
    features.append(1 if tensor.is_contiguous() else 0)
    features.append(1 if tensor.is_cuda else 0)

    # Dimensionality
    features.append(tensor.dim())

    # Stride pattern (simplified)
    strides = list(tensor.stride())[:4]
    features.extend(strides + [0] * (4 - len(strides)))

    return np.array(features, dtype=np.float32)


def extract_workload_features(
    operation: str,
    input_shapes: list,
    output_shape: tuple,
    dtype: str = "float32"
) -> np.ndarray:
    """
    Extrahiert Features aus einer Workload-Beschreibung.

    Args:
        operation: Name der Operation (z.B. "matmul", "conv2d", "rasterize")
        input_shapes: Liste von Input-Shapes
        output_shape: Output-Shape
        dtype: Datentyp

    Returns:
        Feature vector für Strategie-Auswahl
    """
    # Operation encoding (simple hash)
    op_hash = hash(operation) % 1000

    features = [op_hash]

    # Flatten input shapes
    for shape in input_shapes[:2]:  # Max 2 inputs
        flat = list(shape)[:4]
        features.extend(flat + [0] * (4 - len(flat)))

    # Output shape
    out_flat = list(output_shape)[:4]
    features.extend(out_flat + [0] * (4 - len(out_flat)))

    # Dtype encoding
    dtype_map = {"float32": 4, "float16": 2, "int32": 4, "int64": 8}
    features.append(dtype_map.get(dtype, 4))

    # Compute intensity estimate
    total_elements = np.prod(output_shape)
    features.append(np.log10(total_elements + 1))

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    # Test
    print("=== GPU Service Client Test ===")

    # Test mit Context Manager
    features = np.random.rand(20).astype(np.float32)

    print("\n1. Context Manager Test:")
    with optimize_workload(features, baseline_ms=10.0) as opt:
        print(f"   Strategy: {opt.strategy}")
        print(f"   Params: {get_strategy_params(opt.strategy)}")
        time.sleep(0.01)  # Simulate work
    print(f"   Elapsed: {opt.elapsed_ms:.2f}ms")

    # Test manuelle API
    print("\n2. Manual API Test:")
    strategy, embedding = get_strategy(features)
    print(f"   Strategy: {strategy}")
    print(f"   Embedding shape: {embedding.shape if embedding is not None else None}")

    # Stats
    print(f"\n3. Stats: {get_stats()}")
