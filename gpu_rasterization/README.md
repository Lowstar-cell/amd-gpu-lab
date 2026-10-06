# AMD GPU 3D Rasterization Project

## Ziel
Hochperformante 3D Rasterisierung auf AMD GPUs mit direkter Hardware-Kontrolle für:
- **Training**: GPU-beschleunigtes Training mit Custom Kernels
- **Datensammlung**: Effiziente Vertex/Fragment-Datensammlung
- **Embedding**: GPU-native Embedding-Berechnungen

## Architektur-Ebenen

### 1. HIP (Highest Level)
- Portabler C++ Code für AMD/NVIDIA
- Schnelle Entwicklung, gute Performance

### 2. OpenCL (Portable)
- Fallback für breitere Hardware-Unterstützung
- Standard-konform

### 3. GCN/RDNA Assembly (Lowest Level)
- Direkter Zugriff auf GPU ISA (Instruction Set Architecture)
- Maximale Performance
- Hardware-spezifisch optimiert

## Ordnerstruktur
```
gpu_rasterization/
├── kernels/
│   ├── hip/           # HIP-basierte Kernel
│   ├── opencl/        # OpenCL Kernels
│   └── asm/           # GCN/RDNA Assembly
├── data_collection/   # Datensammlung während Rasterisierung
├── embedding/         # GPU-Embedding Implementierung
├── utils/             # Helper Functions
└── tests/             # Benchmarks & Tests
```

## AMD GPU Technologien

### ROCm (Radeon Open Compute)
- Open-Source GPU Computing Platform
- HIP: CUDA-ähnliche API
- rocBLAS, rocFFT, etc.

### ISA Levels
- **GCN** (Graphics Core Next): GCN3, GCN4, GCN5
- **RDNA**: RDNA1, RDNA2, RDNA3
- Direkter Assembly-Zugriff via `s_*` (scalar) und `v_*` (vector) Instructions

## Performance-Strategie
1. Start mit HIP (schnelle Iteration)
2. Profiling mit rocprof
3. Hotspots in Assembly optimieren
4. Datensammlung parallel zur Rasterisierung
