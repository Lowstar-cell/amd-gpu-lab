# CUDA → HIP Translator - Proof of Concept

## Ziel
ML-basierter Code Translator der CUDA Code automatisch zu HIP (ROCm) übersetzt.

## Architektur

### 1. Datensammlung
**Quellen für CUDA↔HIP Paare:**
- `hipify-perl` Output (AMD's offizieller Translator)
- Manuelle kuratierte Beispiele
- Open Source Repos die beide Backends haben
- HIP Dokumentation (Mapping Tables)

**Format:**
```json
{
  "cuda": "__global__ void kernel() { int idx = blockIdx.x * blockDim.x + threadIdx.x; }",
  "hip": "__global__ void kernel() { int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; }",
  "category": "thread_indexing"
}
```

### 2. Model Architecture

**Option A: Transformer (Empfohlen)**
- Encoder: CUDA Code → Latent Representation
- Decoder: Latent → HIP Code
- Attention über Code Tokens
- Vorteil: Besser für längere Abhängigkeiten

**Option B: LSTM Seq2Seq (Schneller PoC)**
- Wiederverwendung der existierenden LSTM Komponenten
- Encoder-Decoder mit Attention
- Vorteil: Schnelle Integration

### 3. Tokenization

**Character-level** (einfach):
```python
tokens = list(code_string)
```

**Token-level** (besser):
```python
# BPE oder WordPiece auf Code
vocab = {"__global__", "void", "int", "blockIdx", "threadIdx", ...}
```

### 4. Training

**Supervised Learning:**
```
Input:  [CUDA tokens]
Target: [HIP tokens]
Loss:   CrossEntropy
```

**Metriken:**
- Token Accuracy
- Exact Match (ganzes Snippet korrekt)
- BLEU Score (Text Similarity)
- **Compilability** (wichtigste Metrik!)

### 5. Inference Pipeline

```
CUDA Code → Tokenize → Model → Detokenize → HIP Code → Validate (hipcc)
```

## PoC Implementation Plan

### Phase 1: Minimal Dataset (100-500 Beispiele)
```python
examples = [
    # Basic API translations
    ("cudaMalloc", "hipMalloc"),
    ("cudaMemcpy", "hipMemcpy"),
    ("__syncthreads()", "__syncthreads()"),  # identical

    # Thread indexing
    ("blockIdx.x", "hipBlockIdx_x"),
    ("threadIdx.x", "hipThreadIdx_x"),

    # More complex patterns
    ("__global__ void kernel()", "__global__ void kernel()"),
    # ...
]
```

### Phase 2: Simple LSTM Seq2Seq
- Reuse existing model.py Encoder
- Add LSTM Decoder
- Character-level tokenization (einfachster Start)

### Phase 3: Evaluation
```python
def evaluate_translation(cuda_code, hip_code):
    # 1. Token accuracy
    # 2. Compilability test
    subprocess.run(["hipcc", "-c", hip_code])
    # 3. Runtime equivalence (advanced)
```

## Quick Start

```bash
# 1. Create training data
python cuda_translator/create_dataset.py

# 2. Train model
python cuda_translator/train_translator.py --epochs 50

# 3. Test translation
python cuda_translator/translate.py --input kernel.cu --output kernel.hip
```

## Limitations

**PoC Scope:**
- Nur einfache Kernel (< 50 Lines)
- Fokus auf API calls nicht Algorithmen
- Keine komplexe Control Flow Analyse

**Future Work:**
- AST-basierte Features
- Compiler Feedback Loop
- Multi-GPU optimizations
- Performance prediction
