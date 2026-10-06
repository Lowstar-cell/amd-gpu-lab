# CUDA → HIP Translator - Quick Start

## Was ist das?

Ein **ML-basierter Code Translator** der CUDA Code automatisch zu AMD HIP/ROCm Code übersetzt.

## Architektur

```
CUDA Code → Seq2Seq Model → HIP Code
```

**Model**: LSTM Encoder-Decoder mit Attention
**Training**: Supervised Learning auf 30+ CUDA↔HIP Paaren
**Tokenization**: Character-level

## Quick Start (3 Minuten)

### 1. Training

```bash
python3 cuda_translator/train_translator.py --epochs 100 --batch-size 16
```

**Training Zeit**: ~2-5 Minuten (CPU) / ~30 Sekunden (GPU)
**Output**: `cuda_translator/translator.pt`

### 2. Translation

```bash
# Einzelnes Snippet
python3 cuda_translator/translate.py "cudaMalloc(&ptr, size);"

# Datei übersetzen
python3 cuda_translator/translate.py \
    --input test_examples/simple_kernel.cu \
    --output test_examples/simple_kernel_translated.hip

# Interaktiver Modus
python3 cuda_translator/translate.py --interactive
```

### 3. Evaluation

```bash
python3 cuda_translator/evaluate.py
```

**Metriken**:
- Exact Match Rate
- Character Accuracy
- API Coverage (CUDA→HIP API Translation)

## Beispiele

### Einfache API Calls

```
CUDA: cudaMalloc(&ptr, size);
HIP:  hipMalloc(&ptr, size);

CUDA: cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
HIP:  hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
```

### Thread Indexing

```
CUDA: int idx = blockIdx.x * blockDim.x + threadIdx.x;
HIP:  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
```

### Kernel Launch

```
CUDA: kernel<<<grid, block>>>(args);
HIP:  hipLaunchKernelGGL(kernel, grid, block, 0, 0, args);
```

## Dateien

```
cuda_translator/
├── README_POC.md              # Detaillierte Dokumentation
├── QUICKSTART.md              # Diese Datei
├── training_data.json         # 30+ CUDA↔HIP Beispiele
├── translator_model.py        # Seq2Seq Model + Tokenizer
├── train_translator.py        # Training Script
├── translate.py               # Inference Script
├── evaluate.py                # Evaluation Script
└── test_examples/
    ├── simple_kernel.cu       # Test CUDA Kernel
    └── simple_kernel.hip      # Erwartete HIP Ausgabe
```

## Erwartete Ergebnisse

**Nach 100 Epochs**:
- Character Accuracy: ~70-85%
- Exact Match: ~40-60% (auf einfachen Beispielen)
- API Coverage: ~80-95%

**Limitation**: Aktuell nur einfache Kernels (<50 Lines)

## Verbesserungen

### Mehr Trainingsdaten

```bash
# training_data.json erweitern
nano cuda_translator/training_data.json
```

### Besseres Model (Transformer)

```python
# In translator_model.py:
# Ersetze LSTM durch Transformer
# - Bessere Long-Range Dependencies
# - Paralleles Training
# - State-of-the-Art für Code Translation
```

### Compiler Feedback Loop

```python
# Validiere HIP Output
subprocess.run(["hipcc", "-c", "translated.hip"])
# Falls Compile Error → Re-train mit Korrektur
```

## Nächste Schritte

1. **Mehr Daten sammeln**:
   - Open Source CUDA Repos durchsuchen
   - AMD's hipify-perl nutzen für Ground Truth
   - Community Contributions

2. **Model verbessern**:
   - Transformer Architecture
   - AST-basierte Features
   - Multi-Task Learning (Translation + Performance Prediction)

3. **Production Ready**:
   - WebUI für einfache Nutzung
   - API Server
   - Integration in AMD Tools

4. **Advanced Features**:
   - Performance-aware Translation (schnellste HIP Variante wählen)
   - Multi-GPU Optimization Hints
   - Error Explanation (warum wurde X zu Y übersetzt?)

## Support

Bei Fragen:
1. Siehe `README_POC.md` für Details
2. Prüfe `training_data.json` für Beispiele
3. Nutze `--interactive` Modus zum Experimentieren
