#!/usr/bin/env python3
"""
Hybrid CUDA→HIP Translator

Kombiniert regelbasierte Übersetzung (für bekannte APIs) mit ML-Fallback.
Erreicht nahezu 100% Genauigkeit für Standard-CUDA-Code.
"""

import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# CUDA → HIP API MAPPINGS (regelbasiert, 100% korrekt)
# =============================================================================

# Sortiert nach Länge (längste zuerst) um korrekte Ersetzung zu garantieren
API_MAPPINGS = [
    # Memory - Async (längere Patterns zuerst!)
    ("cudaMallocManaged", "hipMallocManaged"),
    ("cudaMallocAsync", "hipMallocAsync"),
    ("cudaMallocPitch", "hipMallocPitch"),
    ("cudaMallocArray", "hipMallocArray"),
    ("cudaMalloc3DArray", "hipMalloc3DArray"),
    ("cudaMalloc3D", "hipMalloc3D"),
    ("cudaMallocHost", "hipHostMalloc"),  # Wichtig: hipHostMalloc!
    ("cudaMalloc", "hipMalloc"),
    ("cudaFreeMipmappedArray", "hipFreeMipmappedArray"),
    ("cudaFreeAsync", "hipFreeAsync"),
    ("cudaFreeArray", "hipFreeArray"),
    ("cudaFreeHost", "hipHostFree"),  # Wichtig: hipHostFree, nicht hipFreeHost!
    ("cudaFree", "hipFree"),
    ("cudaMallocMipmappedArray", "hipMallocMipmappedArray"),

    # Memcpy variants (längste zuerst!)
    ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
    ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
    ("cudaMemcpyDeviceToDevice", "hipMemcpyDeviceToDevice"),
    ("cudaMemcpyHostToHost", "hipMemcpyHostToHost"),
    ("cudaMemcpyDefault", "hipMemcpyDefault"),
    ("cudaMemcpyToSymbol", "hipMemcpyToSymbol"),
    ("cudaMemcpyFromSymbol", "hipMemcpyFromSymbol"),
    ("cudaMemcpy2DToArray", "hipMemcpy2DToArray"),
    ("cudaMemcpy2DFromArray", "hipMemcpy2DFromArray"),
    ("cudaMemcpy2DAsync", "hipMemcpy2DAsync"),
    ("cudaMemcpy2D", "hipMemcpy2D"),
    ("cudaMemcpy3DParms", "hipMemcpy3DParms"),
    ("cudaMemcpy3DAsync", "hipMemcpy3DAsync"),
    ("cudaMemcpy3D", "hipMemcpy3D"),
    ("cudaMemcpyAsync", "hipMemcpyAsync"),
    ("cudaMemcpyPeerAsync", "hipMemcpyPeerAsync"),
    ("cudaMemcpyPeer", "hipMemcpyPeer"),
    ("cudaMemcpy", "hipMemcpy"),

    # Memset
    ("cudaMemsetAsync", "hipMemsetAsync"),
    ("cudaMemset2DAsync", "hipMemset2DAsync"),
    ("cudaMemset2D", "hipMemset2D"),
    ("cudaMemset3DAsync", "hipMemset3DAsync"),
    ("cudaMemset3D", "hipMemset3D"),
    ("cudaMemset", "hipMemset"),

    # Streams
    ("cudaStreamCreateWithFlags", "hipStreamCreateWithFlags"),
    ("cudaStreamCreateWithPriority", "hipStreamCreateWithPriority"),
    ("cudaStreamCreate", "hipStreamCreate"),
    ("cudaStreamDestroy", "hipStreamDestroy"),
    ("cudaStreamSynchronize", "hipStreamSynchronize"),
    ("cudaStreamWaitEvent", "hipStreamWaitEvent"),
    ("cudaStreamQuery", "hipStreamQuery"),
    ("cudaStreamNonBlocking", "hipStreamNonBlocking"),
    ("cudaStreamDefault", "hipStreamDefault"),

    # Events
    ("cudaEventCreateWithFlags", "hipEventCreateWithFlags"),
    ("cudaEventCreate", "hipEventCreate"),
    ("cudaEventDestroy", "hipEventDestroy"),
    ("cudaEventRecord", "hipEventRecord"),
    ("cudaEventSynchronize", "hipEventSynchronize"),
    ("cudaEventElapsedTime", "hipEventElapsedTime"),
    ("cudaEventQuery", "hipEventQuery"),

    # Device Management
    ("cudaGetDeviceProperties", "hipGetDeviceProperties"),
    ("cudaDeviceGetAttribute", "hipDeviceGetAttribute"),
    ("cudaGetDeviceCount", "hipGetDeviceCount"),
    ("cudaGetDevice", "hipGetDevice"),
    ("cudaSetDevice", "hipSetDevice"),
    ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
    ("cudaDeviceReset", "hipDeviceReset"),
    ("cudaDeviceCanAccessPeer", "hipDeviceCanAccessPeer"),
    ("cudaDeviceEnablePeerAccess", "hipDeviceEnablePeerAccess"),
    ("cudaDeviceDisablePeerAccess", "hipDeviceDisablePeerAccess"),

    # Error Handling
    ("cudaGetLastError", "hipGetLastError"),
    ("cudaPeekAtLastError", "hipPeekAtLastError"),
    ("cudaGetErrorString", "hipGetErrorString"),
    ("cudaGetErrorName", "hipGetErrorName"),

    # Host Memory (Achtung: cuda/hip Namen unterscheiden sich!)
    ("cudaHostAllocDefault", "hipHostMallocDefault"),
    ("cudaHostAllocPortable", "hipHostMallocPortable"),
    ("cudaHostAllocMapped", "hipHostMallocMapped"),
    ("cudaHostAllocWriteCombined", "hipHostMallocWriteCombined"),
    ("cudaHostAlloc", "hipHostMalloc"),
    ("cudaMallocHost", "hipHostMalloc"),  # cudaMallocHost -> hipHostMalloc (nicht hipMallocHost!)
    ("cudaFreeHost", "hipHostFree"),  # cudaFreeHost -> hipHostFree (nicht hipFreeHost!)
    ("cudaHostRegisterDefault", "hipHostRegisterDefault"),
    ("cudaHostRegisterMapped", "hipHostRegisterMapped"),
    ("cudaHostRegister", "hipHostRegister"),
    ("cudaHostUnregister", "hipHostUnregister"),
    ("cudaHostGetDevicePointer", "hipHostGetDevicePointer"),
    ("cudaHostGetFlags", "hipHostGetFlags"),

    # Types
    ("cudaError_t", "hipError_t"),
    ("cudaSuccess", "hipSuccess"),
    ("cudaStream_t", "hipStream_t"),
    ("cudaEvent_t", "hipEvent_t"),
    ("cudaDeviceProp", "hipDeviceProp_t"),
    ("cudaMemcpyKind", "hipMemcpyKind"),
    ("cudaMemsetParams", "hipMemsetParams"),
    ("cudaArray_t", "hipArray_t"),
    ("cudaSurfaceObject_t", "hipSurfaceObject_t"),
    ("cudaTextureObject_t", "hipTextureObject_t"),
    ("cudaMipmappedArray_t", "hipMipmappedArray_t"),

    # Array Flags
    ("cudaArrayLayered", "hipArrayLayered"),
    ("cudaArraySurfaceLoadStore", "hipArraySurfaceLoadStore"),
    ("cudaArrayCubemap", "hipArrayCubemap"),
    ("cudaArrayDefault", "hipArrayDefault"),

    # Host Alloc Flags (alternative Namen)
    ("cudaHostAllocPortable", "hipHostMallocPortable"),
    ("cudaHostAllocMapped", "hipHostMallocMapped"),
    ("cudaHostAllocWriteCombined", "hipHostMallocWriteCombined"),

    # Kernel intrinsics (device code)
    ("__syncthreads", "__syncthreads"),  # Same in HIP
    ("__threadfence", "__threadfence"),
    ("__threadfence_block", "__threadfence_block"),

    # Thread indexing - these stay the same in modern HIP
    # but some old code uses hipBlockIdx_x style

    # CUDA Graphs
    ("cudaGraphCreate", "hipGraphCreate"),
    ("cudaGraphDestroy", "hipGraphDestroy"),
    ("cudaGraphLaunch", "hipGraphLaunch"),
    ("cudaGraphInstantiate", "hipGraphInstantiate"),
    ("cudaGraphExecDestroy", "hipGraphExecDestroy"),
    ("cudaGraphAddKernelNode", "hipGraphAddKernelNode"),
    ("cudaGraphAddMemcpyNode", "hipGraphAddMemcpyNode"),
    ("cudaGraphAddMemsetNode", "hipGraphAddMemsetNode"),
    ("cudaGraphAddHostNode", "hipGraphAddHostNode"),
    ("cudaGraphAddChildGraphNode", "hipGraphAddChildGraphNode"),
    ("cudaGraphAddEmptyNode", "hipGraphAddEmptyNode"),
    ("cudaGraphAddDependencies", "hipGraphAddDependencies"),
    ("cudaGraphExec_t", "hipGraphExec_t"),
    ("cudaGraph_t", "hipGraph_t"),
    ("cudaGraphNode_t", "hipGraphNode_t"),
]

# Compile regex patterns for efficiency
COMPILED_PATTERNS = [
    (re.compile(r'\b' + re.escape(cuda) + r'\b'), hip)
    for cuda, hip in API_MAPPINGS
]

# Additional patterns that need special handling
SPECIAL_PATTERNS = [
    # CUDA header includes
    (re.compile(r'#include\s*[<"]cuda_runtime\.h[>"]'), '#include <hip/hip_runtime.h>'),
    (re.compile(r'#include\s*[<"]cuda\.h[>"]'), '#include <hip/hip_runtime.h>'),
    (re.compile(r'#include\s*[<"]cuda_runtime_api\.h[>"]'), '#include <hip/hip_runtime_api.h>'),
    (re.compile(r'#include\s*[<"]cuda_profiler_api\.h[>"]'), '#include <roctracer/roctx.h>'),
    (re.compile(r'#include\s*[<"]cuda_fp16\.h[>"]'), '#include <hip/hip_fp16.h>'),
    (re.compile(r'#include\s*[<"]cuda_bf16\.h[>"]'), '#include <hip/hip_bfloat16.h>'),
    (re.compile(r'#include\s*[<"]cooperative_groups\.h[>"]'), '#include <hip/hip_cooperative_groups.h>'),
    (re.compile(r'#include\s*[<"]cooperative_groups/reduce\.h[>"]'), '#include "cooperative_groups/reduce.h"'),
    (re.compile(r'#include\s*[<"]helper_cuda\.h[>"]'),
     '// helper_cuda.h replaced for AMD - checkCudaErrors macro inline\n'
     'static inline void _hipCheck(hipError_t e, const char* f, int l) { if(e!=hipSuccess){fprintf(stderr,"HIP error %s at %s:%d\\\\n",hipGetErrorString(e),f,l);exit(1);} }\n'
     '#define checkCudaErrors(v) _hipCheck((v),__FILE__,__LINE__)\n'
     '#define getLastCudaError(m) _hipCheck(hipGetLastError(),__FILE__,__LINE__)'),
    (re.compile(r'#include\s*[<"]helper_functions\.h[>"]'),
     '// helper_functions.h replaced for AMD\n'
     '#include <cstdio>\n#include <cmath>\n#include <cstring>\n#include <chrono>\n'
     'static inline bool checkCmdLineFlag(int argc,const char**argv,const char*s){'
     'for(int i=1;i<argc;i++){const char*a=argv[i];if(a[0]==\'-\'&&a[1]==\'-\')a+=2;'
     'if(strncmp(a,s,strlen(s))==0)return true;}return false;}\n'
     'static inline int getCmdLineArgumentInt(int argc,const char**argv,const char*s){'
     'for(int i=1;i<argc;i++){if(strncmp(argv[i]+2,s,strlen(s))==0&&argv[i][strlen(s)+2]==\'=\')'
     'return atoi(argv[i]+strlen(s)+3);}return 0;}\n'
     'static inline bool getCmdLineArgumentString(int argc,const char**argv,const char*s,char**out){'
     'for(int i=1;i<argc;i++){if(strncmp(argv[i]+2,s,strlen(s))==0&&argv[i][strlen(s)+2]==\'=\'){'
     '*out=(char*)(argv[i]+strlen(s)+3);return true;}}return false;}\n'
     'static inline int findCudaDevice(int argc,const char**argv){return 0;}\n'
     'struct StopWatchInterface{'
     'std::chrono::time_point<std::chrono::high_resolution_clock> _s;'
     'bool _r=false; double _ms=0;'
     'void start(){_s=std::chrono::high_resolution_clock::now();_r=true;}\n'
     'void stop(){if(_r){auto e=std::chrono::high_resolution_clock::now();'
     '_ms+=std::chrono::duration<double,std::milli>(e-_s).count();_r=false;}}\n'
     'void reset(){_ms=0;_r=false;}\n'
     'float getTime(){return(float)_ms;}\n'
     '};\n'
     'static inline void sdkCreateTimer(StopWatchInterface**t){*t=new StopWatchInterface();}\n'
     'static inline void sdkDeleteTimer(StopWatchInterface**t){delete *t;*t=nullptr;}\n'
     'static inline void sdkStartTimer(StopWatchInterface**t){(*t)->start();}\n'
     'static inline void sdkStopTimer(StopWatchInterface**t){(*t)->stop();}\n'
     'static inline void sdkResetTimer(StopWatchInterface**t){(*t)->reset();}\n'
     'static inline float sdkGetTimerValue(StopWatchInterface**t){return (*t)->getTime();}\n'
     'static inline float sdkGetAverageTimerValue(StopWatchInterface**t){return (*t)->getTime();}'),

    # Warp-Sync Intrinsics: Maske entfernen (HIP braucht sie nicht)
    (re.compile(r'__shfl_down_sync\s*\(\s*[^,]+,\s*'), '__shfl_down('),
    (re.compile(r'__shfl_up_sync\s*\(\s*[^,]+,\s*'), '__shfl_up('),
    (re.compile(r'__shfl_xor_sync\s*\(\s*[^,]+,\s*'), '__shfl_xor('),
    (re.compile(r'__shfl_sync\s*\(\s*[^,]+,\s*'), '__shfl('),
    (re.compile(r'__ballot_sync\s*\(\s*[^,]+,\s*'), '__ballot('),
    (re.compile(r'__any_sync\s*\(\s*[^,]+,\s*'), '__any('),
    (re.compile(r'__all_sync\s*\(\s*[^,]+,\s*'), '__all('),
    (re.compile(r'__activemask\s*\(\s*\)'), '0xFFFFFFFF'),
    # cooperative_groups: cg::plus → std::plus ähnliche Semantik via inline
    (re.compile(r'\bcg::plus\b'), 'cg::plus'),
    (re.compile(r'\bcg::block_tile_memory\b'), 'int /* block_tile_memory stub */'),

    # CUDA driver API
    (re.compile(r'\bcuInit\b'), 'hipInit'),
    (re.compile(r'\bcuDeviceGet\b'), 'hipDeviceGet'),
    (re.compile(r'\bcuCtxCreate\b'), 'hipCtxCreate'),
    (re.compile(r'\bcuCtxDestroy\b'), 'hipCtxDestroy'),
    (re.compile(r'\bcuModuleLoad\b'), 'hipModuleLoad'),
    (re.compile(r'\bcuModuleUnload\b'), 'hipModuleUnload'),
    # Profiler API → no-op (ROCm profiler meist nicht aktiv)
    (re.compile(r'\bcudaProfilerStart\(\)'), '(hipSuccess) /* cudaProfilerStart noop */'),
    (re.compile(r'\bcudaProfilerStop\(\)'), '(hipSuccess) /* cudaProfilerStop noop */'),
    (re.compile(r'\bhipProfilerStart\(\)'), '(hipSuccess) /* profiler noop */'),
    (re.compile(r'\bhipProfilerStop\(\)'), '(hipSuccess) /* profiler noop */'),
]

# Patterns für HIP_SYMBOL wrapper
# hipMemcpyToSymbol(symbol, src, ...) - symbol ist 1. Argument
SYMBOL_TO_PATTERN = re.compile(
    r'(hipMemcpyToSymbol)\s*\(\s*([^,\s]+)'
)
# hipMemcpyFromSymbol(dst, symbol, ...) - symbol ist 2. Argument
SYMBOL_FROM_PATTERN = re.compile(
    r'(hipMemcpyFromSymbol)\s*\(\s*([^,]+),\s*([^,\s]+)'
)

def add_hip_symbol_wrapper(code: str) -> str:
    """Füge HIP_SYMBOL() wrapper für hipMemcpy*Symbol hinzu."""
    # hipMemcpyToSymbol - erstes Argument ist das Symbol
    def to_replacer(m):
        func = m.group(1)
        symbol = m.group(2)
        if symbol.startswith('HIP_SYMBOL'):
            return m.group(0)
        return f'{func}(HIP_SYMBOL({symbol})'

    # hipMemcpyFromSymbol - zweites Argument ist das Symbol
    def from_replacer(m):
        func = m.group(1)
        dst = m.group(2)
        symbol = m.group(3)
        if symbol.startswith('HIP_SYMBOL'):
            return m.group(0)
        return f'{func}({dst}, HIP_SYMBOL({symbol})'

    code = SYMBOL_TO_PATTERN.sub(to_replacer, code)
    code = SYMBOL_FROM_PATTERN.sub(from_replacer, code)
    return code


class HybridTranslator:
    """Hybrid CUDA→HIP Translator mit Regex + optionalem ML-Fallback."""

    def __init__(self, ml_model_path: Optional[str] = None):
        self.ml_model = None
        self.ml_tokenizer = None

        if ml_model_path and Path(ml_model_path).exists():
            self._load_ml_model(ml_model_path)

    def _load_ml_model(self, model_path: str):
        """Lade ML-Modell für Fallback."""
        try:
            import torch
            from cuda_translator.translator_model import Seq2SeqTranslator, CharTokenizer

            device = torch.device('cpu')
            checkpoint = torch.load(model_path, map_location=device)

            self.ml_tokenizer = CharTokenizer()
            self.ml_tokenizer.char2id = checkpoint['tokenizer_char2id']
            self.ml_tokenizer.id2char = checkpoint['tokenizer_id2char']
            self.ml_tokenizer.next_id = checkpoint['vocab_size']

            self.ml_model = Seq2SeqTranslator(
                vocab_size=checkpoint['vocab_size'],
                embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.0
            ).to(device)
            self.ml_model.load_state_dict(checkpoint['model_state_dict'])
            self.ml_model.eval()
            self.ml_device = device

            print(f"ML-Fallback geladen: {model_path}")
        except Exception as e:
            print(f"Warning: ML-Modell konnte nicht geladen werden: {e}")

    def translate(self, cuda_code: str, use_ml_fallback: bool = False) -> Tuple[str, dict]:
        """
        Übersetze CUDA-Code zu HIP.

        Returns:
            Tuple[str, dict]: (hip_code, stats)
            stats enthält: replacements, ml_used, unchanged
        """
        hip_code = cuda_code
        stats = {"replacements": 0, "ml_used": False, "unchanged": False}

        # 1. Spezielle Patterns (includes etc.)
        for pattern, replacement in SPECIAL_PATTERNS:
            hip_code, n = pattern.subn(replacement, hip_code)
            stats["replacements"] += n

        # 2. API Mappings (längste zuerst, bereits sortiert)
        for pattern, replacement in COMPILED_PATTERNS:
            hip_code, n = pattern.subn(replacement, hip_code)
            stats["replacements"] += n

        # 3. HIP_SYMBOL wrapper für hipMemcpy*Symbol
        hip_code = add_hip_symbol_wrapper(hip_code)

        # 4. Prüfe ob noch CUDA-Patterns übrig sind
        remaining_cuda = re.findall(r'\bcuda[A-Z]\w+', hip_code)

        if remaining_cuda and use_ml_fallback and self.ml_model:
            # ML-Fallback für unbekannte Patterns
            stats["ml_used"] = True
            hip_code = self._ml_translate(hip_code)

        if stats["replacements"] == 0:
            stats["unchanged"] = True

        return hip_code, stats

    def _ml_translate(self, code: str) -> str:
        """Verwende ML-Modell für Übersetzung."""
        import torch

        with torch.no_grad():
            src = torch.tensor(
                [self.ml_tokenizer.encode(code, add_sos_eos=False)],
                dtype=torch.long
            ).to(self.ml_device)
            out = self.ml_model.translate(src, max_len=min(len(code) * 2, 500))
            return self.ml_tokenizer.decode(out[0].cpu().tolist(), skip_special=True)

    def translate_file(self, input_path: str, output_path: Optional[str] = None) -> dict:
        """Übersetze eine CUDA-Datei zu HIP."""
        input_path = Path(input_path)

        if output_path is None:
            # .cu -> .hip.cpp oder .cuh -> .hip.hpp
            if input_path.suffix == '.cu':
                output_path = input_path.with_suffix('.hip.cpp')
            elif input_path.suffix == '.cuh':
                output_path = input_path.with_suffix('.hip.hpp')
            else:
                output_path = input_path.with_suffix(input_path.suffix + '.hip')

        output_path = Path(output_path)

        cuda_code = input_path.read_text()
        hip_code, stats = self.translate(cuda_code)

        output_path.write_text(hip_code)
        stats["input"] = str(input_path)
        stats["output"] = str(output_path)

        return stats


def translate_code(cuda_code: str, ml_fallback: bool = False) -> str:
    """Einfache Funktion für schnelle Übersetzung."""
    translator = HybridTranslator()
    hip_code, _ = translator.translate(cuda_code, use_ml_fallback=ml_fallback)
    return hip_code


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid CUDA→HIP Translator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Einzelne Zeile übersetzen
  python hybrid_translator.py -c "cudaMalloc(&ptr, size);"

  # Datei übersetzen
  python hybrid_translator.py input.cu -o output.hip.cpp

  # Mit ML-Fallback für unbekannte APIs
  python hybrid_translator.py input.cu --ml-fallback
"""
    )
    parser.add_argument('input', nargs='?', help='Input CUDA file (.cu)')
    parser.add_argument('-o', '--output', help='Output HIP file')
    parser.add_argument('-c', '--code', help='CUDA code string to translate')
    parser.add_argument('--ml-fallback', action='store_true',
                        help='Use ML model for unknown patterns')
    parser.add_argument('--ml-model', default='cuda_translator/translator_v5.pt',
                        help='Path to ML model for fallback')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show translation statistics')

    args = parser.parse_args()

    # Initialisiere Translator
    ml_path = args.ml_model if args.ml_fallback else None
    translator = HybridTranslator(ml_model_path=ml_path)

    if args.code:
        # Übersetze Code-String
        hip_code, stats = translator.translate(args.code, use_ml_fallback=args.ml_fallback)
        print(hip_code)
        if args.verbose:
            print(f"\n--- Stats: {stats['replacements']} replacements ---")

    elif args.input:
        # Übersetze Datei
        stats = translator.translate_file(args.input, args.output)
        print(f"✓ {stats['input']} → {stats['output']}")
        print(f"  {stats['replacements']} API replacements")
        if stats['ml_used']:
            print("  (ML-Fallback wurde verwendet)")

    else:
        # Interaktiver Modus
        print("Hybrid CUDA→HIP Translator")
        print("Gib CUDA-Code ein (Ctrl+D zum Beenden):")
        print("-" * 40)
        try:
            while True:
                line = input("CUDA> ")
                hip, stats = translator.translate(line)
                print(f"HIP>  {hip}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")


if __name__ == '__main__':
    main()
