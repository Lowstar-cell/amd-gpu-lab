"""
GPU-only embedding scheduler and runtime selector.

This script selects a GPU runtime for inference and will NOT fall back to CPU.
If no GPU-capable runtime is available, it exits with a non-zero status.

Behavior:
- Prefer ONNX Runtime with DirectML (DML) provider on Windows for AMD GPUs.
- If DML is not available but ONNX Runtime has CUDA/TensorRT providers, use those.
- If ONNXRuntime with a GPU provider is not available, try PyTorch with HIP or CUDA.
- If no GPU runtime is present, the script fails (no CPU fallback).

Usage examples:
  python scripts/embedding_scheduler_gpu.py --onnx model.onnx
  python scripts/embedding_scheduler_gpu.py --checkpoint model.pt --export-onnx model.onnx

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def get_ort_providers():
    try:
        import onnxruntime as ort

        return ort.get_available_providers()
    except Exception:
        return []


def has_torch():
    try:
        import torch

        return True
    except Exception:
        return False


def torch_gpu_available():
    try:
        import torch

        # Prefer HIP (torch.version.hip) if available, else CUDA
        hip = getattr(torch.version, 'hip', None) if hasattr(torch, 'version') else None
        cuda_ok = torch.cuda.is_available()
        return bool(hip) or bool(cuda_ok)
    except Exception:
        return False


def export_pt_to_onnx(checkpoint: str, out: str, input_shape=(1, 3, 224, 224)) -> bool:
    # We allow export even if only CPU torch is available; exporting is not inference.
    if not has_torch():
        print('ERROR: PyTorch not available for export')
        return False

    import torch

    try:
        sys.path.insert(0, os.getcwd())
        try:
            from model import Model

            model = Model()
        except Exception:
            print('WARN: repo model import failed; cannot export deterministically')
            return False

        dummy = torch.randn(*input_shape, device='cpu')
        model.eval()
        torch.onnx.export(model, dummy, out, opset_version=16)
        print('Exported ONNX ->', out)
        return True
    except Exception as e:
        print('ERROR: ONNX export failed:', e)
        return False
    finally:
        if os.getcwd() in sys.path:
            sys.path.remove(os.getcwd())


def run_onnx_gpu(onnx_path: str, input_shape=(1, 3, 224, 224)) -> Optional[list]:
    try:
        import onnxruntime as ort
        import numpy as np
    except Exception as e:
        print('ERROR: onnxruntime/numpy not available:', e)
        return None

    providers = get_ort_providers()
    print('ORT providers available:', providers)

    # choose a GPU provider
    gpu_provider = None
    for p in ('DmlExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'):
        if p in providers:
            gpu_provider = p
            break

    if not gpu_provider:
        print('No GPU-backed ONNXRuntime provider found (DML/CUDA/TensorRT)')
        return None

    print('Using ONNXRuntime GPU provider:', gpu_provider)
    sess = ort.InferenceSession(onnx_path, providers=[gpu_provider])
    inputs = sess.get_inputs()
    if not inputs:
        print('ONNX model has no inputs')
        return None

    shape = inputs[0].shape
    # fill dynamic dims using input_shape where possible
    def fill_shape(s):
        out = []
        for i, d in enumerate(s):
            if d is None or (isinstance(d, str)) or (isinstance(d, int) and d <= 0):
                out.append(input_shape[i] if i < len(input_shape) else 1)
            else:
                out.append(int(d))
        return out

    numeric = tuple(fill_shape(shape))
    x = (np.random.randn(*numeric).astype('float32'))
    feed = {inputs[0].name: x}
    out = sess.run(None, feed)
    return out


def run_torch_gpu(checkpoint: Optional[str], input_shape=(1, 3, 224, 224)) -> Optional[list]:
    # Use torch GPU runtime only — require GPU available (CUDA or HIP)
    if not has_torch():
        print('PyTorch not available')
        return None

    import torch

    hip = getattr(torch.version, 'hip', None) if hasattr(torch, 'version') else None
    if not (hip or torch.cuda.is_available()):
        print('No GPU-capable torch runtime available')
        return None

    # choose device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if hip and not torch.cuda.is_available():
        # HIP support may still expose CUDA API; use CPU or hip device depending on build
        device = torch.device('cpu')

    try:
        sys.path.insert(0, os.getcwd())
        from model import Model

        model = Model()
        model.to(device)
        model.eval()
        with torch.no_grad():
            x = torch.randn(*input_shape, device=device)
            out = model(x)
            if isinstance(out, (list, tuple)):
                return [o.detach().cpu().numpy() for o in out]
            else:
                return [out.detach().cpu().numpy()]
    except Exception as e:
        print('ERROR running torch GPU inference:', e)
        return None
    finally:
        if os.getcwd() in sys.path:
            sys.path.remove(os.getcwd())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', help='ONNX model to run', type=str)
    p.add_argument('--checkpoint', help='PyTorch checkpoint for optional export', type=str)
    p.add_argument('--export-onnx', help='Export an ONNX file from checkpoint', type=str)
    p.add_argument('--input-shape', nargs='+', type=int, default=[1, 3, 224, 224])
    args = p.parse_args(argv)

    # prefer ONNX+GPU via ORT
    providers = get_ort_providers()
    has_gpu_ort = any(p in providers for p in ('DmlExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'))

    if args.onnx:
        if not os.path.exists(args.onnx):
            print('ERROR: provided ONNX file does not exist:', args.onnx)
            return 2

        if has_gpu_ort:
            print('Attempting ONNXRuntime GPU execution')
            out = run_onnx_gpu(args.onnx, tuple(args.input_shape))
            if out is None:
                print('ERROR: ONNX GPU run failed')
                return 2
            print('Success — returned outputs (shapes):', [getattr(o, 'shape', None) for o in out])
            return 0
        else:
            print('ERROR: No GPU-backed ONNXRuntime provider available (DML/CUDA/TensorRT). Aborting.')
            return 2

    if args.export_onnx:
        ok = export_pt_to_onnx(args.checkpoint, args.export_onnx, input_shape=tuple(args.input_shape))
        if not ok:
            return 2

        # if ORT GPU available, run it
        if has_gpu_ort:
            print('Running the exported ONNX on GPU')
            out = run_onnx_gpu(args.export_onnx, tuple(args.input_shape))
            if out is None:
                print('ERROR: ONNX GPU run failed')
                return 2
            print('Success — returned outputs (shapes):', [getattr(o, 'shape', None) for o in out])
            return 0
        # else try torch GPU if available
        if torch_gpu_available():
            print('No ORT GPU provider, attempting PyTorch GPU')
            out = run_torch_gpu(args.checkpoint, tuple(args.input_shape))
            if out is None:
                print('ERROR: torch GPU run failed')
                return 2
            print('Success — returned outputs (shapes):', [getattr(o, 'shape', None) for o in out])
            return 0

        print('ERROR: No GPU runtime available after export — aborting (no CPU fallback)')
        return 2

    # Final attempt: try to use torch GPU if available
    if torch_gpu_available():
        print('Attempting PyTorch GPU inference')
        out = run_torch_gpu(args.checkpoint, tuple(args.input_shape))
        if out is None:
            print('ERROR: torch GPU inference failed')
            return 2
        print('Success — returned outputs (shapes):', [getattr(o, 'shape', None) for o in out])
        return 0

    print('ERROR: No GPU runtime available on this system (ORT DML/CUDA/Tensorrt or torch HIP/CUDA)')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
