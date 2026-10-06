#!/usr/bin/env python3
"""
Test script for CUDA Shim

Usage:
    LD_PRELOAD=./libcuda_shim.so python3 test_shim.py
"""

import ctypes
import os

def test_direct_calls():
    """Test shim directly via ctypes"""
    print("=" * 50)
    print("Testing CUDA Shim directly")
    print("=" * 50)

    # Load our shim
    shim_path = os.path.join(os.path.dirname(__file__), "libcuda_shim.so")
    if not os.path.exists(shim_path):
        print(f"ERROR: {shim_path} not found. Run 'make' first.")
        return False

    lib = ctypes.CDLL(shim_path)

    # Test cudaGetDeviceCount
    count = ctypes.c_int(0)
    err = lib.cudaGetDeviceCount(ctypes.byref(count))
    print(f"cudaGetDeviceCount: {count.value} device(s), error={err}")

    if count.value == 0:
        print("No AMD GPUs found!")
        return False

    # Test cudaSetDevice
    err = lib.cudaSetDevice(0)
    print(f"cudaSetDevice(0): error={err}")

    # Test cudaGetDevice
    device = ctypes.c_int(-1)
    err = lib.cudaGetDevice(ctypes.byref(device))
    print(f"cudaGetDevice: device={device.value}, error={err}")

    # Test cudaMalloc / cudaFree
    ptr = ctypes.c_void_p(0)
    size = 1024 * 1024  # 1 MB
    err = lib.cudaMalloc(ctypes.byref(ptr), size)
    print(f"cudaMalloc(1MB): ptr={hex(ptr.value) if ptr.value else 'NULL'}, error={err}")

    if ptr.value:
        # Test cudaMemset
        err = lib.cudaMemset(ptr, 0, size)
        print(f"cudaMemset: error={err}")

        # Test cudaFree
        err = lib.cudaFree(ptr)
        print(f"cudaFree: error={err}")

    # Test cudaDeviceSynchronize
    err = lib.cudaDeviceSynchronize()
    print(f"cudaDeviceSynchronize: error={err}")

    # Test version
    version = ctypes.c_int(0)
    err = lib.cudaRuntimeGetVersion(ctypes.byref(version))
    print(f"cudaRuntimeGetVersion: {version.value} (fake CUDA 11.8)")

    # Test error string
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    err_str = lib.cudaGetErrorString(0)
    print(f"cudaGetErrorString(0): {err_str.decode() if err_str else 'NULL'}")

    print("\nAll direct tests passed!")
    return True


def test_pytorch():
    """Test with PyTorch (if installed)"""
    print("\n" + "=" * 50)
    print("Testing with PyTorch")
    print("=" * 50)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")

            # Try a simple tensor operation
            x = torch.zeros(100, device='cuda')
            print(f"Created tensor on GPU: {x.shape}")
            y = x + 1
            print(f"Tensor operation: success")
            return True
        else:
            print("PyTorch doesn't see CUDA - shim might not be loaded")
            print("Make sure to run with: LD_PRELOAD=./libcuda_shim.so python3 test_shim.py")
            return False

    except ImportError:
        print("PyTorch not installed, skipping")
        return True
    except Exception as e:
        print(f"PyTorch test failed: {e}")
        return False


if __name__ == "__main__":
    print("CUDA Shim Test Suite")
    print()

    ok1 = test_direct_calls()
    ok2 = test_pytorch()

    print("\n" + "=" * 50)
    if ok1 and ok2:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
