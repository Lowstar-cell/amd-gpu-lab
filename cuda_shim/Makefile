# CUDA Shim Makefile
# Full CUDA compatibility layer for AMD GPUs

HIPCC ?= hipcc
CXXFLAGS = -fPIC -O2 -Wall -std=c++17
LDFLAGS = -shared

# Source files
SRCS = cuda_shim.cpp cuda_driver_shim.cpp nvrtc_shim.cpp nvjitlink_shim.cpp

# Output - one combined library
SHIM_LIB = libcuda_shim.so

.PHONY: all clean install test test-pytorch wrapper uninstall

all: $(SHIM_LIB) symlinks

$(SHIM_LIB): $(SRCS)
	$(HIPCC) $(CXXFLAGS) $(LDFLAGS) -lhiprtc -o $@ $^
	@echo ""
	@echo "=== Build complete: $(SHIM_LIB) ==="

# Create all necessary symlinks
symlinks: $(SHIM_LIB)
	@echo "Creating library symlinks..."
	ln -sf $(SHIM_LIB) libcuda.so.1
	ln -sf $(SHIM_LIB) libcuda.so
	ln -sf $(SHIM_LIB) libcudart.so.12
	ln -sf $(SHIM_LIB) libcudart.so.11
	ln -sf $(SHIM_LIB) libcudart.so
	ln -sf $(SHIM_LIB) libnvrtc.so.12
	ln -sf $(SHIM_LIB) libnvrtc.so
	ln -sf $(SHIM_LIB) libnvrtc-builtins.so.12.8
	ln -sf $(SHIM_LIB) libnvrtc-builtins.so.12
	ln -sf $(SHIM_LIB) libnvrtc-builtins.so
	ln -sf $(SHIM_LIB) libnvJitLink.so.12
	ln -sf $(SHIM_LIB) libnvJitLink.so
	@echo ""
	@echo "=== Symlinks created ==="
	@echo ""
	@echo "To use:"
	@echo "  export LD_LIBRARY_PATH=$(PWD):\$$LD_LIBRARY_PATH"
	@echo "  python3 your_script.py"
	@echo ""
	@echo "Or use the wrapper:"
	@echo "  make wrapper"
	@echo "  ./cuda-shim python3 your_script.py"

clean:
	rm -f $(SHIM_LIB) libcuda.so* libcudart.so* libnvrtc*.so* libnvJitLink.so* cuda-shim

# Install system-wide
install: $(SHIM_LIB) symlinks
	@echo "Installing CUDA shim to /usr/local/lib/cuda-shim/"
	sudo mkdir -p /usr/local/lib/cuda-shim
	sudo cp -P $(SHIM_LIB) lib*.so* /usr/local/lib/cuda-shim/
	@echo ""
	@echo "=== Installed! ==="
	@echo ""
	@echo "Add to your shell profile:"
	@echo "  export LD_LIBRARY_PATH=/usr/local/lib/cuda-shim:\$$LD_LIBRARY_PATH"

uninstall:
	sudo rm -rf /usr/local/lib/cuda-shim

# Quick API test
test: $(SHIM_LIB) symlinks
	@echo "=== Testing CUDA Shim APIs ==="
	@LD_LIBRARY_PATH=$(PWD):$$LD_LIBRARY_PATH CUDA_SHIM_VERBOSE=1 python3 -c "\
import ctypes; \
lib = ctypes.CDLL('$(PWD)/$(SHIM_LIB)'); \
count = ctypes.c_int(0); \
print('--- Runtime API ---'); \
lib.cudaGetDeviceCount(ctypes.byref(count)); \
print(f'cudaGetDeviceCount: {count.value} GPU(s)'); \
print('--- Driver API ---'); \
lib.cuInit(0); \
lib.cuDeviceGetCount(ctypes.byref(count)); \
print(f'cuDeviceGetCount: {count.value} GPU(s)'); \
print('--- NVRTC API ---'); \
major = ctypes.c_int(0); minor = ctypes.c_int(0); \
lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)); \
print(f'nvrtcVersion: {major.value}.{minor.value}'); \
print('--- All APIs working! ---')"

# Full PyTorch test
test-pytorch: $(SHIM_LIB) symlinks
	@echo "=== Testing with PyTorch ==="
	LD_LIBRARY_PATH=$(PWD):$$LD_LIBRARY_PATH CUDA_SHIM_VERBOSE=1 python3 -c "\
import torch; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
if torch.cuda.is_available(): \
    print(f'Device count: {torch.cuda.device_count()}'); \
    print(f'Device 0: {torch.cuda.get_device_name(0)}'); \
    x = torch.randn(3, 3, device='cuda'); \
    print(f'Tensor on GPU: {x.shape}'); \
    print('SUCCESS!'); \
"

# Create convenient wrapper script
wrapper: $(SHIM_LIB) symlinks
	@echo '#!/bin/bash' > cuda-shim
	@echo '# CUDA Shim wrapper - runs commands with AMD GPU CUDA compatibility' >> cuda-shim
	@echo 'SCRIPT_DIR="$$(cd "$$(dirname "$${BASH_SOURCE[0]}")" && pwd)"' >> cuda-shim
	@echo 'export LD_LIBRARY_PATH="$$SCRIPT_DIR:$$LD_LIBRARY_PATH"' >> cuda-shim
	@echo 'export CUDA_SHIM_VERBOSE=$${CUDA_SHIM_VERBOSE:-0}' >> cuda-shim
	@echo 'exec "$$@"' >> cuda-shim
	@chmod +x cuda-shim
	@echo ""
	@echo "=== Created: cuda-shim ==="
	@echo ""
	@echo "Usage:"
	@echo "  ./cuda-shim python3 train.py"
	@echo "  ./cuda-shim python3 -c 'import torch; print(torch.cuda.is_available())'"
	@echo ""
	@echo "With debug output:"
	@echo "  CUDA_SHIM_VERBOSE=1 ./cuda-shim python3 train.py"
