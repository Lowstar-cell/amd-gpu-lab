/*
 * CUDA Driver API Shim
 *
 * Implements the CUDA Driver API (cuXxx functions) on top of HIP.
 * This is what PyTorch and other frameworks actually link against.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

extern "C" {

// =============================================================================
// CUDA Driver API Types
// =============================================================================

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;
typedef unsigned long long CUdeviceptr;

// Error codes
#define CUDA_SUCCESS                    0
#define CUDA_ERROR_INVALID_VALUE        1
#define CUDA_ERROR_OUT_OF_MEMORY        2
#define CUDA_ERROR_NOT_INITIALIZED      3
#define CUDA_ERROR_INVALID_DEVICE       101
#define CUDA_ERROR_NOT_FOUND            500

// Flags
#define CU_CTX_SCHED_AUTO               0x00
#define CU_CTX_MAP_HOST                 0x08

// =============================================================================
// Helper
// =============================================================================

static int g_initialized = 0;

static CUresult hip_to_cu_error(hipError_t err) {
    switch (err) {
        case hipSuccess:                return CUDA_SUCCESS;
        case hipErrorInvalidValue:      return CUDA_ERROR_INVALID_VALUE;
        case hipErrorOutOfMemory:       return CUDA_ERROR_OUT_OF_MEMORY;
        case hipErrorNotInitialized:    return CUDA_ERROR_NOT_INITIALIZED;
        case hipErrorInvalidDevice:     return CUDA_ERROR_INVALID_DEVICE;
        default:                        return (int)err;
    }
}

// =============================================================================
// Initialization
// =============================================================================

CUresult cuInit(unsigned int flags) {
    (void)flags;
    // HIP doesn't need explicit init
    g_initialized = 1;
    return CUDA_SUCCESS;
}

// =============================================================================
// Device Management
// =============================================================================

CUresult cuDeviceGetCount(int* count) {
    return hip_to_cu_error(hipGetDeviceCount(count));
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    // In CUDA driver API, CUdevice is just an int
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev);
    if (err == hipSuccess) {
        strncpy(name, prop.name, len - 1);
        name[len - 1] = '\0';
    }
    return hip_to_cu_error(err);
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev);
    if (err == hipSuccess) {
        *bytes = prop.totalGlobalMem;
    }
    return hip_to_cu_error(err);
}

CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
    // Map CUDA attributes to HIP
    hipDeviceAttribute_t hipAttr;

    switch (attrib) {
        case 1:  hipAttr = hipDeviceAttributeMaxThreadsPerBlock; break;
        case 2:  hipAttr = hipDeviceAttributeMaxBlockDimX; break;
        case 3:  hipAttr = hipDeviceAttributeMaxBlockDimY; break;
        case 4:  hipAttr = hipDeviceAttributeMaxBlockDimZ; break;
        case 5:  hipAttr = hipDeviceAttributeMaxGridDimX; break;
        case 6:  hipAttr = hipDeviceAttributeMaxGridDimY; break;
        case 7:  hipAttr = hipDeviceAttributeMaxGridDimZ; break;
        case 8:  hipAttr = hipDeviceAttributeMaxSharedMemoryPerBlock; break;
        case 10: hipAttr = hipDeviceAttributeWarpSize; break;
        case 14: hipAttr = hipDeviceAttributeClockRate; break;
        case 16: hipAttr = hipDeviceAttributeMultiprocessorCount; break;
        case 21: // Compute capability major
            {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, dev);
                *pi = prop.major;
                return CUDA_SUCCESS;
            }
        case 75: // Compute capability minor
            {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, dev);
                *pi = prop.minor;
                return CUDA_SUCCESS;
            }
        default:
            *pi = 0;
            return CUDA_SUCCESS; // Return success but 0 for unknown attrs
    }

    return hip_to_cu_error(hipDeviceGetAttribute(pi, hipAttr, dev));
}

CUresult cuDeviceGetUuid(void* uuid, CUdevice dev) {
    // Generate a fake but consistent UUID based on device index
    // Real UUID would require deeper HIP integration
    unsigned char* u = (unsigned char*)uuid;
    memset(u, 0, 16);
    u[0] = 'A'; u[1] = 'M'; u[2] = 'D'; u[3] = dev;
    return CUDA_SUCCESS;
}

// =============================================================================
// Context Management (HIP uses implicit context, so these are mostly stubs)
// =============================================================================

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    (void)flags;
    hipError_t err = hipSetDevice(dev);
    // Return a fake context handle
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    return hip_to_cu_error(err);
}

CUresult cuCtxDestroy(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (ctx) {
        int dev = (int)(uintptr_t)ctx - 1;
        return hip_to_cu_error(hipSetDevice(dev));
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    int dev;
    hipError_t err = hipGetDevice(&dev);
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    return hip_to_cu_error(err);
}

CUresult cuCtxSynchronize(void) {
    return hip_to_cu_error(hipDeviceSynchronize());
}

CUresult cuCtxGetDevice(CUdevice* device) {
    return hip_to_cu_error(hipGetDevice(device));
}

// Primary context (what PyTorch uses)
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    return cuCtxCreate(pctx, 0, dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    (void)dev;
    *flags = 0;
    *active = 1;  // Always active
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    (void)dev;
    (void)flags;
    return CUDA_SUCCESS;
}

// =============================================================================
// Memory Management
// =============================================================================

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    void* ptr;
    hipError_t err = hipMalloc(&ptr, bytesize);
    *dptr = (CUdeviceptr)ptr;
    return hip_to_cu_error(err);
}

CUresult cuMemFree(CUdeviceptr dptr) {
    return hip_to_cu_error(hipFree((void*)dptr));
}

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
    return hip_to_cu_error(hipHostMalloc(pp, bytesize, hipHostMallocDefault));
}

CUresult cuMemFreeHost(void* p) {
    return hip_to_cu_error(hipHostFree(p));
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    return hip_to_cu_error(hipMemcpy((void*)dstDevice, srcHost, ByteCount, hipMemcpyHostToDevice));
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return hip_to_cu_error(hipMemcpy(dstHost, (void*)srcDevice, ByteCount, hipMemcpyDeviceToHost));
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    return hip_to_cu_error(hipMemcpy((void*)dstDevice, (void*)srcDevice, ByteCount, hipMemcpyDeviceToDevice));
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    return hip_to_cu_error(hipMemcpyAsync((void*)dstDevice, srcHost, ByteCount, hipMemcpyHostToDevice, (hipStream_t)hStream));
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return hip_to_cu_error(hipMemcpyAsync(dstHost, (void*)srcDevice, ByteCount, hipMemcpyDeviceToHost, (hipStream_t)hStream));
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    return hip_to_cu_error(hipMemset((void*)dstDevice, uc, N));
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    // HIP doesn't have direct D32, do it via memset (limited)
    return hip_to_cu_error(hipMemset((void*)dstDevice, ui & 0xFF, N * 4));
}

CUresult cuMemGetInfo(size_t* free, size_t* total) {
    return hip_to_cu_error(hipMemGetInfo(free, total));
}

CUresult cuPointerGetAttribute(void* data, int attribute, CUdeviceptr ptr) {
    // Minimal implementation - return device memory type
    if (attribute == 1) {  // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
        *(int*)data = 2;   // CU_MEMORYTYPE_DEVICE
    }
    return CUDA_SUCCESS;
}

// =============================================================================
// Stream Management
// =============================================================================

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    (void)Flags;
    return hip_to_cu_error(hipStreamCreate((hipStream_t*)phStream));
}

CUresult cuStreamDestroy(CUstream hStream) {
    return hip_to_cu_error(hipStreamDestroy((hipStream_t)hStream));
}

CUresult cuStreamSynchronize(CUstream hStream) {
    return hip_to_cu_error(hipStreamSynchronize((hipStream_t)hStream));
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    (void)Flags;
    return hip_to_cu_error(hipStreamWaitEvent((hipStream_t)hStream, (hipEvent_t)hEvent, 0));
}

CUresult cuStreamQuery(CUstream hStream) {
    return hip_to_cu_error(hipStreamQuery((hipStream_t)hStream));
}

// =============================================================================
// Event Management
// =============================================================================

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    (void)Flags;
    return hip_to_cu_error(hipEventCreate((hipEvent_t*)phEvent));
}

CUresult cuEventDestroy(CUevent hEvent) {
    return hip_to_cu_error(hipEventDestroy((hipEvent_t)hEvent));
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return hip_to_cu_error(hipEventRecord((hipEvent_t)hEvent, (hipStream_t)hStream));
}

CUresult cuEventSynchronize(CUevent hEvent) {
    return hip_to_cu_error(hipEventSynchronize((hipEvent_t)hEvent));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    return hip_to_cu_error(hipEventElapsedTime(pMilliseconds, (hipEvent_t)hStart, (hipEvent_t)hEnd));
}

CUresult cuEventQuery(CUevent hEvent) {
    return hip_to_cu_error(hipEventQuery((hipEvent_t)hEvent));
}

// =============================================================================
// Version
// =============================================================================

CUresult cuDriverGetVersion(int* driverVersion) {
    *driverVersion = 12080;  // CUDA 12.8 for PyTorch cu128
    return CUDA_SUCCESS;
}

// =============================================================================
// Stubs for other commonly used functions
// =============================================================================

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    *pStr = hipGetErrorString((hipError_t)error);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
    *pStr = hipGetErrorName((hipError_t)error);
    return CUDA_SUCCESS;
}

// Module/Function loading - stubs for now
CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    (void)fname;
    *module = NULL;
    return CUDA_ERROR_NOT_FOUND;  // Not implemented yet
}

CUresult cuModuleUnload(CUmodule hmod) {
    (void)hmod;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    (void)hmod;
    (void)name;
    *hfunc = NULL;
    return CUDA_ERROR_NOT_FOUND;
}

// Library handle (for CUDA 11+ features)
CUresult cuLibraryLoadData(void** library, const void* code, void* jitOptions,
                           void* jitOptionsValues, unsigned int numJitOptions,
                           void* libraryOptions, void* libraryOptionValues,
                           unsigned int numLibraryOptions) {
    (void)library; (void)code; (void)jitOptions; (void)jitOptionsValues;
    (void)numJitOptions; (void)libraryOptions; (void)libraryOptionValues;
    (void)numLibraryOptions;
    return CUDA_ERROR_NOT_FOUND;
}

// =============================================================================
// Profiler stubs
// =============================================================================

CUresult cuProfilerStart(void) { return CUDA_SUCCESS; }
CUresult cuProfilerStop(void) { return CUDA_SUCCESS; }

} // extern "C"
