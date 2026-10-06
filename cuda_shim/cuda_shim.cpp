/*
 * CUDA Shim Implementation
 *
 * Translates CUDA Runtime API calls to HIP at runtime.
 * Compile with: hipcc -shared -fPIC -o libcuda_shim.so cuda_shim.cpp
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

// We need to export C symbols
extern "C" {

// Store last error for cudaGetLastError()
static thread_local hipError_t g_lastError = hipSuccess;

// =============================================================================
// Helper: Convert HIP error to CUDA error
// =============================================================================

static int hip_to_cuda_error(hipError_t err) {
    g_lastError = err;
    switch (err) {
        case hipSuccess:                    return 0;  // cudaSuccess
        case hipErrorInvalidValue:          return 1;  // cudaErrorInvalidValue
        case hipErrorOutOfMemory:           return 2;  // cudaErrorMemoryAllocation
        case hipErrorNotInitialized:        return 3;  // cudaErrorInitializationError
        case hipErrorInvalidDevice:         return 10; // cudaErrorInvalidDevice
        case hipErrorNotReady:              return 34; // cudaErrorNotReady
        default:                            return (int)err;
    }
}

// =============================================================================
// Memory Management
// =============================================================================

int cudaMalloc(void** devPtr, size_t size) {
    hipError_t err = hipMalloc(devPtr, size);
    return hip_to_cuda_error(err);
}

int cudaFree(void* devPtr) {
    hipError_t err = hipFree(devPtr);
    return hip_to_cuda_error(err);
}

int cudaMallocHost(void** ptr, size_t size) {
    hipError_t err = hipHostMalloc(ptr, size, hipHostMallocDefault);
    return hip_to_cuda_error(err);
}

int cudaFreeHost(void* ptr) {
    hipError_t err = hipHostFree(ptr);
    return hip_to_cuda_error(err);
}

int cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    // HIP managed memory
    hipError_t err = hipMallocManaged(devPtr, size, hipMemAttachGlobal);
    return hip_to_cuda_error(err);
}

int cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    // Map CUDA flags to HIP flags
    unsigned int hipFlags = 0;
    if (flags & 0x02) hipFlags |= hipHostMallocPortable;  // cudaHostAllocPortable
    if (flags & 0x04) hipFlags |= hipHostMallocMapped;    // cudaHostAllocMapped
    if (flags & 0x08) hipFlags |= hipHostMallocWriteCombined;

    hipError_t err = hipHostMalloc(ptr, size, hipFlags);
    return hip_to_cuda_error(err);
}

// =============================================================================
// Memory Copy
// =============================================================================

static hipMemcpyKind cuda_to_hip_memcpy_kind(int kind) {
    switch (kind) {
        case 0: return hipMemcpyHostToHost;
        case 1: return hipMemcpyHostToDevice;
        case 2: return hipMemcpyDeviceToHost;
        case 3: return hipMemcpyDeviceToDevice;
        case 4: return hipMemcpyDefault;
        default: return hipMemcpyDefault;
    }
}

int cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    hipError_t err = hipMemcpy(dst, src, count, cuda_to_hip_memcpy_kind(kind));
    return hip_to_cuda_error(err);
}

int cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, hipStream_t stream) {
    hipError_t err = hipMemcpyAsync(dst, src, count, cuda_to_hip_memcpy_kind(kind), stream);
    return hip_to_cuda_error(err);
}

int cudaMemset(void* devPtr, int value, size_t count) {
    hipError_t err = hipMemset(devPtr, value, count);
    return hip_to_cuda_error(err);
}

int cudaMemsetAsync(void* devPtr, int value, size_t count, hipStream_t stream) {
    hipError_t err = hipMemsetAsync(devPtr, value, count, stream);
    return hip_to_cuda_error(err);
}

// =============================================================================
// Device Management
// =============================================================================

int cudaGetDeviceCount(int* count) {
    hipError_t err = hipGetDeviceCount(count);
    return hip_to_cuda_error(err);
}

int cudaGetDevice(int* device) {
    hipError_t err = hipGetDevice(device);
    return hip_to_cuda_error(err);
}

int cudaSetDevice(int device) {
    hipError_t err = hipSetDevice(device);
    return hip_to_cuda_error(err);
}

int cudaDeviceSynchronize(void) {
    hipError_t err = hipDeviceSynchronize();
    return hip_to_cuda_error(err);
}

int cudaDeviceReset(void) {
    hipError_t err = hipDeviceReset();
    return hip_to_cuda_error(err);
}

// Device properties - map HIP props to CUDA struct layout
int cudaGetDeviceProperties(void* prop, int device) {
    hipDeviceProp_t hipProp;
    hipError_t err = hipGetDeviceProperties(&hipProp, device);

    if (err == hipSuccess && prop) {
        // Cast to our CUDA-compatible struct
        // This is the tricky part - struct layouts must match!
        struct CudaDeviceProp {
            char name[256];
            size_t totalGlobalMem;
            size_t sharedMemPerBlock;
            int regsPerBlock;
            int warpSize;
            int maxThreadsPerBlock;
            int maxThreadsDim[3];
            int maxGridSize[3];
            int clockRate;
            int major;
            int minor;
            int multiProcessorCount;
        };

        CudaDeviceProp* cudaProp = (CudaDeviceProp*)prop;
        strncpy(cudaProp->name, hipProp.name, 255);
        cudaProp->name[255] = '\0';
        cudaProp->totalGlobalMem = hipProp.totalGlobalMem;
        cudaProp->sharedMemPerBlock = hipProp.sharedMemPerBlock;
        cudaProp->regsPerBlock = hipProp.regsPerBlock;
        cudaProp->warpSize = hipProp.warpSize;
        cudaProp->maxThreadsPerBlock = hipProp.maxThreadsPerBlock;
        memcpy(cudaProp->maxThreadsDim, hipProp.maxThreadsDim, sizeof(int) * 3);
        memcpy(cudaProp->maxGridSize, hipProp.maxGridSize, sizeof(int) * 3);
        cudaProp->clockRate = hipProp.clockRate;
        cudaProp->major = hipProp.major;
        cudaProp->minor = hipProp.minor;
        cudaProp->multiProcessorCount = hipProp.multiProcessorCount;
    }

    return hip_to_cuda_error(err);
}

// =============================================================================
// Stream Management
// =============================================================================

int cudaStreamCreate(hipStream_t* pStream) {
    hipError_t err = hipStreamCreate(pStream);
    return hip_to_cuda_error(err);
}

int cudaStreamCreateWithFlags(hipStream_t* pStream, unsigned int flags) {
    unsigned int hipFlags = 0;
    if (flags & 0x01) hipFlags |= hipStreamNonBlocking;

    hipError_t err = hipStreamCreateWithFlags(pStream, hipFlags);
    return hip_to_cuda_error(err);
}

int cudaStreamDestroy(hipStream_t stream) {
    hipError_t err = hipStreamDestroy(stream);
    return hip_to_cuda_error(err);
}

int cudaStreamSynchronize(hipStream_t stream) {
    hipError_t err = hipStreamSynchronize(stream);
    return hip_to_cuda_error(err);
}

int cudaStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    hipError_t err = hipStreamWaitEvent(stream, event, flags);
    return hip_to_cuda_error(err);
}

int cudaStreamQuery(hipStream_t stream) {
    hipError_t err = hipStreamQuery(stream);
    return hip_to_cuda_error(err);
}

// =============================================================================
// Event Management
// =============================================================================

int cudaEventCreate(hipEvent_t* event) {
    hipError_t err = hipEventCreate(event);
    return hip_to_cuda_error(err);
}

int cudaEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    unsigned int hipFlags = 0;
    if (flags & 0x01) hipFlags |= hipEventBlockingSync;
    if (flags & 0x02) hipFlags |= hipEventDisableTiming;

    hipError_t err = hipEventCreateWithFlags(event, hipFlags);
    return hip_to_cuda_error(err);
}

int cudaEventDestroy(hipEvent_t event) {
    hipError_t err = hipEventDestroy(event);
    return hip_to_cuda_error(err);
}

int cudaEventRecord(hipEvent_t event, hipStream_t stream) {
    hipError_t err = hipEventRecord(event, stream);
    return hip_to_cuda_error(err);
}

int cudaEventSynchronize(hipEvent_t event) {
    hipError_t err = hipEventSynchronize(event);
    return hip_to_cuda_error(err);
}

int cudaEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end) {
    hipError_t err = hipEventElapsedTime(ms, start, end);
    return hip_to_cuda_error(err);
}

int cudaEventQuery(hipEvent_t event) {
    hipError_t err = hipEventQuery(event);
    return hip_to_cuda_error(err);
}

// =============================================================================
// Error Handling
// =============================================================================

int cudaGetLastError(void) {
    hipError_t err = g_lastError;
    g_lastError = hipSuccess;
    return hip_to_cuda_error(err);
}

int cudaPeekAtLastError(void) {
    return hip_to_cuda_error(g_lastError);
}

const char* cudaGetErrorString(int error) {
    // Map back to HIP error and get string
    return hipGetErrorString((hipError_t)error);
}

const char* cudaGetErrorName(int error) {
    return hipGetErrorName((hipError_t)error);
}

// =============================================================================
// Version Info - Fake CUDA version to make apps happy
// =============================================================================

int cudaDriverGetVersion(int* driverVersion) {
    // Return CUDA 12.8 (12080) - for PyTorch cu128
    if (driverVersion) *driverVersion = 12080;
    return 0;
}

int cudaRuntimeGetVersion(int* runtimeVersion) {
    // Return CUDA 12.8
    if (runtimeVersion) *runtimeVersion = 12080;
    return 0;
}

// =============================================================================
// Initialization message
// =============================================================================

__attribute__((constructor))
static void cuda_shim_init() {
    const char* verbose = getenv("CUDA_SHIM_VERBOSE");
    if (verbose && verbose[0] == '1') {
        fprintf(stderr, "[CUDA-SHIM] AMD GPU Compatibility Layer loaded\n");
        fprintf(stderr, "[CUDA-SHIM] CUDA calls will be translated to HIP\n");

        int deviceCount = 0;
        hipGetDeviceCount(&deviceCount);
        fprintf(stderr, "[CUDA-SHIM] Found %d AMD GPU(s)\n", deviceCount);
    }
}

} // extern "C"
