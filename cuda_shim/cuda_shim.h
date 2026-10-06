/*
 * CUDA Shim for AMD GPUs
 *
 * Fake CUDA Runtime API that internally calls HIP.
 * Use with LD_PRELOAD or as libcuda.so replacement.
 */

#ifndef CUDA_SHIM_H
#define CUDA_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CUDA Types (must match NVIDIA's definitions exactly!)
// =============================================================================

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

// Error codes
#define cudaSuccess                     0
#define cudaErrorInvalidValue           1
#define cudaErrorMemoryAllocation       2
#define cudaErrorInitializationError    3
#define cudaErrorInvalidDevice          10
#define cudaErrorInvalidMemcpyDirection 21
#define cudaErrorNotReady               34

// Memory copy directions
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
} cudaMemcpyKind;

// Device properties (simplified - extend as needed)
struct cudaDeviceProp {
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
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    // ... more fields as needed
};

// Stream flags
#define cudaStreamDefault      0x00
#define cudaStreamNonBlocking  0x01

// Event flags
#define cudaEventDefault       0x00
#define cudaEventBlockingSync  0x01
#define cudaEventDisableTiming 0x02

// =============================================================================
// Memory Management
// =============================================================================

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);

// =============================================================================
// Memory Copy
// =============================================================================

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

// =============================================================================
// Device Management
// =============================================================================

cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);

// =============================================================================
// Stream Management
// =============================================================================

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamQuery(cudaStream_t stream);

// =============================================================================
// Event Management
// =============================================================================

cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventQuery(cudaEvent_t event);

// =============================================================================
// Error Handling
// =============================================================================

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

// =============================================================================
// Version Info
// =============================================================================

cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);

#ifdef __cplusplus
}
#endif

#endif // CUDA_SHIM_H
