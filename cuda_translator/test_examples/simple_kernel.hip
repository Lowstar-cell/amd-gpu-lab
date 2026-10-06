// Simple HIP vector addition kernel (expected output)

#include <hip/hip_runtime.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    dim3 block(256);
    dim3 grid((n + 255) / 256);

    hipLaunchKernelGGL(vectorAdd, grid, block, 0, 0, d_a, d_b, d_c, n);

    hipDeviceSynchronize();

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
