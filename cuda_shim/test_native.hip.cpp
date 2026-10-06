// Test CUDA code - will be compiled with hipcc via our shim
#include <stdio.h>

// Simple kernel
__global__ void addKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);

    // Host arrays
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    // Copy to device
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Sync
    hipDeviceSynchronize();

    // Copy result back
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS! Vector addition of %d elements completed.\n", N);
        printf("Sample: %f + %f = %f\n", h_a[0], h_b[0], h_c[0]);
        printf("Sample: %f + %f = %f\n", h_a[N-1], h_b[N-1], h_c[N-1]);
    } else {
        printf("FAILED: %d errors\n", errors);
    }

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return errors;
}
