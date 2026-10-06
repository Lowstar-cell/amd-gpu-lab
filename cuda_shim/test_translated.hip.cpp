#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 2048;
    float *h_a = (float*)malloc(N*sizeof(float));
    float *h_b = (float*)malloc(N*sizeof(float));
    float *h_c = (float*)malloc(N*sizeof(float));
    for(int i=0; i<N; i++) { h_a[i]=i; h_b[i]=i*2; }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N*sizeof(float));
    hipMalloc(&d_b, N*sizeof(float));
    hipMalloc(&d_c, N*sizeof(float));
    hipMemcpy(d_a, h_a, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N*sizeof(float), hipMemcpyHostToDevice);

    vectorAdd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    hipMemcpy(h_c, d_c, N*sizeof(float), hipMemcpyDeviceToHost);

    printf("CUDA Code on AMD GPU: %.0f + %.0f = %.0f\n", h_a[N-1], h_b[N-1], h_c[N-1]);

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
