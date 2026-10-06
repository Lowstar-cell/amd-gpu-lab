// AMD GPU 3D Rasterizer - HIP Implementation
// Kompilieren: hipcc -o rasterizer rasterizer.hip

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

// Triangle Struktur
struct Triangle {
    float v0[4];  // Vertex 0: x, y, z, w
    float v1[4];  // Vertex 1
    float v2[4];  // Vertex 2
    float color[3]; // RGB
};

// Framebuffer
struct Framebuffer {
    float* color;   // RGB buffer
    float* depth;   // Z-buffer
    int width;
    int height;
};

// Edge Function für Triangle Coverage Test
__device__ float edgeFunction(const float* a, const float* b, const float* c) {
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

// 3D Rasterizer Kernel - jeder Thread rastert ein Pixel
__global__ void rasterizeTriangles(
    const Triangle* triangles,
    int numTriangles,
    Framebuffer fb,
    float* dataCollectionBuffer  // Für Training/Embedding
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= fb.width || py >= fb.height) return;

    int pixelIdx = py * fb.width + px;
    float pixelCenter[2] = {px + 0.5f, py + 0.5f};

    float closestZ = fb.depth[pixelIdx];
    float finalColor[3] = {fb.color[pixelIdx*3], fb.color[pixelIdx*3+1], fb.color[pixelIdx*3+2]};

    // Über alle Triangles iterieren
    for (int t = 0; t < numTriangles; t++) {
        const Triangle& tri = triangles[t];

        // Edge functions
        float w0 = edgeFunction(tri.v1, tri.v2, pixelCenter);
        float w1 = edgeFunction(tri.v2, tri.v0, pixelCenter);
        float w2 = edgeFunction(tri.v0, tri.v1, pixelCenter);

        // Coverage test (alle edges müssen positive sein)
        if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
            // Barycentrische Koordinaten
            float area = edgeFunction(tri.v0, tri.v1, tri.v2);
            if (area == 0.0f) continue;

            w0 /= area;
            w1 /= area;
            w2 /= area;

            // Depth interpolation
            float z = w0 * tri.v0[2] + w1 * tri.v1[2] + w2 * tri.v2[2];

            // Z-Test
            if (z < closestZ) {
                closestZ = z;

                // Color
                finalColor[0] = tri.color[0];
                finalColor[1] = tri.color[1];
                finalColor[2] = tri.color[2];

                // DATENSAMMLUNG: Barycentrische Koordinaten für Training
                int dataIdx = (pixelIdx * numTriangles + t) * 3;
                if (dataCollectionBuffer != nullptr) {
                    dataCollectionBuffer[dataIdx + 0] = w0;
                    dataCollectionBuffer[dataIdx + 1] = w1;
                    dataCollectionBuffer[dataIdx + 2] = w2;
                }
            }
        }
    }

    // Write back
    fb.depth[pixelIdx] = closestZ;
    fb.color[pixelIdx * 3 + 0] = finalColor[0];
    fb.color[pixelIdx * 3 + 1] = finalColor[1];
    fb.color[pixelIdx * 3 + 2] = finalColor[2];
}

// Vertex Shader Stage
__global__ void vertexShader(
    const float* vertices,      // Input vertices (x,y,z)
    int numVertices,
    float* transformedVertices, // Output (x,y,z,w)
    const float* mvpMatrix      // Model-View-Projection 4x4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    // Matrix-Vector Multiplikation
    float v[4] = {vertices[idx*3], vertices[idx*3+1], vertices[idx*3+2], 1.0f};
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i] += mvpMatrix[i*4 + j] * v[j];
        }
    }

    // Perspective divide
    if (result[3] != 0.0f) {
        result[0] /= result[3];
        result[1] /= result[3];
        result[2] /= result[3];
    }

    transformedVertices[idx*4 + 0] = result[0];
    transformedVertices[idx*4 + 1] = result[1];
    transformedVertices[idx*4 + 2] = result[2];
    transformedVertices[idx*4 + 3] = result[3];
}

// Host Code
extern "C" {
    void launchRasterizer(
        const Triangle* h_triangles,
        int numTriangles,
        int width,
        int height,
        float* h_outputColor,
        float* h_dataCollection
    ) {
        // Device memory allocation
        Triangle* d_triangles;
        float *d_color, *d_depth, *d_dataCollection;

        hipMalloc(&d_triangles, numTriangles * sizeof(Triangle));
        hipMalloc(&d_color, width * height * 3 * sizeof(float));
        hipMalloc(&d_depth, width * height * sizeof(float));
        hipMalloc(&d_dataCollection, width * height * numTriangles * 3 * sizeof(float));

        // Copy data to device
        hipMemcpy(d_triangles, h_triangles, numTriangles * sizeof(Triangle), hipMemcpyHostToDevice);

        // Initialize depth buffer
        float farZ = 1e10f;
        for (int i = 0; i < width * height; i++) {
            hipMemcpy(&d_depth[i], &farZ, sizeof(float), hipMemcpyHostToDevice);
        }

        // Setup framebuffer
        Framebuffer fb;
        fb.color = d_color;
        fb.depth = d_depth;
        fb.width = width;
        fb.height = height;

        // Launch kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((width + 15) / 16, (height + 15) / 16);

        hipLaunchKernelGGL(rasterizeTriangles, gridDim, blockDim, 0, 0,
            d_triangles, numTriangles, fb, d_dataCollection);

        // Copy back
        hipMemcpy(h_outputColor, d_color, width * height * 3 * sizeof(float), hipMemcpyDeviceToHost);
        if (h_dataCollection) {
            hipMemcpy(h_dataCollection, d_dataCollection,
                width * height * numTriangles * 3 * sizeof(float), hipMemcpyDeviceToHost);
        }

        // Cleanup
        hipFree(d_triangles);
        hipFree(d_color);
        hipFree(d_depth);
        hipFree(d_dataCollection);
    }
}
