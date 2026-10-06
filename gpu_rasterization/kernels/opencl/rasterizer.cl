// AMD GPU 3D Rasterizer - OpenCL Implementation
// Fallback für breitere Hardware-Unterstützung

typedef struct {
    float v0[4];
    float v1[4];
    float v2[4];
    float color[3];
} Triangle;

// Edge function
float edgeFunction(__global const float* a, __global const float* b, float2 c) {
    return (c.x - a[0]) * (b[1] - a[1]) - (c.y - a[1]) * (b[0] - a[0]);
}

// Main rasterization kernel
__kernel void rasterizeTriangles(
    __global const Triangle* triangles,
    int numTriangles,
    __global float* colorBuffer,
    __global float* depthBuffer,
    int width,
    int height,
    __global float* dataCollection  // Datensammlung für Training
) {
    int px = get_global_id(0);
    int py = get_global_id(1);

    if (px >= width || py >= height) return;

    int pixelIdx = py * width + px;
    float2 pixelCenter = (float2)(px + 0.5f, py + 0.5f);

    float closestZ = depthBuffer[pixelIdx];
    float3 finalColor = (float3)(
        colorBuffer[pixelIdx*3],
        colorBuffer[pixelIdx*3+1],
        colorBuffer[pixelIdx*3+2]
    );

    for (int t = 0; t < numTriangles; t++) {
        __global const Triangle* tri = &triangles[t];

        // Edge tests
        float w0 = edgeFunction(tri->v1, tri->v2, pixelCenter);
        float w1 = edgeFunction(tri->v2, tri->v0, pixelCenter);
        float w2 = edgeFunction(tri->v0, tri->v1, pixelCenter);

        if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
            float area = edgeFunction(tri->v0, tri->v1,
                (float2)(tri->v2[0], tri->v2[1]));

            if (fabs(area) < 0.0001f) continue;

            // Barycentric coordinates
            w0 /= area;
            w1 /= area;
            w2 /= area;

            // Interpolate depth
            float z = w0 * tri->v0[2] + w1 * tri->v1[2] + w2 * tri->v2[2];

            if (z < closestZ) {
                closestZ = z;
                finalColor = (float3)(tri->color[0], tri->color[1], tri->color[2]);

                // Datensammlung
                if (dataCollection != 0) {
                    int dataIdx = (pixelIdx * numTriangles + t) * 3;
                    dataCollection[dataIdx + 0] = w0;
                    dataCollection[dataIdx + 1] = w1;
                    dataCollection[dataIdx + 2] = w2;
                }
            }
        }
    }

    depthBuffer[pixelIdx] = closestZ;
    colorBuffer[pixelIdx * 3 + 0] = finalColor.x;
    colorBuffer[pixelIdx * 3 + 1] = finalColor.y;
    colorBuffer[pixelIdx * 3 + 2] = finalColor.z;
}

// Vertex transformation kernel
__kernel void transformVertices(
    __global const float* vertices,
    __global float* transformed,
    __global const float* mvpMatrix,
    int numVertices
) {
    int idx = get_global_id(0);
    if (idx >= numVertices) return;

    float4 v = (float4)(
        vertices[idx*3],
        vertices[idx*3+1],
        vertices[idx*3+2],
        1.0f
    );

    float4 result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    // Matrix multiplication
    for (int i = 0; i < 4; i++) {
        result.s0 += mvpMatrix[0*4+i] * v[i];
        result.s1 += mvpMatrix[1*4+i] * v[i];
        result.s2 += mvpMatrix[2*4+i] * v[i];
        result.s3 += mvpMatrix[3*4+i] * v[i];
    }

    // Perspective divide
    if (result.w != 0.0f) {
        result.xyz /= result.w;
    }

    transformed[idx*4 + 0] = result.x;
    transformed[idx*4 + 1] = result.y;
    transformed[idx*4 + 2] = result.z;
    transformed[idx*4 + 3] = result.w;
}

// Data collection kernel - sammelt Features während Rasterisierung
__kernel void collectRasterizationData(
    __global const float* barycentricCoords,
    __global const Triangle* triangles,
    __global float* features,  // Output für Embedding-Training
    int numPixels,
    int numTriangles
) {
    int pixelIdx = get_global_id(0);
    if (pixelIdx >= numPixels) return;

    // Für jedes Triangle, extrahiere Features
    for (int t = 0; t < numTriangles; t++) {
        int coordIdx = (pixelIdx * numTriangles + t) * 3;
        float w0 = barycentricCoords[coordIdx + 0];
        float w1 = barycentricCoords[coordIdx + 1];
        float w2 = barycentricCoords[coordIdx + 2];

        // Feature-Vektor: [barycentric coords, interpolated depth, coverage]
        int featureIdx = (pixelIdx * numTriangles + t) * 5;
        features[featureIdx + 0] = w0;
        features[featureIdx + 1] = w1;
        features[featureIdx + 2] = w2;

        __global const Triangle* tri = &triangles[t];
        float interpDepth = w0 * tri->v0[2] + w1 * tri->v1[2] + w2 * tri->v2[2];
        features[featureIdx + 3] = interpDepth;

        float coverage = (w0 >= 0 && w1 >= 0 && w2 >= 0) ? 1.0f : 0.0f;
        features[featureIdx + 4] = coverage;
    }
}
