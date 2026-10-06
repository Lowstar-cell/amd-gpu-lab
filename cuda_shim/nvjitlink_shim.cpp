/*
 * nvJitLink Shim
 *
 * Stub implementation for NVIDIA JIT Linker.
 * Most apps don't actually need this - they use nvrtc directly.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

// =============================================================================
// nvJitLink Types
// =============================================================================

typedef enum {
    NVJITLINK_SUCCESS = 0,
    NVJITLINK_ERROR_UNRECOGNIZED_OPTION = 1,
    NVJITLINK_ERROR_MISSING_ARCH = 2,
    NVJITLINK_ERROR_INVALID_INPUT = 3,
    NVJITLINK_ERROR_PTX_COMPILE = 4,
    NVJITLINK_ERROR_NVVM_COMPILE = 5,
    NVJITLINK_ERROR_INTERNAL = 6,
    NVJITLINK_ERROR_THREADPOOL = 7
} nvJitLinkResult;

typedef enum {
    NVJITLINK_INPUT_NONE = 0,
    NVJITLINK_INPUT_CUBIN = 1,
    NVJITLINK_INPUT_PTX = 2,
    NVJITLINK_INPUT_LTOIR = 3,
    NVJITLINK_INPUT_FATBIN = 4,
    NVJITLINK_INPUT_OBJECT = 5,
    NVJITLINK_INPUT_LIBRARY = 6,
    NVJITLINK_INPUT_ANY = 10
} nvJitLinkInputType;

struct _nvJitLinkHandle {
    std::vector<std::vector<char>> inputs;
    std::vector<char> linked_output;
    std::string log;
    bool linked;
};

typedef struct _nvJitLinkHandle* nvJitLinkHandle;

// =============================================================================
// nvJitLink API
// =============================================================================

nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle,
                                 uint32_t numOptions,
                                 const char* const* options) {
    (void)numOptions;
    (void)options;

    if (!handle) return NVJITLINK_ERROR_INVALID_INPUT;

    *handle = new _nvJitLinkHandle();
    (*handle)->linked = false;

    const char* verbose = getenv("CUDA_SHIM_VERBOSE");
    if (verbose && verbose[0] == '1') {
        fprintf(stderr, "[NVJITLINK-SHIM] Created linker handle\n");
    }

    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle) {
    if (!handle || !*handle) return NVJITLINK_ERROR_INVALID_INPUT;

    delete *handle;
    *handle = nullptr;

    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle,
                                  nvJitLinkInputType inputType,
                                  const void* data,
                                  size_t size,
                                  const char* name) {
    (void)inputType;
    (void)name;

    if (!handle || !data) return NVJITLINK_ERROR_INVALID_INPUT;

    // Store the input data
    std::vector<char> input((const char*)data, (const char*)data + size);
    handle->inputs.push_back(std::move(input));

    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle,
                                  nvJitLinkInputType inputType,
                                  const char* fileName) {
    (void)inputType;

    if (!handle || !fileName) return NVJITLINK_ERROR_INVALID_INPUT;

    // Read file and add as data
    FILE* f = fopen(fileName, "rb");
    if (!f) return NVJITLINK_ERROR_INVALID_INPUT;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<char> data(size);
    fread(data.data(), 1, size, f);
    fclose(f);

    handle->inputs.push_back(std::move(data));

    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) {
    if (!handle) return NVJITLINK_ERROR_INVALID_INPUT;

    // For now, just concatenate all inputs as "linked" output
    // Real linking would require more sophisticated handling
    for (const auto& input : handle->inputs) {
        handle->linked_output.insert(handle->linked_output.end(),
                                      input.begin(), input.end());
    }

    handle->linked = true;
    handle->log = "Linking completed (shim passthrough)";

    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) {
    if (!handle || !size) return NVJITLINK_ERROR_INVALID_INPUT;
    if (!handle->linked) return NVJITLINK_ERROR_INTERNAL;

    *size = handle->linked_output.size();
    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) {
    if (!handle || !cubin) return NVJITLINK_ERROR_INVALID_INPUT;
    if (!handle->linked) return NVJITLINK_ERROR_INTERNAL;

    memcpy(cubin, handle->linked_output.data(), handle->linked_output.size());
    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) {
    return nvJitLinkGetLinkedCubinSize(handle, size);
}

nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) {
    return nvJitLinkGetLinkedCubin(handle, ptx);
}

nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) {
    if (!handle || !size) return NVJITLINK_ERROR_INVALID_INPUT;
    *size = handle->log.size() + 1;
    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) {
    if (!handle || !log) return NVJITLINK_ERROR_INVALID_INPUT;
    strcpy(log, handle->log.c_str());
    return NVJITLINK_SUCCESS;
}

nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) {
    return nvJitLinkGetErrorLogSize(handle, size);
}

nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) {
    return nvJitLinkGetErrorLog(handle, log);
}

nvJitLinkResult nvJitLinkVersion(unsigned int* major, unsigned int* minor) {
    if (major) *major = 12;
    if (minor) *minor = 8;
    return NVJITLINK_SUCCESS;
}

// =============================================================================
// Initialization
// =============================================================================

__attribute__((constructor))
static void nvjitlink_shim_init() {
    const char* verbose = getenv("CUDA_SHIM_VERBOSE");
    if (verbose && verbose[0] == '1') {
        fprintf(stderr, "[NVJITLINK-SHIM] JIT Linker shim loaded\n");
    }
}

} // extern "C"
