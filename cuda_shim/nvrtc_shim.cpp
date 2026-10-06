/*
 * NVRTC Shim - CUDA Runtime Compiler on AMD
 *
 * Translates CUDA kernel code to HIP and compiles via hiprtc.
 * This allows JIT compilation of CUDA kernels on AMD GPUs.
 */

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <regex>

extern "C" {

// =============================================================================
// NVRTC Types
// =============================================================================

typedef enum {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;

// Opaque program handle
struct _nvrtcProgram {
    hiprtcProgram hip_program;
    std::string original_src;
    std::string translated_src;
    std::string compile_log;
    std::vector<char> compiled_code;
    bool compiled;
};

typedef struct _nvrtcProgram* nvrtcProgram;

// =============================================================================
// CUDA to HIP Source Translation (simplified inline version)
// =============================================================================

static std::string translate_cuda_to_hip(const std::string& cuda_src) {
    std::string hip_src = cuda_src;

    // Include replacement
    hip_src = std::regex_replace(hip_src, std::regex(R"(#include\s*[<"]cuda_runtime\.h[>"])"), "#include <hip/hip_runtime.h>");
    hip_src = std::regex_replace(hip_src, std::regex(R"(#include\s*[<"]cuda\.h[>"])"), "#include <hip/hip_runtime.h>");

    // API replacements (order matters - longer patterns first)
    static const std::pair<std::regex, const char*> replacements[] = {
        // Memory
        {std::regex(R"(\bcudaMallocManaged\b)"), "hipMallocManaged"},
        {std::regex(R"(\bcudaMallocHost\b)"), "hipHostMalloc"},
        {std::regex(R"(\bcudaMalloc\b)"), "hipMalloc"},
        {std::regex(R"(\bcudaFreeHost\b)"), "hipHostFree"},
        {std::regex(R"(\bcudaFree\b)"), "hipFree"},

        // Memcpy
        {std::regex(R"(\bcudaMemcpyHostToDevice\b)"), "hipMemcpyHostToDevice"},
        {std::regex(R"(\bcudaMemcpyDeviceToHost\b)"), "hipMemcpyDeviceToHost"},
        {std::regex(R"(\bcudaMemcpyDeviceToDevice\b)"), "hipMemcpyDeviceToDevice"},
        {std::regex(R"(\bcudaMemcpyAsync\b)"), "hipMemcpyAsync"},
        {std::regex(R"(\bcudaMemcpy\b)"), "hipMemcpy"},
        {std::regex(R"(\bcudaMemsetAsync\b)"), "hipMemsetAsync"},
        {std::regex(R"(\bcudaMemset\b)"), "hipMemset"},

        // Streams & Events
        {std::regex(R"(\bcudaStreamCreate\b)"), "hipStreamCreate"},
        {std::regex(R"(\bcudaStreamDestroy\b)"), "hipStreamDestroy"},
        {std::regex(R"(\bcudaStreamSynchronize\b)"), "hipStreamSynchronize"},
        {std::regex(R"(\bcudaEventCreate\b)"), "hipEventCreate"},
        {std::regex(R"(\bcudaEventDestroy\b)"), "hipEventDestroy"},
        {std::regex(R"(\bcudaEventRecord\b)"), "hipEventRecord"},
        {std::regex(R"(\bcudaEventSynchronize\b)"), "hipEventSynchronize"},

        // Device
        {std::regex(R"(\bcudaDeviceSynchronize\b)"), "hipDeviceSynchronize"},
        {std::regex(R"(\bcudaGetDeviceCount\b)"), "hipGetDeviceCount"},
        {std::regex(R"(\bcudaSetDevice\b)"), "hipSetDevice"},
        {std::regex(R"(\bcudaGetDevice\b)"), "hipGetDevice"},

        // Types
        {std::regex(R"(\bcudaError_t\b)"), "hipError_t"},
        {std::regex(R"(\bcudaSuccess\b)"), "hipSuccess"},
        {std::regex(R"(\bcudaStream_t\b)"), "hipStream_t"},
        {std::regex(R"(\bcudaEvent_t\b)"), "hipEvent_t"},

        // Error handling
        {std::regex(R"(\bcudaGetLastError\b)"), "hipGetLastError"},
        {std::regex(R"(\bcudaGetErrorString\b)"), "hipGetErrorString"},
    };

    for (const auto& [pattern, replacement] : replacements) {
        hip_src = std::regex_replace(hip_src, pattern, replacement);
    }

    return hip_src;
}

// =============================================================================
// Helper
// =============================================================================

static nvrtcResult hiprtc_to_nvrtc(hiprtcResult err) {
    switch (err) {
        case HIPRTC_SUCCESS: return NVRTC_SUCCESS;
        case HIPRTC_ERROR_OUT_OF_MEMORY: return NVRTC_ERROR_OUT_OF_MEMORY;
        case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE: return NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
        case HIPRTC_ERROR_INVALID_INPUT: return NVRTC_ERROR_INVALID_INPUT;
        case HIPRTC_ERROR_INVALID_PROGRAM: return NVRTC_ERROR_INVALID_PROGRAM;
        case HIPRTC_ERROR_INVALID_OPTION: return NVRTC_ERROR_INVALID_OPTION;
        case HIPRTC_ERROR_COMPILATION: return NVRTC_ERROR_COMPILATION;
        default: return NVRTC_ERROR_INTERNAL_ERROR;
    }
}

// =============================================================================
// NVRTC API Implementation
// =============================================================================

const char* nvrtcGetErrorString(nvrtcResult result) {
    switch (result) {
        case NVRTC_SUCCESS: return "NVRTC_SUCCESS";
        case NVRTC_ERROR_OUT_OF_MEMORY: return "NVRTC_ERROR_OUT_OF_MEMORY";
        case NVRTC_ERROR_PROGRAM_CREATION_FAILURE: return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
        case NVRTC_ERROR_INVALID_INPUT: return "NVRTC_ERROR_INVALID_INPUT";
        case NVRTC_ERROR_INVALID_PROGRAM: return "NVRTC_ERROR_INVALID_PROGRAM";
        case NVRTC_ERROR_INVALID_OPTION: return "NVRTC_ERROR_INVALID_OPTION";
        case NVRTC_ERROR_COMPILATION: return "NVRTC_ERROR_COMPILATION";
        case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
        default: return "NVRTC_ERROR_UNKNOWN";
    }
}

nvrtcResult nvrtcVersion(int* major, int* minor) {
    // Report NVRTC 12.8 to match our fake CUDA version
    if (major) *major = 12;
    if (minor) *minor = 8;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog,
                                const char* src,
                                const char* name,
                                int numHeaders,
                                const char* const* headers,
                                const char* const* includeNames) {
    if (!prog || !src) {
        return NVRTC_ERROR_INVALID_INPUT;
    }

    // Allocate program struct
    nvrtcProgram p = new _nvrtcProgram();
    p->compiled = false;
    p->original_src = src;

    // Translate CUDA to HIP
    p->translated_src = translate_cuda_to_hip(p->original_src);

    // Debug output
    const char* debug = getenv("CUDA_SHIM_DEBUG");
    if (debug && debug[0] == '1') {
        fprintf(stderr, "[NVRTC-SHIM] Translated source:\n%s\n", p->translated_src.c_str());
    }

    // Create HIP RTC program with translated source
    hiprtcResult err = hiprtcCreateProgram(
        &p->hip_program,
        p->translated_src.c_str(),
        name ? name : "cuda_kernel.cu",
        numHeaders,
        const_cast<const char**>(headers),
        const_cast<const char**>(includeNames)
    );

    if (err != HIPRTC_SUCCESS) {
        delete p;
        return hiprtc_to_nvrtc(err);
    }

    *prog = p;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
    if (!prog || !*prog) {
        return NVRTC_ERROR_INVALID_PROGRAM;
    }

    hiprtcDestroyProgram(&(*prog)->hip_program);
    delete *prog;
    *prog = nullptr;

    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
                                 int numOptions,
                                 const char* const* options) {
    if (!prog) {
        return NVRTC_ERROR_INVALID_PROGRAM;
    }

    // Convert CUDA compile options to HIP options
    std::vector<const char*> hip_options;
    std::vector<std::string> hip_option_storage;

    for (int i = 0; i < numOptions; i++) {
        std::string opt = options[i];

        // Translate some common options
        if (opt.find("--gpu-architecture=") == 0 || opt.find("-arch=") == 0) {
            // Skip CUDA arch options - hiprtc figures it out
            continue;
        }
        if (opt.find("-gencode") == 0) {
            continue;
        }

        // Keep other options
        hip_option_storage.push_back(opt);
        hip_options.push_back(hip_option_storage.back().c_str());
    }

    // Compile with hiprtc
    hiprtcResult err = hiprtcCompileProgram(
        prog->hip_program,
        hip_options.size(),
        hip_options.empty() ? nullptr : hip_options.data()
    );

    // Get compile log
    size_t logSize;
    hiprtcGetProgramLogSize(prog->hip_program, &logSize);
    if (logSize > 1) {
        prog->compile_log.resize(logSize);
        hiprtcGetProgramLog(prog->hip_program, &prog->compile_log[0]);
    }

    if (err != HIPRTC_SUCCESS) {
        const char* debug = getenv("CUDA_SHIM_DEBUG");
        if (debug && debug[0] == '1') {
            fprintf(stderr, "[NVRTC-SHIM] Compilation failed:\n%s\n", prog->compile_log.c_str());
        }
        return hiprtc_to_nvrtc(err);
    }

    // Get compiled code
    size_t codeSize;
    hiprtcGetCodeSize(prog->hip_program, &codeSize);
    prog->compiled_code.resize(codeSize);
    hiprtcGetCode(prog->hip_program, prog->compiled_code.data());

    prog->compiled = true;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) {
    if (!prog || !logSizeRet) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    *logSizeRet = prog->compile_log.size() + 1;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
    if (!prog || !log) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    strcpy(log, prog->compile_log.c_str());
    return NVRTC_SUCCESS;
}

// PTX is NVIDIA-specific, but some apps expect it
// We return our HIP code size instead (it's actually AMD GCN/RDNA code)
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) {
    if (!prog || !ptxSizeRet) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    if (!prog->compiled) {
        return NVRTC_ERROR_INVALID_PROGRAM;
    }
    *ptxSizeRet = prog->compiled_code.size();
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) {
    if (!prog || !ptx) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    if (!prog->compiled) {
        return NVRTC_ERROR_INVALID_PROGRAM;
    }
    memcpy(ptx, prog->compiled_code.data(), prog->compiled_code.size());
    return NVRTC_SUCCESS;
}

// CUBIN is the compiled binary - we return AMD binary
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) {
    return nvrtcGetPTXSize(prog, cubinSizeRet);
}

nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) {
    return nvrtcGetPTX(prog, cubin);
}

// Code is generic term for compiled output
nvrtcResult nvrtcGetCodeSize(nvrtcProgram prog, size_t* codeSizeRet) {
    return nvrtcGetPTXSize(prog, codeSizeRet);
}

nvrtcResult nvrtcGetCode(nvrtcProgram prog, char* code) {
    return nvrtcGetPTX(prog, code);
}

// Name mangling support
nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) {
    if (!prog || !name_expression) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    return hiprtc_to_nvrtc(hiprtcAddNameExpression(prog->hip_program, name_expression));
}

nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                 const char* name_expression,
                                 const char** lowered_name) {
    if (!prog || !name_expression || !lowered_name) {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    return hiprtc_to_nvrtc(hiprtcGetLoweredName(prog->hip_program, name_expression, lowered_name));
}

// =============================================================================
// Initialization
// =============================================================================

__attribute__((constructor))
static void nvrtc_shim_init() {
    const char* verbose = getenv("CUDA_SHIM_VERBOSE");
    if (verbose && verbose[0] == '1') {
        fprintf(stderr, "[NVRTC-SHIM] CUDA Runtime Compiler shim loaded (using hiprtc)\n");
    }
}

} // extern "C"
