#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#include <GLFW/glfw3native.h>
#include "gpu_performance_api/gpu_perf_api.h"
#include "gpu_performance_api/gpu_perf_api_interface_loader.h"

int main(int argc, char** argv) {
    const char* lib_path = nullptr;
    const char* out_path = "gpa_sample.csv";
    int iterations = 10;
    if (argc >= 2) lib_path = argv[1];
    if (argc >= 3) out_path = argv[2];
    if (argc >= 4) iterations = atoi(argv[3]);

    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_CONTEXT_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(640, 480, "GPA Instrument Sample", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // initialize GPA
    GpaStatus status = GpaInitialize(kGpaInitializeDefaultBit);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaInitialize failed: " << status << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    status = GpaInterfaceLoaderLoadApi(kGpaApiOpengl, lib_path);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaInterfaceLoaderLoadApi failed: " << status << std::endl;
    }

    // get native GLX context
    GLXContext glxCtx = glfwGetGLXContext(window);
    if (!glxCtx) {
        std::cerr << "Failed to get GLX context" << std::endl;
    }

    GpaContextId gpaCtx = NULL;
    status = GpaOpenContext((void*)glxCtx, kGpaOpenContextEnableHardwareCountersBit, &gpaCtx);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaOpenContext failed: " << status << std::endl;
    }

    GpaSessionId session = NULL;
    status = GpaCreateSession(gpaCtx, kGpaSessionSampleTypeDiscreteCounter, &session);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaCreateSession failed: " << status << std::endl;
    }

    status = GpaBeginSession(session);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaBeginSession failed: " << status << std::endl;
    }

    // enumerate counters
    GpaUInt32 num = 0;
    status = GpaGetNumCounters(session, &num);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaGetNumCounters failed: " << status << std::endl;
    }
    std::cerr << "CPU: num counters=" << num << std::endl;

    // enable a small set of counters (first 4)
    std::vector<int> enabled;
    for (GpaUInt32 i = 0; i < num && i < 4; ++i) {
        status = GpaEnableCounter(session, i);
        if (status == kGpaStatusOk) enabled.push_back(i);
    }

    // query pass count
    GpaUInt32 passCount = 0;
    status = GpaGetPassCount(session, &passCount);
    std::cerr << "pass count: " << passCount << std::endl;

    // Query per-counter data types & usage types for decoding
    std::vector<GpaDataType> counter_data_types(enabled.size(), kGpaDataTypeUint64);
    std::vector<GpaUsageType> counter_usage_types(enabled.size(), kGpaUsageTypeLast);
    for (size_t idx = 0; idx < enabled.size(); ++idx) {
        GpaDataType dt = kGpaDataTypeUint64;
        GpaUsageType ut = kGpaUsageTypeUnknown;
        GpaStatus dtStatus = GpaGetCounterDataType(session, enabled[idx], &dt);
        if (dtStatus == kGpaStatusOk) counter_data_types[idx] = dt;
        GpaStatus utStatus = GpaGetCounterUsageType(session, enabled[idx], &ut);
        if (utStatus == kGpaStatusOk) counter_usage_types[idx] = ut;
    }

    // begin sample capture and draw a few frames
    std::ofstream out(out_path);
    out << "iter,";
    std::ofstream meta_out(std::string(out_path) + ".meta.json");
    meta_out << "{" << std::endl;
    GpaUInt32 deviceId = 0, revisionId = 0;
    if (GpaGetDeviceAndRevisionId(gpaCtx, &deviceId, &revisionId) == kGpaStatusOk) {
        meta_out << "\"device_id\": " << (int)deviceId << ",\n";
    } else {
        meta_out << "\"device_id\": " << (int)0 << ",\n";
    }
    meta_out << "\"counters\": [\n";
    for (size_t i = 0; i < enabled.size(); ++i) {
        out << "counter_" << enabled[i] << (i + 1 < enabled.size() ? "," : "\n");
        const char* counter_name = "";
        GpaGetCounterName(session, enabled[i], &counter_name);
        const char* usage_str = "";
        GpaGetUsageTypeAsStr(counter_usage_types[i], &usage_str);
        meta_out << "  {\"index\": " << enabled[i] << ", \"name\": \"" << counter_name << "\", \"usage\": \"" << usage_str << "\" }" << (i + 1 < enabled.size() ? ",\n" : "\n");
    }
    // Log selected counters
    std::cerr << "Enabled counters: ";
    for (auto c : enabled) std::cerr << c << " ";
    std::cerr << std::endl;
    meta_out << "]\n}" << std::endl;
    meta_out.close();

    for (int it = 0; it < iterations; ++it) {
        for (GpaUInt32 pass = 0; pass < passCount; ++pass) {
            std::cerr << "pass " << pass << std::endl;
            // begin sample
            GpaSampleId sample = 0;
            status = GpaBeginSample(session, nullptr, &sample);
            if (status != kGpaStatusOk) std::cerr << "GpaBeginSample failed: " << status << std::endl;

            // do some GL work
            glClearColor((float)it/iterations, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glfwSwapBuffers(window);

            status = GpaEndSample(session);
            if (status != kGpaStatusOk) std::cerr << "GpaEndSample failed: " << status << std::endl;

            // get sample result
            size_t size = 0;
            status = GpaGetSampleResultSize(session, sample, &size);
            if (status != kGpaStatusOk) {
                std::cerr << "GpaGetSampleResultSize failed: " << status << std::endl;
            } else {
                std::vector<char> buffer(size);
                status = GpaGetSampleResult(session, sample, size, buffer.data());
                if (status != kGpaStatusOk) {
                    std::cerr << "GpaGetSampleResult failed: " << status << std::endl;
                    out << it << "," << 0;
                    for (size_t ii = 1; ii < enabled.size(); ++ii) out << "," << 0;
                    out << "\n";
                } else {
                    // Interpret buffer using data types per counter, normalize based on GpaUsageType, and write numeric doubles
                    size_t offset = 0;
                    out << it;
                    for (size_t j = 0; j < enabled.size(); ++j) {
                        double val = 0.0;
                        GpaUsageType ut = counter_usage_types[j];
                        size_t remaining = (offset >= size ? 0 : (size - offset));
                        GpaDataType dt = counter_data_types[j];
                        if (dt == kGpaDataTypeFloat32 && remaining >= sizeof(float)) {
                            float v = *reinterpret_cast<float*>(buffer.data() + offset);
                            val = static_cast<double>(v);
                            offset += sizeof(float);
                        } else if (dt == kGpaDataTypeFloat64 && remaining >= sizeof(double)) {
                            double v = *reinterpret_cast<double*>(buffer.data() + offset);
                            val = v;
                            offset += sizeof(double);
                        } else if ((dt == kGpaDataTypeUint64 || dt == kGpaDataTypeInt64) && remaining >= sizeof(uint64_t)) {
                            uint64_t v = *reinterpret_cast<uint64_t*>(buffer.data() + offset);
                            val = static_cast<double>(v);
                            offset += sizeof(uint64_t);
                        } else if ((dt == kGpaDataTypeUint32 || dt == kGpaDataTypeInt32) && remaining >= sizeof(uint32_t)) {
                            uint32_t v = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
                            val = static_cast<double>(v);
                            offset += sizeof(uint32_t);
                        } else {
                            // fallback: read as uint64
                            if (remaining >= sizeof(uint64_t)) {
                                uint64_t v = *reinterpret_cast<uint64_t*>(buffer.data() + offset);
                                val = static_cast<double>(v);
                                offset += sizeof(uint64_t);
                            }
                        }
                        // normalize by usage type
                        switch (ut) {
                            case kGpaUsageTypePercentage:
                                val = val / 100.0;
                                break;
                            case kGpaUsageTypeMilliseconds:
                                val = val / 1000.0;
                                break;
                            case kGpaUsageTypeNanoseconds:
                                val = val / 1e9;
                                break;
                            case kGpaUsageTypeBytes:
                                val = val / (1024.0 * 1024.0); // to MB
                                break;
                            case kGpaUsageTypeKilobytes:
                                val = val / 1024.0; // to MB
                                break;
                            case kGpaUsageTypeCycles:
                                val = val / 1e6;
                                break;
                            case kGpaUsageTypeItems:
                            case kGpaUsageTypeRatio:
                            default:
                                break;
                        }
                        out << "," << val;
                    }
                    out << "\n";
                }
            }
        }
    }

    out.close();

    if (session) GpaEndSession(session);
    if (session) GpaDeleteSession(session);
    if (gpaCtx) GpaCloseContext(gpaCtx);
    GpaDestroy();

    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Done: wrote to " << out_path << std::endl;
    return 0;
}
