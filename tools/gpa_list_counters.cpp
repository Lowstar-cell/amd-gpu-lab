#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "gpu_performance_api/gpu_perf_api.h"
#include "gpu_performance_api/gpu_perf_api_interface_loader.h"

int main(int argc, char** argv) {
    const char* lib_path = nullptr;
    const char* out_path = "gpa_counters.csv";
    if (argc >= 2) lib_path = argv[1];
    if (argc >= 3) out_path = argv[2];

    GpaStatus s = GpaInitialize(kGpaInitializeDefaultBit);
    if (s != kGpaStatusOk) {
        std::cerr << "GpaInitialize failed: " << s << std::endl;
        return 1;
    }

    // Load API (try OpenGL first)
    GpaStatus status = GpaInterfaceLoaderLoadApi(kGpaApiOpengl, lib_path);
    if (status != kGpaStatusOk) {
        // try Vulkan
        status = GpaInterfaceLoaderLoadApi(kGpaApiVulkan, lib_path);
    }
    if (status != kGpaStatusOk) {
        std::cerr << "Failed to load GPA API libraries: " << status << std::endl;
        GpaDestroy();
        return 2;
    }

    // Now attempt to query counters via GpaGetNumCounters
    GpaUInt32 num = 0;
    GpaContextId ctx = NULL;
    GpaSessionId session = NULL;
    // Try to open context with a NULL api_context - this may work with some APIs
    s = GpaOpenContext(NULL, kGpaOpenContextDefaultBit | kGpaOpenContextEnableHardwareCountersBit, &ctx);
    if (s != kGpaStatusOk) {
        std::cerr << "GpaOpenContext failed: " << s << std::endl;
    } else {
        s = GpaCreateSession(ctx, kGpaSessionSampleTypeDiscreteCounter, &session);
        if (s != kGpaStatusOk) {
            std::cerr << "GpaCreateSession failed: " << s << std::endl;
        } else {
            s = GpaGetNumCounters(session, &num);
        }
    }
    if (s != kGpaStatusOk) {
        std::cerr << "GpaGetNumCounters failed: " << s << std::endl;
        GpaDestroy();
        return 3;
    }

    std::ofstream out(out_path);
    out << "index,name,group,description" << std::endl;
    for (GpaUInt32 i = 0; i < num; ++i) {
        const char* name = nullptr;
        const char* desc = nullptr;
        const char* group = nullptr;
        GpaGetCounterName(session, i, &name);
        GpaGetCounterDescription(session, i, &desc);
        GpaGetCounterGroup(session, i, &group);
        out << i << ",\"" << (name ? name : "") << "\",\"" << (group) << "\",\"" << (desc ? desc : "") << "\"" << std::endl;
    }
    out.close();

    if (session) {
        GpaDeleteSession(session);
    }
    if (ctx) {
        GpaCloseContext(ctx);
    }
    GpaDestroy();
    std::cout << "Wrote counters to " << out_path << std::endl;
    return 0;
}
