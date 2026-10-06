#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <vulkan/vulkan.h>
#include "gpu_performance_api/gpu_perf_api.h"
#include "gpu_performance_api/gpu_perf_api_interface_loader.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                   VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                   const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                   void* pUserData) {
    std::cerr << "Vulkan: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

int main(int argc, char** argv) {
    const char* lib_path = nullptr;
    const char* out_path = "gpa_sample_vk.csv";
    int iterations = 10;
    if (argc >= 2) lib_path = argv[1];
    if (argc >= 3) out_path = argv[2];
    if (argc >= 4) iterations = atoi(argv[3]);

    VkResult res;
    VkInstance instance;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPA Vulkan Sample";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    res = vkCreateInstance(&createInfo, nullptr, &instance);
    if (res != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance\n";
        return 1;
    }

    // pick first physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "Failed to find GPUs with Vulkan support\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());
    VkPhysicalDevice physicalDevice = physicalDevices[0];

    // find queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    int queueFamilyIndex = -1;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == -1) {
        std::cerr << "Failed to find a graphics queue\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = (uint32_t)queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device;
    res = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    if (res != VK_SUCCESS) {
        std::cerr << "Failed to create logical device\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Prepare GPA
    GpaStatus status = GpaInitialize(kGpaInitializeDefaultBit);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaInitialize failed: " << status << std::endl;
    }

    status = GpaInterfaceLoaderLoadApi(kGpaApiVulkan, lib_path);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaInterfaceLoaderLoadApi failed: " << status << std::endl;
    }

    // Use VkDevice as API context
    GpaContextId gpaCtx = NULL;
    status = GpaOpenContext((void*)device, kGpaOpenContextEnableHardwareCountersBit, &gpaCtx);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaOpenContext failed: " << status << std::endl;
    }

    // Create session
    GpaSessionId session = NULL;
    status = GpaCreateSession(gpaCtx, kGpaSessionSampleTypeDiscreteCounter, &session);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaCreateSession failed: " << status << std::endl;
    }
    status = GpaBeginSession(session);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaBeginSession failed: " << status << std::endl;
    }

    GpaUInt32 num = 0;
    status = GpaGetNumCounters(session, &num);
    if (status != kGpaStatusOk) {
        std::cerr << "GpaGetNumCounters failed: " << status << std::endl;
    }
    std::cerr << "num counters (Vulkan): " << num << std::endl;

    std::vector<int> enabled;
    for (GpaUInt32 i = 0; i < num && i < 4; ++i) {
        status = GpaEnableCounter(session, i);
        if (status == kGpaStatusOk) enabled.push_back(i);
    }

    GpaUInt32 passCount = 0;
    status = GpaGetPassCount(session, &passCount);
    std::cerr << "pass count: " << passCount << std::endl;

    // Create a queue and command pool to submit a dummy command buffer
    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = 0;
    VkCommandPool commandPool;
    res = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
    if (res != VK_SUCCESS) {
        std::cerr << "Failed to create command pool\n";
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    std::ofstream out(out_path);
    out << "iter,";
    std::ofstream meta_out(std::string(out_path) + ".meta.json");
    // Create counter metadata JSON
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

    // Query and record counter data types & usage types
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

    for (int it = 0; it < iterations; ++it) {
        for (GpaUInt32 pass = 0; pass < passCount; ++pass) {
            std::cerr << "pass " << pass << std::endl;
            // begin sample
            GpaSampleId sample = 0;
            status = GpaBeginSample(session, nullptr, &sample);
            if (status != kGpaStatusOk) std::cerr << "GpaBeginSample failed: " << status << std::endl;

            // record and submit a no-op command buffer
            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            vkEndCommandBuffer(commandBuffer);
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);

            status = GpaEndSample(session);
            if (status != kGpaStatusOk) std::cerr << "GpaEndSample failed: " << status << std::endl;

                size_t size = 0;
            status = GpaGetSampleResultSize(session, sample, &size);
            if (status != kGpaStatusOk) {
                std::cerr << "GpaGetSampleResultSize failed: " << status << std::endl;
            } else {
                std::vector<char> buffer(size);
                status = GpaGetSampleResult(session, sample, size, buffer.data());
                if (status != kGpaStatusOk) {
                    std::cerr << "GpaGetSampleResult failed: " << status << std::endl;
                    out << it << ",0";
                    for (size_t ii = 1; ii < enabled.size(); ++ii) out << "," << 0;
                    out << "\n";
                } else {
                    // Interpret buffer based on data type for each counter, normalize based on GpaUsageType, and write as double
                    size_t offset = 0;
                    out << it;
                    for (size_t j = 0; j < enabled.size(); ++j) {
                        double val = 0.0;
                        // safe check for not running off the buffer
                        size_t remaining = (offset >= size ? 0 : (size - offset));
                        GpaDataType dt = counter_data_types[j];
                        GpaUsageType ut = counter_usage_types[j];
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
                            // fallback: try to read as uint64
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
                                val = val / (1024.0 * 1024.0); // convert to MB
                                break;
                            case kGpaUsageTypeKilobytes:
                                val = val / 1024.0; // to MB
                                break;
                            case kGpaUsageTypeCycles:
                                val = val / 1e6; // approximate scaling to keep manageable
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

    // cleanup
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
    if (session) GpaEndSession(session);
    if (session) GpaDeleteSession(session);
    if (gpaCtx) GpaCloseContext(gpaCtx);
    GpaDestroy();
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);

    std::cout << "Done: wrote to " << out_path << std::endl;
    return 0;
}
