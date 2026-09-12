#include <vulkan/vulkan.h>

#include "weights.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

void check(VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with Vulkan error " +
            std::to_string(result));
    }
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string api_version(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(kGiB);
}

struct Arguments {
    bool list_only = false;
    bool validation = false;
    bool smoke_test = true;
    std::string device_selector = "auto";
    std::optional<std::uint64_t> vram_limit;
    std::optional<std::string> model_path;
    std::uint64_t reserve = 192ull * kMiB;
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "  --list                     list usable Vulkan compute devices\n"
        << "  --device <auto|index|text> select a device\n"
        << "  --vram-limit-gib <auto|N>  optional hard ceiling over live budget\n"
        << "  --model <weights.rvf>       validate and upload native weights\n"
        << "  --reserve-mib <N>          live-budget safety reserve (default 192)\n"
        << "  --validation               enable Vulkan validation layers\n"
        << "  --no-smoke-test            report capabilities without dispatching\n"
        << "  --help                     show this message\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + option);
            }
            return argv[index];
        };

        if (option == "--list") {
            arguments.list_only = true;
        } else if (option == "--validation") {
            arguments.validation = true;
        } else if (option == "--no-smoke-test") {
            arguments.smoke_test = false;
        } else if (option == "--device") {
            arguments.device_selector = value();
        } else if (option == "--reserve-mib") {
            arguments.reserve =
                static_cast<std::uint64_t>(std::stoull(value())) * kMiB;
        } else if (option == "--vram-limit-gib") {
            const std::string limit = value();
            if (limit != "auto") {
                const double parsed = std::stod(limit);
                if (!(parsed > 0.0)) {
                    throw std::runtime_error("VRAM limit must be positive");
                }
                arguments.vram_limit =
                    static_cast<std::uint64_t>(parsed * static_cast<double>(kGiB));
            }
        } else if (option == "--model") {
            arguments.model_path = value();
        } else if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    return arguments;
}

bool has_instance_extension(std::string_view name) {
    std::uint32_t count = 0;
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
          "enumerate instance extensions");
    std::vector<VkExtensionProperties> properties(count);
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data()),
          "enumerate instance extensions");
    return std::any_of(properties.begin(), properties.end(), [&](const auto& item) {
        return name == item.extensionName;
    });
}

bool has_layer(std::string_view name) {
    std::uint32_t count = 0;
    check(vkEnumerateInstanceLayerProperties(&count, nullptr),
          "enumerate instance layers");
    std::vector<VkLayerProperties> properties(count);
    check(vkEnumerateInstanceLayerProperties(&count, properties.data()),
          "enumerate instance layers");
    return std::any_of(properties.begin(), properties.end(), [&](const auto& item) {
        return name == item.layerName;
    });
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Vulkan validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}

struct Instance {
    VkInstance handle = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

    ~Instance() {
        if (messenger != VK_NULL_HANDLE && handle != VK_NULL_HANDLE) {
            auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(handle, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy) {
                destroy(handle, messenger, nullptr);
            }
        }
        if (handle != VK_NULL_HANDLE) {
            vkDestroyInstance(handle, nullptr);
        }
    }
};

Instance create_instance(bool validation) {
    Instance instance;
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    VkInstanceCreateFlags flags = 0;

    if (validation) {
        if (!has_layer("VK_LAYER_KHRONOS_validation")) {
            throw std::runtime_error(
                "--validation requested but VK_LAYER_KHRONOS_validation is absent");
        }
        layers.push_back("VK_LAYER_KHRONOS_validation");
        if (has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif

    std::uint32_t loader_version = VK_API_VERSION_1_0;
    const auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerate_instance_version) {
        check(enumerate_instance_version(&loader_version),
              "query Vulkan loader version");
    }
    if (loader_version < VK_API_VERSION_1_1) {
        throw std::runtime_error("Vulkan 1.1 or newer is required");
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "RealViformer Vulkan";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "rvf-vulkan";
    application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = std::min(loader_version, VK_API_VERSION_1_2);

    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.flags = flags;
    create.pApplicationInfo = &application;
    create.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create.ppEnabledLayerNames = layers.data();
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check(vkCreateInstance(&create, nullptr, &instance.handle), "create instance");

    if (validation && has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance.handle,
                                  "vkCreateDebugUtilsMessengerEXT"));
        if (create_debug) {
            VkDebugUtilsMessengerCreateInfoEXT debug{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug.pfnUserCallback = debug_callback;
            check(create_debug(instance.handle, &debug, nullptr, &instance.messenger),
                  "create debug messenger");
        }
    }
    return instance;
}

struct DeviceCandidate {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    std::uint32_t compute_queue = std::numeric_limits<std::uint32_t>::max();
    bool memory_budget = false;
    bool portability_subset = false;
    bool storage_16 = false;
    bool arithmetic_16 = false;
};

void refresh_memory_properties(DeviceCandidate& device) {
    device.budget = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 memory{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    if (device.memory_budget) {
        memory.pNext = &device.budget;
    }
    vkGetPhysicalDeviceMemoryProperties2(device.handle, &memory);
    device.memory = memory.memoryProperties;
}

std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    check(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
          "enumerate device extensions");
    std::vector<VkExtensionProperties> extensions(count);
    check(vkEnumerateDeviceExtensionProperties(
              device, nullptr, &count, extensions.data()),
          "enumerate device extensions");
    return extensions;
}

bool contains_extension(const std::vector<VkExtensionProperties>& extensions,
                        std::string_view name) {
    return std::any_of(extensions.begin(), extensions.end(), [&](const auto& item) {
        return name == item.extensionName;
    });
}

std::vector<DeviceCandidate> enumerate_devices(VkInstance instance) {
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr),
          "enumerate physical devices");
    if (count == 0) {
        throw std::runtime_error("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> physical_devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, physical_devices.data()),
          "enumerate physical devices");

    std::vector<DeviceCandidate> candidates;
    for (VkPhysicalDevice physical : physical_devices) {
        DeviceCandidate candidate;
        candidate.handle = physical;
        vkGetPhysicalDeviceProperties(physical, &candidate.properties);

        std::uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical, &queue_count, queues.data());
        for (std::uint32_t index = 0; index < queue_count; ++index) {
            if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                candidate.compute_queue = index;
                break;
            }
        }
        if (candidate.compute_queue == std::numeric_limits<std::uint32_t>::max()) {
            for (std::uint32_t index = 0; index < queue_count; ++index) {
                if (queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    candidate.compute_queue = index;
                    break;
                }
            }
        }
        if (candidate.compute_queue == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }

        const auto extensions = device_extensions(physical);
        candidate.memory_budget = contains_extension(
            extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        candidate.portability_subset = contains_extension(
            extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

        refresh_memory_properties(candidate);

        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDevice16BitStorageFeatures storage{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        VkPhysicalDeviceShaderFloat16Int8Features float16{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        features.pNext = &storage;
        storage.pNext = &float16;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        candidate.storage_16 = storage.storageBuffer16BitAccess == VK_TRUE;
        candidate.arithmetic_16 = float16.shaderFloat16 == VK_TRUE;
        candidates.push_back(candidate);
    }
    if (candidates.empty()) {
        throw std::runtime_error("no Vulkan device has a compute queue");
    }
    return candidates;
}

std::uint32_t largest_local_heap(const DeviceCandidate& device) {
    std::uint32_t selected = std::numeric_limits<std::uint32_t>::max();
    VkDeviceSize selected_size = 0;
    for (std::uint32_t index = 0; index < device.memory.memoryHeapCount; ++index) {
        const auto& heap = device.memory.memoryHeaps[index];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            heap.size > selected_size) {
            selected = index;
            selected_size = heap.size;
        }
    }
    if (selected == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("device has no local memory heap");
    }
    return selected;
}

void print_device(const DeviceCandidate& device, std::size_t index) {
    const std::uint32_t heap = largest_local_heap(device);
    const std::uint64_t size = device.memory.memoryHeaps[heap].size;
    const std::uint64_t budget =
        device.memory_budget ? device.budget.heapBudget[heap] : size;
    const std::uint64_t usage =
        device.memory_budget ? device.budget.heapUsage[heap] : 0;
    std::cout << '[' << index << "] " << device.properties.deviceName
              << " (vendor 0x" << std::hex << device.properties.vendorID
              << ", device 0x" << device.properties.deviceID << std::dec << ")\n"
              << "    Vulkan " << api_version(device.properties.apiVersion)
              << ", queue family " << device.compute_queue
              << ", local heap " << std::fixed << std::setprecision(2)
              << gib(size) << " GiB\n"
              << "    live budget " << gib(budget) << " GiB, process usage "
              << gib(usage) << " GiB, VK_EXT_memory_budget "
              << (device.memory_budget ? "yes" : "no") << "\n"
              << "    FP16 storage " << (device.storage_16 ? "yes" : "no")
              << ", FP16 arithmetic "
              << (device.arithmetic_16 ? "yes" : "no") << '\n';
}

std::size_t select_device(const std::vector<DeviceCandidate>& devices,
                          const std::string& selector) {
    if (selector != "auto") {
        try {
            std::size_t consumed = 0;
            const std::size_t index = std::stoull(selector, &consumed);
            if (consumed == selector.size() && index < devices.size()) {
                return index;
            }
        } catch (const std::exception&) {
        }
        const std::string needle = lower(selector);
        for (std::size_t index = 0; index < devices.size(); ++index) {
            if (lower(devices[index].properties.deviceName).find(needle) !=
                std::string::npos) {
                return index;
            }
        }
        throw std::runtime_error("device selector did not match: " + selector);
    }

    auto score = [](const DeviceCandidate& device) {
        std::uint64_t value = device.memory.memoryHeaps[largest_local_heap(device)].size;
        if (device.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            value += 1ull << 62;
        } else if (device.properties.deviceType ==
                   VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            value += 1ull << 61;
        }
        return value;
    };
    std::size_t best = 0;
    for (std::size_t index = 1; index < devices.size(); ++index) {
        if (score(devices[index]) > score(devices[best])) {
            best = index;
        }
    }
    return best;
}

struct LogicalDevice {
    VkDevice handle = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;

    ~LogicalDevice() {
        if (handle != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(handle);
            vkDestroyDevice(handle, nullptr);
        }
    }
};

LogicalDevice create_logical_device(const DeviceCandidate& physical) {
    LogicalDevice logical;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = physical.compute_queue;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;

    std::vector<const char*> extensions;
    if (physical.memory_budget) {
        extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (physical.portability_subset) {
        extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif

    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue;
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physical.handle, &create, nullptr, &logical.handle),
          "create logical device");
    vkGetDeviceQueue(logical.handle, physical.compute_queue, 0, &logical.queue);
    return logical;
}

struct MemoryPlan {
    std::uint32_t heap = 0;
    std::uint64_t heap_size = 0;
    std::uint64_t live_budget = 0;
    std::uint64_t process_usage = 0;
    std::uint64_t reserve = 0;
    std::uint64_t ceiling = 0;
};

MemoryPlan plan_memory(DeviceCandidate& device,
                       const Arguments& arguments) {
    // The live budget changes with desktop activity. Re-query at each future
    // inference planning boundary instead of treating startup as permanent.
    refresh_memory_properties(device);
    MemoryPlan plan;
    plan.heap = largest_local_heap(device);
    plan.heap_size = device.memory.memoryHeaps[plan.heap].size;
    plan.live_budget = device.memory_budget
                           ? device.budget.heapBudget[plan.heap]
                           : plan.heap_size;
    plan.process_usage =
        device.memory_budget ? device.budget.heapUsage[plan.heap] : 0;
    const std::uint64_t free_budget = plan.live_budget > plan.process_usage
                                          ? plan.live_budget - plan.process_usage
                                          : 0;
    plan.reserve = std::min(arguments.reserve, free_budget / 4);
    plan.ceiling = free_budget - plan.reserve;
    if (arguments.vram_limit) {
        plan.ceiling = std::min(plan.ceiling, *arguments.vram_limit);
    }
    return plan;
}

std::uint32_t find_memory_type(
    const VkPhysicalDeviceMemoryProperties& memory,
    std::uint32_t allowed,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred = 0) {
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        const VkMemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
        if ((allowed & (1u << index)) && (flags & required) == required &&
            (flags & preferred) == preferred) {
            return index;
        }
    }
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        const VkMemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
        if ((allowed & (1u << index)) && (flags & required) == required) {
            return index;
        }
    }
    throw std::runtime_error("no compatible Vulkan memory type found");
}

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkMemoryPropertyFlags properties = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept {
        *this = std::move(other);
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            destroy();
            device = other.device;
            handle = other.handle;
            memory = other.memory;
            size = other.size;
            properties = other.properties;
            other.handle = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
        }
        return *this;
    }
    ~Buffer() { destroy(); }

    void destroy() {
        if (handle != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, handle, nullptr);
            handle = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
};

Buffer create_buffer(VkDevice device,
                     const VkPhysicalDeviceMemoryProperties& memory_properties,
                     VkDeviceSize size,
                     VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags required,
                     VkMemoryPropertyFlags preferred = 0) {
    Buffer buffer;
    buffer.device = device;
    buffer.size = size;

    VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create.size = size;
    create.usage = usage;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device, &create, nullptr, &buffer.handle),
          "create buffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
    const std::uint32_t memory_type = find_memory_type(
        memory_properties, requirements.memoryTypeBits, required, preferred);
    buffer.properties = memory_properties.memoryTypes[memory_type].propertyFlags;

    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    check(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory),
          "allocate buffer memory");
    check(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0),
          "bind buffer memory");
    return buffer;
}

std::vector<std::uint32_t> read_shader(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open shader: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        throw std::runtime_error("invalid SPIR-V size: " + path);
    }
    file.seekg(0);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4);
    if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
        throw std::runtime_error("cannot read shader: " + path);
    }
    return words;
}

void run_smoke_test(const DeviceCandidate& physical,
                    const LogicalDevice& logical,
                    const MemoryPlan& plan) {
    constexpr std::uint32_t element_count = 1u << 20;
    constexpr float alpha = 1.25f;
    const VkDeviceSize vector_bytes =
        static_cast<VkDeviceSize>(element_count) * sizeof(float);
    const std::uint64_t local_bytes = 3ull * vector_bytes;
    if (local_bytes > plan.ceiling) {
        throw std::runtime_error("smoke test exceeds the dynamic VRAM ceiling");
    }

    Buffer staging_input = create_buffer(
        logical.handle, physical.memory, vector_bytes * 2,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer staging_output = create_buffer(
        logical.handle, physical.memory, vector_bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer input_a = create_buffer(
        logical.handle, physical.memory, vector_bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer input_b = create_buffer(
        logical.handle, physical.memory, vector_bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer output_c = create_buffer(
        logical.handle, physical.memory, vector_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    void* mapped = nullptr;
    check(vkMapMemory(logical.handle, staging_input.memory, 0,
                      staging_input.size, 0, &mapped),
          "map staging input");
    auto* values = static_cast<float*>(mapped);
    for (std::uint32_t index = 0; index < element_count; ++index) {
        values[index] = static_cast<float>(index) * 0.25f;
        values[element_count + index] =
            static_cast<float>(element_count - index) * 0.5f;
    }
    if (!(staging_input.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = staging_input.memory;
        range.size = VK_WHOLE_SIZE;
        check(vkFlushMappedMemoryRanges(logical.handle, 1, &range),
              "flush staging input");
    }
    vkUnmapMemory(logical.handle, staging_input.memory);

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_create{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_create.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_create.pBindings = bindings.data();
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(logical.handle, &layout_create, nullptr,
                                      &descriptor_layout),
          "create descriptor layout");

    struct PushConstants {
        std::uint32_t count;
        float alpha;
    };
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_create{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create.setLayoutCount = 1;
    pipeline_layout_create.pSetLayouts = &descriptor_layout;
    pipeline_layout_create.pushConstantRangeCount = 1;
    pipeline_layout_create.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(logical.handle, &pipeline_layout_create, nullptr,
                                 &pipeline_layout),
          "create pipeline layout");

    const auto shader_words = read_shader(RVF_AXPY_SHADER_PATH);
    VkShaderModuleCreateInfo shader_create{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_create.codeSize = shader_words.size() * sizeof(std::uint32_t);
    shader_create.pCode = shader_words.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    check(vkCreateShaderModule(logical.handle, &shader_create, nullptr, &shader),
          "create shader module");

    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_create{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_create.stage = stage;
    pipeline_create.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateComputePipelines(logical.handle, VK_NULL_HANDLE, 1,
                                   &pipeline_create, nullptr, &pipeline),
          "create compute pipeline");

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;
    VkDescriptorPoolCreateInfo pool_create{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_create.maxSets = 1;
    pool_create.poolSizeCount = 1;
    pool_create.pPoolSizes = &pool_size;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(logical.handle, &pool_create, nullptr,
                                 &descriptor_pool),
          "create descriptor pool");
    VkDescriptorSetAllocateInfo set_allocate{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_allocate.descriptorPool = descriptor_pool;
    set_allocate.descriptorSetCount = 1;
    set_allocate.pSetLayouts = &descriptor_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check(vkAllocateDescriptorSets(logical.handle, &set_allocate, &descriptor_set),
          "allocate descriptor set");

    const std::array<VkDescriptorBufferInfo, 3> buffer_info{{
        {input_a.handle, 0, vector_bytes},
        {input_b.handle, 0, vector_bytes},
        {output_c.handle, 0, vector_bytes},
    }};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (std::uint32_t index = 0; index < writes.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &buffer_info[index];
    }
    vkUpdateDescriptorSets(logical.handle, static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    VkCommandPoolCreateInfo command_pool_create{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_create.queueFamilyIndex = physical.compute_queue;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(logical.handle, &command_pool_create, nullptr,
                              &command_pool),
          "create command pool");
    VkCommandBufferAllocateInfo command_allocate{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocate.commandPool = command_pool;
    command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(logical.handle, &command_allocate, &command),
          "allocate command buffer");

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "begin command buffer");
    VkBufferCopy copy_a{0, 0, vector_bytes};
    VkBufferCopy copy_b{vector_bytes, 0, vector_bytes};
    vkCmdCopyBuffer(command, staging_input.handle, input_a.handle, 1, &copy_a);
    vkCmdCopyBuffer(command, staging_input.handle, input_b.handle, 1, &copy_b);

    std::array<VkBufferMemoryBarrier, 2> input_barriers{};
    for (std::uint32_t index = 0; index < input_barriers.size(); ++index) {
        auto& barrier = input_barriers[index];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = index == 0 ? input_a.handle : input_b.handle;
        barrier.size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(input_barriers.size()),
                         input_barriers.data(), 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    const PushConstants parameters{element_count, alpha};
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(parameters), &parameters);
    vkCmdDispatch(command, (element_count + 255) / 256, 1, 1);

    VkBufferMemoryBarrier output_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output_barrier.buffer = output_c.handle;
    output_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &output_barrier, 0, nullptr);
    VkBufferCopy copy_output{0, 0, vector_bytes};
    vkCmdCopyBuffer(command, output_c.handle, staging_output.handle, 1,
                    &copy_output);
    check(vkEndCommandBuffer(command), "end command buffer");

    VkFenceCreateInfo fence_create{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(logical.handle, &fence_create, nullptr, &fence),
          "create fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const auto start = std::chrono::steady_clock::now();
    check(vkQueueSubmit(logical.queue, 1, &submit, fence), "submit smoke test");
    check(vkWaitForFences(logical.handle, 1, &fence, VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "wait for smoke test");
    const auto elapsed = std::chrono::steady_clock::now() - start;

    check(vkMapMemory(logical.handle, staging_output.memory, 0,
                      staging_output.size, 0, &mapped),
          "map staging output");
    if (!(staging_output.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = staging_output.memory;
        range.size = VK_WHOLE_SIZE;
        check(vkInvalidateMappedMemoryRanges(logical.handle, 1, &range),
              "invalidate staging output");
    }
    const auto* output = static_cast<const float*>(mapped);
    float maximum_error = 0.0f;
    for (std::uint32_t index = 0; index < element_count; ++index) {
        const float expected = alpha * (static_cast<float>(index) * 0.25f) +
                               static_cast<float>(element_count - index) * 0.5f;
        maximum_error = std::max(maximum_error, std::abs(output[index] - expected));
    }
    vkUnmapMemory(logical.handle, staging_output.memory);

    vkDestroyFence(logical.handle, fence, nullptr);
    vkDestroyCommandPool(logical.handle, command_pool, nullptr);
    vkDestroyDescriptorPool(logical.handle, descriptor_pool, nullptr);
    vkDestroyPipeline(logical.handle, pipeline, nullptr);
    vkDestroyShaderModule(logical.handle, shader, nullptr);
    vkDestroyPipelineLayout(logical.handle, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(logical.handle, descriptor_layout, nullptr);

    if (maximum_error > 1e-4f) {
        throw std::runtime_error("compute smoke test produced incorrect results");
    }
    std::cout << "Compute smoke test: PASS (" << element_count
              << " FP32 elements, max error " << maximum_error << ", "
              << std::chrono::duration<double, std::milli>(elapsed).count()
              << " ms submit-to-fence)\n";
}

void upload_and_verify_weights(const DeviceCandidate& physical,
                               const LogicalDevice& logical,
                               const MemoryPlan& plan,
                               const rvf::WeightFile& weights) {
    const std::span<const std::byte> data = weights.data();
    if (data.empty()) {
        throw std::runtime_error("native model contains no weight data");
    }
    if (data.size() > plan.ceiling) {
        throw std::runtime_error("model weights exceed the dynamic VRAM ceiling");
    }

    Buffer upload = create_buffer(
        logical.handle, physical.memory, data.size(),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer download = create_buffer(
        logical.handle, physical.memory, data.size(),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer device_weights = create_buffer(
        logical.handle, physical.memory, data.size(),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    void* mapped = nullptr;
    check(vkMapMemory(logical.handle, upload.memory, 0, upload.size, 0, &mapped),
          "map model upload buffer");
    std::memcpy(mapped, data.data(), data.size());
    if (!(upload.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = upload.memory;
        range.size = VK_WHOLE_SIZE;
        check(vkFlushMappedMemoryRanges(logical.handle, 1, &range),
              "flush model upload buffer");
    }
    vkUnmapMemory(logical.handle, upload.memory);

    VkCommandPoolCreateInfo pool_create{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_create.queueFamilyIndex = physical.compute_queue;
    VkCommandPool pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(logical.handle, &pool_create, nullptr, &pool),
          "create model upload command pool");
    VkCommandBufferAllocateInfo command_allocate{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocate.commandPool = pool;
    command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(logical.handle, &command_allocate, &command),
          "allocate model upload command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "begin model upload");
    VkBufferCopy region{0, 0, data.size()};
    vkCmdCopyBuffer(command, upload.handle, device_weights.handle, 1, &region);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = device_weights.handle;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
    vkCmdCopyBuffer(command, device_weights.handle, download.handle, 1, &region);
    check(vkEndCommandBuffer(command), "end model upload");

    VkFenceCreateInfo fence_create{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(logical.handle, &fence_create, nullptr, &fence),
          "create model upload fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const auto start = std::chrono::steady_clock::now();
    check(vkQueueSubmit(logical.queue, 1, &submit, fence), "submit model upload");
    check(vkWaitForFences(logical.handle, 1, &fence, VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "wait for model upload");
    const auto elapsed = std::chrono::steady_clock::now() - start;

    check(vkMapMemory(logical.handle, download.memory, 0, download.size, 0,
                      &mapped),
          "map model verification buffer");
    if (!(download.properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = download.memory;
        range.size = VK_WHOLE_SIZE;
        check(vkInvalidateMappedMemoryRanges(logical.handle, 1, &range),
              "invalidate model verification buffer");
    }
    const bool matches = std::memcmp(mapped, data.data(), data.size()) == 0;
    vkUnmapMemory(logical.handle, download.memory);
    vkDestroyFence(logical.handle, fence, nullptr);
    vkDestroyCommandPool(logical.handle, pool, nullptr);
    if (!matches) {
        throw std::runtime_error("GPU model upload verification failed");
    }
    std::cout << "Model upload: PASS (" << weights.tensors().size()
              << " tensors, " << std::fixed << std::setprecision(2)
              << static_cast<double>(data.size()) / static_cast<double>(kMiB)
              << " MiB, upload plus readback "
              << std::chrono::duration<double, std::milli>(elapsed).count()
              << " ms)\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        Instance instance = create_instance(arguments.validation);
        const auto devices = enumerate_devices(instance.handle);
        for (std::size_t index = 0; index < devices.size(); ++index) {
            print_device(devices[index], index);
        }
        if (arguments.list_only) {
            return 0;
        }

        const std::size_t selected_index =
            select_device(devices, arguments.device_selector);
        DeviceCandidate physical = devices[selected_index];
        const MemoryPlan plan = plan_memory(physical, arguments);
        std::cout << "Selected device: [" << selected_index << "] "
                  << physical.properties.deviceName << '\n'
                  << "Dynamic local-memory ceiling: " << std::fixed
                  << std::setprecision(2) << gib(plan.ceiling) << " GiB"
                  << " (live budget " << gib(plan.live_budget)
                  << " GiB, reserve " << gib(plan.reserve) << " GiB)\n";
        if (plan.ceiling < 64ull * kMiB) {
            throw std::runtime_error(
                "less than 64 MiB remains under the dynamic memory ceiling");
        }

        LogicalDevice logical = create_logical_device(physical);
        if (arguments.smoke_test) {
            run_smoke_test(physical, logical, plan);
        }
        if (arguments.model_path) {
            const rvf::WeightFile weights(*arguments.model_path);
            upload_and_verify_weights(physical, logical, plan, weights);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
