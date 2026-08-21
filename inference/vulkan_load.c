/* vulkan_load.c — 动态加载 libvulkan / vulkan-1.dll 并创建 compute 队列设备 */
#include "vulkan_load.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

struct VulkanFns {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceQueue GetDeviceQueue;
};

static void* g_lib;
static struct VulkanFns g_fn;

static void* load_sym(void* lib, const char* name)
{
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static int open_lib(void)
{
    if (g_lib) return 0;
#ifdef _WIN32
    g_lib = (void*)LoadLibraryA("vulkan-1.dll");
#else
# ifdef __APPLE__
    g_lib = dlopen("libvulkan.1.dylib", RTLD_NOW);
    if (!g_lib) g_lib = dlopen("libMoltenVK.dylib", RTLD_NOW);
# else
    g_lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!g_lib) g_lib = dlopen("libvulkan.so", RTLD_NOW);
# endif
#endif
    if (!g_lib) return -1;
    g_fn.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)load_sym(g_lib, "vkGetInstanceProcAddr");
    if (!g_fn.GetInstanceProcAddr) return -1;
    return 0;
}

#define LOAD_I(name) do { \
    g_fn.name = (PFN_vk##name)g_fn.GetInstanceProcAddr(inst, "vk"#name); \
    if (!g_fn.name) return -1; \
} while (0)

static int load_instance_fns(VkInstance inst)
{
    LOAD_I(DestroyInstance);
    LOAD_I(EnumeratePhysicalDevices);
    LOAD_I(GetPhysicalDeviceProperties);
    LOAD_I(GetPhysicalDeviceQueueFamilyProperties);
    LOAD_I(CreateDevice);
    return 0;
}

int vulkan_try_init(VulkanCtx* ctx, int device_id, char* err, size_t errlen)
{
    if (!ctx) return -1;
    if (open_lib() != 0) {
        if (err && errlen) snprintf(err, errlen, "vulkan loader not found");
        return -1;
    }

    g_fn.CreateInstance = (PFN_vkCreateInstance)g_fn.GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!g_fn.CreateInstance) {
        if (err && errlen) snprintf(err, errlen, "vkCreateInstance missing");
        return -1;
    }

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "yllm";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "yllm";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = g_fn.CreateInstance(&ici, NULL, &inst);
    if (r != VK_SUCCESS) {
        if (err && errlen) snprintf(err, errlen, "vkCreateInstance failed (%d)", (int)r);
        return -1;
    }
    if (load_instance_fns(inst) != 0) {
        g_fn.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "load instance fns failed");
        return -1;
    }

    uint32_t ndev = 0;
    g_fn.EnumeratePhysicalDevices(inst, &ndev, NULL);
    if (ndev == 0) {
        g_fn.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "no Vulkan physical devices");
        return -1;
    }
    VkPhysicalDevice* phys = (VkPhysicalDevice*)calloc(ndev, sizeof(VkPhysicalDevice));
    if (!phys) {
        g_fn.DestroyInstance(inst, NULL);
        return -1;
    }
    g_fn.EnumeratePhysicalDevices(inst, &ndev, phys);
    int idx = device_id;
    if (idx < 0 || (uint32_t)idx >= ndev) idx = 0;
    VkPhysicalDevice pd = phys[idx];

    VkPhysicalDeviceProperties props;
    g_fn.GetPhysicalDeviceProperties(pd, &props);
    ylog_info("vulkan: physical[%d/%u] %s api=%u.%u",
              idx, ndev, props.deviceName,
              VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion));

    uint32_t nq = 0;
    g_fn.GetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties* qfp = (VkQueueFamilyProperties*)calloc(nq, sizeof(VkQueueFamilyProperties));
    if (!qfp) {
        free(phys);
        g_fn.DestroyInstance(inst, NULL);
        return -1;
    }
    g_fn.GetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfp);
    uint32_t qfam = (uint32_t)~0u;
    uint32_t i;
    for (i = 0; i < nq; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qfam = i;
            break;
        }
    }
    free(qfp);
    free(phys);
    if (qfam == (uint32_t)~0u) {
        g_fn.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "no compute queue family");
        return -1;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice dev = VK_NULL_HANDLE;
    r = g_fn.CreateDevice(pd, &dci, NULL, &dev);
    if (r != VK_SUCCESS) {
        g_fn.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "vkCreateDevice failed (%d)", (int)r);
        return -1;
    }

    PFN_vkGetDeviceQueue GetDeviceQueue =
        (PFN_vkGetDeviceQueue)g_fn.GetInstanceProcAddr(inst, "vkGetDeviceQueue");
    PFN_vkDestroyDevice DestroyDevice =
        (PFN_vkDestroyDevice)g_fn.GetInstanceProcAddr(inst, "vkDestroyDevice");
    if (!GetDeviceQueue || !DestroyDevice) {
        if (DestroyDevice) DestroyDevice(dev, NULL);
        g_fn.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "device fns missing");
        return -1;
    }
    g_fn.GetDeviceQueue = GetDeviceQueue;
    g_fn.DestroyDevice = DestroyDevice;

    VkQueue queue = VK_NULL_HANDLE;
    GetDeviceQueue(dev, qfam, 0, &queue);

    ctx->instance = (void*)inst;
    ctx->phys = (void*)pd;
    ctx->device = (void*)dev;
    ctx->queue = (void*)queue;
    ctx->queue_family = qfam;
    ctx->host_shim = 0;
    ctx->device_id = idx;
    return 0;
}

void vulkan_shutdown(VulkanCtx* ctx)
{
    if (!ctx) return;
    VkDevice dev = (VkDevice)ctx->device;
    VkInstance inst = (VkInstance)ctx->instance;
    if (dev && g_fn.DestroyDevice) g_fn.DestroyDevice(dev, NULL);
    if (inst && g_fn.DestroyInstance) g_fn.DestroyInstance(inst, NULL);
    ctx->instance = NULL;
    ctx->phys = NULL;
    ctx->device = NULL;
    ctx->queue = NULL;
}
