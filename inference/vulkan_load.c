/* vulkan_load.c — 动态加载 Vulkan + 创建设备; 填充 vulkan_api() */
#include "vulkan_load.h"
#include "vulkan_api.h"
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

static void* g_lib;
static VulkanApi g_api;
static PFN_vkCreateInstance g_CreateInstance;
static PFN_vkEnumeratePhysicalDevices g_EnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceProperties g_GetPhysicalDeviceProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties g_GetPhysicalDeviceQueueFamilyProperties;

VulkanApi* vulkan_api(void) { return &g_api; }

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
    g_api.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)load_sym(g_lib, "vkGetInstanceProcAddr");
    return g_api.GetInstanceProcAddr ? 0 : -1;
}

#define GI(name) (PFN_vk##name)g_api.GetInstanceProcAddr(inst, "vk"#name)
#define GD(name) do { \
    g_api.name = (PFN_vk##name)g_api.GetDeviceProcAddr(dev, "vk"#name); \
    if (!g_api.name) g_api.name = (PFN_vk##name)g_api.GetInstanceProcAddr(inst, "vk"#name); \
    if (!g_api.name) return -1; \
} while (0)

static int load_device_fns(VkInstance inst, VkDevice dev)
{
    g_api.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)g_api.GetInstanceProcAddr(inst, "vkGetDeviceProcAddr");
    if (!g_api.GetDeviceProcAddr) return -1;
    g_api.DestroyInstance = GI(DestroyInstance);
    g_api.GetPhysicalDeviceMemoryProperties = GI(GetPhysicalDeviceMemoryProperties);
    g_api.DestroyDevice = GI(DestroyDevice);
    g_api.GetDeviceQueue = GI(GetDeviceQueue);
    GD(CreateBuffer);
    GD(DestroyBuffer);
    GD(GetBufferMemoryRequirements);
    GD(AllocateMemory);
    GD(FreeMemory);
    GD(BindBufferMemory);
    GD(MapMemory);
    GD(UnmapMemory);
    GD(CreateShaderModule);
    GD(DestroyShaderModule);
    GD(CreateDescriptorSetLayout);
    GD(DestroyDescriptorSetLayout);
    GD(CreatePipelineLayout);
    GD(DestroyPipelineLayout);
    GD(CreateComputePipelines);
    GD(DestroyPipeline);
    GD(CreateDescriptorPool);
    GD(DestroyDescriptorPool);
    GD(AllocateDescriptorSets);
    GD(UpdateDescriptorSets);
    GD(CreateCommandPool);
    GD(DestroyCommandPool);
    GD(AllocateCommandBuffers);
    GD(BeginCommandBuffer);
    GD(EndCommandBuffer);
    GD(CmdBindPipeline);
    GD(CmdBindDescriptorSets);
    GD(CmdPushConstants);
    GD(CmdDispatch);
    GD(QueueSubmit);
    GD(QueueWaitIdle);
    GD(CreateFence);
    GD(DestroyFence);
    GD(WaitForFences);
    GD(ResetFences);
    GD(ResetCommandBuffer);
    return 0;
}

int vulkan_try_init(VulkanCtx* ctx, int device_id, char* err, size_t errlen)
{
    if (!ctx) return -1;
    memset(&g_api, 0, sizeof(g_api));
    if (open_lib() != 0) {
        if (err && errlen) snprintf(err, errlen, "vulkan loader not found");
        return -1;
    }
    g_CreateInstance = (PFN_vkCreateInstance)g_api.GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!g_CreateInstance) {
        if (err && errlen) snprintf(err, errlen, "vkCreateInstance missing");
        return -1;
    }

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "yllm";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = g_CreateInstance(&ici, NULL, &inst);
    if (r != VK_SUCCESS) {
        if (err && errlen) snprintf(err, errlen, "vkCreateInstance failed (%d)", (int)r);
        return -1;
    }

    g_EnumeratePhysicalDevices = GI(EnumeratePhysicalDevices);
    g_GetPhysicalDeviceProperties = GI(GetPhysicalDeviceProperties);
    g_GetPhysicalDeviceQueueFamilyProperties = GI(GetPhysicalDeviceQueueFamilyProperties);
    g_api.CreateDevice = GI(CreateDevice);
    g_api.DestroyInstance = GI(DestroyInstance);
    if (!g_EnumeratePhysicalDevices || !g_api.CreateDevice) {
        if (g_api.DestroyInstance) g_api.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "instance fns missing");
        return -1;
    }

    uint32_t ndev = 0;
    g_EnumeratePhysicalDevices(inst, &ndev, NULL);
    if (ndev == 0) {
        g_api.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "no Vulkan physical devices");
        return -1;
    }
    VkPhysicalDevice* phys = (VkPhysicalDevice*)calloc(ndev, sizeof(VkPhysicalDevice));
    if (!phys) { g_api.DestroyInstance(inst, NULL); return -1; }
    g_EnumeratePhysicalDevices(inst, &ndev, phys);
    int idx = device_id;
    if (idx < 0 || (uint32_t)idx >= ndev) idx = 0;
    VkPhysicalDevice pd = phys[idx];

    VkPhysicalDeviceProperties props;
    g_GetPhysicalDeviceProperties(pd, &props);
    ylog_info("vulkan: physical[%d/%u] %s api=%u.%u",
              idx, ndev, props.deviceName,
              VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion));

    uint32_t nq = 0;
    g_GetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties* qfp = (VkQueueFamilyProperties*)calloc(nq, sizeof(VkQueueFamilyProperties));
    if (!qfp) {
        free(phys);
        g_api.DestroyInstance(inst, NULL);
        return -1;
    }
    g_GetPhysicalDeviceQueueFamilyProperties(pd, &nq, qfp);
    uint32_t qfam = (uint32_t)~0u, i;
    for (i = 0; i < nq; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }
    free(qfp);
    free(phys);
    if (qfam == (uint32_t)~0u) {
        g_api.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "no compute queue");
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
    r = g_api.CreateDevice(pd, &dci, NULL, &dev);
    if (r != VK_SUCCESS) {
        g_api.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "vkCreateDevice failed (%d)", (int)r);
        return -1;
    }
    if (load_device_fns(inst, dev) != 0) {
        g_api.DestroyDevice(dev, NULL);
        g_api.DestroyInstance(inst, NULL);
        if (err && errlen) snprintf(err, errlen, "load device fns failed");
        return -1;
    }

    VkQueue queue = VK_NULL_HANDLE;
    g_api.GetDeviceQueue(dev, qfam, 0, &queue);

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
    if (!ctx || ctx->host_shim) return;
    VulkanApi* a = &g_api;
    VkDevice dev = (VkDevice)ctx->device;
    VkInstance inst = (VkInstance)ctx->instance;
    if (!dev) return;

    if (ctx->pipeline && a->DestroyPipeline)
        a->DestroyPipeline(dev, (VkPipeline)ctx->pipeline, NULL);
    if (ctx->pipe_layout && a->DestroyPipelineLayout)
        a->DestroyPipelineLayout(dev, (VkPipelineLayout)ctx->pipe_layout, NULL);
    if (ctx->shader && a->DestroyShaderModule)
        a->DestroyShaderModule(dev, (VkShaderModule)ctx->shader, NULL);
    if (ctx->desc_pool && a->DestroyDescriptorPool)
        a->DestroyDescriptorPool(dev, (VkDescriptorPool)ctx->desc_pool, NULL);
    if (ctx->desc_layout && a->DestroyDescriptorSetLayout)
        a->DestroyDescriptorSetLayout(dev, (VkDescriptorSetLayout)ctx->desc_layout, NULL);
    if (ctx->cmd_pool && a->DestroyCommandPool)
        a->DestroyCommandPool(dev, (VkCommandPool)ctx->cmd_pool, NULL);
    if (ctx->fence && a->DestroyFence)
        a->DestroyFence(dev, (VkFence)ctx->fence, NULL);

    if (ctx->buf_x && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_x, NULL);
    if (ctx->buf_y && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_y, NULL);
    if (ctx->buf_w && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_w, NULL);
    if (ctx->mem_x && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_x, NULL);
    if (ctx->mem_y && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_y, NULL);
    if (ctx->mem_w && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_w, NULL);
    free(ctx->host_w);

    if (a->DestroyDevice) a->DestroyDevice(dev, NULL);
    if (inst && a->DestroyInstance) a->DestroyInstance(inst, NULL);
    memset(ctx, 0, sizeof(*ctx));
    ctx->host_shim = 1;
}
