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
    GD(CmdPipelineBarrier);
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
    ctx->integrated_gpu =
        (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 1 : 0;
    ctx->max_ssbo_range = props.limits.maxStorageBufferRange;
    ylog_info("vulkan: physical[%d/%u] %s api=%u.%u type=%u ssbo_max=%zuMB",
              idx, ndev, props.deviceName,
              VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
              (unsigned)props.deviceType,
              (size_t)(ctx->max_ssbo_range / (1024 * 1024)));

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

static void destroy_pipe(VulkanApi* a, VkDevice dev,
                         void* pipe, void* layout, void* shader,
                         void* pool, void* dsl)
{
    if (pipe && a->DestroyPipeline) a->DestroyPipeline(dev, (VkPipeline)pipe, NULL);
    if (layout && a->DestroyPipelineLayout)
        a->DestroyPipelineLayout(dev, (VkPipelineLayout)layout, NULL);
    if (shader && a->DestroyShaderModule)
        a->DestroyShaderModule(dev, (VkShaderModule)shader, NULL);
    if (pool && a->DestroyDescriptorPool)
        a->DestroyDescriptorPool(dev, (VkDescriptorPool)pool, NULL);
    if (dsl && a->DestroyDescriptorSetLayout)
        a->DestroyDescriptorSetLayout(dev, (VkDescriptorSetLayout)dsl, NULL);
}

void vulkan_shutdown(VulkanCtx* ctx)
{
    if (!ctx || ctx->host_shim) return;
    VulkanApi* a = &g_api;
    VkDevice dev = (VkDevice)ctx->device;
    VkInstance inst = (VkInstance)ctx->instance;
    if (!dev) return;

    if (ctx->queue && a->QueueWaitIdle)
        a->QueueWaitIdle((VkQueue)ctx->queue);

    if (a->UnmapMemory) {
        VkDevice dev = (VkDevice)ctx->device;
        if (ctx->map_x) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_x); ctx->map_x = NULL; }
        if (ctx->map_y) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_y); ctx->map_y = NULL; }
        if (ctx->map_o0) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_o0); ctx->map_o0 = NULL; }
        if (ctx->map_o1) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_o1); ctx->map_o1 = NULL; }
        if (ctx->map_o2) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_o2); ctx->map_o2 = NULL; }
        if (ctx->map_wn) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wn); ctx->map_wn = NULL; }
        if (ctx->map_wq) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq); ctx->map_wq = NULL; }
        if (ctx->map_kv) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_kv); ctx->map_kv = NULL; }
        if (ctx->map_emb) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_emb); ctx->map_emb = NULL; }
        if (ctx->map_logits) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_logits); ctx->map_logits = NULL; }
        if (ctx->map_bias) { a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_bias); ctx->map_bias = NULL; }
    }

    destroy_pipe(a, dev, ctx->rms_pipeline, ctx->rms_pipe_layout, ctx->rms_shader,
                 ctx->rms_desc_pool, ctx->rms_desc_layout);
    destroy_pipe(a, dev, ctx->q8k_pipeline, ctx->q8k_pipe_layout, ctx->q8k_shader,
                 ctx->q8k_desc_pool, ctx->q8k_desc_layout);
    destroy_pipe(a, dev, ctx->gemv_pipeline, ctx->gemv_pipe_layout, ctx->gemv_shader,
                 ctx->gemv_desc_pool, ctx->gemv_desc_layout);
    if (ctx->gemv_q6k_pipeline && a->DestroyPipeline)
        a->DestroyPipeline(dev, (VkPipeline)ctx->gemv_q6k_pipeline, NULL);
    if (ctx->gemv_q6k_shader && a->DestroyShaderModule)
        a->DestroyShaderModule(dev, (VkShaderModule)ctx->gemv_q6k_shader, NULL);
    destroy_pipe(a, dev, ctx->swi_pipeline, ctx->swi_pipe_layout, ctx->swi_shader,
                 ctx->swi_desc_pool, ctx->swi_desc_layout);
    destroy_pipe(a, dev, ctx->attn_pipeline, ctx->attn_pipe_layout, ctx->attn_shader,
                 ctx->attn_desc_pool, ctx->attn_desc_layout);
    destroy_pipe(a, dev, ctx->rope_pipeline, ctx->rope_pipe_layout, ctx->rope_shader,
                 ctx->rope_desc_pool, ctx->rope_desc_layout);
    destroy_pipe(a, dev, ctx->add_pipeline, ctx->add_pipe_layout, ctx->add_shader,
                 ctx->add_desc_pool, ctx->add_desc_layout);
    destroy_pipe(a, dev, ctx->bias_pipeline, ctx->bias_pipe_layout, ctx->bias_shader,
                 ctx->bias_desc_pool, ctx->bias_desc_layout);
    destroy_pipe(a, dev, ctx->skv_pipeline, ctx->skv_pipe_layout, ctx->skv_shader,
                 ctx->skv_desc_pool, ctx->skv_desc_layout);
    destroy_pipe(a, dev, ctx->embed_pipeline, ctx->embed_pipe_layout, ctx->embed_shader,
                 ctx->embed_desc_pool, ctx->embed_desc_layout);
    if (ctx->cmd_pool && a->DestroyCommandPool)
        a->DestroyCommandPool(dev, (VkCommandPool)ctx->cmd_pool, NULL);
    if (ctx->fence && a->DestroyFence)
        a->DestroyFence(dev, (VkFence)ctx->fence, NULL);

    if (ctx->buf_x && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_x, NULL);
    if (ctx->buf_y && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_y, NULL);
    if (ctx->buf_o0 && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_o0, NULL);
    if (ctx->buf_o1 && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_o1, NULL);
    if (ctx->buf_o2 && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_o2, NULL);
    if (ctx->buf_wn && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_wn, NULL);
    if (ctx->buf_wq && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_wq, NULL);
    if (ctx->buf_emb && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_emb, NULL);
    if (ctx->buf_logits && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_logits, NULL);
    if (ctx->buf_bias && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_bias, NULL);
    if (ctx->buf_kv && a->DestroyBuffer) a->DestroyBuffer(dev, (VkBuffer)ctx->buf_kv, NULL);
    if (ctx->mem_x && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_x, NULL);
    if (ctx->mem_y && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_y, NULL);
    if (ctx->mem_o0 && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_o0, NULL);
    if (ctx->mem_o1 && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_o1, NULL);
    if (ctx->mem_o2 && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_o2, NULL);
    if (ctx->mem_wn && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_wn, NULL);
    if (ctx->mem_wq && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_wq, NULL);
    if (ctx->mem_emb && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_emb, NULL);
    if (ctx->mem_logits && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_logits, NULL);
    if (ctx->mem_bias && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_bias, NULL);
    if (ctx->mem_kv && a->FreeMemory) a->FreeMemory(dev, (VkDeviceMemory)ctx->mem_kv, NULL);
    free(ctx->host_w);
    free(ctx->host_w2);
    free(ctx->host_wq);
    free(ctx->wq_off);

    if (a->DestroyDevice) a->DestroyDevice(dev, NULL);
    if (inst && a->DestroyInstance) a->DestroyInstance(inst, NULL);
    memset(ctx, 0, sizeof(*ctx));
    ctx->host_shim = 1;
}
