/* vulkan_compute.c — RMSNorm SPIR-V dispatch */
#include "vulkan_compute.h"
#include "vulkan_api.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

typedef struct {
    uint32_t n;
    float eps;
} RmsPush;

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t type_bits,
                                 VkMemoryPropertyFlags props)
{
    VulkanApi* a = vulkan_api();
    VkPhysicalDeviceMemoryProperties mp;
    a->GetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t i;
    for (i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return (uint32_t)~0u;
}

static int create_host_buffer(VulkanCtx* ctx, size_t bytes,
                              VkBuffer* out_buf, VkDeviceMemory* out_mem)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    VkPhysicalDevice pd = (VkPhysicalDevice)ctx->phys;

    VkBufferCreateInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (a->CreateBuffer(dev, &bi, NULL, out_buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    a->GetBufferMemoryRequirements(dev, *out_buf, &req);
    uint32_t mi = find_memory_type(pd, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mi == (uint32_t)~0u) return -1;

    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mi;
    if (a->AllocateMemory(dev, &ai, NULL, out_mem) != VK_SUCCESS) return -1;
    if (a->BindBufferMemory(dev, *out_buf, *out_mem, 0) != VK_SUCCESS) return -1;
    return 0;
}

static uint32_t* read_spv(const char* path, size_t* out_nwords)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 4 || (sz % 4) != 0) { fclose(f); return NULL; }
    uint32_t* code = (uint32_t*)malloc((size_t)sz);
    if (!code) { fclose(f); return NULL; }
    if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
        free(code);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_nwords = (size_t)sz / 4;
    return code;
}

static const char* find_spv(const char* override_path, char* buf, size_t bufn)
{
    const char* cands[] = {
        override_path,
        "inference/shaders/rmsnorm.spv",
        "shaders/rmsnorm.spv",
        "../inference/shaders/rmsnorm.spv",
        NULL
    };
    int i;
    for (i = 0; cands[i]; i++) {
        if (!cands[i] || !cands[i][0]) continue;
        FILE* f = fopen(cands[i], "rb");
        if (f) {
            fclose(f);
            snprintf(buf, bufn, "%s", cands[i]);
            return buf;
        }
    }
    const char* env = getenv("YLLM_SHADER_DIR");
    if (env && env[0]) {
        snprintf(buf, bufn, "%s/rmsnorm.spv", env);
        FILE* f = fopen(buf, "rb");
        if (f) { fclose(f); return buf; }
    }
    return NULL;
}

int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden, const char* spv_path,
                         char* err, size_t errlen)
{
    if (!ctx || ctx->host_shim || !ctx->device) {
        if (err && errlen) snprintf(err, errlen, "no native device");
        return -1;
    }
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t bytes = (size_t)hidden * 4;
    if (bytes < 64) bytes = 64;
    ctx->buf_bytes = bytes;
    ctx->hidden = hidden;

    char pathbuf[512];
    const char* path = find_spv(spv_path, pathbuf, sizeof(pathbuf));
    if (!path) {
        if (err && errlen) snprintf(err, errlen, "rmsnorm.spv not found (set YLLM_SHADER_DIR)");
        return -1;
    }
    size_t nwords = 0;
    uint32_t* code = read_spv(path, &nwords);
    if (!code) {
        if (err && errlen) snprintf(err, errlen, "read spv failed: %s", path);
        return -1;
    }

    VkShaderModuleCreateInfo smi = {0};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = nwords * 4;
    smi.pCode = code;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult r = a->CreateShaderModule(dev, &smi, NULL, &shader);
    free(code);
    if (r != VK_SUCCESS) {
        if (err && errlen) snprintf(err, errlen, "CreateShaderModule %d", (int)r);
        return -1;
    }
    ctx->shader = (void*)shader;

    VkDescriptorSetLayoutBinding binds[3];
    memset(binds, 0, sizeof(binds));
    uint32_t b;
    for (b = 0; b < 3; b++) {
        binds[b].binding = b;
        binds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[b].descriptorCount = 1;
        binds[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli = {0};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3;
    dli.pBindings = binds;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    if (a->CreateDescriptorSetLayout(dev, &dli, NULL, &dsl) != VK_SUCCESS) return -1;
    ctx->desc_layout = (void*)dsl;

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(RmsPush);

    VkPipelineLayoutCreateInfo pli = {0};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    if (a->CreatePipelineLayout(dev, &pli, NULL, &pl) != VK_SUCCESS) return -1;
    ctx->pipe_layout = (void*)pl;

    VkComputePipelineCreateInfo cpi = {0};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = shader;
    cpi.stage.pName = "main";
    cpi.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (a->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe) != VK_SUCCESS)
        return -1;
    ctx->pipeline = (void*)pipe;

    VkDescriptorPoolSize dps = {0};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpi = {0};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &dps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (a->CreateDescriptorPool(dev, &dpi, NULL, &pool) != VK_SUCCESS) return -1;
    ctx->desc_pool = (void*)pool;

    VkDescriptorSetAllocateInfo dai = {0};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (a->AllocateDescriptorSets(dev, &dai, &dset) != VK_SUCCESS) return -1;
    ctx->desc_set = (void*)dset;

    VkBuffer bx, by, bw;
    VkDeviceMemory mx, my, mw;
    if (create_host_buffer(ctx, bytes, &bx, &mx) != 0) return -1;
    if (create_host_buffer(ctx, bytes, &by, &my) != 0) return -1;
    if (create_host_buffer(ctx, bytes, &bw, &mw) != 0) return -1;
    ctx->buf_x = (void*)bx; ctx->mem_x = (void*)mx;
    ctx->buf_y = (void*)by; ctx->mem_y = (void*)my;
    ctx->buf_w = (void*)bw; ctx->mem_w = (void*)mw;

    VkDescriptorBufferInfo bis[3];
    VkWriteDescriptorSet writes[3];
    memset(bis, 0, sizeof(bis));
    memset(writes, 0, sizeof(writes));
    VkBuffer bufs[3] = { bx, by, bw };
    for (b = 0; b < 3; b++) {
        bis[b].buffer = bufs[b];
        bis[b].offset = 0;
        bis[b].range = bytes;
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = dset;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &bis[b];
    }
    a->UpdateDescriptorSets(dev, 3, writes, 0, NULL);

    VkCommandPoolCreateInfo cpool = {0};
    cpool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpool.queueFamilyIndex = ctx->queue_family;
    VkCommandPool cmdpool = VK_NULL_HANDLE;
    if (a->CreateCommandPool(dev, &cpool, NULL, &cmdpool) != VK_SUCCESS) return -1;
    ctx->cmd_pool = (void*)cmdpool;

    VkCommandBufferAllocateInfo cai = {0};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = cmdpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (a->AllocateCommandBuffers(dev, &cai, &cmd) != VK_SUCCESS) return -1;
    ctx->cmd = (void*)cmd;

    VkFenceCreateInfo fi = {0};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (a->CreateFence(dev, &fi, NULL, &fence) != VK_SUCCESS) return -1;
    ctx->fence = (void*)fence;

    ctx->host_w = (float*)malloc(bytes);
    if (!ctx->host_w) return -1;

    ctx->compute_ready = 1;
    ylog_info("vulkan: rmsnorm compute ready (spv=%s hidden=%u)", path, hidden);
    return 0;
}

int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps)
{
    if (!ctx || !ctx->compute_ready || !y || !x || !w || n == 0) return -1;
    if ((size_t)n * 4 > ctx->buf_bytes) return -1;
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t nbytes = (size_t)n * 4;

    void* px = NULL; void* py = NULL; void* pw = NULL;
    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_x, 0, nbytes, 0, &px) != VK_SUCCESS)
        return -1;
    memcpy(px, x, nbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_x);

    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_w, 0, nbytes, 0, &pw) != VK_SUCCESS)
        return -1;
    memcpy(pw, w, nbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_w);

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
    a->EndCommandBuffer(cmd);

    VkFence fence = (VkFence)ctx->fence;
    a->ResetFences(dev, 1, &fence);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (a->QueueSubmit((VkQueue)ctx->queue, 1, &si, fence) != VK_SUCCESS) return -1;
    if (a->WaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull) != VK_SUCCESS)
        return -1;

    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_y, 0, nbytes, 0, &py) != VK_SUCCESS)
        return -1;
    memcpy(y, py, nbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_y);
    return 0;
}
