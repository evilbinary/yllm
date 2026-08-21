/* vulkan_compute.c — RMSNorm + Q4_K gemv SPIR-V dispatch */
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

typedef struct {
    uint32_t out_n;
    uint32_t in_n;
    uint32_t w_off;
    uint32_t _pad;
} GemvPush;

typedef struct {
    uint32_t n;
} SwiPush;

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
    if (bytes < 16) bytes = 16;

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

static const char* find_spv_named(const char* name, char* buf, size_t bufn)
{
    const char* prefixes[] = {
        "inference/shaders/",
        "shaders/",
        "../inference/shaders/",
        NULL
    };
    int i;
    for (i = 0; prefixes[i]; i++) {
        snprintf(buf, bufn, "%s%s", prefixes[i], name);
        FILE* f = fopen(buf, "rb");
        if (f) { fclose(f); return buf; }
    }
    const char* env = getenv("YLLM_SHADER_DIR");
    if (env && env[0]) {
        snprintf(buf, bufn, "%s/%s", env, name);
        FILE* f = fopen(buf, "rb");
        if (f) { fclose(f); return buf; }
    }
    return NULL;
}

static int create_ssbo_pipeline(VulkanCtx* ctx, const char* spv_name,
                                size_t push_size,
                                VkShaderModule* out_shader,
                                VkDescriptorSetLayout* out_dsl,
                                VkPipelineLayout* out_pl,
                                VkPipeline* out_pipe,
                                VkDescriptorPool* out_pool,
                                VkDescriptorSet* out_dset,
                                VkBuffer bufs[3], size_t ranges[3],
                                char* err, size_t errlen)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    char pathbuf[512];
    const char* path = find_spv_named(spv_name, pathbuf, sizeof(pathbuf));
    if (!path) {
        if (err && errlen) snprintf(err, errlen, "%s not found", spv_name);
        return -1;
    }
    size_t nwords = 0;
    uint32_t* code = read_spv(path, &nwords);
    if (!code) {
        if (err && errlen) snprintf(err, errlen, "read %s failed", path);
        return -1;
    }
    VkShaderModuleCreateInfo smi = {0};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = nwords * 4;
    smi.pCode = code;
    VkResult r = a->CreateShaderModule(dev, &smi, NULL, out_shader);
    free(code);
    if (r != VK_SUCCESS) {
        if (err && errlen) snprintf(err, errlen, "CreateShaderModule %s %d", spv_name, (int)r);
        return -1;
    }

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
    if (a->CreateDescriptorSetLayout(dev, &dli, NULL, out_dsl) != VK_SUCCESS) return -1;

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = (uint32_t)push_size;
    VkPipelineLayoutCreateInfo pli = {0};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = out_dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (a->CreatePipelineLayout(dev, &pli, NULL, out_pl) != VK_SUCCESS) return -1;

    VkComputePipelineCreateInfo cpi = {0};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = *out_shader;
    cpi.stage.pName = "main";
    cpi.layout = *out_pl;
    if (a->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, out_pipe) != VK_SUCCESS)
        return -1;

    VkDescriptorPoolSize dps = {0};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpi = {0};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &dps;
    if (a->CreateDescriptorPool(dev, &dpi, NULL, out_pool) != VK_SUCCESS) return -1;

    VkDescriptorSetAllocateInfo dai = {0};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = *out_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = out_dsl;
    if (a->AllocateDescriptorSets(dev, &dai, out_dset) != VK_SUCCESS) return -1;

    VkDescriptorBufferInfo bis[3];
    VkWriteDescriptorSet writes[3];
    memset(bis, 0, sizeof(bis));
    memset(writes, 0, sizeof(writes));
    for (b = 0; b < 3; b++) {
        bis[b].buffer = bufs[b];
        bis[b].offset = 0;
        bis[b].range = ranges[b];
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = *out_dset;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &bis[b];
    }
    a->UpdateDescriptorSets(dev, 3, writes, 0, NULL);
    ylog_info("vulkan: pipeline %s ready (%s)", spv_name, path);
    return 0;
}

static int submit_and_wait(VulkanCtx* ctx, VkCommandBuffer cmd)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    VkFence fence = (VkFence)ctx->fence;
    a->ResetFences(dev, 1, &fence);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (a->QueueSubmit((VkQueue)ctx->queue, 1, &si, fence) != VK_SUCCESS) return -1;
    if (a->WaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull) != VK_SUCCESS)
        return -1;
    return 0;
}

int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden,
                         uint32_t max_in, uint32_t max_out,
                         size_t total_wq_bytes, uint32_t n_layers, uint32_t nslot,
                         char* err, size_t errlen)
{
    if (!ctx || ctx->host_shim || !ctx->device) {
        if (err && errlen) snprintf(err, errlen, "no native device");
        return -1;
    }
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;

    if (max_in < hidden) max_in = hidden;
    if (max_out < hidden) max_out = hidden;
    ctx->hidden = hidden;
    ctx->max_in = max_in;
    ctx->max_out = max_out;
    ctx->n_layers = n_layers;
    ctx->wq_nslot = nslot ? nslot : 24;
    ctx->x_bytes = (size_t)max_in * 4;
    ctx->y_bytes = (size_t)max_out * 4;
    ctx->wn_bytes = (size_t)hidden * 4;
    /* 至少留 8 行自检空间 */
    if (total_wq_bytes < 144 * 8) total_wq_bytes = 144 * 8;
    /* 4 字节对齐(uint SSBO) */
    total_wq_bytes = (total_wq_bytes + 3u) & ~(size_t)3u;
    ctx->wq_bytes = total_wq_bytes;
    ctx->wq_resident = 0;

    size_t ntab = (size_t)n_layers * ctx->wq_nslot;
    ctx->wq_off = (uint64_t*)malloc(ntab * sizeof(uint64_t));
    if (!ctx->wq_off) return -1;
    {
        size_t i;
        for (i = 0; i < ntab; i++) ctx->wq_off[i] = (uint64_t)~0ull;
    }

    VkBuffer bx = VK_NULL_HANDLE, by = VK_NULL_HANDLE;
    VkDeviceMemory mx = VK_NULL_HANDLE, my = VK_NULL_HANDLE;
    if (create_host_buffer(ctx, ctx->x_bytes, &bx, &mx) != 0) return -1;
    if (create_host_buffer(ctx, ctx->y_bytes, &by, &my) != 0) return -1;
    ctx->buf_x = (void*)bx; ctx->mem_x = (void*)mx;
    ctx->buf_y = (void*)by; ctx->mem_y = (void*)my;

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

    {
        VkBuffer bw = VK_NULL_HANDLE;
        VkDeviceMemory mw = VK_NULL_HANDLE;
        if (create_host_buffer(ctx, ctx->wn_bytes, &bw, &mw) != 0) return -1;
        ctx->buf_wn = (void*)bw; ctx->mem_wn = (void*)mw;
        VkBuffer bufs[3] = { bx, by, bw };
        size_t ranges[3] = { ctx->x_bytes, ctx->y_bytes, ctx->wn_bytes };
        VkShaderModule sh = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        if (create_ssbo_pipeline(ctx, "rmsnorm.spv", sizeof(RmsPush),
                                 &sh, &dsl, &pl, &pipe, &pool, &dset,
                                 bufs, ranges, err, errlen) != 0)
            return -1;
        ctx->rms_shader = (void*)sh;
        ctx->rms_desc_layout = (void*)dsl;
        ctx->rms_pipe_layout = (void*)pl;
        ctx->rms_pipeline = (void*)pipe;
        ctx->rms_desc_pool = (void*)pool;
        ctx->rms_desc_set = (void*)dset;
        ctx->host_w = (float*)malloc(ctx->wn_bytes);
        if (!ctx->host_w) return -1;
        ctx->compute_ready = 1;
    }

    {
        VkBuffer bw = VK_NULL_HANDLE;
        VkDeviceMemory mw = VK_NULL_HANDLE;
        char gerr[256];
        VkBuffer bo0 = VK_NULL_HANDLE, bo1 = VK_NULL_HANDLE, bo2 = VK_NULL_HANDLE;
        VkDeviceMemory mo0 = VK_NULL_HANDLE, mo1 = VK_NULL_HANDLE, mo2 = VK_NULL_HANDLE;
        if (create_host_buffer(ctx, ctx->wq_bytes, &bw, &mw) != 0 ||
            create_host_buffer(ctx, ctx->y_bytes, &bo0, &mo0) != 0 ||
            create_host_buffer(ctx, ctx->y_bytes, &bo1, &mo1) != 0 ||
            create_host_buffer(ctx, ctx->y_bytes, &bo2, &mo2) != 0) {
            ylog_warn("vulkan: gemv buffers alloc failed");
        } else {
            ctx->buf_wq = (void*)bw; ctx->mem_wq = (void*)mw;
            ctx->buf_o0 = (void*)bo0; ctx->mem_o0 = (void*)mo0;
            ctx->buf_o1 = (void*)bo1; ctx->mem_o1 = (void*)mo1;
            ctx->buf_o2 = (void*)bo2; ctx->mem_o2 = (void*)mo2;
            VkBuffer bufs[3] = { bx, by, bw };
            size_t ranges[3] = { ctx->x_bytes, ctx->y_bytes, ctx->wq_bytes };
            VkShaderModule sh = VK_NULL_HANDLE;
            VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
            VkPipelineLayout pl = VK_NULL_HANDLE;
            VkPipeline pipe = VK_NULL_HANDLE;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VkDescriptorSet dset = VK_NULL_HANDLE;
            if (create_ssbo_pipeline(ctx, "gemv_q4k.spv", sizeof(GemvPush),
                                     &sh, &dsl, &pl, &pipe, &pool, &dset,
                                     bufs, ranges, gerr, sizeof(gerr)) != 0) {
                ylog_warn("vulkan: gemv pipeline failed (%s)", gerr);
            } else {
                /* 扩容 pool: 再建 3 个 set (y→o0/o1/o2, x=buf_y) */
                a->DestroyDescriptorPool(dev, pool, NULL);
                VkDescriptorPoolSize dps = {0};
                dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                dps.descriptorCount = 12;
                VkDescriptorPoolCreateInfo dpi = {0};
                dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                dpi.maxSets = 4;
                dpi.poolSizeCount = 1;
                dpi.pPoolSizes = &dps;
                if (a->CreateDescriptorPool(dev, &dpi, NULL, &pool) != VK_SUCCESS) {
                    ylog_warn("vulkan: gemv desc pool recreate failed");
                } else {
                    VkDescriptorSetLayout layouts[4] = { dsl, dsl, dsl, dsl };
                    VkDescriptorSet sets[4];
                    VkDescriptorSetAllocateInfo dai = {0};
                    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    dai.descriptorPool = pool;
                    dai.descriptorSetCount = 4;
                    dai.pSetLayouts = layouts;
                    if (a->AllocateDescriptorSets(dev, &dai, sets) != VK_SUCCESS) {
                        ylog_warn("vulkan: gemv desc sets alloc failed");
                    } else {
                        VkBuffer xb[4] = { bx, by, by, by };
                        VkBuffer yb[4] = { by, bo0, bo1, bo2 };
                        uint32_t si;
                        for (si = 0; si < 4; si++) {
                            VkDescriptorBufferInfo bis[3];
                            VkWriteDescriptorSet writes[3];
                            memset(bis, 0, sizeof(bis));
                            memset(writes, 0, sizeof(writes));
                            VkBuffer trip[3] = { xb[si], yb[si], bw };
                            size_t rng[3] = { ctx->x_bytes, ctx->y_bytes, ctx->wq_bytes };
                            uint32_t b;
                            for (b = 0; b < 3; b++) {
                                bis[b].buffer = trip[b];
                                bis[b].range = rng[b];
                                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                writes[b].dstSet = sets[si];
                                writes[b].dstBinding = b;
                                writes[b].descriptorCount = 1;
                                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                writes[b].pBufferInfo = &bis[b];
                            }
                            a->UpdateDescriptorSets(dev, 3, writes, 0, NULL);
                        }
                        ctx->gemv_shader = (void*)sh;
                        ctx->gemv_desc_layout = (void*)dsl;
                        ctx->gemv_pipe_layout = (void*)pl;
                        ctx->gemv_pipeline = (void*)pipe;
                        ctx->gemv_desc_pool = (void*)pool;
                        ctx->gemv_desc_set = (void*)sets[0];
                        ctx->gemv_ds0 = (void*)sets[1];
                        ctx->gemv_ds1 = (void*)sets[2];
                        ctx->gemv_ds2 = (void*)sets[3];
                        ctx->gemv_ready = 1;
                        ctx->fuse_ready = 1;
                        ylog_info("vulkan: gemv_q4k+fuse ready max_in=%u max_out=%u wq=%zuMB",
                                  max_in, max_out, ctx->wq_bytes / (1024 * 1024));
                    }
                }
            }
        }
    }

    /* SwiGLU: gate=o0 up=o1 out=buf_y */
    if (ctx->fuse_ready && ctx->buf_o0 && ctx->buf_o1 && ctx->buf_y) {
        char serr[256];
        VkBuffer bufs[3] = {
            (VkBuffer)ctx->buf_o0, (VkBuffer)ctx->buf_o1, (VkBuffer)ctx->buf_y
        };
        size_t ranges[3] = { ctx->y_bytes, ctx->y_bytes, ctx->y_bytes };
        VkShaderModule sh = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        if (create_ssbo_pipeline(ctx, "swiglu.spv", sizeof(SwiPush),
                                 &sh, &dsl, &pl, &pipe, &pool, &dset,
                                 bufs, ranges, serr, sizeof(serr)) != 0) {
            ylog_warn("vulkan: swiglu pipeline failed (%s)", serr);
        } else {
            ctx->swi_shader = (void*)sh;
            ctx->swi_desc_layout = (void*)dsl;
            ctx->swi_pipe_layout = (void*)pl;
            ctx->swi_pipeline = (void*)pipe;
            ctx->swi_desc_pool = (void*)pool;
            ctx->swi_desc_set = (void*)dset;
            ctx->swi_ready = 1;
            ylog_info("vulkan: swiglu ready");
        }
    }
    return 0;
}

int vulkan_wq_upload(VulkanCtx* ctx, const void* blob, size_t bytes)
{
    if (!ctx || !ctx->gemv_ready || !blob || bytes == 0) return -1;
    if (bytes > ctx->wq_bytes) return -1;
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    void* pw = NULL;
    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wq, 0, bytes, 0, &pw) != VK_SUCCESS)
        return -1;
    memcpy(pw, blob, bytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq);
    ctx->wq_resident = 1;
    ylog_info("vulkan: Q4_K weights resident %zuMB", bytes / (1024 * 1024));
    return 0;
}

int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps)
{
    if (!ctx || !ctx->compute_ready || !y || !x || !w || n == 0) return -1;
    if ((size_t)n * 4 > ctx->x_bytes || (size_t)n * 4 > ctx->y_bytes ||
        (size_t)n * 4 > ctx->wn_bytes)
        return -1;
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t nbytes = (size_t)n * 4;
    void* px = NULL; void* py = NULL; void* pw = NULL;

    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_x, 0, nbytes, 0, &px) != VK_SUCCESS)
        return -1;
    memcpy(px, x, nbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_x);
    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wn, 0, nbytes, 0, &pw) != VK_SUCCESS)
        return -1;
    memcpy(pw, w, nbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wn);

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rms_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->rms_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rms_pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rms_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_y, 0, nbytes, 0, &py) != VK_SUCCESS)
        return -1;
    memcpy(y, py, nbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_y);
    return 0;
}

int vulkan_k_gemv_q4k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off)
{
    if (!ctx || !ctx->gemv_ready || !y || !x) return -1;
    if (out == 0 || in == 0 || (in % 256) != 0) return -1;
    if (in > ctx->max_in || out > ctx->max_out) return -1;
    size_t wbytes = (size_t)out * ((size_t)(in / 256) * 144);
    if (w_byte_off + wbytes > ctx->wq_bytes) return -1;
    if (w_byte_off > 0xffffffffull) return -1;

    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t xbytes = (size_t)in * 4;
    size_t ybytes = (size_t)out * 4;
    void* px = NULL; void* py = NULL;

    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_x, 0, xbytes, 0, &px) != VK_SUCCESS)
        return -1;
    memcpy(px, x, xbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_x);

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->gemv_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->gemv_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->gemv_pipe_layout, 0, 1, &ds, 0, NULL);
    GemvPush push;
    push.out_n = out;
    push.in_n = in;
    push.w_off = (uint32_t)w_byte_off;
    push._pad = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, out, 1, 1);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_y, 0, ybytes, 0, &py) != VK_SUCCESS)
        return -1;
    memcpy(y, py, ybytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_y);
    return 0;
}

int vulkan_k_gemv_q4k_host(VulkanCtx* ctx, float* y, const float* x,
                           const uint8_t* w, uint32_t out, uint32_t in)
{
    if (!ctx || !w) return -1;
    size_t wbytes = (size_t)out * ((size_t)(in / 256) * 144);
    if (wbytes > ctx->wq_bytes) return -1;
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    void* pw = NULL;
    if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wq, 0, wbytes, 0, &pw) != VK_SUCCESS)
        return -1;
    memcpy(pw, w, wbytes);
    a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq);
    return vulkan_k_gemv_q4k(ctx, y, x, out, in, 0);
}

static void cmd_barrier_compute(VkCommandBuffer cmd)
{
    VulkanApi* a = vulkan_api();
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    a->CmdPipelineBarrier(cmd,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &mb, 0, NULL, 0, NULL);
}

static void cmd_rmsnorm(VulkanCtx* ctx, VkCommandBuffer cmd, uint32_t n, float eps)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rms_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->rms_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rms_pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rms_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
}

static void cmd_gemv(VulkanCtx* ctx, VkCommandBuffer cmd, void* dset,
                     uint32_t out, uint32_t in, uint32_t w_off)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->gemv_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->gemv_pipe_layout, 0, 1, &ds, 0, NULL);
    GemvPush push;
    push.out_n = out;
    push.in_n = in;
    push.w_off = w_off;
    push._pad = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, out, 1, 1);
}

static int map_copy(VulkanCtx* ctx, void* mem, const void* src, size_t nbytes)
{
    VulkanApi* a = vulkan_api();
    void* p = NULL;
    if (a->MapMemory((VkDevice)ctx->device, (VkDeviceMemory)mem, 0, nbytes, 0, &p) != VK_SUCCESS)
        return -1;
    memcpy(p, src, nbytes);
    a->UnmapMemory((VkDevice)ctx->device, (VkDeviceMemory)mem);
    return 0;
}

static int map_read(VulkanCtx* ctx, void* mem, void* dst, size_t nbytes)
{
    VulkanApi* a = vulkan_api();
    void* p = NULL;
    if (a->MapMemory((VkDevice)ctx->device, (VkDeviceMemory)mem, 0, nbytes, 0, &p) != VK_SUCCESS)
        return -1;
    memcpy(dst, p, nbytes);
    a->UnmapMemory((VkDevice)ctx->device, (VkDeviceMemory)mem);
    return 0;
}

int vulkan_fused_norm_qkv(VulkanCtx* ctx,
                          const float* x, const float* wn, uint32_t hidden, float eps,
                          float* q, float* k, float* v, uint32_t kv_dim,
                          uint64_t off_q, uint64_t off_k, uint64_t off_v)
{
    if (!ctx || !ctx->fuse_ready || !ctx->wq_resident || !x || !wn || !q || !k || !v)
        return -1;
    if (hidden > ctx->max_in || hidden > ctx->max_out || kv_dim > ctx->max_out)
        return -1;
    if (off_q > 0xffffffffull || off_k > 0xffffffffull || off_v > 0xffffffffull)
        return -1;

    size_t xb = (size_t)hidden * 4;
    size_t kb = (size_t)kv_dim * 4;
    if (map_copy(ctx, ctx->mem_x, x, xb) != 0) return -1;
    if (map_copy(ctx, ctx->mem_wn, wn, xb) != 0) return -1;

    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    cmd_rmsnorm(ctx, cmd, hidden, eps);
    cmd_barrier_compute(cmd);
    /* Q/K/V 只读 buf_y, 写不同 o* — 可并行 dispatch */
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, hidden, hidden, (uint32_t)off_q);
    cmd_gemv(ctx, cmd, ctx->gemv_ds1, kv_dim, hidden, (uint32_t)off_k);
    cmd_gemv(ctx, cmd, ctx->gemv_ds2, kv_dim, hidden, (uint32_t)off_v);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (map_read(ctx, ctx->mem_o0, q, xb) != 0) return -1;
    if (map_read(ctx, ctx->mem_o1, k, kb) != 0) return -1;
    if (map_read(ctx, ctx->mem_o2, v, kb) != 0) return -1;
    return 0;
}

int vulkan_fused_norm_gate_up(VulkanCtx* ctx,
                              const float* x, const float* wn, uint32_t hidden, float eps,
                              float* gate, float* up, uint32_t inter,
                              uint64_t off_gate, uint64_t off_up)
{
    if (!ctx || !ctx->fuse_ready || !ctx->wq_resident || !x || !wn || !gate || !up)
        return -1;
    if (hidden > ctx->max_in || inter > ctx->max_out) return -1;
    if (off_gate > 0xffffffffull || off_up > 0xffffffffull) return -1;

    size_t xb = (size_t)hidden * 4;
    size_t ib = (size_t)inter * 4;
    if (map_copy(ctx, ctx->mem_x, x, xb) != 0) return -1;
    if (map_copy(ctx, ctx->mem_wn, wn, xb) != 0) return -1;

    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    cmd_rmsnorm(ctx, cmd, hidden, eps);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, inter, hidden, (uint32_t)off_gate);
    cmd_gemv(ctx, cmd, ctx->gemv_ds1, inter, hidden, (uint32_t)off_up);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (map_read(ctx, ctx->mem_o0, gate, ib) != 0) return -1;
    if (map_read(ctx, ctx->mem_o1, up, ib) != 0) return -1;
    return 0;
}

static void cmd_swiglu(VulkanCtx* ctx, VkCommandBuffer cmd, uint32_t n)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->swi_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->swi_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->swi_pipe_layout, 0, 1, &ds, 0, NULL);
    SwiPush push;
    push.n = n;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->swi_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, (n + 255u) / 256u, 1, 1);
}

int vulkan_fused_ffn(VulkanCtx* ctx,
                     const float* x, const float* wn, uint32_t hidden, float eps,
                     float* out, uint32_t inter,
                     uint64_t off_gate, uint64_t off_up, uint64_t off_down)
{
    if (!ctx || !ctx->fuse_ready || !ctx->swi_ready || !ctx->wq_resident ||
        !x || !wn || !out)
        return -1;
    if (hidden > ctx->max_in || inter > ctx->max_out || hidden > ctx->max_out)
        return -1;
    if (off_gate > 0xffffffffull || off_up > 0xffffffffull || off_down > 0xffffffffull)
        return -1;

    size_t xb = (size_t)hidden * 4;
    if (map_copy(ctx, ctx->mem_x, x, xb) != 0) return -1;
    if (map_copy(ctx, ctx->mem_wn, wn, xb) != 0) return -1;

    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);

    cmd_rmsnorm(ctx, cmd, hidden, eps);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, inter, hidden, (uint32_t)off_gate);
    cmd_gemv(ctx, cmd, ctx->gemv_ds1, inter, hidden, (uint32_t)off_up);
    cmd_barrier_compute(cmd);
    cmd_swiglu(ctx, cmd, inter);
    cmd_barrier_compute(cmd);
    /* swiglu → buf_y; down: x=buf_y → o0 */
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, hidden, inter, (uint32_t)off_down);

    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;
    return map_read(ctx, ctx->mem_o0, out, xb);
}
