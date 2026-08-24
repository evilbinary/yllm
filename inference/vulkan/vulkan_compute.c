/* vulkan_compute.c — RMSNorm + Q4_K gemv SPIR-V dispatch */
#include "vulkan_compute.h"
#include "vulkan_api.h"
#include "yllm.h"
#include "llf.h"
#include "matvec.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

typedef struct {
    uint32_t hidden;
} EmbedPush;

typedef struct {
    uint32_t n;
    float eps;
    uint32_t w_off;
    uint32_t _pad;
} RmsPush;

typedef struct {
    uint32_t out_n;
    uint32_t in_n;
    uint32_t w_off;
    uint32_t serial; /* iGPU: 1 */
} GemvPush;

static uint32_t gemv_serial_flag(const VulkanCtx* ctx)
{
    /* 默认并行; YLLM_VK_GEMV_SERIAL=1 才单线程(仅排查 iGPU 归约) */
    (void)ctx;
    const char* e = getenv("YLLM_VK_GEMV_SERIAL");
    return (e && e[0] == '1') ? 1u : 0u;
}

typedef struct {
    uint32_t n;
} Q8kPush;

typedef struct {
    uint32_t n;
} SwiPush;

typedef struct {
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t pos;
    uint32_t mode; /* 0=llama 1=qwen */
    float theta;
    uint32_t _pad; /* 与 GLSL 20→24 对齐, 避免 theta 读脏 */
} RopePush;

typedef struct {
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_dim;
    uint32_t max_seq;
    uint32_t pos;
    uint32_t k_slot;
    uint32_t v_slot;
} AttnPush;

typedef struct {
    uint32_t n;
} AddPush;

typedef struct {
    uint32_t n;
    uint32_t bias_off;
} BiasPush;

typedef struct {
    uint32_t n;
    uint32_t dst_off;
} StoreKvPush;

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

/* HOST_VISIBLE。dGPU 优先 DEVICE_LOCAL(ReBAR)；iGPU 优先 HOST_CACHED，
 * 否则 CPU memcpy 到 WC/uncached UMA 后 GPU 可能读到脏数据。 */
static uint32_t find_ssbo_memory(VkPhysicalDevice pd, uint32_t type_bits, int igpu,
                                 VkMemoryPropertyFlags* out_flags)
{
    VulkanApi* a = vulkan_api();
    VkPhysicalDeviceMemoryProperties mp;
    a->GetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t i, best = (uint32_t)~0u;
    int best_score = -1;
    VkMemoryPropertyFlags best_flags = 0;
    for (i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f;
        int score;
        if ((type_bits & (1u << i)) == 0) continue;
        f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
        score = 0;
        if (igpu) {
            if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += 8;
            if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score += 4;
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += 1;
        } else {
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += 8;
            if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score += 2;
            if ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == 0) score += 1;
        }
        if (score > best_score) {
            best_score = score;
            best = i;
            best_flags = f;
        }
    }
    if (out_flags) *out_flags = best_flags;
    return best;
}

/* 纯显存: 不要 HOST_VISIBLE, 避免 ReBAR 上 compute-compute 不可见 */
static uint32_t find_vram_memory(VkPhysicalDevice pd, uint32_t type_bits,
                                 VkMemoryPropertyFlags* out_flags)
{
    VulkanApi* a = vulkan_api();
    VkPhysicalDeviceMemoryProperties mp;
    a->GetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t i, best = (uint32_t)~0u;
    int best_score = -1;
    VkMemoryPropertyFlags best_flags = 0;
    for (i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f;
        int score;
        if ((type_bits & (1u << i)) == 0) continue;
        f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) continue;
        score = 4;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) score += 8;
        else score -= 4;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score -= 2;
        if (score > best_score) {
            best_score = score;
            best = i;
            best_flags = f;
        }
    }
    if (out_flags) *out_flags = best_flags;
    return best;
}

static int create_host_buffer(VulkanCtx* ctx, size_t bytes,
                              VkBuffer* out_buf, VkDeviceMemory* out_mem,
                              void** persist_map)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    VkPhysicalDevice pd = (VkPhysicalDevice)ctx->phys;
    if (bytes < 16) bytes = 16;

    VkBufferCreateInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (a->CreateBuffer(dev, &bi, NULL, out_buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    VkMemoryPropertyFlags mflags = 0;
    uint32_t mi;
    a->GetBufferMemoryRequirements(dev, *out_buf, &req);
    mi = find_ssbo_memory(pd, req.memoryTypeBits, ctx->integrated_gpu, &mflags);
    if (mi == (uint32_t)~0u)
        mi = find_memory_type(pd, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mi == (uint32_t)~0u) return -1;
    {
        static int logged;
        if (!logged) {
            logged = 1;
            ylog_info("vulkan: SSBO mem type=%u flags=0x%x%s%s", mi, (unsigned)mflags,
                      (mflags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : " sysmem",
                      (mflags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " CACHED" : "");
        }
        if (bytes >= (1u << 20) && (mflags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)
            ylog_warn("vulkan: SSBO %zuMB not DEVICE_LOCAL (type=%u flags=0x%x); token-batch may be wrong",
                      bytes / (1024 * 1024), mi, (unsigned)mflags);
    }

    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mi;
    if (a->AllocateMemory(dev, &ai, NULL, out_mem) != VK_SUCCESS) return -1;
    if (a->BindBufferMemory(dev, *out_buf, *out_mem, 0) != VK_SUCCESS) return -1;
    if (persist_map) {
        void* p = NULL;
        /* 必须覆盖整个 allocation, 否则 Flush/Invalidate VK_WHOLE_SIZE 非法 */
        if (a->MapMemory(dev, *out_mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS)
            return -1;
        *persist_map = p;
    }
    return 0;
}

/* dGPU 激活/KV: 优先不带 HOST_VISIBLE 的 DEVICE_LOCAL. */
static int create_scratch_buffer(VulkanCtx* ctx, size_t bytes,
                                 VkBuffer* out_buf, VkDeviceMemory* out_mem,
                                 void** persist_map)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    VkPhysicalDevice pd = (VkPhysicalDevice)ctx->phys;
    if (ctx->integrated_gpu || getenv("YLLM_VK_MAP_SCRATCH") ||
        !getenv("YLLM_VK_VRAM_SCRATCH"))
        return create_host_buffer(ctx, bytes, out_buf, out_mem, persist_map);
    if (bytes < 16) bytes = 16;
    {
        VkBufferCreateInfo bi = {0};
        VkMemoryRequirements req;
        VkMemoryPropertyFlags mflags = 0;
        uint32_t mi;
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (a->CreateBuffer(dev, &bi, NULL, out_buf) != VK_SUCCESS) return -1;
        a->GetBufferMemoryRequirements(dev, *out_buf, &req);
        mi = find_vram_memory(pd, req.memoryTypeBits, &mflags);
        if (mi == (uint32_t)~0u) {
            a->DestroyBuffer(dev, *out_buf, NULL);
            return create_host_buffer(ctx, bytes, out_buf, out_mem, persist_map);
        }
        {
            static int logged;
            VkMemoryAllocateInfo ai = {0};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mi;
            if (a->AllocateMemory(dev, &ai, NULL, out_mem) != VK_SUCCESS) {
                a->DestroyBuffer(dev, *out_buf, NULL);
                return -1;
            }
            if (a->BindBufferMemory(dev, *out_buf, *out_mem, 0) != VK_SUCCESS) {
                a->FreeMemory(dev, *out_mem, NULL);
                a->DestroyBuffer(dev, *out_buf, NULL);
                return -1;
            }
            if (persist_map) *persist_map = NULL;
            if (!logged) {
                logged = 1;
                ylog_info("vulkan: scratch mem type=%u flags=0x%x%s",
                          mi, (unsigned)mflags,
                          (mflags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " ReBAR" : " VRAM");
            }
            return 0;
        }
    }
}

static VkBuffer mem_to_buf(VulkanCtx* ctx, void* mem)
{
    if (mem == ctx->mem_x) return (VkBuffer)ctx->buf_x;
    if (mem == ctx->mem_y) return (VkBuffer)ctx->buf_y;
    if (mem == ctx->mem_o0) return (VkBuffer)ctx->buf_o0;
    if (mem == ctx->mem_o1) return (VkBuffer)ctx->buf_o1;
    if (mem == ctx->mem_o2) return (VkBuffer)ctx->buf_o2;
    if (mem == ctx->mem_wn) return (VkBuffer)ctx->buf_wn;
    if (mem == ctx->mem_wq) return (VkBuffer)ctx->buf_wq;
    if (mem == ctx->mem_kv) return (VkBuffer)ctx->buf_kv;
    if (mem == ctx->mem_bias) return (VkBuffer)ctx->buf_bias;
    if (mem == ctx->mem_logits) return (VkBuffer)ctx->buf_logits;
    if (mem == ctx->mem_emb) return (VkBuffer)ctx->buf_emb;
    return VK_NULL_HANDLE;
}

static void* vk_map_slot(VulkanCtx* ctx, void* mem, size_t off)
{
    if (mem == ctx->mem_x && ctx->map_x) return (uint8_t*)ctx->map_x + off;
    if (mem == ctx->mem_y && ctx->map_y) return (uint8_t*)ctx->map_y + off;
    if (mem == ctx->mem_o0 && ctx->map_o0) return (uint8_t*)ctx->map_o0 + off;
    if (mem == ctx->mem_o1 && ctx->map_o1) return (uint8_t*)ctx->map_o1 + off;
    if (mem == ctx->mem_o2 && ctx->map_o2) return (uint8_t*)ctx->map_o2 + off;
    if (mem == ctx->mem_wn && ctx->map_wn) return (uint8_t*)ctx->map_wn + off;
    if (mem == ctx->mem_wq && ctx->map_wq) return (uint8_t*)ctx->map_wq + off;
    if (mem == ctx->mem_kv && ctx->map_kv) return (uint8_t*)ctx->map_kv + off;
    if (mem == ctx->mem_bias && ctx->map_bias) return (uint8_t*)ctx->map_bias + off;
    return NULL;
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
        "inference/vulkan/shaders/",
        "vulkan/shaders/",
        "shaders/",
        "../inference/vulkan/shaders/",
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

static int create_compute_pipeline(VulkanCtx* ctx, const char* spv_name,
                                   VkPipelineLayout pl,
                                   VkShaderModule* out_shader, VkPipeline* out_pipe,
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
    uint32_t* code = NULL;
    size_t nwords = 0;
    code = read_spv(path, &nwords);
    if (!code) {
        if (err && errlen) snprintf(err, errlen, "read %s failed", path);
        return -1;
    }
    VkShaderModuleCreateInfo smi = {0};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = nwords * 4;
    smi.pCode = code;
    VkShaderModule sh = VK_NULL_HANDLE;
    VkResult r = a->CreateShaderModule(dev, &smi, NULL, &sh);
    free(code);
    if (r != VK_SUCCESS) {
        if (err && errlen) snprintf(err, errlen, "CreateShaderModule %s %d", spv_name, (int)r);
        return -1;
    }
    VkComputePipelineCreateInfo cpi = {0};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sh;
    cpi.stage.pName = "main";
    cpi.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (a->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe) != VK_SUCCESS) {
        a->DestroyShaderModule(dev, sh, NULL);
        if (err && errlen) snprintf(err, errlen, "CreateComputePipelines %s failed", spv_name);
        return -1;
    }
    *out_shader = sh;
    *out_pipe = pipe;
    ylog_info("vulkan: pipeline %s ready (%s)", spv_name, path);
    return 0;
}

static void cmd_barrier_host_to_shader(VkCommandBuffer cmd, VkBuffer buf)
{
    VulkanApi* a = vulkan_api();
    VkBufferMemoryBarrier bb = {0};
    if (!buf) return;
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = buf;
    bb.size = VK_WHOLE_SIZE;
    a->CmdPipelineBarrier(cmd,
                          VK_PIPELINE_STAGE_HOST_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, NULL, 1, &bb, 0, NULL);
}

static void cmd_barrier_shader_to_host(VkCommandBuffer cmd, VkBuffer buf)
{
    VulkanApi* a = vulkan_api();
    VkBufferMemoryBarrier bb = {0};
    if (!buf) return;
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = buf;
    bb.size = VK_WHOLE_SIZE;
    a->CmdPipelineBarrier(cmd,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT,
                          0, 0, NULL, 1, &bb, 0, NULL);
}
static void host_flush_mem(VulkanCtx* ctx, void* mem)
{
    VulkanApi* a = vulkan_api();
    VkMappedMemoryRange rng = {0};
    if (!ctx || !mem || !a->FlushMappedMemoryRanges) return;
    rng.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    rng.memory = (VkDeviceMemory)mem;
    rng.size = VK_WHOLE_SIZE;
    a->FlushMappedMemoryRanges((VkDevice)ctx->device, 1, &rng);
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
    if (a->InvalidateMappedMemoryRanges) {
        VkDeviceMemory mems[8];
        VkMappedMemoryRange rng[8];
        uint32_t n = 0, i;
        if (ctx->map_x) mems[n++] = (VkDeviceMemory)ctx->mem_x;
        if (ctx->map_y) mems[n++] = (VkDeviceMemory)ctx->mem_y;
        if (ctx->map_o0) mems[n++] = (VkDeviceMemory)ctx->mem_o0;
        if (ctx->map_o1) mems[n++] = (VkDeviceMemory)ctx->mem_o1;
        if (ctx->map_o2) mems[n++] = (VkDeviceMemory)ctx->mem_o2;
        if (ctx->map_kv) mems[n++] = (VkDeviceMemory)ctx->mem_kv;
        if (ctx->map_logits) mems[n++] = (VkDeviceMemory)ctx->mem_logits;
        if (ctx->map_stage) mems[n++] = (VkDeviceMemory)ctx->mem_stage;
        memset(rng, 0, sizeof(rng));
        for (i = 0; i < n; i++) {
            rng[i].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            rng[i].memory = mems[i];
            rng[i].size = VK_WHOLE_SIZE;
        }
        if (n) a->InvalidateMappedMemoryRanges(dev, n, rng);
    }
    if (ctx->prof_on) ctx->prof_submit_n++;
    return 0;
}

static void host_invalidate_scratch(VulkanCtx* ctx)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev;
    VkDeviceMemory mems[8];
    VkMappedMemoryRange rng[8];
    uint32_t n = 0, i;
    if (!ctx || !a->InvalidateMappedMemoryRanges) return;
    dev = (VkDevice)ctx->device;
    if (ctx->map_x) mems[n++] = (VkDeviceMemory)ctx->mem_x;
    if (ctx->map_y) mems[n++] = (VkDeviceMemory)ctx->mem_y;
    if (ctx->map_o0) mems[n++] = (VkDeviceMemory)ctx->mem_o0;
    if (ctx->map_o1) mems[n++] = (VkDeviceMemory)ctx->mem_o1;
    if (ctx->map_o2) mems[n++] = (VkDeviceMemory)ctx->mem_o2;
    if (ctx->map_kv) mems[n++] = (VkDeviceMemory)ctx->mem_kv;
    if (ctx->map_logits) mems[n++] = (VkDeviceMemory)ctx->mem_logits;
    if (ctx->map_stage) mems[n++] = (VkDeviceMemory)ctx->mem_stage;
    memset(rng, 0, sizeof(rng));
    for (i = 0; i < n; i++) {
        rng[i].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        rng[i].memory = mems[i];
        rng[i].size = VK_WHOLE_SIZE;
    }
    if (n) a->InvalidateMappedMemoryRanges(dev, n, rng);
}

/* finish=0: 只 submit, 用 semaphore 链到下一次; finish=1: 等 fence 并清 tokb_i */
static int submit_chained(VulkanCtx* ctx, VkCommandBuffer cmd, int finish)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    VkFence fence = finish ? (VkFence)ctx->fence : VK_NULL_HANDLE;
    VkSubmitInfo si = {0};
    VkPipelineStageFlags wait_st = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSemaphore wait_sem = VK_NULL_HANDLE;
    VkSemaphore sig_sem = VK_NULL_HANDLE;

    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (ctx->tokb_i > 0 && ctx->sem_ring && ctx->tokb_i - 1u < ctx->cmd_n) {
        wait_sem = (VkSemaphore)ctx->sem_ring[ctx->tokb_i - 1u];
        if (wait_sem) {
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &wait_sem;
            si.pWaitDstStageMask = &wait_st;
        }
    }
    if (!finish && ctx->sem_ring && ctx->tokb_i < ctx->cmd_n) {
        sig_sem = (VkSemaphore)ctx->sem_ring[ctx->tokb_i];
        if (sig_sem) {
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &sig_sem;
        }
    }
    if (finish && fence) a->ResetFences(dev, 1, &fence);
    if (a->QueueSubmit((VkQueue)ctx->queue, 1, &si, fence) != VK_SUCCESS) return -1;
    ctx->tokb_i++;
    if (finish) {
        if (!fence) {
            a->QueueWaitIdle((VkQueue)ctx->queue);
        } else if (a->WaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull) != VK_SUCCESS)
            return -1;
        host_invalidate_scratch(ctx);
        ctx->tokb_i = 0;
        if (ctx->prof_on) ctx->prof_submit_n++;
    } else if (ctx->prof_on) {
        ctx->prof_submit_n++;
    }
    return 0;
}

static VkCommandBuffer tokb_cmd(VulkanCtx* ctx)
{
    if (ctx && ctx->tokb_chain && ctx->cmd_ring && ctx->tokb_i < ctx->cmd_n &&
        ctx->cmd_ring[ctx->tokb_i])
        return (VkCommandBuffer)ctx->cmd_ring[ctx->tokb_i];
    return ctx ? (VkCommandBuffer)ctx->cmd : VK_NULL_HANDLE;
}

static int stage_copy(VulkanCtx* ctx, VkBuffer dst, const void* src, size_t nbytes, int h2d)
{
    VulkanApi* a = vulkan_api();
    VkBuffer stg;
    VkBufferCopy cp;
    if (!ctx || !ctx->map_stage || !ctx->buf_stage || nbytes == 0) return -1;
    if (nbytes > ctx->stage_bytes) return -1;
    stg = (VkBuffer)ctx->buf_stage;
    if (h2d) {
        memcpy(ctx->map_stage, src, nbytes);
        host_flush_mem(ctx, ctx->mem_stage);
    }
    if (ctx->cmd_open) {
        VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
        VkBufferMemoryBarrier bb = {0};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.size = VK_WHOLE_SIZE;
        if (h2d) {
            bb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            bb.buffer = stg;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        } else {
            bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            bb.buffer = dst;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        }
        memset(&cp, 0, sizeof(cp));
        cp.size = nbytes;
        if (h2d) a->CmdCopyBuffer(cmd, stg, dst, 1, &cp);
        else a->CmdCopyBuffer(cmd, dst, stg, 1, &cp);
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        if (h2d) {
            bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            bb.buffer = dst;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        } else {
            bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            bb.buffer = stg;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        }
        return 0;
    }
    {
        VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
        VkCommandBufferBeginInfo bi = {0};
        VkBufferMemoryBarrier bb = {0};
        a->ResetCommandBuffer(cmd, 0);
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        a->BeginCommandBuffer(cmd, &bi);
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.size = VK_WHOLE_SIZE;
        if (h2d) {
            bb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            bb.buffer = stg;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        }
        memset(&cp, 0, sizeof(cp));
        cp.size = nbytes;
        if (h2d) a->CmdCopyBuffer(cmd, stg, dst, 1, &cp);
        else a->CmdCopyBuffer(cmd, dst, stg, 1, &cp);
        if (!h2d) {
            bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            bb.buffer = stg;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        } else {
            bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            bb.buffer = dst;
            a->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        }
        if (a->EndCommandBuffer(cmd) != VK_SUCCESS) return -1;
        if (submit_and_wait(ctx, cmd) != 0) return -1;
    }
    if (!h2d) {
        if (a->InvalidateMappedMemoryRanges && ctx->mem_stage) {
            VkMappedMemoryRange rng = {0};
            rng.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            rng.memory = (VkDeviceMemory)ctx->mem_stage;
            rng.size = VK_WHOLE_SIZE;
            a->InvalidateMappedMemoryRanges((VkDevice)ctx->device, 1, &rng);
        }
    }
    return 0;
}

static int map_copy(VulkanCtx* ctx, void* mem, const void* src, size_t nbytes);
static int map_read(VulkanCtx* ctx, void* mem, void* dst, size_t nbytes);

void vulkan_gpu_discard(VulkanCtx* ctx)
{
    if (!ctx || !ctx->cmd_open) return;
    VulkanApi* a = vulkan_api();
    a->ResetCommandBuffer((VkCommandBuffer)ctx->cmd, 0);
    ctx->cmd_open = 0;
}

int vulkan_gpu_flush(VulkanCtx* ctx)
{
    if (!ctx) return 0;
    if (ctx->tokb_i > 0 && ctx->tokb_chain) {
        /* 半截 token: 等队列清空, 避免 semaphore 残留 signaled */
        VulkanApi* a = vulkan_api();
        if (ctx->queue && a->QueueWaitIdle)
            a->QueueWaitIdle((VkQueue)ctx->queue);
        ctx->tokb_i = 0;
        ctx->cmd_open = 0;
        return 0;
    }
    if (!ctx->cmd_open) return 0;
    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;
    ctx->cmd_open = 0;
    return 0;
}

static uint32_t norm_w_off1(VulkanCtx* ctx, uint32_t layer)
{
    return (layer - 1u) * 2u * ctx->hidden;
}

static uint32_t norm_w_off2(VulkanCtx* ctx, uint32_t layer)
{
    return norm_w_off1(ctx, layer) + ctx->hidden;
}

static uint32_t norm_w_final(VulkanCtx* ctx)
{
    return ctx->n_blocks * 2u * ctx->hidden;
}

static uint32_t tokb_group(VulkanCtx* ctx)
{
    /* 默认 1: 多层同一 CB 在 NVIDIA 上数值错误。YLLM_VK_TOKB_GRP 才合并录制 */
    const char* eg = getenv("YLLM_VK_TOKB_GRP");
    uint32_t grp = eg ? (uint32_t)atoi(eg) : 1u;
    if (grp < 1u) grp = 1u;
    if (ctx->n_blocks && grp > ctx->n_blocks) grp = ctx->n_blocks;
    return grp;
}

int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden,
                         uint32_t max_in, uint32_t max_out,
                         size_t total_wq_bytes, uint32_t n_layers, uint32_t nslot,
                         uint32_t lm_vocab, char* err, size_t errlen)
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
    {
        uint32_t nb = ctx->n_blocks ? ctx->n_blocks
                                    : (n_layers > 3u ? n_layers - 3u : 1u);
        ctx->wn_bytes = (size_t)(2u * nb + 1u) * hidden * 4;
    }
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
    if (create_scratch_buffer(ctx, ctx->x_bytes, &bx, &mx, &ctx->map_x) != 0) return -1;
    if (create_scratch_buffer(ctx, ctx->y_bytes, &by, &my, &ctx->map_y) != 0) return -1;
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
    {
        uint32_t ncmd = n_layers + 8u;
        uint32_t i;
        VkCommandBuffer tmp[256];
        if (ncmd < 8u) ncmd = 8u;
        if (ncmd > 256u) ncmd = 256u;
        cai.commandBufferCount = ncmd;
        if (a->AllocateCommandBuffers(dev, &cai, tmp) != VK_SUCCESS) {
            cai.commandBufferCount = 1;
            if (a->AllocateCommandBuffers(dev, &cai, tmp) != VK_SUCCESS) return -1;
            ncmd = 1;
        }
        ctx->cmd_n = ncmd;
        ctx->cmd = (void*)tmp[0];
        ctx->cmd_ring = (void**)calloc(ncmd, sizeof(void*));
        if (ctx->cmd_ring) {
            for (i = 0; i < ncmd; i++) ctx->cmd_ring[i] = (void*)tmp[i];
        }
        ctx->tokb_i = 0;
        ctx->tokb_chain = 0;
        ctx->sem_ring = (void**)calloc(ncmd, sizeof(void*));
        if (ctx->sem_ring && a->CreateSemaphore && ncmd > 1u) {
            VkSemaphoreCreateInfo sci = {0};
            sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (i = 0; i < ncmd; i++) {
                VkSemaphore s = VK_NULL_HANDLE;
                if (a->CreateSemaphore(dev, &sci, NULL, &s) != VK_SUCCESS) {
                    ctx->tokb_chain = 0;
                    break;
                }
                ctx->sem_ring[i] = (void*)s;
            }
            if (i == ncmd && ctx->cmd_ring) ctx->tokb_chain = 1;
        }
        ctx->tokb_i = 0;
    }
    VkFenceCreateInfo fi = {0};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (a->CreateFence(dev, &fi, NULL, &fence) != VK_SUCCESS) return -1;
    ctx->fence = (void*)fence;

    {
        VkBuffer bw = VK_NULL_HANDLE;
        VkDeviceMemory mw = VK_NULL_HANDLE;
        if (create_host_buffer(ctx, ctx->wn_bytes, &bw, &mw, &ctx->map_wn) != 0) return -1;
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
        if (lm_vocab > 0) {
            a->DestroyDescriptorPool(dev, pool, NULL);
            VkDescriptorPoolSize dps = {0};
            dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            dps.descriptorCount = 6;
            VkDescriptorPoolCreateInfo dpi = {0};
            dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpi.maxSets = 2;
            dpi.poolSizeCount = 1;
            dpi.pPoolSizes = &dps;
            VkDescriptorPool rpool = VK_NULL_HANDLE;
            if (a->CreateDescriptorPool(dev, &dpi, NULL, &rpool) != VK_SUCCESS)
                return -1;
            VkDescriptorSetLayout rl = (VkDescriptorSetLayout)dsl;
            VkDescriptorSetLayout rlayouts[2] = { rl, rl };
            VkDescriptorSet rsets[2];
            VkDescriptorSetAllocateInfo dai = {0};
            dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dai.descriptorPool = rpool;
            dai.descriptorSetCount = 2;
            dai.pSetLayouts = rlayouts;
            if (a->AllocateDescriptorSets(dev, &dai, rsets) != VK_SUCCESS)
                return -1;
            {
                VkDescriptorBufferInfo bis[3];
                VkWriteDescriptorSet writes[3];
                uint32_t si;
                for (si = 0; si < 2; si++) {
                    VkBuffer yb = (si == 0) ? by : bx;
                    VkBuffer trip[3] = { bx, yb, bw };
                    size_t rng[3] = { ctx->x_bytes, ctx->y_bytes, ctx->wn_bytes };
                    memset(bis, 0, sizeof(bis));
                    memset(writes, 0, sizeof(writes));
                    uint32_t b;
                    for (b = 0; b < 3; b++) {
                        bis[b].buffer = trip[b];
                        bis[b].range = rng[b];
                        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        writes[b].dstSet = rsets[si];
                        writes[b].dstBinding = b;
                        writes[b].descriptorCount = 1;
                        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        writes[b].pBufferInfo = &bis[b];
                    }
                    a->UpdateDescriptorSets(dev, 3, writes, 0, NULL);
                }
            }
            ctx->rms_desc_pool = (void*)rpool;
            ctx->rms_desc_set = (void*)rsets[0];
            ctx->rms_ds_inplace = (void*)rsets[1];
        }
        ctx->host_w = (float*)malloc(ctx->wn_bytes);
        ctx->host_w2 = (float*)malloc(ctx->wn_bytes);
        if (!ctx->host_w || !ctx->host_w2) return -1;
        ctx->compute_ready = 1;
    }

    {
        VkBuffer bw = VK_NULL_HANDLE;
        VkDeviceMemory mw = VK_NULL_HANDLE;
        char gerr[256];
        VkBuffer bo0 = VK_NULL_HANDLE, bo1 = VK_NULL_HANDLE, bo2 = VK_NULL_HANDLE;
        VkDeviceMemory mo0 = VK_NULL_HANDLE, mo1 = VK_NULL_HANDLE, mo2 = VK_NULL_HANDLE;
        VkBuffer blogits = VK_NULL_HANDLE;
        VkDeviceMemory mlogits = VK_NULL_HANDLE;
        if (lm_vocab > 0) {
            ctx->logits_bytes = (size_t)lm_vocab * 4;
            if (create_host_buffer(ctx, ctx->logits_bytes, &blogits, &mlogits,
                                   &ctx->map_logits) != 0) {
                ylog_warn("vulkan: buf_logits alloc failed vocab=%u", lm_vocab);
                ctx->logits_bytes = 0;
                blogits = VK_NULL_HANDLE;
            } else {
                ctx->buf_logits = (void*)blogits;
                ctx->mem_logits = (void*)mlogits;
            }
        }
        if (create_host_buffer(ctx, ctx->wq_bytes, &bw, &mw, &ctx->map_wq) != 0 ||
            create_scratch_buffer(ctx, ctx->y_bytes, &bo0, &mo0, &ctx->map_o0) != 0 ||
            create_scratch_buffer(ctx, ctx->y_bytes, &bo1, &mo1, &ctx->map_o1) != 0 ||
            create_scratch_buffer(ctx, ctx->y_bytes, &bo2, &mo2, &ctx->map_o2) != 0) {
            ylog_warn("vulkan: gemv buffers alloc failed");
        } else {
            ctx->buf_wq = (void*)bw; ctx->mem_wq = (void*)mw;
            ctx->buf_o0 = (void*)bo0; ctx->mem_o0 = (void*)mo0;
            ctx->buf_o1 = (void*)bo1; ctx->mem_o1 = (void*)mo1;
            ctx->buf_o2 = (void*)bo2; ctx->mem_o2 = (void*)mo2;
            ctx->stage_bytes = ctx->y_bytes;
            if (ctx->x_bytes > ctx->stage_bytes) ctx->stage_bytes = ctx->x_bytes;
            if (ctx->logits_bytes > ctx->stage_bytes) ctx->stage_bytes = ctx->logits_bytes;
            {
                VkBuffer bst = VK_NULL_HANDLE;
                VkDeviceMemory mst = VK_NULL_HANDLE;
                if (create_host_buffer(ctx, ctx->stage_bytes, &bst, &mst, &ctx->map_stage) != 0) {
                    ylog_warn("vulkan: staging buffer alloc failed");
                    ctx->map_stage = NULL;
                } else {
                    ctx->buf_stage = (void*)bst;
                    ctx->mem_stage = (void*)mst;
                }
            }
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
                /* 扩容 pool: ds0/1/2 (x=buf_y→o*) + ds_xo (x=o2→y) */
                a->DestroyDescriptorPool(dev, pool, NULL);
                VkDescriptorPoolSize dps = {0};
                dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                dps.descriptorCount = blogits ? 18 : 15;
                VkDescriptorPoolCreateInfo dpi = {0};
                dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                dpi.maxSets = blogits ? 6 : 5;
                dpi.poolSizeCount = 1;
                dpi.pPoolSizes = &dps;
                if (a->CreateDescriptorPool(dev, &dpi, NULL, &pool) != VK_SUCCESS) {
                    ylog_warn("vulkan: gemv desc pool recreate failed");
                } else {
                    uint32_t nsets = blogits ? 6u : 5u;
                    VkDescriptorSetLayout layouts[6];
                    VkDescriptorSet sets[6];
                    uint32_t si;
                    for (si = 0; si < nsets; si++) layouts[si] = dsl;
                    VkDescriptorSetAllocateInfo dai = {0};
                    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    dai.descriptorPool = pool;
                    dai.descriptorSetCount = nsets;
                    dai.pSetLayouts = layouts;
                    if (a->AllocateDescriptorSets(dev, &dai, sets) != VK_SUCCESS) {
                        ylog_warn("vulkan: gemv desc sets alloc failed");
                    } else {
                        VkBuffer xb[6] = { bx, by, by, by, bo2, bx };
                        VkBuffer yb[6] = { by, bo0, bo1, bo2, by, blogits ? blogits : by };
                        for (si = 0; si < nsets; si++) {
                            VkDescriptorBufferInfo bis[3];
                            VkWriteDescriptorSet writes[3];
                            memset(bis, 0, sizeof(bis));
                            memset(writes, 0, sizeof(writes));
                            VkBuffer trip[3] = { xb[si], yb[si], bw };
                            size_t xrng = (si == 0 || si == 5) ? ctx->x_bytes : ctx->y_bytes;
                            size_t yrng = (si == 5 && blogits) ? ctx->logits_bytes : ctx->y_bytes;
                            size_t rng[3] = { xrng, yrng, ctx->wq_bytes };
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
                        ctx->gemv_ds_xo = (void*)sets[4];
                        if (blogits) ctx->gemv_ds_lm = (void*)sets[5];
                        ctx->attn_o_ready = 1;
                        ctx->gemv_ready = 1;
                        ctx->fuse_ready = 1;
                        ylog_info("vulkan: gemv_q4k+fuse ready max_in=%u max_out=%u wq=%zuMB%s",
                                  max_in, max_out, ctx->wq_bytes / (1024 * 1024),
                                  blogits ? " lm_buf=1" : "");
                        {
                            char q6err[256];
                            VkShaderModule q6sh = VK_NULL_HANDLE;
                            VkPipeline q6pipe = VK_NULL_HANDLE;
                            if (create_compute_pipeline(ctx, "gemv_q6k.spv",
                                                        (VkPipelineLayout)pl,
                                                        &q6sh, &q6pipe,
                                                        q6err, sizeof(q6err)) != 0) {
                                ylog_warn("vulkan: gemv_q6k pipeline failed (%s)", q6err);
                            } else {
                                ctx->gemv_q6k_shader = (void*)q6sh;
                                ctx->gemv_q6k_pipeline = (void*)q6pipe;
                                ctx->gemv_q6k_ready = 1;
                                ylog_info("vulkan: gemv_q6k ready");
                            }
                        }
                        if (blogits && ctx->gemv_q6k_ready) {
                            char q8err[256];
                            VkBuffer bufs3[3] = { bx, bx, bx };
                            size_t ranges3[3] = { ctx->x_bytes, ctx->x_bytes, ctx->x_bytes };
                            VkShaderModule q8sh = VK_NULL_HANDLE;
                            VkDescriptorSetLayout q8dsl = VK_NULL_HANDLE;
                            VkPipelineLayout q8pl = VK_NULL_HANDLE;
                            VkPipeline q8pipe = VK_NULL_HANDLE;
                            VkDescriptorPool q8pool = VK_NULL_HANDLE;
                            VkDescriptorSet q8dset = VK_NULL_HANDLE;
                            if (create_ssbo_pipeline(ctx, "q8k_quant.spv", sizeof(Q8kPush),
                                                     &q8sh, &q8dsl, &q8pl, &q8pipe, &q8pool, &q8dset,
                                                     bufs3, ranges3, q8err, sizeof(q8err)) != 0) {
                                ylog_warn("vulkan: q8k_quant pipeline failed (%s)", q8err);
                            } else {
                                ctx->q8k_shader = (void*)q8sh;
                                ctx->q8k_desc_layout = (void*)q8dsl;
                                ctx->q8k_pipe_layout = (void*)q8pl;
                                ctx->q8k_pipeline = (void*)q8pipe;
                                ctx->q8k_desc_pool = (void*)q8pool;
                                ctx->q8k_desc_set = (void*)q8dset;
                                ctx->q8k_ready = 1;
                                ylog_info("vulkan: q8k_quant ready");
                            }
                        }
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

    /* Q4_K embed: 上传一行 → buf_x */
    if (ctx->gemv_ready && ctx->buf_x && hidden >= 256 && (hidden % 256) == 0) {
        char eerr[256];
        uint32_t nb = hidden / 256;
        ctx->emb_bytes = (size_t)nb * 144;
        VkBuffer be = VK_NULL_HANDLE;
        VkDeviceMemory me = VK_NULL_HANDLE;
        if (create_host_buffer(ctx, ctx->emb_bytes, &be, &me, &ctx->map_emb) == 0) {
            ctx->buf_emb = (void*)be;
            ctx->mem_emb = (void*)me;
            VkShaderModule sh = VK_NULL_HANDLE;
            VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
            VkPipelineLayout pl = VK_NULL_HANDLE;
            VkPipeline pipe = VK_NULL_HANDLE;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VkDescriptorSet dset = VK_NULL_HANDLE;
            /* create_ssbo_pipeline expects 3 bindings; pad with duplicate emb */
            VkBuffer bufs3[3] = { bx, be, be };
            size_t ranges3[3] = { ctx->x_bytes, ctx->emb_bytes, ctx->emb_bytes };
            if (create_ssbo_pipeline(ctx, "embed_q4k.spv", sizeof(EmbedPush),
                                     &sh, &dsl, &pl, &pipe, &pool, &dset,
                                     bufs3, ranges3, eerr, sizeof(eerr)) == 0) {
                ctx->embed_shader = (void*)sh;
                ctx->embed_desc_layout = (void*)dsl;
                ctx->embed_pipe_layout = (void*)pl;
                ctx->embed_pipeline = (void*)pipe;
                ctx->embed_desc_pool = (void*)pool;
                ctx->embed_desc_set = (void*)dset;
                ctx->embed_ready = 1;
                ylog_info("vulkan: embed_q4k ready hidden=%u", hidden);
            } else {
                ylog_warn("vulkan: embed_q4k pipeline failed (%s)", eerr);
            }
        }
    }
    return 0;
}

int vulkan_wq_upload(VulkanCtx* ctx, const void* blob, size_t bytes)
{
    if (!ctx || !ctx->gemv_ready || !blob || bytes == 0) return -1;
    if (bytes > ctx->wq_bytes) return -1;
    if (ctx->map_wq) {
        memcpy(ctx->map_wq, blob, bytes);
        host_flush_mem(ctx, ctx->mem_wq);
    } else {
        VulkanApi* a = vulkan_api();
        VkDevice dev = (VkDevice)ctx->device;
        const size_t chunk = 16u * 1024u * 1024u;
        size_t off = 0;
        while (off < bytes) {
            size_t n = bytes - off;
            if (n > chunk) n = chunk;
            void* pw = NULL;
            if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wq, off, n, 0, &pw) != VK_SUCCESS)
                return -1;
            memcpy(pw, (const uint8_t*)blob + off, n);
            a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq);
            off += n;
        }
    }
    ctx->wq_resident = 1;
    ylog_info("vulkan: Q4_K weights resident %zuMB", bytes / (1024 * 1024));
    return 0;
}

int vulkan_wq_upload_range(VulkanCtx* ctx, const void* src, size_t dst_off, size_t bytes)
{
    if (!ctx || !ctx->mem_wq || !src || bytes == 0) return -1;
    if (dst_off + bytes > ctx->wq_bytes) return -1;
    if (ctx->map_wq) {
        memcpy((uint8_t*)ctx->map_wq + dst_off, src, bytes);
        host_flush_mem(ctx, ctx->mem_wq);
        return 0;
    }
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    const size_t chunk = 16u * 1024u * 1024u;
    size_t done = 0;
    while (done < bytes) {
        size_t n = bytes - done;
        if (n > chunk) n = chunk;
        void* pw = NULL;
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wq, dst_off + done, n, 0, &pw) != VK_SUCCESS)
            return -1;
        memcpy(pw, (const uint8_t*)src + done, n);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq);
        done += n;
    }
    return 0;
}

int vulkan_stream_layer(VulkanCtx* ctx, uint32_t layer)
{
    if (!ctx || !ctx->wq_stream || !ctx->host_wq || !ctx->mem_wq) return -1;
    if (ctx->stream_layer == layer) return 0;
    uint32_t s;
    uint64_t lo = (uint64_t)~0ull, hi = 0;
    for (s = 0; s < ctx->wq_nslot; s++) {
        uint64_t off = ctx->wq_off[(size_t)layer * ctx->wq_nslot + s];
        if (off == (uint64_t)~0ull) continue;
        if (off < lo) lo = off;
        /* 估每槽最大: 用到 layer_wq_max 覆盖 */
        if (off > hi) hi = off;
    }
    if (lo == (uint64_t)~0ull) return -1;
    size_t nbytes = ctx->layer_wq_max;
    if (nbytes == 0) nbytes = ctx->wq_bytes;
    if (lo + nbytes > ctx->host_wq_bytes)
        nbytes = ctx->host_wq_bytes - (size_t)lo;
    if (nbytes > ctx->wq_bytes) nbytes = ctx->wq_bytes;
    if (ctx->map_wq) {
        memcpy(ctx->map_wq, ctx->host_wq + (size_t)lo, nbytes);
        host_flush_mem(ctx, ctx->mem_wq);
    } else {
        const size_t chunk = 16u * 1024u * 1024u;
        size_t done = 0;
        VulkanApi* a = vulkan_api();
        VkDevice dev = (VkDevice)ctx->device;
        while (done < nbytes) {
            size_t n = nbytes - done;
            if (n > chunk) n = chunk;
            void* pw = NULL;
            if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wq, done, n, 0, &pw) != VK_SUCCESS)
                return -1;
            memcpy(pw, ctx->host_wq + (size_t)lo + done, n);
            a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq);
            done += n;
        }
    }
    ctx->stream_layer = layer;
    ctx->stream_base = lo;
    (void)hi;
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
    if (map_copy(ctx, ctx->mem_x, x, nbytes) != 0) return -1;
    if (map_copy(ctx, ctx->mem_wn, w, nbytes) != 0) return -1;

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_x);
    cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wn);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rms_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->rms_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rms_pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    push.w_off = 0;
    push._pad = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rms_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
    cmd_barrier_shader_to_host(cmd, (VkBuffer)ctx->buf_y);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (map_read(ctx, ctx->mem_y, y, nbytes) != 0) return -1;
    return 0;
}

/* buf_x 已有效; 结果写回 buf_x, 不 D2H */
int vulkan_k_rmsnorm_inplace(VulkanCtx* ctx, const float* w, uint32_t n, float eps)
{
    if (!ctx || !ctx->compute_ready || !ctx->rms_ds_inplace || !w || n == 0) return -1;
    if ((size_t)n * 4 > ctx->x_bytes || (size_t)n * 4 > ctx->wn_bytes) return -1;
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t nbytes = (size_t)n * 4;
    if (ctx->map_wn) { memcpy(ctx->map_wn, w, nbytes); host_flush_mem(ctx, ctx->mem_wn); }
    else {
        void* pw = NULL;
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wn, 0, nbytes, 0, &pw) != VK_SUCCESS)
            return -1;
        memcpy(pw, w, nbytes);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wn);
    }
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rms_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->rms_ds_inplace;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rms_pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    push.w_off = 0;
    push._pad = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rms_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;
    ctx->x_on_dev = 1;
    return 0;
}

int vulkan_k_lm_gemv(VulkanCtx* ctx, float* y, const float* x, int x_on_dev,
                     uint32_t vocab, uint32_t hidden, uint64_t w_byte_off, uint32_t dtype)
{
    if (!ctx || !ctx->gemv_ds_lm || !ctx->map_logits || !y) return -1;
    if (vocab == 0 || hidden == 0 || (hidden % 256) != 0) return -1;
    if ((size_t)vocab * 4 > ctx->logits_bytes) return -1;
    size_t rowb = (dtype == DT_Q6K) ? 210u : 144u;
    size_t wbytes = (size_t)vocab * ((size_t)(hidden / 256) * rowb);
    if (w_byte_off + wbytes > ctx->wq_bytes) return -1;
    if (w_byte_off > 0xffffffffull) return -1;

    int q6 = (dtype == DT_Q6K);
    if (q6 && !ctx->gemv_q6k_ready) return -1;
    if (!q6 && !ctx->gemv_ready) return -1;

    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t xbytes = (size_t)hidden * 4;

    if (!x_on_dev) {
        if (!x) return -1;
        if (map_copy(ctx, ctx->mem_x, x, xbytes) != 0) return -1;
        ctx->x_on_dev = 1;
    }

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    VkPipeline pipe = q6 ? (VkPipeline)ctx->gemv_q6k_pipeline : (VkPipeline)ctx->gemv_pipeline;
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->gemv_ds_lm;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->gemv_pipe_layout, 0, 1, &ds, 0, NULL);
    GemvPush push;
    push.out_n = vocab;
    push.in_n = hidden;
    push.w_off = (uint32_t)w_byte_off;
    push.serial = gemv_serial_flag(ctx);
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, vocab, 1, 1);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    size_t ybytes = (size_t)vocab * 4;
    if (ctx->map_logits) memcpy(y, ctx->map_logits, ybytes);
    else {
        void* py = NULL;
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_logits, 0, ybytes, 0, &py) != VK_SUCCESS)
            return -1;
        memcpy(y, py, ybytes);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_logits);
    }
    return 0;
}

int vulkan_k_gemv_q4k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off)
{
    if (!ctx || !ctx->gemv_ready || !y || !x) return -1;
    if (out == 0 || in == 0 || (in % 256) != 0) return -1;
    if (out > ctx->max_out) return -1;
    /* in>max_in: FFN down, 输入走 buf_y(gemv_ds0), 输出 o0 */
    int wide = (in > ctx->max_in);
    if (wide) {
        if (in > ctx->max_out || !ctx->gemv_ds0) return -1;
    }
    size_t wbytes = (size_t)out * ((size_t)(in / 256) * 144);
    if (w_byte_off + wbytes > ctx->wq_bytes) return -1;
    if (w_byte_off > 0xffffffffull) return -1;

    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t xbytes = (size_t)in * 4;
    size_t ybytes = (size_t)out * 4;
    void* xmem = wide ? ctx->mem_y : ctx->mem_x;
    void* ymem = wide ? ctx->mem_o0 : ctx->mem_y;
    void* dset = wide ? ctx->gemv_ds0 : ctx->gemv_desc_set;

    {
        void* px = vk_map_slot(ctx, xmem, 0);
        if (px) { memcpy(px, x, xbytes); host_flush_mem(ctx, xmem); }
        else if (map_copy(ctx, xmem, x, xbytes) != 0)
            return -1;
    }

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    cmd_barrier_host_to_shader(cmd, (VkBuffer)(wide ? ctx->buf_y : ctx->buf_x));
    cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wq);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->gemv_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->gemv_pipe_layout, 0, 1, &ds, 0, NULL);
    GemvPush push;
    push.out_n = out;
    push.in_n = in;
    push.w_off = (uint32_t)w_byte_off;
    push.serial = gemv_serial_flag(ctx);
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, out, 1, 1);
    cmd_barrier_shader_to_host(cmd, (VkBuffer)(wide ? ctx->buf_o0 : ctx->buf_y));
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (map_read(ctx, ymem, y, ybytes) != 0) return -1;
    return 0;
}

int vulkan_k_gemv_q6k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off)
{
    if (!ctx || !ctx->gemv_q6k_ready || !y || !x) return -1;
    if (out == 0 || in == 0 || (in % 256) != 0) return -1;
    if (out > ctx->max_out) return -1;
    if (in > ctx->max_in) return -1;
    size_t wbytes = (size_t)out * ((size_t)(in / 256) * 210);
    if (w_byte_off + wbytes > ctx->wq_bytes) return -1;
    if (w_byte_off > 0xffffffffull) return -1;

    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t xbytes = (size_t)in * 4;
    size_t ybytes = (size_t)out * 4;

    if (map_copy(ctx, ctx->mem_x, x, xbytes) != 0) return -1;

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->gemv_q6k_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->gemv_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->gemv_pipe_layout, 0, 1, &ds, 0, NULL);
    GemvPush push;
    push.out_n = out;
    push.in_n = in;
    push.w_off = (uint32_t)w_byte_off;
    push.serial = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, out, 1, 1);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    if (map_read(ctx, ctx->mem_y, y, ybytes) != 0) return -1;
    return 0;
}

int vulkan_k_gemv_q4k_host(VulkanCtx* ctx, float* y, const float* x,
                           const uint8_t* w, uint32_t out, uint32_t in)
{
    if (!ctx || !w) return -1;
    /* 常驻/流式权在 GPU wq[0..); 禁止把单矩阵盖到 offset 0 */
    if (ctx->wq_resident) return -1;
    size_t wbytes = (size_t)out * ((size_t)(in / 256) * 144);
    if (wbytes > ctx->wq_bytes) return -1;
    if (ctx->map_wq) {
        memcpy(ctx->map_wq, w, wbytes);
        host_flush_mem(ctx, ctx->mem_wq);
    }
    else {
        VulkanApi* a = vulkan_api();
        VkDevice dev = (VkDevice)ctx->device;
        void* pw = NULL;
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_wq, 0, wbytes, 0, &pw) != VK_SUCCESS)
            return -1;
        memcpy(pw, w, wbytes);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_wq);
    }
    return vulkan_k_gemv_q4k(ctx, y, x, out, in, 0);
}

static void cmd_barrier_compute(VkCommandBuffer cmd)
{
    VulkanApi* a = vulkan_api();
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    a->CmdPipelineBarrier(cmd,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &mb, 0, NULL, 0, NULL);
}

static void cmd_barrier_buf(VkCommandBuffer cmd, VkBuffer buf, VkDeviceSize sz)
{
    if (!buf || sz == 0) return;
    VulkanApi* a = vulkan_api();
    VkBufferMemoryBarrier bb = {0};
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = buf;
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;
    (void)sz;
    a->CmdPipelineBarrier(cmd,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, NULL, 1, &bb, 0, NULL);
}

static void cmd_barrier_layer(VulkanCtx* ctx, VkCommandBuffer cmd)
{
    VulkanApi* a = vulkan_api();
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    a->CmdPipelineBarrier(cmd,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          0, 1, &mb, 0, NULL, 0, NULL);
    cmd_barrier_buf(cmd, (VkBuffer)ctx->buf_x, (VkDeviceSize)ctx->x_bytes);
    cmd_barrier_buf(cmd, (VkBuffer)ctx->buf_y, (VkDeviceSize)ctx->y_bytes);
    if (ctx->buf_o0)
        cmd_barrier_buf(cmd, (VkBuffer)ctx->buf_o0, (VkDeviceSize)ctx->y_bytes);
    if (ctx->buf_o1)
        cmd_barrier_buf(cmd, (VkBuffer)ctx->buf_o1, (VkDeviceSize)ctx->y_bytes);
    if (ctx->buf_o2)
        cmd_barrier_buf(cmd, (VkBuffer)ctx->buf_o2, (VkDeviceSize)ctx->y_bytes);
    if (ctx->buf_kv)
        cmd_barrier_buf(cmd, (VkBuffer)ctx->buf_kv, (VkDeviceSize)ctx->kv_bytes);
}

static void cmd_rmsnorm(VulkanCtx* ctx, VkCommandBuffer cmd, uint32_t n, float eps,
                        uint32_t w_off)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rms_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->rms_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rms_pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    push.w_off = w_off;
    push._pad = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rms_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
}

static void cmd_rmsnorm_inplace(VulkanCtx* ctx, VkCommandBuffer cmd, uint32_t n, float eps,
                                uint32_t w_off)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rms_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->rms_ds_inplace;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rms_pipe_layout, 0, 1, &ds, 0, NULL);
    RmsPush push;
    push.n = n;
    push.eps = eps;
    push.w_off = w_off;
    push._pad = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rms_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, 1, 1, 1);
}

static void cmd_q8k_quant(VulkanCtx* ctx, VkCommandBuffer cmd, uint32_t n)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->q8k_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->q8k_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->q8k_pipe_layout, 0, 1, &ds, 0, NULL);
    Q8kPush push;
    push.n = n;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->q8k_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, n / 256u, 1, 1);
}

static void cmd_gemv_q6k(VulkanCtx* ctx, VkCommandBuffer cmd, void* dset,
                         uint32_t out, uint32_t in, uint32_t w_off)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->gemv_q6k_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->gemv_pipe_layout, 0, 1, &ds, 0, NULL);
    GemvPush push;
    push.out_n = out;
    push.in_n = in;
    push.w_off = w_off;
    push.serial = 0;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, out, 1, 1);
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
    push.serial = gemv_serial_flag(ctx);
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->gemv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, out, 1, 1);
}

static int map_copy(VulkanCtx* ctx, void* mem, const void* src, size_t nbytes)
{
    void* p = vk_map_slot(ctx, mem, 0);
    if (p) {
        memcpy(p, src, nbytes);
        host_flush_mem(ctx, mem);
        return 0;
    }
    {
        VkBuffer dst = mem_to_buf(ctx, mem);
        if (!dst || stage_copy(ctx, dst, src, nbytes, 1) != 0) return -1;
        return 0;
    }
}

/* final norm(in-place) + 可选 q8k + lm gemv → logits; 单次 submit */
int vulkan_k_lm_fused(VulkanCtx* ctx, float* y, const float* w_norm,
                      uint32_t hidden, float eps, uint32_t vocab,
                      uint64_t w_byte_off, uint32_t dtype,
                      int upload_x, const float* host_x)
{
    if (!ctx || !ctx->rms_ds_inplace || !ctx->gemv_ds_lm || !ctx->map_logits || !y)
        return -1;
    if (!ctx->norm_ready && !w_norm) return -1;
    if (vocab == 0 || hidden == 0 || (hidden % 256) != 0) return -1;
    if ((size_t)vocab * 4 > ctx->logits_bytes) return -1;

    int q6 = (dtype == DT_Q6K);
    if (q6) {
        if (!ctx->gemv_q6k_ready || !ctx->q8k_ready) return -1;
    } else if (dtype == DT_Q4K) {
        if (!ctx->gemv_ready) return -1;
    } else {
        return -1;
    }

    size_t rowb = q6 ? 210u : 144u;
    size_t wbytes = (size_t)vocab * ((size_t)(hidden / 256) * rowb);
    if (w_byte_off + wbytes > ctx->wq_bytes) return -1;
    if (w_byte_off > 0xffffffffull) return -1;

    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    size_t xbytes = (size_t)hidden * 4;
    size_t wnbytes = (size_t)hidden * 4;
    uint32_t w_off = ctx->norm_ready ? norm_w_final(ctx) : 0u;

    if (upload_x && host_x && !ctx->cmd_open) {
        if (map_copy(ctx, ctx->mem_x, host_x, xbytes) != 0) return -1;
        ctx->x_on_dev = 1;
    } else if (!ctx->x_on_dev && !ctx->cmd_open) {
        return -1;
    }

    if (!ctx->norm_ready) {
        if (map_copy(ctx, ctx->mem_wn, w_norm, wnbytes) != 0) return -1;
    }

    VkCommandBuffer cmd;
    if (ctx->tokb_chain && ctx->tokb_i > 0 && !ctx->cmd_open) {
        cmd = tokb_cmd(ctx);
        a->ResetCommandBuffer(cmd, 0);
        {
            VkCommandBufferBeginInfo bi = {0};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            a->BeginCommandBuffer(cmd, &bi);
        }
    } else if (!ctx->cmd_open) {
        cmd = (VkCommandBuffer)ctx->cmd;
        a->ResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi = {0};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        a->BeginCommandBuffer(cmd, &bi);
    } else {
        cmd = (VkCommandBuffer)ctx->cmd;
        cmd_barrier_layer(ctx, cmd);
    }

    cmd_rmsnorm_inplace(ctx, cmd, hidden, eps, w_off);
    cmd_barrier_compute(cmd);
    if (q6) {
        cmd_q8k_quant(ctx, cmd, hidden);
        cmd_barrier_compute(cmd);
    }
    if (q6)
        cmd_gemv_q6k(ctx, cmd, ctx->gemv_ds_lm, vocab, hidden, (uint32_t)w_byte_off);
    else
        cmd_gemv(ctx, cmd, ctx->gemv_ds_lm, vocab, hidden, (uint32_t)w_byte_off);

    a->EndCommandBuffer(cmd);
    if (ctx->tokb_chain && ctx->tokb_i > 0) {
        if (submit_chained(ctx, cmd, 1) != 0) return -1;
    } else if (submit_and_wait(ctx, cmd) != 0) {
        return -1;
    }
    ctx->cmd_open = 0;

    ctx->x_on_dev = 1;
    size_t ybytes = (size_t)vocab * 4;
    if (ctx->map_logits) memcpy(y, ctx->map_logits, ybytes);
    else {
        void* py = NULL;
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_logits, 0, ybytes, 0, &py) != VK_SUCCESS)
            return -1;
        memcpy(y, py, ybytes);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_logits);
    }
    return 0;
}

static int map_read(VulkanCtx* ctx, void* mem, void* dst, size_t nbytes)
{
    void* p = vk_map_slot(ctx, mem, 0);
    if (p) {
        memcpy(dst, p, nbytes);
        return 0;
    }
    {
        VkBuffer srcb = mem_to_buf(ctx, mem);
        if (!srcb || stage_copy(ctx, srcb, NULL, nbytes, 0) != 0) return -1;
        memcpy(dst, ctx->map_stage, nbytes);
        return 0;
    }
}

int vulkan_k_embed_q4k(VulkanCtx* ctx, float* host_y, const uint8_t* table,
                       uint32_t token, uint32_t hidden)
{
    if (!ctx || !ctx->embed_ready || !table || hidden == 0 || (hidden % 256) != 0)
        return -1;
    if ((size_t)hidden * 4 > ctx->x_bytes) return -1;
    uint32_t nb = hidden / 256;
    size_t rowb = (size_t)nb * 144;
    const uint8_t* row = table + (size_t)token * rowb;
    if (rowb > ctx->emb_bytes) return -1;
    if (map_copy(ctx, ctx->mem_emb, row, rowb) != 0) return -1;

    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->embed_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->embed_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->embed_pipe_layout, 0, 1, &ds, 0, NULL);
    EmbedPush push;
    push.hidden = hidden;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->embed_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, nb, 1, 1);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    ctx->x_on_dev = 1;
    if (host_y)
        return map_read(ctx, ctx->mem_x, host_y, (size_t)hidden * 4);
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
    cmd_rmsnorm(ctx, cmd, hidden, eps, 0);
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
    cmd_rmsnorm(ctx, cmd, hidden, eps, 0);
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

static void cmd_rope(VulkanCtx* ctx, VkCommandBuffer cmd, void* dset,
                     uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                     uint32_t mode, float theta)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rope_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rope_pipe_layout, 0, 1, &ds, 0, NULL);
    RopePush push;
    memset(&push, 0, sizeof(push));
    push.n_heads = n_heads;
    push.head_dim = head_dim;
    push.pos = pos;
    push.mode = mode;
    push.theta = theta;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->rope_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, n_heads, 1, 1);
}

static void cmd_attn(VulkanCtx* ctx, VkCommandBuffer cmd, uint32_t layer, uint32_t pos)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->attn_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->attn_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->attn_pipe_layout, 0, 1, &ds, 0, NULL);
    AttnPush push;
    push.n_heads = ctx->n_heads;
    push.n_kv_heads = ctx->n_kv_heads;
    push.head_dim = ctx->head_dim;
    push.kv_dim = ctx->kv_dim;
    push.max_seq = ctx->max_seq;
    push.pos = pos;
    push.k_slot = layer;
    push.v_slot = ctx->n_blocks + layer;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->attn_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, ctx->n_heads, 1, 1);
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

    cmd_rmsnorm(ctx, cmd, hidden, eps, 0);
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

static int alloc_multi_ds(VulkanCtx* ctx, const char* spv, size_t push_sz,
                          uint32_t nsets, VkBuffer triples[][3], size_t ranges[][3],
                          void** out_shader, void** out_dsl, void** out_pl, void** out_pipe,
                          void** out_pool, void** out_sets)
{
    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;
    char err[256];
    VkShaderModule sh = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool0 = VK_NULL_HANDLE;
    VkDescriptorSet dset0 = VK_NULL_HANDLE;
    if (create_ssbo_pipeline(ctx, spv, push_sz, &sh, &dsl, &pl, &pipe, &pool0, &dset0,
                             triples[0], ranges[0], err, sizeof(err)) != 0) {
        ylog_warn("vulkan: %s failed (%s)", spv, err);
        return -1;
    }
    a->DestroyDescriptorPool(dev, pool0, NULL);
    VkDescriptorPoolSize dps = {0};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 3u * nsets;
    VkDescriptorPoolCreateInfo dpi = {0};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = nsets;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &dps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (a->CreateDescriptorPool(dev, &dpi, NULL, &pool) != VK_SUCCESS) return -1;
    VkDescriptorSetLayout* layouts = (VkDescriptorSetLayout*)malloc(sizeof(*layouts) * nsets);
    VkDescriptorSet* sets = (VkDescriptorSet*)malloc(sizeof(*sets) * nsets);
    if (!layouts || !sets) { free(layouts); free(sets); return -1; }
    uint32_t i;
    for (i = 0; i < nsets; i++) layouts[i] = dsl;
    VkDescriptorSetAllocateInfo dai = {0};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool;
    dai.descriptorSetCount = nsets;
    dai.pSetLayouts = layouts;
    if (a->AllocateDescriptorSets(dev, &dai, sets) != VK_SUCCESS) {
        free(layouts); free(sets);
        return -1;
    }
    for (i = 0; i < nsets; i++) {
        VkDescriptorBufferInfo bis[3];
        VkWriteDescriptorSet writes[3];
        memset(bis, 0, sizeof(bis));
        memset(writes, 0, sizeof(writes));
        uint32_t b;
        for (b = 0; b < 3; b++) {
            bis[b].buffer = triples[i][b];
            bis[b].range = ranges[i][b];
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = sets[i];
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &bis[b];
        }
        a->UpdateDescriptorSets(dev, 3, writes, 0, NULL);
        out_sets[i] = (void*)sets[i];
    }
    free(layouts);
    free(sets);
    *out_shader = (void*)sh;
    *out_dsl = (void*)dsl;
    *out_pl = (void*)pl;
    *out_pipe = (void*)pipe;
    *out_pool = (void*)pool;
    return 0;
}

static int setup_block_ops(VulkanCtx* ctx)
{
    if (!ctx || !ctx->fuse_ready || !ctx->buf_x || !ctx->buf_y || !ctx->buf_o0 || !ctx->buf_kv)
        return -1;
    VkBuffer bx = (VkBuffer)ctx->buf_x;
    VkBuffer by = (VkBuffer)ctx->buf_y;
    VkBuffer bo0 = (VkBuffer)ctx->buf_o0;
    VkBuffer bo1 = (VkBuffer)ctx->buf_o1;
    VkBuffer bo2 = (VkBuffer)ctx->buf_o2;
    VkBuffer bkv = (VkBuffer)ctx->buf_kv;
    size_t xb = ctx->x_bytes;
    size_t yb = ctx->y_bytes;
    size_t kvb = ctx->kv_bytes;

    {
        VkBuffer triples[2][3] = { { bx, by, bx }, { bx, bo0, bx } };
        size_t ranges[2][3] = { { xb, yb, xb }, { xb, yb, xb } };
        void* sets[2];
        if (alloc_multi_ds(ctx, "add.spv", sizeof(AddPush), 2, triples, ranges,
                           &ctx->add_shader, &ctx->add_desc_layout, &ctx->add_pipe_layout,
                           &ctx->add_pipeline, &ctx->add_desc_pool, sets) != 0)
            return -1;
        ctx->add_ds_xy = sets[0];
        ctx->add_ds_xo = sets[1];
        ctx->add_ready = 1;
    }

    ctx->bias_bytes = (size_t)ctx->hidden * 4u * 3u;
    if (ctx->max_out > ctx->hidden)
        ctx->bias_bytes = (size_t)ctx->max_out * 4u * 3u;
    {
        VkBuffer bb = VK_NULL_HANDLE;
        VkDeviceMemory mb = VK_NULL_HANDLE;
        if (create_host_buffer(ctx, ctx->bias_bytes, &bb, &mb, &ctx->map_bias) != 0) return -1;
        ctx->buf_bias = (void*)bb;
        ctx->mem_bias = (void*)mb;
        VkBuffer triples[3][3] = {
            { bo0, bb, bb }, { bo1, bb, bb }, { bo2, bb, bb }
        };
        size_t ranges[3][3] = {
            { yb, ctx->bias_bytes, ctx->bias_bytes },
            { yb, ctx->bias_bytes, ctx->bias_bytes },
            { yb, ctx->bias_bytes, ctx->bias_bytes }
        };
        void* sets[3];
        if (alloc_multi_ds(ctx, "bias.spv", sizeof(BiasPush), 3, triples, ranges,
                           &ctx->bias_shader, &ctx->bias_desc_layout, &ctx->bias_pipe_layout,
                           &ctx->bias_pipeline, &ctx->bias_desc_pool, sets) != 0)
            return -1;
        ctx->bias_ds_q = sets[0];
        ctx->bias_ds_k = sets[1];
        ctx->bias_ds_v = sets[2];
        ctx->bias_ready = 1;
    }

    {
        VkBuffer triples[2][3] = { { bo1, bkv, bkv }, { bo2, bkv, bkv } };
        size_t ranges[2][3] = { { yb, kvb, kvb }, { yb, kvb, kvb } };
        void* sets[2];
        if (alloc_multi_ds(ctx, "store_kv.spv", sizeof(StoreKvPush), 2, triples, ranges,
                           &ctx->skv_shader, &ctx->skv_desc_layout, &ctx->skv_pipe_layout,
                           &ctx->skv_pipeline, &ctx->skv_desc_pool, sets) != 0)
            return -1;
        ctx->skv_ds_k = sets[0];
        ctx->skv_ds_v = sets[1];
        ctx->skv_ready = 1;
    }

    ctx->block_ready = (ctx->add_ready && ctx->skv_ready && ctx->rope_ready &&
                        ctx->attn_ready && ctx->swi_ready && ctx->attn_o_ready) ? 1 : 0;
    ylog_info("vulkan: block_ops add=%d bias=%d skv=%d block=%d",
              ctx->add_ready, ctx->bias_ready, ctx->skv_ready, ctx->block_ready);
    return 0;
}

int vulkan_attn_setup(VulkanCtx* ctx, uint32_t n_blocks, uint32_t max_seq,
                      uint32_t kv_dim, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t head_dim, char* err, size_t errlen)
{
    if (!ctx || !ctx->device || ctx->host_shim) return -1;
    if (!ctx->fuse_ready || !ctx->buf_o0 || !ctx->buf_o2) {
        if (err && errlen) snprintf(err, errlen, "fuse buffers missing");
        return -1;
    }
    ctx->n_blocks = n_blocks;
    ctx->max_seq = max_seq;
    ctx->kv_dim = kv_dim;
    ctx->n_heads = n_heads;
    ctx->n_kv_heads = n_kv_heads;
    ctx->head_dim = head_dim;
    ctx->kv_slots = 2u * n_blocks + 1u;
    ctx->kv_bytes = (size_t)ctx->kv_slots * max_seq * kv_dim * 4;

    VkBuffer bkv = VK_NULL_HANDLE;
    VkDeviceMemory mkv = VK_NULL_HANDLE;
    if (create_scratch_buffer(ctx, ctx->kv_bytes, &bkv, &mkv, &ctx->map_kv) != 0) {
        if (err && errlen) snprintf(err, errlen, "kv buffer alloc %zu", ctx->kv_bytes);
        return -1;
    }
    ctx->buf_kv = (void*)bkv;
    ctx->mem_kv = (void*)mkv;
    if (ctx->map_kv) memset(ctx->map_kv, 0, ctx->kv_bytes);
    else {
        VulkanApi* a = vulkan_api();
        VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
        VkCommandBufferBeginInfo bi = {0};
        if (!a->CmdFillBuffer || !ctx->cmd) return -1;
        a->ResetCommandBuffer(cmd, 0);
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        a->BeginCommandBuffer(cmd, &bi);
        a->CmdFillBuffer(cmd, bkv, 0, ctx->kv_bytes, 0);
        if (a->EndCommandBuffer(cmd) != VK_SUCCESS) return -1;
        if (submit_and_wait(ctx, cmd) != 0) return -1;
    }

    VkBuffer bufs[3] = { (VkBuffer)ctx->buf_o0, bkv, (VkBuffer)ctx->buf_o2 };
    size_t ranges[3] = { ctx->y_bytes, ctx->kv_bytes, ctx->y_bytes };
    VkShaderModule sh = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (create_ssbo_pipeline(ctx, "attn_decode.spv", sizeof(AttnPush),
                             &sh, &dsl, &pl, &pipe, &pool, &dset,
                             bufs, ranges, err, errlen) != 0) {
        VulkanApi* a = vulkan_api();
        a->DestroyBuffer((VkDevice)ctx->device, bkv, NULL);
        a->FreeMemory((VkDevice)ctx->device, mkv, NULL);
        ctx->buf_kv = NULL;
        ctx->mem_kv = NULL;
        return -1;
    }
    ctx->attn_shader = (void*)sh;
    ctx->attn_desc_layout = (void*)dsl;
    ctx->attn_pipe_layout = (void*)pl;
    ctx->attn_pipeline = (void*)pipe;
    ctx->attn_desc_pool = (void*)pool;
    ctx->attn_desc_set = (void*)dset;
    ctx->attn_ready = 1;
    ylog_info("vulkan: attn_decode ready heads=%u kv=%u seq=%u cache=%zuMB",
              n_heads, n_kv_heads, max_seq, ctx->kv_bytes / (1024 * 1024));

    /* RoPE: binding0 = o0/o1; binding1/2 占位 */
    {
        char rerr[256];
        VulkanApi* a = vulkan_api();
        VkDevice dev = (VkDevice)ctx->device;
        VkBuffer o0 = (VkBuffer)ctx->buf_o0;
        VkBuffer o1 = (VkBuffer)ctx->buf_o1;
        VkBuffer rbufs[3] = { o0, o0, o0 };
        size_t rranges[3] = { ctx->y_bytes, ctx->y_bytes, ctx->y_bytes };
        VkShaderModule rsh = VK_NULL_HANDLE;
        VkDescriptorSetLayout rdsl = VK_NULL_HANDLE;
        VkPipelineLayout rpl = VK_NULL_HANDLE;
        VkPipeline rpipe = VK_NULL_HANDLE;
        VkDescriptorPool rpool = VK_NULL_HANDLE;
        VkDescriptorSet rdset = VK_NULL_HANDLE;
        if (create_ssbo_pipeline(ctx, "rope.spv", sizeof(RopePush),
                                 &rsh, &rdsl, &rpl, &rpipe, &rpool, &rdset,
                                 rbufs, rranges, rerr, sizeof(rerr)) != 0) {
            ylog_warn("vulkan: rope pipeline failed (%s)", rerr);
        } else {
            a->DestroyDescriptorPool(dev, rpool, NULL);
            VkDescriptorPoolSize dps = {0};
            dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            dps.descriptorCount = 6;
            VkDescriptorPoolCreateInfo dpi = {0};
            dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpi.maxSets = 2;
            dpi.poolSizeCount = 1;
            dpi.pPoolSizes = &dps;
            if (a->CreateDescriptorPool(dev, &dpi, NULL, &rpool) == VK_SUCCESS) {
                VkDescriptorSetLayout layouts[2] = { rdsl, rdsl };
                VkDescriptorSet sets[2];
                VkDescriptorSetAllocateInfo dai = {0};
                dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dai.descriptorPool = rpool;
                dai.descriptorSetCount = 2;
                dai.pSetLayouts = layouts;
                if (a->AllocateDescriptorSets(dev, &dai, sets) == VK_SUCCESS) {
                    VkBuffer xb[2] = { o0, o1 };
                    uint32_t si;
                    for (si = 0; si < 2; si++) {
                        VkDescriptorBufferInfo bis[3];
                        VkWriteDescriptorSet writes[3];
                        memset(bis, 0, sizeof(bis));
                        memset(writes, 0, sizeof(writes));
                        uint32_t b;
                        for (b = 0; b < 3; b++) {
                            bis[b].buffer = (b == 0) ? xb[si] : xb[si];
                            bis[b].range = ctx->y_bytes;
                            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            writes[b].dstSet = sets[si];
                            writes[b].dstBinding = b;
                            writes[b].descriptorCount = 1;
                            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            writes[b].pBufferInfo = &bis[b];
                        }
                        a->UpdateDescriptorSets(dev, 3, writes, 0, NULL);
                    }
                    ctx->rope_shader = (void*)rsh;
                    ctx->rope_desc_layout = (void*)rdsl;
                    ctx->rope_pipe_layout = (void*)rpl;
                    ctx->rope_pipeline = (void*)rpipe;
                    ctx->rope_desc_pool = (void*)rpool;
                    ctx->rope_ds_q = (void*)sets[0];
                    ctx->rope_ds_k = (void*)sets[1];
                    ctx->rope_ready = 1;
                    ylog_info("vulkan: rope ready");
                }
            }
        }
    }
    (void)setup_block_ops(ctx);
    ctx->use_gpu_rope = (!ctx->integrated_gpu && ctx->rope_ready && ctx->skv_ready) ? 1 : 0;
    if (ctx->use_gpu_rope)
        ylog_info("vulkan: gpu_rope + single-submit block enabled");
    return 0;
}

int vulkan_k_attn_decode(VulkanCtx* ctx,
                         const float* q, const float* k, const float* v, float* att_out,
                         uint32_t layer, uint32_t pos,
                         uint16_t* host_k_row, uint16_t* host_v_row,
                         uint64_t off_o)
{
    if (!ctx || !ctx->attn_ready || !q || !k || !v || !att_out) return -1;
    if (pos >= ctx->max_seq || layer > ctx->n_blocks) return -1;

    int fuse_o = (off_o != (uint64_t)~0ull);
    if (fuse_o) {
        if (!ctx->attn_o_ready || !ctx->gemv_ds_xo || !ctx->wq_resident) return -1;
        if (off_o > 0xffffffffull) return -1;
        uint32_t hidden = ctx->n_heads * ctx->head_dim;
        if (hidden > ctx->max_in || hidden > ctx->max_out) return -1;
        size_t wbytes = (size_t)hidden * ((size_t)(hidden / 256) * 144);
        if ((hidden % 256) != 0 || off_o + wbytes > ctx->wq_bytes) return -1;
    }

    uint32_t kv_dim = ctx->kv_dim;
    uint32_t hidden = ctx->n_heads * ctx->head_dim;
    size_t qbytes = (size_t)hidden * 4;
    size_t rowb = (size_t)kv_dim * 4;
    uint32_t k_slot = layer;
    uint32_t v_slot = ctx->n_blocks + layer;
    size_t k_off = ((size_t)k_slot * ctx->max_seq + pos) * kv_dim * 4;
    size_t v_off = ((size_t)v_slot * ctx->max_seq + pos) * kv_dim * 4;

    VulkanApi* a = vulkan_api();
    VkDevice dev = (VkDevice)ctx->device;

    if (map_copy(ctx, ctx->mem_o0, q, qbytes) != 0) return -1;

    if (ctx->map_kv) {
        memcpy((uint8_t*)ctx->map_kv + k_off, k, rowb);
        memcpy((uint8_t*)ctx->map_kv + v_off, v, rowb);
    } else {
        void* p = NULL;
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_kv, k_off, rowb, 0, &p) != VK_SUCCESS)
            return -1;
        memcpy(p, k, rowb);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_kv);
        if (a->MapMemory(dev, (VkDeviceMemory)ctx->mem_kv, v_off, rowb, 0, &p) != VK_SUCCESS)
            return -1;
        memcpy(p, v, rowb);
        a->UnmapMemory(dev, (VkDeviceMemory)ctx->mem_kv);
    }

    if (host_k_row && host_v_row) {
        uint32_t j;
        for (j = 0; j < kv_dim; j++) {
            host_k_row[j] = f32_to_f16(k[j]);
            host_v_row[j] = f32_to_f16(v[j]);
        }
    }

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    a->ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    a->BeginCommandBuffer(cmd, &bi);
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->attn_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)ctx->attn_desc_set;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->attn_pipe_layout, 0, 1, &ds, 0, NULL);
    AttnPush push;
    push.n_heads = ctx->n_heads;
    push.n_kv_heads = ctx->n_kv_heads;
    push.head_dim = ctx->head_dim;
    push.kv_dim = ctx->kv_dim;
    push.max_seq = ctx->max_seq;
    push.pos = pos;
    push.k_slot = k_slot;
    push.v_slot = v_slot;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->attn_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, ctx->n_heads, 1, 1);

    if (fuse_o) {
        cmd_barrier_compute(cmd);
        cmd_gemv(ctx, cmd, ctx->gemv_ds_xo, hidden, hidden, (uint32_t)off_o);
    }

    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;
    if (fuse_o)
        return map_read(ctx, ctx->mem_y, att_out, qbytes);
    return map_read(ctx, ctx->mem_o2, att_out, qbytes);
}

/* rmsnorm+QKV; 可选 bias/qk-norm; CPU RoPE; KV 行; attn(+O)
 * Q 留在 o0。RoPE 走 host 与 engine 一致(GPU rope 在部分 iGPU 上 push/数值不稳)。 */
int vulkan_fused_qkv_rope_attn(VulkanCtx* ctx,
                               const float* x, const float* wn, uint32_t hidden, float eps,
                               float* out, uint32_t kv_dim,
                               uint64_t off_q, uint64_t off_k, uint64_t off_v, uint64_t off_o,
                               uint32_t layer, uint32_t pos, uint32_t rope_mode, float theta,
                               const float* bq, const float* bk, const float* bv,
                               const uint8_t* qnorm, uint32_t qnorm_dtype,
                               const uint8_t* knorm, uint32_t knorm_dtype,
                               uint16_t* host_k_row, uint16_t* host_v_row)
{
    if (!ctx || !ctx->fuse_ready || !ctx->wq_resident ||
        !ctx->attn_ready || !x || !wn || !out)
        return -1;
    if (hidden != ctx->n_heads * ctx->head_dim || kv_dim != ctx->kv_dim) return -1;
    if (hidden > ctx->max_in || hidden > ctx->max_out || kv_dim > ctx->max_out) return -1;
    if (pos >= ctx->max_seq || layer > ctx->n_blocks) return -1;
    if (off_q > 0xffffffffull || off_k > 0xffffffffull || off_v > 0xffffffffull)
        return -1;

    int fuse_o = (off_o != (uint64_t)~0ull);
    if (fuse_o) {
        if (!ctx->attn_o_ready || !ctx->gemv_ds_xo) return -1;
        if (off_o > 0xffffffffull) return -1;
        size_t wbytes = (size_t)hidden * ((size_t)(hidden / 256) * 144);
        if ((hidden % 256) != 0 || off_o + wbytes > ctx->wq_bytes) return -1;
    }

    size_t xb = (size_t)hidden * 4;
    size_t rowb = (size_t)kv_dim * 4;
    uint32_t k_slot = layer;
    uint32_t v_slot = ctx->n_blocks + layer;
    size_t k_off = ((size_t)k_slot * ctx->max_seq + pos) * kv_dim * 4;
    size_t v_off = ((size_t)v_slot * ctx->max_seq + pos) * kv_dim * 4;

    if (map_copy(ctx, ctx->mem_x, x, xb) != 0) return -1;
    if (map_copy(ctx, ctx->mem_wn, wn, xb) != 0) return -1;

    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    a->ResetCommandBuffer(cmd, 0);
    a->BeginCommandBuffer(cmd, &bi);
    cmd_rmsnorm(ctx, cmd, hidden, eps, 0);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, hidden, hidden, (uint32_t)off_q);
    cmd_gemv(ctx, cmd, ctx->gemv_ds1, kv_dim, hidden, (uint32_t)off_k);
    cmd_gemv(ctx, cmd, ctx->gemv_ds2, kv_dim, hidden, (uint32_t)off_v);
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;

    /* bias / qk-norm / RoPE 在 host 就地改 o* */
    {
        uint32_t j, hh;
        float* q = (float*)ctx->map_o0;
        if (!q) return -1;
        if (bq) for (j = 0; j < hidden; j++) q[j] += bq[j];
        if (qnorm) {
            for (hh = 0; hh < ctx->n_heads; hh++)
                rmsnorm(q + (size_t)hh * ctx->head_dim, q + (size_t)hh * ctx->head_dim,
                        qnorm, ctx->head_dim, eps, qnorm_dtype);
        }
        for (hh = 0; hh < ctx->n_heads; hh++) {
            if (rope_mode)
                rope_inplace_qwen(q + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
            else
                rope_inplace(q + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
        }

        float* k = (float*)ctx->map_o1;
        if (!k) return -1;
        if (bk) for (j = 0; j < kv_dim; j++) k[j] += bk[j];
        if (knorm) {
            for (hh = 0; hh < ctx->n_kv_heads; hh++)
                rmsnorm(k + (size_t)hh * ctx->head_dim, k + (size_t)hh * ctx->head_dim,
                        knorm, ctx->head_dim, eps, knorm_dtype);
        }
        for (hh = 0; hh < ctx->n_kv_heads; hh++) {
            if (rope_mode)
                rope_inplace_qwen(k + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
            else
                rope_inplace(k + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
        }

        if (bv) {
            float* v = (float*)ctx->map_o2;
            if (!v) return -1;
            for (j = 0; j < kv_dim; j++) v[j] += bv[j];
        }
    }

    /* o1=K o2=V → GPU KV; 可选同步 host f16 */
    {
        if (!ctx->map_o1 || !ctx->map_kv) return -1;
        memcpy((uint8_t*)ctx->map_kv + k_off, ctx->map_o1, rowb);
        if (host_k_row) {
            uint32_t j;
            const float* kf = (const float*)ctx->map_o1;
            for (j = 0; j < kv_dim; j++) host_k_row[j] = f32_to_f16(kf[j]);
        }

        memcpy((uint8_t*)ctx->map_kv + v_off, ctx->map_o2, rowb);
        if (host_v_row) {
            uint32_t j;
            const float* vf = (const float*)ctx->map_o2;
            for (j = 0; j < kv_dim; j++) host_v_row[j] = f32_to_f16(vf[j]);
        }
    }

    a->ResetCommandBuffer(cmd, 0);
    a->BeginCommandBuffer(cmd, &bi);
    cmd_attn(ctx, cmd, layer, pos);
    if (fuse_o) {
        cmd_barrier_compute(cmd);
        cmd_gemv(ctx, cmd, ctx->gemv_ds_xo, hidden, hidden, (uint32_t)off_o);
    }
    a->EndCommandBuffer(cmd);
    if (submit_and_wait(ctx, cmd) != 0) return -1;
    if (fuse_o)
        return map_read(ctx, ctx->mem_y, out, xb);
    return map_read(ctx, ctx->mem_o2, out, xb);
}

static void cmd_add(VulkanCtx* ctx, VkCommandBuffer cmd, void* dset, uint32_t n)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->add_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->add_pipe_layout, 0, 1, &ds, 0, NULL);
    AddPush push;
    push.n = n;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->add_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, (n + 255u) / 256u, 1, 1);
}

static void cmd_bias(VulkanCtx* ctx, VkCommandBuffer cmd, void* dset, uint32_t n, uint32_t off)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->bias_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->bias_pipe_layout, 0, 1, &ds, 0, NULL);
    BiasPush push;
    push.n = n;
    push.bias_off = off;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->bias_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, (n + 255u) / 256u, 1, 1);
}

static void cmd_store_kv(VulkanCtx* ctx, VkCommandBuffer cmd, void* dset,
                         uint32_t n, uint32_t dst_off)
{
    VulkanApi* a = vulkan_api();
    a->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->skv_pipeline);
    VkDescriptorSet ds = (VkDescriptorSet)dset;
    a->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->skv_pipe_layout, 0, 1, &ds, 0, NULL);
    StoreKvPush push;
    push.n = n;
    push.dst_off = dst_off;
    a->CmdPushConstants(cmd, (VkPipelineLayout)ctx->skv_pipe_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    a->CmdDispatch(cmd, (n + 255u) / 256u, 1, 1);
}

void vulkan_mark_x_host(VulkanCtx* ctx)
{
    if (ctx) ctx->x_on_dev = 0;
}

int vulkan_upload_x(VulkanCtx* ctx, const float* x, uint32_t hidden)
{
    if (!ctx || !x || hidden == 0) return -1;
    size_t nbytes = (size_t)hidden * 4;
    if (nbytes > ctx->x_bytes) return -1;
    if (map_copy(ctx, ctx->mem_x, x, nbytes) != 0) return -1;
    ctx->x_on_dev = 1;
    return 0;
}

int vulkan_sync_x_to_host(VulkanCtx* ctx, float* host_x, uint32_t hidden)
{
    if (!ctx || !host_x || !ctx->mem_x) return -1;
    if (!ctx->x_on_dev) return 0;
    return map_read(ctx, ctx->mem_x, host_x, (size_t)hidden * 4);
}

/* 整层两次 submit: 激活留 buf_x。无 qk-norm。
 * off_* 为 GPU 相对偏移(调用方已用 wq_off / stream 折算)。 */
int vulkan_fused_block(VulkanCtx* ctx,
                       float* host_x, int upload_x,
                       const float* wn1, const float* wn2,
                       uint32_t hidden, float eps, float theta, uint32_t rope_mode,
                       uint32_t kv_dim, uint32_t inter,
                       uint64_t off_q, uint64_t off_k, uint64_t off_v, uint64_t off_o,
                       uint64_t off_g, uint64_t off_u, uint64_t off_d,
                       uint32_t layer, uint32_t pos,
                       const float* bq, const float* bk, const float* bv,
                       uint16_t* host_k_row, uint16_t* host_v_row,
                       int sync_host)
{
    if (!ctx || !ctx->block_ready || !ctx->wq_resident) return -1;
    if (!ctx->norm_ready && (!wn1 || !wn2)) return -1;
    if (hidden != ctx->n_heads * ctx->head_dim || kv_dim != ctx->kv_dim) return -1;
    if (hidden > ctx->max_in || inter > ctx->max_out || hidden > ctx->max_out) return -1;
    if (pos >= ctx->max_seq || layer > ctx->n_blocks) return -1;
    if (off_q > 0xffffffffull || off_k > 0xffffffffull || off_v > 0xffffffffull ||
        off_o > 0xffffffffull || off_g > 0xffffffffull || off_u > 0xffffffffull ||
        off_d > 0xffffffffull)
        return -1;

    /* 流式: 确保本层已上传; 勿再减 stream_base(调用方 off 已是相对) */
    if (ctx->wq_stream) {
        if (ctx->stream_layer != layer) {
            if (vulkan_stream_layer(ctx, layer) != 0) return -1;
        }
    }

    size_t xb = (size_t)hidden * 4;
    uint32_t w1_off = ctx->norm_ready ? norm_w_off1(ctx, layer) : 0u;
    uint32_t w2_off = ctx->norm_ready ? norm_w_off2(ctx, layer) : 0u;

    if ((upload_x || !ctx->x_on_dev) && !ctx->cmd_open) {
        if (!host_x) return -1;
        if (map_copy(ctx, ctx->mem_x, host_x, xb) != 0) return -1;
        ctx->x_on_dev = 1;
    } else if (!ctx->x_on_dev && !ctx->cmd_open) {
        return -1;
    }
    if (!ctx->norm_ready) {
        if (map_copy(ctx, ctx->mem_wn, wn1, xb) != 0) return -1;
    }

    /* 打包 bias 到 buf_bias: [0,h) q, [h,h+kv) k, [h+kv,...) v */
    uint32_t bq_off = 0, bk_off = hidden, bv_off = hidden + kv_dim;
    if (bq || bk || bv) {
        if (!ctx->bias_ready) return -1;
        if (ctx->map_bias) {
            memset(ctx->map_bias, 0, ctx->bias_bytes);
            if (bq) memcpy((float*)ctx->map_bias + bq_off, bq, xb);
            if (bk) memcpy((float*)ctx->map_bias + bk_off, bk, (size_t)kv_dim * 4);
            if (bv) memcpy((float*)ctx->map_bias + bv_off, bv, (size_t)kv_dim * 4);
        } else {
            void* p = NULL;
            VulkanApi* a0 = vulkan_api();
            if (a0->MapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->mem_bias,
                              0, ctx->bias_bytes, 0, &p) != VK_SUCCESS)
                return -1;
            memset(p, 0, ctx->bias_bytes);
            if (bq) memcpy((float*)p + bq_off, bq, xb);
            if (bk) memcpy((float*)p + bk_off, bk, (size_t)kv_dim * 4);
            if (bv) memcpy((float*)p + bv_off, bv, (size_t)kv_dim * 4);
            a0->UnmapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->mem_bias);
        }
    }

    VulkanApi* a = vulkan_api();
    VkCommandBuffer cmd = (VkCommandBuffer)ctx->cmd;
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    int gpu_rope = ctx->use_gpu_rope && ctx->rope_ready && ctx->skv_ready &&
                   !host_k_row && !host_v_row;
    int batch = ctx->token_batch && !sync_host && gpu_rope && !getenv("YLLM_VK_NOTOKB");
    int last_layer = (layer >= ctx->n_blocks);
    int chain = batch && ctx->tokb_chain && ctx->cmd_n > 1u && !ctx->map_x &&
                !getenv("YLLM_VK_TOKB_ONECB");
    uint32_t tgrp = tokb_group(ctx);
    int defer_lm = !chain && batch && last_layer && ctx->lm_fused;
    int defer_submit = !chain && batch && !last_layer && ((layer % tgrp) != 0u);
    int finish = 0;

    if (chain) {
        if (ctx->cmd_open) vulkan_gpu_discard(ctx);
        cmd = tokb_cmd(ctx);
        a->ResetCommandBuffer(cmd, 0);
        a->BeginCommandBuffer(cmd, &bi);
        if (ctx->tokb_i == 0) {
            cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_x);
            cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wq);
            cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wn);
            if (ctx->buf_bias)
                cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_bias);
        }
        finish = last_layer && !ctx->lm_fused;
    } else if (!gpu_rope) {
        if (ctx->cmd_open) vulkan_gpu_discard(ctx);
        a->ResetCommandBuffer(cmd, 0);
        a->BeginCommandBuffer(cmd, &bi);
        cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_x);
        cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wq);
        cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wn);
    } else if (!ctx->cmd_open) {
        a->ResetCommandBuffer(cmd, 0);
        a->BeginCommandBuffer(cmd, &bi);
        ctx->cmd_open = 1;
        cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_x);
        cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wq);
        cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_wn);
        if (ctx->buf_bias)
            cmd_barrier_host_to_shader(cmd, (VkBuffer)ctx->buf_bias);
    } else {
        cmd_barrier_layer(ctx, cmd);
    }

    cmd_rmsnorm(ctx, cmd, hidden, eps, w1_off);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, hidden, hidden, (uint32_t)off_q);
    cmd_gemv(ctx, cmd, ctx->gemv_ds1, kv_dim, hidden, (uint32_t)off_k);
    cmd_gemv(ctx, cmd, ctx->gemv_ds2, kv_dim, hidden, (uint32_t)off_v);
    cmd_barrier_compute(cmd);
    if (bq) cmd_bias(ctx, cmd, ctx->bias_ds_q, hidden, bq_off);
    if (bk) cmd_bias(ctx, cmd, ctx->bias_ds_k, kv_dim, bk_off);
    if (bv) cmd_bias(ctx, cmd, ctx->bias_ds_v, kv_dim, bv_off);

    if (gpu_rope) {
        cmd_barrier_compute(cmd);
        cmd_rope(ctx, cmd, ctx->rope_ds_q, ctx->n_heads, ctx->head_dim,
                 pos, rope_mode, theta);
        cmd_barrier_compute(cmd);
        cmd_rope(ctx, cmd, ctx->rope_ds_k, ctx->n_kv_heads, ctx->head_dim,
                 pos, rope_mode, theta);
        cmd_barrier_compute(cmd);
        {
            uint32_t k_el = (uint32_t)(((size_t)layer * ctx->max_seq + pos) * kv_dim);
            uint32_t v_el = (uint32_t)(((size_t)(ctx->n_blocks + layer) * ctx->max_seq + pos) * kv_dim);
            cmd_store_kv(ctx, cmd, ctx->skv_ds_k, kv_dim, k_el);
            cmd_store_kv(ctx, cmd, ctx->skv_ds_v, kv_dim, v_el);
        }
        cmd_barrier_compute(cmd);
        if (!ctx->norm_ready) {
            if (map_copy(ctx, ctx->mem_wn, wn2, xb) != 0) return -1;
        }
    } else {
        a->EndCommandBuffer(cmd);
        if (submit_and_wait(ctx, cmd) != 0) return -1;

        /* CPU RoPE + 可选 host KV f16 */
        {
            uint32_t hh;
            float* q = (float*)ctx->map_o0;
            if (!q) return -1;
            for (hh = 0; hh < ctx->n_heads; hh++) {
                if (rope_mode)
                    rope_inplace_qwen(q + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
                else
                    rope_inplace(q + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
            }

            float* k = (float*)ctx->map_o1;
            if (!k) return -1;
            for (hh = 0; hh < ctx->n_kv_heads; hh++) {
                if (rope_mode)
                    rope_inplace_qwen(k + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
                else
                    rope_inplace(k + (size_t)hh * ctx->head_dim, ctx->head_dim, pos, theta);
            }
            if (host_k_row) {
                uint32_t j;
                for (j = 0; j < kv_dim; j++) host_k_row[j] = f32_to_f16(k[j]);
            }

            if (host_v_row) {
                float* vf = (float*)ctx->map_o2;
                if (!vf) return -1;
                uint32_t j;
                for (j = 0; j < kv_dim; j++) host_v_row[j] = f32_to_f16(vf[j]);
            }
        }

        if (map_copy(ctx, ctx->mem_wn, wn2, xb) != 0) return -1;

        if (ctx->cmd_open) vulkan_gpu_discard(ctx);
        a->ResetCommandBuffer(cmd, 0);
        a->BeginCommandBuffer(cmd, &bi);
        {
            uint32_t k_el = (uint32_t)(((size_t)layer * ctx->max_seq + pos) * kv_dim);
            uint32_t v_el = (uint32_t)(((size_t)(ctx->n_blocks + layer) * ctx->max_seq + pos) * kv_dim);
            cmd_store_kv(ctx, cmd, ctx->skv_ds_k, kv_dim, k_el);
            cmd_store_kv(ctx, cmd, ctx->skv_ds_v, kv_dim, v_el);
        }
        cmd_barrier_compute(cmd);
    }

    cmd_attn(ctx, cmd, layer, pos);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds_xo, hidden, hidden, (uint32_t)off_o);
    cmd_barrier_compute(cmd);
    cmd_add(ctx, cmd, ctx->add_ds_xy, hidden);
    cmd_barrier_compute(cmd);
    /* FFN: wn 已是 wn2 或分片偏移 w2_off */
    cmd_rmsnorm(ctx, cmd, hidden, eps, w2_off);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, inter, hidden, (uint32_t)off_g);
    cmd_gemv(ctx, cmd, ctx->gemv_ds1, inter, hidden, (uint32_t)off_u);
    cmd_barrier_compute(cmd);
    cmd_swiglu(ctx, cmd, inter);
    cmd_barrier_compute(cmd);
    cmd_gemv(ctx, cmd, ctx->gemv_ds0, hidden, inter, (uint32_t)off_d);
    cmd_barrier_compute(cmd);
    cmd_add(ctx, cmd, ctx->add_ds_xo, hidden);

    if (batch && defer_lm) {
        ctx->x_on_dev = 1;
        return 0;
    }
    if (defer_submit) {
        ctx->x_on_dev = 1;
        return 0;
    }

    if (sync_host && host_x)
        cmd_barrier_shader_to_host(cmd, (VkBuffer)ctx->buf_x);
    a->EndCommandBuffer(cmd);
    if (chain) {
        if (submit_chained(ctx, cmd, finish) != 0) return -1;
    } else if (submit_and_wait(ctx, cmd) != 0) {
        return -1;
    }
    ctx->cmd_open = 0;
    ctx->x_on_dev = 1;

    if (sync_host && host_x) {
        if (map_read(ctx, ctx->mem_x, host_x, xb) != 0) return -1;
        ctx->x_on_dev = 0;
    }
    return 0;
}
