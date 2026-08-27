/* device_vulkan.c — Vulkan 设备后端(Android / iOS·MoltenVK / PC) */
#include "device.h"
#include "yllm.h"
#include "llf.h"
#include "log.h"
#include "vulkan_ctx.h"
#include "vulkan_load.h"
#include "vulkan_compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matvec.h"

void vulkan_attach_fwd(Engine* e);
int vulkan_selftest_rmsnorm(VulkanCtx* ctx);
int vulkan_selftest_gemv_q4k(VulkanCtx* ctx);
int vulkan_selftest_gemv_q4k_real(VulkanCtx* ctx);
int vulkan_selftest_gemv_q6k(VulkanCtx* ctx);

static int tensor_out_in(const LlfTensorMeta* mt, uint32_t* out, uint32_t* in)
{
    if (mt->ndim < 2 || mt->shape[0] == 0 || mt->shape[1] == 0) return -1;
    uint32_t a = mt->shape[0], b = mt->shape[1];
    *out = a;
    *in = b;
    if (mt->dtype == DT_Q4K || mt->dtype == DT_Q6K) {
        size_t rowb = (mt->dtype == DT_Q4K) ? 144u : 210u;
        size_t e0 = (size_t)a * ((size_t)(b / 256) * rowb);
        size_t e1 = (size_t)b * ((size_t)(a / 256) * rowb);
        if (e0 == mt->size && e1 == mt->size) {
            /* 方阵歧义少见; 两侧都合法时取较大维为 out(vocab) */
            if (a >= b) { *out = a; *in = b; }
            else { *out = b; *in = a; }
        } else if (e0 != mt->size && e1 == mt->size) {
            *out = b; *in = a;
        } else if (e0 != mt->size && e1 != mt->size) {
            return -1;
        }
    }
    return 0;
}

/* Q4_K 行列可对调且字节相同(gemma O/down: 1536×4096 与 4096×1536 同体积).
 * 较大维=out 对 lm_head 对, 对 O/down 会把 in 收成 hidden, GPU gemv 与 CPU 不一致. */
static int tensor_out_in_slot(const LlfTensorMeta* mt, uint32_t slot, uint32_t hidden,
                              uint32_t* out, uint32_t* in)
{
    uint32_t a, b;
    if (tensor_out_in(mt, out, in) != 0) return -1;
    if (hidden == 0 || mt->ndim < 2) return 0;
    if (mt->dtype != DT_Q4K && mt->dtype != DT_Q6K) return 0;
    a = mt->shape[0];
    b = mt->shape[1];
    {
        size_t rowb = (mt->dtype == DT_Q4K) ? 144u : 210u;
        size_t e0 = (size_t)a * ((size_t)(b / 256) * rowb);
        size_t e1 = (size_t)b * ((size_t)(a / 256) * rowb);
        if (e0 != mt->size || e1 != mt->size) return 0;
    }
    if (slot == SLOT_O || slot == SLOT_DOWN || slot == SLOT_PLE_PROJ) {
        if (a == hidden) { *out = a; *in = b; }
        else if (b == hidden) { *out = b; *in = a; }
    } else {
        if (b == hidden) { *out = a; *in = b; }
        else if (a == hidden) { *out = b; *in = a; }
    }
    return 0;
}

/* bank=0 不填充. 张量不跨 SSBO 边界 */
static size_t wq_bank_align(size_t cursor, size_t nbytes, size_t bank)
{
    size_t used;
    if (bank == 0 || nbytes == 0 || nbytes > bank) return cursor;
    used = cursor % bank;
    if (used && used + nbytes > bank)
        cursor += bank - used;
    return cursor;
}

static size_t block_q4k_bytes(const LlModel* m, uint32_t layer)
{
    size_t sum = 0;
    uint32_t s, hidx, nt;
    if (layer >= m->n_layers) return 0;
    hidx = m->base_idx[layer];
    nt = m->dir[layer].n_tensors;
    for (s = 0; s < nt; s++) {
        const LlfTensorMeta* mt = &m->metas[hidx + s];
        uint32_t o, i;
        if (mt->dtype != DT_Q4K || mt->size == 0) continue;
        if (tensor_out_in_slot(mt, s, m->h.hidden, &o, &i) != 0 || (i % 256) != 0) continue;
        sum += (size_t)o * ((size_t)(i / 256) * 144);
    }
    return sum;
}

static size_t q4k_packed_bytes(const LlModel* m, size_t bank)
{
    size_t cursor = 0;
    uint32_t li, s;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        uint32_t hidx, nt;
        if (layer >= m->n_layers) continue;
        cursor = wq_bank_align(cursor, block_q4k_bytes(m, layer), bank);
        hidx = m->base_idx[layer];
        nt = m->dir[layer].n_tensors;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            uint32_t o, i;
            size_t nbytes;
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            if (tensor_out_in_slot(mt, s, m->h.hidden, &o, &i) != 0 || (i % 256) != 0) continue;
            nbytes = (size_t)o * ((size_t)(i / 256) * 144);
            cursor = wq_bank_align(cursor, nbytes, bank);
            cursor += nbytes;
        }
    }
    {
        uint32_t layer = m->h.n_blocks + 2;
        if (layer < m->n_layers) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
            uint32_t o, i;
            if (mt->size > 0 && tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                size_t nbytes = 0;
                if (mt->dtype == DT_Q4K)
                    nbytes = (size_t)o * ((size_t)(i / 256) * 144);
                else if (mt->dtype == DT_Q6K)
                    nbytes = (size_t)o * ((size_t)(i / 256) * 210);
                if (nbytes) {
                    if (nbytes != mt->size) nbytes = (size_t)mt->size;
                    cursor = wq_bank_align(cursor, nbytes, bank);
                    cursor += nbytes;
                }
            }
        }
    }
    return cursor;
}

static void scan_block_q4k(const LlModel* m, uint32_t* max_in, uint32_t* max_out,
                           size_t* total_wq)
{
    uint32_t hidden = m->h.hidden;
    *max_in = hidden;
    *max_out = hidden;
    *total_wq = 0;
    uint32_t li;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > BLOCK_TENSORS) nt = BLOCK_TENSORS;
        uint32_t s;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in_slot(mt, s, hidden, &o, &i) != 0) continue;
            if ((i % 256) != 0) continue;
            if (o > *max_out) *max_out = o;
            if (i > *max_in) *max_in = i;
            *total_wq += (size_t)o * ((size_t)(i / 256) * 144);
        }
    }
    /* lm_head 权体积计入 wq; 不抬 max_out(vocab 过大), gemv 分块写 logits */
    {
        uint32_t layer = m->h.n_blocks + 2;
        if (layer < m->n_layers) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
            if (mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    if (mt->dtype == DT_Q4K)
                        *total_wq += (size_t)o * ((size_t)(i / 256) * 144);
                    else if (mt->dtype == DT_Q6K)
                        *total_wq += (size_t)o * ((size_t)(i / 256) * 210);
                }
            }
        }
    }
}

static size_t max_block_q4k_bytes(const LlModel* m)
{
    size_t mx = 0;
    uint32_t li, s;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        size_t layer_sz = 0;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > BLOCK_TENSORS) nt = BLOCK_TENSORS;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in_slot(mt, s, m->h.hidden, &o, &i) != 0 || (i % 256) != 0) continue;
            layer_sz += (size_t)o * ((size_t)(i / 256) * 144);
        }
        if (layer_sz > mx) mx = layer_sz;
    }
    return mx;
}

static int pack_norm_slice(float* dst, const uint8_t* wbytes, uint32_t n, uint32_t dtype)
{
    if (dtype == DT_F32) {
        memcpy(dst, wbytes, (size_t)n * 4);
        return 0;
    }
    if (dtype == DT_F16) {
        uint32_t i;
        const uint16_t* wh = (const uint16_t*)wbytes;
        for (i = 0; i < n; i++) dst[i] = f16_to_f32(wh[i]);
        return 0;
    }
    return -1;
}

/* 各层 norm1/norm2 + final norm 分片写入 mem_wn, 供 token-batch 使用 */
static int vk_pack_norms(Engine* e, VulkanCtx* ctx)
{
    LlModel* m = &e->ws.model;
    uint32_t n_blocks = m->h.n_blocks;
    uint32_t hidden = m->h.hidden;
    size_t need = (size_t)(2u * n_blocks + 1u) * hidden * 4;
    if (!ctx->map_wn || need > ctx->wn_bytes) return -1;

    float* dst = (float*)ctx->map_wn;
    uint32_t li;
    for (li = 0; li < n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) return -1;
        const uint8_t* base =
            (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
        const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > BLOCK_TENSORS) nt = BLOCK_TENSORS;
        if (mt[SLOT_NORM1].size == 0 || mt[SLOT_NORM2].size == 0) return -1;
        if (pack_norm_slice(dst + (size_t)li * 2u * hidden,
                            base + mt[SLOT_NORM1].offset, hidden,
                            mt[SLOT_NORM1].dtype) != 0)
            return -1;
        if (pack_norm_slice(dst + (size_t)(li * 2u + 1u) * hidden,
                            base + mt[SLOT_NORM2].offset, hidden,
                            mt[SLOT_NORM2].dtype) != 0)
            return -1;
    }
    {
        uint32_t layer = n_blocks + 1;
        if (layer >= m->n_layers) return -1;
        const uint8_t* base =
            (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
        const LlfTensorMeta* tm = &m->metas[m->base_idx[layer]];
        if (pack_norm_slice(dst + (size_t)n_blocks * 2u * hidden,
                            base + tm->offset, hidden, tm->dtype) != 0)
            return -1;
    }
    ctx->norm_ready = 1;
    return 0;
}

/* 打包块内 Q4_K → 连续 blob, 填 wq_off[layer*nslot+slot] */
static int pack_upload_q4k(Engine* e, VulkanCtx* ctx)
{
    LlModel* m = &e->ws.model;
    uint32_t nslot = ctx->wq_nslot;
    size_t bank = (!ctx->wq_stream && ctx->wq_nbank > 1) ? ctx->wq_bank_size : 0;
    size_t total = q4k_packed_bytes(m, bank);
    uint32_t li, s;
    if (total == 0) return -1;
    if (!ctx->wq_stream && total > ctx->wq_bytes) {
        ylog_warn("vulkan: Q4_K pack size %zu vs buf %zu", total, ctx->wq_bytes);
        return -1;
    }

    uint8_t* blob = (uint8_t*)malloc(total);
    if (!blob) return -1;
    size_t cursor = 0;
    ctx->lm_ready = 0;
    ctx->lm_off = (uint64_t)~0ull;
    ctx->lm_dtype = 0;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        const uint8_t* base =
            (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > nslot) nt = nslot;
        cursor = wq_bank_align(cursor, block_q4k_bytes(m, layer), bank);
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in_slot(mt, s, m->h.hidden, &o, &i) != 0 || (i % 256) != 0) continue;
            size_t nbytes = (size_t)o * ((size_t)(i / 256) * 144);
            if (mt->size != 0 && mt->size != nbytes) {
                ylog_warn("vulkan: Q4_K size mismatch slot=%u layer=%u calc=%zu meta=%llu; use calc",
                          s, layer, nbytes, (unsigned long long)mt->size);
            }
            cursor = wq_bank_align(cursor, nbytes, bank);
            if (cursor + nbytes > total) {
                free(blob);
                return -1;
            }
            memcpy(blob + cursor, base + mt->offset, nbytes);
            ctx->wq_off[(size_t)layer * nslot + s] = cursor;
            cursor += nbytes;
        }
    }
    {
        uint32_t layer = m->h.n_blocks + 2;
        if (layer < m->n_layers) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
            const uint8_t* base =
                (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
            if (mt->dtype == DT_Q4K && mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    size_t nbytes = (size_t)o * ((size_t)(i / 256) * 144);
                    if (nbytes != mt->size) nbytes = (size_t)mt->size;
                    cursor = wq_bank_align(cursor, nbytes, bank);
                    if (cursor + nbytes <= total) {
                        memcpy(blob + cursor, base + mt->offset, nbytes);
                        ctx->lm_off = cursor;
                        ctx->lm_out = o;
                        ctx->lm_in = i;
                        ctx->lm_dtype = DT_Q4K;
                        ctx->wq_off[(size_t)layer * nslot + 0] = cursor;
                        cursor += nbytes;
                        ctx->lm_ready = 1;
                    } else {
                        ylog_warn("vulkan: lm_head pack overflow cursor=%zu need=%zu total=%zu",
                                  cursor, nbytes, total);
                    }
                } else {
                    ylog_warn("vulkan: lm_head skip shape dtype=%u size=%llu ndim=%u s0=%u s1=%u",
                              mt->dtype, (unsigned long long)mt->size, mt->ndim,
                              mt->shape[0], mt->shape[1]);
                }
            } else if (mt->dtype == DT_Q6K && mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    size_t nbytes = (size_t)o * ((size_t)(i / 256) * 210);
                    if (nbytes != mt->size) nbytes = (size_t)mt->size;
                    cursor = wq_bank_align(cursor, nbytes, bank);
                    if (cursor + nbytes <= total) {
                        memcpy(blob + cursor, base + mt->offset, nbytes);
                        ctx->lm_off = cursor;
                        ctx->lm_out = o;
                        ctx->lm_in = i;
                        ctx->lm_dtype = DT_Q6K;
                        ctx->wq_off[(size_t)layer * nslot + 0] = cursor;
                        cursor += nbytes;
                        ctx->lm_ready = 1;
                    } else {
                        ylog_warn("vulkan: lm_head Q6_K pack overflow cursor=%zu need=%zu total=%zu",
                                  cursor, nbytes, total);
                    }
                } else {
                    ylog_warn("vulkan: lm_head Q6_K skip shape dtype=%u size=%llu ndim=%u s0=%u s1=%u",
                              mt->dtype, (unsigned long long)mt->size, mt->ndim,
                              mt->shape[0], mt->shape[1]);
                }
            } else {
                ylog_info("vulkan: lm_head dtype=%u (CPU fallback)", mt->dtype);
            }
        }
    }
    if (ctx->wq_stream) {
        ctx->host_wq = blob;
        ctx->host_wq_bytes = cursor;
        ctx->wq_resident = 1;
        ctx->stream_layer = (uint32_t)~0u;
        if (ctx->lm_ready) {
            if (ctx->lm_dtype == DT_Q6K)
                ylog_info("vulkan: lm_head Q6_K stream-chunked out=%u in=%u", ctx->lm_out, ctx->lm_in);
            else
                ylog_info("vulkan: lm_head Q4_K stream-chunked out=%u in=%u", ctx->lm_out, ctx->lm_in);
        }
        ylog_info("vulkan: Q4_K stream host=%zuMB layer_gpu=%zuMB",
                  cursor / (1024 * 1024), ctx->wq_bytes / (1024 * 1024));
        return 0;
    }
    int rc = vulkan_wq_upload(ctx, blob, cursor);
    free(blob);
    if (rc == 0 && ctx->lm_ready) {
        const char* tag = (ctx->lm_dtype == DT_Q6K) ? "Q6_K" : "Q4_K";
        ylog_info("vulkan: lm_head %s resident out=%u in=%u", tag, ctx->lm_out, ctx->lm_in);
    }
    return rc;
}

static int vk_load_weights(Engine* e, char* err, size_t errlen)
{
    VulkanCtx* ctx = e->dev && e->dev->handle ? (VulkanCtx*)e->dev->handle : NULL;
    if (!ctx) {
        if (err && errlen) snprintf(err, errlen, "null vulkan ctx");
        return -1;
    }
    e->w_dev = ctx;
    e->d_kv = e->kv;
    e->weights_ready = 1;
    e->device_mode = ctx->host_shim ? DEV_MODE_VULKAN_HOST : DEV_MODE_VULKAN;
    ctx->n_layers = e->ws.model.n_layers;
    ctx->hidden = e->ws.model.h.hidden;
    ctx->n_blocks = e->ws.model.h.n_blocks;
    memcpy(&ctx->norm_eps, &e->ws.model.h.norm_eps_bits, 4);

    if (!ctx->host_shim && e->ops && !e->ops->gpu_fused) {
        ylog_info("vulkan: skip GPU kernels for %s (need arch-specific fused block)",
                  e->ops->name);
        vulkan_attach_fwd(e);
        return 0;
    }

    if (!ctx->host_shim) {
        uint32_t max_in = ctx->hidden, max_out = ctx->hidden;
        size_t total_wq = 0;
        scan_block_q4k(&e->ws.model, &max_in, &max_out, &total_wq);
        ctx->layer_wq_max = 0;
        ctx->stream_base = 0;
        size_t layer_max = max_block_q4k_bytes(&e->ws.model);
        if (layer_max == 0) layer_max = total_wq;
        layer_max = (layer_max + 4095u) & ~(size_t)4095u;
        if (layer_max < 144 * 8) layer_max = 144 * 8;
        ctx->layer_wq_max = layer_max;

        /* 单 SSBO 装不下则拆 bank 常驻; YLLM_VK_STREAM=1 才强制按层拷 */
        ctx->wq_stream = 0;
        size_t gpu_wq = total_wq;
        {
            const char* env = getenv("YLLM_VK_STREAM");
            int force_stream = (env && env[0] == '1') ? 1 : 0;
            int force_resident = (env && env[0] == '0') ? 1 : 0;
            /* 用设备 maxStorageBufferRange. 旧硬砍 256MB 会把 TinyLlama
             * 拆成 3 bank, 从而关掉 token-batch(每层一次 fence). */
            size_t ssbo = (size_t)ctx->max_ssbo_range;
            {
                const char* smb = getenv("YLLM_VK_SSBO_MB");
                unsigned long mb;
                if (smb && smb[0] && (mb = strtoul(smb, NULL, 10)) >= 16ul && mb <= 4096ul)
                    ssbo = (size_t)mb * 1024ull * 1024ull;
            }
            if (ssbo > 0xfffff000ull) ssbo = 0xfffff000ull;
            ssbo &= ~(size_t)4095u;
            if (ssbo < 16u * 1024u * 1024u) ssbo = 16u * 1024u * 1024u;
            if (force_stream) {
                ctx->wq_stream = 1;
                gpu_wq = layer_max;
                ylog_info("vulkan: weight stream on (YLLM_VK_STREAM=1 total=%zuMB layer_gpu=%zuMB)",
                          total_wq / (1024 * 1024), layer_max / (1024 * 1024));
            } else {
                size_t padded = (total_wq > ssbo) ? q4k_packed_bytes(&e->ws.model, ssbo) : total_wq;
                size_t nbank = ssbo ? (padded + ssbo - 1) / ssbo : 1;
                if (!force_resident && (nbank > YLLM_VK_MAX_WQ_BANKS || layer_max > ssbo)) {
                    ctx->wq_stream = 1;
                    gpu_wq = layer_max;
                    ylog_info("vulkan: weight stream on (total=%zuMB layer_gpu=%zuMB banks_need=%zu)",
                              total_wq / (1024 * 1024), layer_max / (1024 * 1024), nbank);
                } else {
                    gpu_wq = padded;
                    ctx->wq_bank_size = ssbo;
                    ctx->wq_nbank = (int)nbank;
                    if (ctx->wq_nbank < 1) ctx->wq_nbank = 1;
                    ylog_info("vulkan: weight resident (total=%zuMB packed=%zuMB ssbo_max=%zuMB banks~%zu igpu=%d)",
                              total_wq / (1024 * 1024), padded / (1024 * 1024),
                              ssbo / (1024 * 1024), nbank, ctx->integrated_gpu);
                }
            }
        }
        char cerr[256];
        uint32_t lm_vocab = 0;
        {
            LlModel* m = &e->ws.model;
            uint32_t layer = m->h.n_blocks + 2;
            if (layer < m->n_layers) {
                const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
                if ((mt->dtype == DT_Q4K || mt->dtype == DT_Q6K) && mt->size > 0) {
                    uint32_t o, i;
                    if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0)
                        lm_vocab = o;
                }
            }
        }
        if (vulkan_compute_setup(ctx, ctx->hidden, max_in, max_out, gpu_wq,
                                 e->ws.model.n_layers, BLOCK_TENSORS, lm_vocab,
                                 cerr, sizeof(cerr)) != 0) {
            ylog_warn("vulkan: compute setup failed (%s); fwd stays CPU", cerr);
        } else {
            if (vulkan_selftest_rmsnorm(ctx) != 0) {
                ylog_warn("vulkan: rmsnorm selftest failed; GPU rmsnorm disabled");
                ctx->compute_ready = 0;
            }
            if (ctx->gemv_ready && vulkan_selftest_gemv_q4k(ctx) != 0) {
                ylog_warn("vulkan: gemv_q4k selftest failed; GPU gemv disabled");
                ctx->gemv_ready = 0;
            }
            if (ctx->gemv_ready && pack_upload_q4k(e, ctx) != 0) {
                ylog_warn("vulkan: Q4_K resident upload failed; gemv falls back host-W");
                ctx->wq_resident = 0;
            }
            if (ctx->gemv_ready && ctx->wq_resident && !getenv("YLLM_VK_NOSELFTEST") &&
                vulkan_selftest_gemv_q4k_real(ctx) != 0) {
                ylog_warn("vulkan: gemv_q4k real-weight selftest failed; GPU gemv disabled");
                ctx->gemv_ready = 0;
            }
            if (ctx->lm_ready && ctx->gemv_q6k_ready && !getenv("YLLM_VK_NOSELFTEST") &&
                vulkan_selftest_gemv_q6k(ctx) != 0) {
                ylog_warn("vulkan: gemv_q6k selftest failed; lm_head CPU fallback");
                ctx->lm_ready = 0;
            }
            if (ctx->lm_ready && ctx->gemv_ds_lm && !ctx->wq_stream &&
                ctx->lm_out > 0 && ctx->logits_bytes >= (size_t)ctx->lm_out * 4) {
                ctx->lm_one_submit = 1;
                ylog_info("vulkan: lm_head one-submit vocab=%u", ctx->lm_out);
                if (ctx->rms_ds_inplace) {
                    if (ctx->lm_dtype == DT_Q4K ||
                        (ctx->lm_dtype == DT_Q6K && ctx->q8k_ready)) {
                        ctx->lm_fused = 1;
                        ylog_info("vulkan: lm_head fused norm+lm vocab=%u", ctx->lm_out);
                    }
                }
            }
            if (ctx->fuse_ready) {
                char aerr[256];
                const LlfHeader* h = &e->ws.model.h;
                uint32_t kv_dim = h->n_kv_heads * h->head_dim;
                if (vulkan_attn_setup(ctx, h->n_blocks, e->max_seq, kv_dim,
                                      h->n_heads, h->n_kv_heads, h->head_dim,
                                      aerr, sizeof(aerr)) != 0) {
                    ylog_warn("vulkan: attn setup failed (%s)", aerr);
                }
            }
            if (ctx->compute_ready && ctx->wq_resident && !ctx->wq_stream) {
                if (vk_pack_norms(e, ctx) == 0) {
                    ylog_info("vulkan: norm weights resident (%u blocks, %zuKB)",
                              e->ws.model.h.n_blocks, ctx->wn_bytes / 1024);
                } else {
                    ylog_warn("vulkan: norm pack failed; token-batch disabled");
                }
            }
            if (ctx->norm_ready && ctx->block_ready && ctx->wq_resident &&
                ctx->use_gpu_rope && !ctx->wq_stream && ctx->wq_nbank <= 1 &&
                !getenv("YLLM_VK_NOTOKB")) {
                ctx->token_batch = 1;
                ylog_info("vulkan: token-batch decode (per-layer fence; YLLM_VK_VRAM_SCRATCH=1 for VRAM+sem)");
            }
            if (getenv("YLLM_VK_NOFUSE")) {
                ctx->block_ready = 0;
                ctx->fuse_ready = 0;
                ctx->attn_ready = 0;
                ctx->embed_ready = 0;
                ctx->use_gpu_rope = 0;
                ctx->token_batch = 0;
                ctx->lm_fused = 0;
                ctx->lm_one_submit = 0;
                ylog_info("vulkan: NOFUSE per-op gemv (no fused block/attn/embed)");
            }
            if (ctx->integrated_gpu && !ctx->gemv_ready && !getenv("YLLM_VK_IGPU_GPU")) {
                ctx->compute_ready = 0;
                ctx->block_ready = 0;
                ctx->fuse_ready = 0;
                ctx->attn_ready = 0;
                ctx->embed_ready = 0;
                ctx->use_gpu_rope = 0;
                ctx->token_batch = 0;
                ctx->lm_ready = 0;
                ctx->lm_one_submit = 0;
                ctx->lm_fused = 0;
                ctx->wq_stream = 0;
                ylog_warn("vulkan: iGPU gemv selftest failed; using CPU fwd (YLLM_VK_IGPU_GPU=1 to force GPU)");
            }
        }
    }

    vulkan_attach_fwd(e);
    ylog_info("vulkan: mode=%s gpu=%d layers=%u hidden=%u rms=%d gemv=%d resident=%d stream=%d fuse=%d swi=%d attn=%d attn_o=%d rope=%d block=%d embed=%d gpu_rope=%d lm=%d lm1=%d lmf=%d norm=%d tokb=%d",
              ctx->host_shim ? "host-shim" : "native",
              ctx->device_id, ctx->n_layers, ctx->hidden,
              ctx->compute_ready, ctx->gemv_ready, ctx->wq_resident, ctx->wq_stream,
              ctx->fuse_ready, ctx->swi_ready, ctx->attn_ready, ctx->attn_o_ready,
              ctx->rope_ready, ctx->block_ready, ctx->embed_ready, ctx->use_gpu_rope,
              ctx->lm_ready, ctx->lm_one_submit, ctx->lm_fused, ctx->norm_ready,
              ctx->token_batch);
    return 0;
}

static void vk_free_dev(Engine* e)
{
    if (e->dev && e->dev->handle) {
        VulkanCtx* ctx = (VulkanCtx*)e->dev->handle;
        vulkan_shutdown(ctx);
        free(ctx);
        e->dev->handle = NULL;
    }
    e->w_dev = NULL;
    e->d_kv = NULL;
    e->weights_ready = 0;
    e->device_mode = DEV_MODE_CPU;
}

Device* device_create_vulkan(int device_id, char* err, size_t errlen)
{
#ifndef YLLM_VULKAN
    (void)device_id;
    if (err && errlen)
        snprintf(err, errlen, "Vulkan backend not built (YLLM_VULKAN=1)");
    return NULL;
#else
    Device* d = (Device*)calloc(1, sizeof(Device));
    VulkanCtx* ctx = (VulkanCtx*)calloc(1, sizeof(VulkanCtx));
    if (!d || !ctx) {
        free(d);
        free(ctx);
        if (err && errlen) snprintf(err, errlen, "oom");
        return NULL;
    }
    ctx->device_id = device_id;
    ctx->host_shim = 1;
#ifdef YLLM_VULKAN_HOST
    ylog_info("vulkan: forced host-shim (YLLM_VULKAN_HOST=1)");
#else
    {
        char verr[256];
        if (vulkan_try_init(ctx, device_id, verr, sizeof(verr)) != 0) {
            ylog_warn("vulkan: native init failed (%s), falling back to host-shim", verr);
            ctx->host_shim = 1;
            ctx->instance = NULL;
            ctx->phys = NULL;
            ctx->device = NULL;
            ctx->queue = NULL;
        } else {
            ylog_info("vulkan: native compute device ready");
        }
    }
#endif
    d->kind = DEV_VULKAN;
    d->id = ctx->device_id;
    d->handle = ctx;
    d->load_weights = vk_load_weights;
    d->free_dev = vk_free_dev;
    d->prefetch_layer = NULL;
    d->release_layer = NULL;
    d->embed = vulkan_embed;
    d->after_cpu_embed = vulkan_after_embed;
    d->final_norm = vulkan_final_norm;
    d->lm_head = vulkan_lm_or_fused;
    d->prefill = vulkan_prefill;
    d->sync_x = vulkan_sync_x;
    d->mark_x_host = vulkan_after_embed;
    return d;
#endif
}
