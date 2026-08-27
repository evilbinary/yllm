#include "arch.h"
#include "yllm.h"
#include "llf.h"
#include "matvec.h"
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

const ArchOps arch_llama_ops = {
    .name = "llama",
    .id = ARCH_LLAMA,
    .cpu_batch_prefill = 0,
    .prefill_batch_min = 16,
    .fwd_block = arch_llama_fwd_block,
    .fwd_block_batch = arch_llama_fwd_block_batch,
};

int arch_llama_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    const uint8_t* base = (const uint8_t*)e->ws.map.base + e->ws.model.dir[layer].offset;
    return arch_llama_fwd_block_at(e, layer, pos, base, e->kv, 0);
}

int arch_llama_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
{
    return arch_llama_fwd_block_batch_rope(e, layer, pos_start, B, 0);
}

/* 批量前向一层: 同时处理 B 个 token(pos 连续: pos_start..pos_start+B-1) */
int arch_llama_fwd_block_batch_rope(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B, int qwen_rope)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[layer].offset;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    float eps, theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    uint32_t hidx = m->base_idx[layer];
    const LlfTensorMeta* mt = &m->metas[hidx];
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;
    uint32_t b;

    /* 1) norm + QKV(批量) */
    for (b = 0; b < B; b++)
        rmsnorm(e->pb2 + (size_t)b * hidden, e->pb + (size_t)b * hidden,
                base + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
    matmul_batch(e->pbq, e->pb2, base + mt[SLOT_Q].offset, hidden, hidden, mt[SLOT_Q].dtype, B);
    matmul_batch(e->pbk, e->pb2, base + mt[SLOT_K].offset, kv_dim, hidden, mt[SLOT_K].dtype, B);
    matmul_batch(e->pbv, e->pb2, base + mt[SLOT_V].offset, kv_dim, hidden, mt[SLOT_V].dtype, B);

    /* 2) bias + QK-norm */
    if (mt[SLOT_QBIAS].size > 0) {
        const float* bq = (const float*)(base + mt[SLOT_QBIAS].offset);
        for (b = 0; b < B; b++) {
            float* q = e->pbq + (size_t)b * hidden;
            uint32_t j;
            for (j = 0; j < hidden; j++) q[j] += bq[j];
        }
    }
    if (mt[SLOT_KBIAS].size > 0) {
        const float* bk = (const float*)(base + mt[SLOT_KBIAS].offset);
        for (b = 0; b < B; b++) {
            float* k = e->pbk + (size_t)b * kv_dim;
            uint32_t j;
            for (j = 0; j < kv_dim; j++) k[j] += bk[j];
        }
    }
    if (mt[SLOT_VBIAS].size > 0) {
        const float* bv = (const float*)(base + mt[SLOT_VBIAS].offset);
        for (b = 0; b < B; b++) {
            float* v = e->pbv + (size_t)b * kv_dim;
            uint32_t j;
            for (j = 0; j < kv_dim; j++) v[j] += bv[j];
        }
    }
    if (mt[SLOT_QNORM].size > 0 || mt[SLOT_KNORM].size > 0) {
        for (b = 0; b < B; b++) {
            float* q = e->pbq + (size_t)b * hidden;
            float* k = e->pbk + (size_t)b * kv_dim;
            uint32_t hh;
            if (mt[SLOT_QNORM].size > 0)
                for (hh = 0; hh < h->n_heads; hh++)
                    rmsnorm(q + (size_t)hh * h->head_dim, q + (size_t)hh * h->head_dim,
                            base + mt[SLOT_QNORM].offset, h->head_dim, eps, mt[SLOT_QNORM].dtype);
            if (mt[SLOT_KNORM].size > 0)
                for (hh = 0; hh < h->n_kv_heads; hh++)
                    rmsnorm(k + (size_t)hh * h->head_dim, k + (size_t)hh * h->head_dim,
                            base + mt[SLOT_KNORM].offset, h->head_dim, eps, mt[SLOT_KNORM].dtype);
        }
    }

    /* 3) RoPE + KV 写入(每 token 独立 pos) */
    for (b = 0; b < B; b++) {
        uint32_t pos = pos_start + b;
        float* q = e->pbq + (size_t)b * hidden;
        float* k = e->pbk + (size_t)b * kv_dim;
        uint32_t hh;
        for (hh = 0; hh < h->n_heads; hh++) {
            if (qwen_rope)
                rope_inplace_qwen(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
            else
                rope_inplace(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        }
        for (hh = 0; hh < h->n_kv_heads; hh++) {
            if (qwen_rope)
                rope_inplace_qwen(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
            else
                rope_inplace(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        }
        {
            uint16_t* kcache = e->kv + (size_t)layer * e->max_seq * kv_dim;
            uint16_t* vcache = e->kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
            uint64_t kvp = (uint64_t)pos * kv_dim;
            const float* kvk = e->pbk + (size_t)b * kv_dim;
            const float* kvv = e->pbv + (size_t)b * kv_dim;
            f32_to_f16_buf(kvk, kcache + kvp, kv_dim);
            f32_to_f16_buf(kvv, vcache + kvp, kv_dim);
        }
    }

    /* 4) 注意力: 每 token 因果关注 0..pos_b; GQA 共用 K/V */
    {
        uint32_t bb;
        float inv_d = 1.0f / sqrtf((float)h->head_dim);
        uint16_t* kcache = e->kv + (size_t)layer * e->max_seq * kv_dim;
        uint16_t* vcache = e->kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
        #pragma omp parallel for schedule(static)
        for (bb = 0; bb < B; bb++) {
            uint32_t pos = pos_start + bb;
            attn_kv_f16(e->pb2 + (size_t)bb * hidden, e->pbq + (size_t)bb * hidden,
                        kcache, vcache, 0, pos,
                        h->n_heads, h->n_kv_heads, h->head_dim, kv_dim, inv_d, 0.0f);
        }
    }

    /* 5) o_proj + 残差, norm2, FFN */
    matmul_batch(e->pbq, e->pb2, base + mt[SLOT_O].offset, hidden, hidden, mt[SLOT_O].dtype, B);
    for (b = 0; b < B; b++) {
        float* xb = e->pb + (size_t)b * hidden;
        float* ob = e->pbq + (size_t)b * hidden;
        add_inplace(xb, ob, hidden);
        rmsnorm(e->pb2 + (size_t)b * hidden, xb, base + mt[SLOT_NORM2].offset,
                hidden, eps, mt[SLOT_NORM2].dtype);
    }
    matmul_batch(e->pbg, e->pb2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype, B);
    matmul_batch(e->pbu, e->pb2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype, B);
    /* swiglu 输出必须是 [B × inter] 布局(down 的输入维度是 inter, 复用 pbg) */
    for (b = 0; b < B; b++)
        swiglu(e->pbg + (size_t)b * inter, e->pbg + (size_t)b * inter, e->pbu + (size_t)b * inter, inter);
    matmul_batch(e->pbq, e->pbg, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype, B);
    for (b = 0; b < B; b++) {
        float* xb = e->pb + (size_t)b * hidden;
        add_inplace(xb, e->pbq + (size_t)b * hidden, hidden);
    }
    return 0;
}

int arch_llama_fwd_block_at(Engine* e, uint32_t layer, uint32_t pos,
                            const uint8_t* base, uint16_t* kv, int qwen_rope)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    float eps;
    float theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    float* x = e->x;
    float* x2 = e->hb;
    float* q = e->hb2;
    float* k = e->hb2 + h->n_heads * h->head_dim;
    float* v = e->hb2 + h->n_heads * h->head_dim + kv_dim;
    float* att_out = e->hb2 + h->n_heads * h->head_dim + 2 * kv_dim;
    uint32_t hidx = m->base_idx[layer];

    const LlfTensorMeta* mt = &m->metas[hidx];
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;
    uint32_t q_dim = h->n_heads * h->head_dim;
    uint32_t hd = h->head_dim;
    uint32_t kvd = kv_dim;
    int has_kv = mt[SLOT_K].size != 0;

    rmsnorm(x2, x, base + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
    if (mt[SLOT_Q].dtype == DT_W4B64) {
        uint32_t nb = hidden / W4B64_BLK;
        int8_t* xq = (int8_t*)alloca((size_t)hidden);
        float* xs = (float*)alloca((size_t)nb * 4);
        int32_t* xsum = (int32_t*)alloca((size_t)nb * 4);
        w4b64_act_quant(x2, xq, xs, xsum, hidden);
        matmul_w4b64_xq(q, xq, xs, xsum, base + mt[SLOT_Q].offset, q_dim, hidden);
        if (has_kv) {
            matmul_w4b64_xq(k, xq, xs, xsum, base + mt[SLOT_K].offset, kvd, hidden);
            if (mt[SLOT_V].size > 0)
                matmul_w4b64_xq(v, xq, xs, xsum, base + mt[SLOT_V].offset, kvd, hidden);
            else
                memcpy(v, k, (size_t)kvd * 4);
        }
    } else {
        matmul(q, x2, base + mt[SLOT_Q].offset, q_dim, hidden, mt[SLOT_Q].dtype);
        if (has_kv) {
            matmul(k, x2, base + mt[SLOT_K].offset, kvd, hidden, mt[SLOT_K].dtype);
            if (mt[SLOT_V].size > 0)
                matmul(v, x2, base + mt[SLOT_V].offset, kvd, hidden, mt[SLOT_V].dtype);
            else
                memcpy(v, k, (size_t)kvd * 4);
        }
    }
    /* 注意力 bias(qwen2.5 gguf 带有非零 bias) */
    if (mt[SLOT_QBIAS].size > 0) {
        const float* bq = (const float*)(base + mt[SLOT_QBIAS].offset);
        uint32_t j;
        for (j = 0; j < q_dim; j++) q[j] += bq[j];
    }
    if (has_kv && mt[SLOT_KBIAS].size > 0) {
        const float* bk = (const float*)(base + mt[SLOT_KBIAS].offset);
        uint32_t j;
        for (j = 0; j < kvd; j++) k[j] += bk[j];
    }
    if (has_kv && mt[SLOT_VBIAS].size > 0) {
        const float* bv = (const float*)(base + mt[SLOT_VBIAS].offset);
        uint32_t j;
        for (j = 0; j < kvd; j++) v[j] += bv[j];
    }

    /* qwen3 QK-norm: 对 q 每头 / k 每 kv 头做 RMSNorm(共享 head_dim 权重), 在 rope 之前 */
    if (mt[SLOT_QNORM].size > 0) {
        uint32_t hh;
        for (hh = 0; hh < h->n_heads; hh++)
            rmsnorm(q + (size_t)hh * hd, q + (size_t)hh * hd,
                    base + mt[SLOT_QNORM].offset, hd, eps, mt[SLOT_QNORM].dtype);
    }
    if (has_kv && mt[SLOT_KNORM].size > 0) {
        uint32_t hh;
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm(k + (size_t)hh * hd, k + (size_t)hh * hd,
                    base + mt[SLOT_KNORM].offset, hd, eps, mt[SLOT_KNORM].dtype);
    }

    uint16_t* kcache = kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
    uint64_t kvp = (uint64_t)pos * kv_dim;
    uint32_t hh;
    for (hh = 0; hh < h->n_heads; hh++) {
        if (qwen_rope)
            rope_inplace_qwen(q + (size_t)hh * hd, hd, pos, theta);
        else
            rope_inplace(q + (size_t)hh * hd, hd, pos, theta);
    }
    if (has_kv) {
        for (hh = 0; hh < h->n_kv_heads; hh++) {
            if (qwen_rope)
                rope_inplace_qwen(k + (size_t)hh * hd, hd, pos, theta);
            else
                rope_inplace(k + (size_t)hh * hd, hd, pos, theta);
        }
        f32_to_f16_buf(k, kcache + kvp, kvd);
        f32_to_f16_buf(v, vcache + kvp, kvd);
    }

    float inv_d = 1.0f / sqrtf((float)hd);
    attn_kv_f16(att_out, q, kcache, vcache, 0, pos,
                h->n_heads, h->n_kv_heads, hd, kv_dim, inv_d, 0.0f);
    memcpy(x2, att_out, (size_t)q_dim * 4);
    matmul(att_out, x2, base + mt[SLOT_O].offset, hidden, q_dim, mt[SLOT_O].dtype);
    add_inplace(x, att_out, hidden);
    rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
    float* fg = e->ffn;
    float* fu = e->ffn + inter;
    matmul(fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
    matmul(fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
    swiglu(x2, fg, fu, inter);
    matmul(att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
    add_inplace(x, att_out, hidden);
    return 0;
}
