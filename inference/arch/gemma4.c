#include "arch.h"
#include "yllm.h"
#include "llf.h"
#include "matvec.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if !defined(_WIN32)
#include <alloca.h>
#endif

const ArchOps arch_gemma4_ops = {
    .name = "gemma4",
    .id = ARCH_GEMMA4,
    .cpu_batch_prefill = 1,
    .prefill_batch_min = 2,
    .alloc = arch_gemma4_alloc,
    .after_embed = arch_gemma4_after_embed,
    .after_embed_batch = arch_gemma4_after_embed_batch,
    .refresh_ple_pp = arch_gemma4_refresh_ple_pp,
    .post_logits = arch_gemma4_post_logits,
    .fwd_block = arch_gemma4_fwd_block,
    .fwd_block_batch = arch_gemma4_fwd_block_batch,
};

static void gemma4_fill_inv_freq(float* out, uint32_t n, uint32_t d, float theta, int fold_self)
{
    uint32_t j;
    for (j = 0; j < n; j++) {
        float f = powf(theta, -2.0f * (float)j / (float)d);
        if (fold_self) {
            float x = out[j];
            if (x > 0.0f) f /= x;
        }
        out[j] = f;
    }
}

int arch_gemma4_alloc(Engine* e)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    uint32_t hidden = m->h.hidden;
    LlfGemma4Ext g4;
    llf_gemma4_ext(&m->h, &g4);
    e->n_ple = g4.n_embd_per_layer;
    if (e->n_ple > 0) {
        uint32_t psz = e->n_ple * m->h.n_blocks;
        e->ple = (float*)ycalloc(psz, 4);
        e->ple_work = (float*)ycalloc(psz, 4);
    }
    {
        const LlfTensorMeta* tm = &m->metas[m->base_idx[0] + SLOT_ROPE_FREQS];
        if (tm->size > 0) {
            uint32_t n = 1, d;
            for (d = 0; d < tm->ndim && d < 4; d++) {
                if (tm->shape[d] > 0) n *= tm->shape[d];
            }
            if (n > 0 && n <= 8192) {
                const uint8_t* p = (const uint8_t*)ws->map.base + m->dir[0].offset + tm->offset;
                e->rope_ff = (float*)ymalloc((size_t)n * 4);
                e->n_rope_ff = n;
                if (tm->dtype == DT_F32) {
                    memcpy(e->rope_ff, p, (size_t)n * 4);
                } else if (tm->dtype == DT_F16) {
                    const uint16_t* hp = (const uint16_t*)p;
                    uint32_t j;
                    for (j = 0; j < n; j++) e->rope_ff[j] = f16_to_f32(hp[j]);
                } else {
                    free(e->rope_ff);
                    e->rope_ff = NULL;
                    e->n_rope_ff = 0;
                }
            }
        }
    }
    {
        float theta_g;
        uint32_t hd_g = m->h.head_dim ? m->h.head_dim : 1;
        uint32_t hd_s = 0, li, swa_il = 0;
        memcpy(&theta_g, &m->h.rope_theta_bits, 4);
        if (e->rope_ff && e->n_rope_ff > 0) {
            uint32_t d = e->n_rope_ff * 2u;
            gemma4_fill_inv_freq(e->rope_ff, e->n_rope_ff, d, theta_g, 1);
        } else if (hd_g >= 2) {
            e->n_rope_ff = hd_g / 2;
            e->rope_ff = (float*)ymalloc((size_t)e->n_rope_ff * 4);
            gemma4_fill_inv_freq(e->rope_ff, e->n_rope_ff, hd_g, theta_g, 0);
        }
        for (li = 1; li <= m->h.n_blocks; li++) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[li]];
            if (!llf_gemma4_is_swa(&g4, li - 1)) continue;
            if (mt[SLOT_Q].ndim >= 2 && hidden > 0 && m->h.n_heads) {
                uint32_t qd = mt[SLOT_Q].shape[0] * mt[SLOT_Q].shape[1] / hidden;
                hd_s = qd / m->h.n_heads;
            }
            swa_il = li - 1;
            break;
        }
        if (hd_s >= 2) {
            float ths = llf_gemma4_rope_theta(&m->h, &g4, swa_il);
            e->n_rope_if_swa = hd_s / 2;
            e->rope_if_swa = (float*)ymalloc((size_t)e->n_rope_if_swa * 4);
            gemma4_fill_inv_freq(e->rope_if_swa, e->n_rope_if_swa, hd_s, ths, 0);
        }
    }
    if (hidden > 0 && e->pb_cap > 0) {
        uint32_t li, q_dim_max = e->pbq_dim;
        for (li = 1; li <= m->h.n_blocks; li++) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[li]];
            if (mt[SLOT_Q].ndim >= 2) {
                uint32_t qd = mt[SLOT_Q].shape[0] * mt[SLOT_Q].shape[1] / hidden;
                if (qd > q_dim_max) q_dim_max = qd;
            }
        }
        if (q_dim_max > e->pbq_dim) {
            free(e->pbq);
            e->pbq = (float*)ycalloc((size_t)e->pb_cap * q_dim_max, 4);
            e->pbq_dim = q_dim_max;
        }
    }
    if (e->n_ple > 0 && e->pb_cap > 0) {
        size_t psz = (size_t)e->n_ple * m->h.n_blocks;
        e->ple_batch = (float*)ycalloc((size_t)e->pb_cap * psz, 4);
        free(e->ple_work);
        e->ple_work = (float*)ycalloc((size_t)e->pb_cap * psz, 4);
    }
    ylog_info("gemma4: n_ple=%u shared_kv=%u swa_win=%u swa_pat=%u rope_freqs=%u",
              e->n_ple, g4.n_kv_shared_layers, g4.swa_window, g4.swa_pattern,
              e->n_rope_ff);
    return 0;
}

static void gemma4_prepare_ple_resid(Engine* e, uint32_t token, const float* resid)
{
    if (!e->n_ple || !e->ple || !e->ple_work || !resid) return;
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* ebase = (const uint8_t*)ws->map.base + m->dir[0].offset;
    const LlfTensorMeta* em = &m->metas[m->base_idx[0]];
    const LlfTensorMeta* tok = &em[SLOT_PLE_TOK];
    const LlfTensorMeta* mproj = &em[SLOT_PLE_MPROJ];
    const LlfTensorMeta* pnorm = &em[SLOT_PLE_PNORM];
    uint32_t n_ple = e->n_ple;
    uint32_t nblk = h->n_blocks;
    uint32_t width = n_ple * nblk;
    uint32_t j, il;
    float eps;
    float ts, ps, isc;
    if (tok->size == 0 || mproj->size == 0) return;
    memcpy(&eps, &h->norm_eps_bits, 4);
    switch (tok->dtype) {
    case DT_F32: embed_f32(e->ple, ebase + tok->offset, token, width); break;
    case DT_Q4K: embed_q4k(e->ple, ebase + tok->offset, token, width); break;
    case DT_Q6K: embed_q6k(e->ple, ebase + tok->offset, token, width); break;
    case DT_Q5K: embed_q5k(e->ple, ebase + tok->offset, token, width); break;
    default: embed_f16(e->ple, ebase + tok->offset, token, width); break;
    }
    ts = sqrtf((float)n_ple);
    for (j = 0; j < width; j++) e->ple[j] *= ts;
    matmul(e->ple_work, resid, ebase + mproj->offset, width, h->hidden, mproj->dtype);
    ps = 1.0f / sqrtf((float)h->hidden);
    for (j = 0; j < width; j++) e->ple_work[j] *= ps;
    isc = 1.0f / sqrtf(2.0f);
    for (il = 0; il < nblk; il++) {
        float* sl = e->ple_work + (size_t)il * n_ple;
        float* pe = e->ple + (size_t)il * n_ple;
        if (pnorm->size > 0)
            rmsnorm(sl, sl, ebase + pnorm->offset, n_ple, eps, pnorm->dtype);
        else
            rmsnorm_unit(sl, sl, n_ple, eps);
        for (j = 0; j < n_ple; j++)
            pe[j] = (sl[j] + pe[j]) * isc;
    }
}

void arch_gemma4_after_embed(Engine* e, uint32_t token)
{
    uint32_t hidden = e->ws.model.h.hidden;
    float scale = sqrtf((float)hidden);
    uint32_t j;
    for (j = 0; j < hidden; j++) e->x[j] *= scale;
    gemma4_prepare_ple_resid(e, token, e->x);
}

void arch_gemma4_refresh_ple_pp(Engine* e, uint32_t token)
{
    if (!e->n_ple || !e->hb) return;
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[0].offset;
    const LlfTensorMeta* tm = &m->metas[m->base_idx[0]];
    float* emb = e->hb;
    uint32_t j;
    switch (tm->dtype) {
    case DT_F32: embed_f32(emb, base + tm->offset, token, h->hidden); break;
    case DT_Q4K: embed_q4k(emb, base + tm->offset, token, h->hidden); break;
    case DT_Q6K: embed_q6k(emb, base + tm->offset, token, h->hidden); break;
    case DT_Q5K: embed_q5k(emb, base + tm->offset, token, h->hidden); break;
    case DT_IQ4XS: embed_iq4xs(emb, base + tm->offset, token, h->hidden); break;
    default: embed_f16(emb, base + tm->offset, token, h->hidden); break;
    }
    {
        float scale = sqrtf((float)h->hidden);
        for (j = 0; j < h->hidden; j++) emb[j] *= scale;
    }
    gemma4_prepare_ple_resid(e, token, emb);
}

static void gemma4_prepare_ple_batch(Engine* e, const uint32_t* tokens, uint32_t B)
{
    if (!e->n_ple || !e->ple_batch || !e->ple_work || B == 0) return;
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* ebase = (const uint8_t*)ws->map.base + m->dir[0].offset;
    const LlfTensorMeta* em = &m->metas[m->base_idx[0]];
    const LlfTensorMeta* tok = &em[SLOT_PLE_TOK];
    const LlfTensorMeta* mproj = &em[SLOT_PLE_MPROJ];
    const LlfTensorMeta* pnorm = &em[SLOT_PLE_PNORM];
    uint32_t n_ple = e->n_ple;
    uint32_t nblk = h->n_blocks;
    uint32_t width = n_ple * nblk;
    uint32_t b, j, il;
    float eps, ts, ps, isc;
    if (tok->size == 0 || mproj->size == 0) return;
    memcpy(&eps, &h->norm_eps_bits, 4);
    ts = sqrtf((float)n_ple);
    ps = 1.0f / sqrtf((float)h->hidden);
    isc = 1.0f / sqrtf(2.0f);
    for (b = 0; b < B; b++) {
        float* pe = e->ple_batch + (size_t)b * width;
        switch (tok->dtype) {
        case DT_F32: embed_f32(pe, ebase + tok->offset, tokens[b], width); break;
        case DT_Q4K: embed_q4k(pe, ebase + tok->offset, tokens[b], width); break;
        case DT_Q6K: embed_q6k(pe, ebase + tok->offset, tokens[b], width); break;
        case DT_Q5K: embed_q5k(pe, ebase + tok->offset, tokens[b], width); break;
        default: embed_f16(pe, ebase + tok->offset, tokens[b], width); break;
        }
        for (j = 0; j < width; j++) pe[j] *= ts;
    }
    matmul_batch(e->ple_work, e->pb, ebase + mproj->offset, width, h->hidden, mproj->dtype, B);
    for (b = 0; b < B; b++) {
        float* sl0 = e->ple_work + (size_t)b * width;
        float* pe = e->ple_batch + (size_t)b * width;
        for (j = 0; j < width; j++) sl0[j] *= ps;
        for (il = 0; il < nblk; il++) {
            float* sl = sl0 + (size_t)il * n_ple;
            float* pel = pe + (size_t)il * n_ple;
            if (pnorm->size > 0)
                rmsnorm(sl, sl, ebase + pnorm->offset, n_ple, eps, pnorm->dtype);
            else
                rmsnorm_unit(sl, sl, n_ple, eps);
            for (j = 0; j < n_ple; j++)
                pel[j] = (sl[j] + pel[j]) * isc;
        }
    }
}

void arch_gemma4_after_embed_batch(Engine* e, const uint32_t* tokens, uint32_t B)
{
    uint32_t hidden = e->ws.model.h.hidden;
    float scale = sqrtf((float)hidden);
    uint32_t b2, j;
    for (b2 = 0; b2 < B; b2++)
        for (j = 0; j < hidden; j++)
            e->pb[(size_t)b2 * hidden + j] *= scale;
    if (e->ple_batch && e->n_ple)
        gemma4_prepare_ple_batch(e, tokens, B);
}

void arch_gemma4_post_logits(Engine* e)
{
    float cap = llf_gemma4_final_cap(&e->ws.model.h);
    if (cap > 0.0f) {
        uint32_t vi, vocab = e->ws.model.h.vocab;
        for (vi = 0; vi < vocab; vi++)
            e->logits[vi] = cap * tanhf(e->logits[vi] / cap);
    }
}

/* gemma4 批量前向: QKV/O/FFN 走 matmul_batch; attn/PLE 按 token */
int arch_gemma4_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[layer].offset;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    float eps, theta;
    const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;
    uint32_t q_dim = h->n_heads * h->head_dim;
    uint32_t hd = h->head_dim;
    uint32_t kvd = kv_dim;
    LlfGemma4Ext g4;
    uint32_t il = layer > 0 ? layer - 1 : 0;
    int has_kv = 1;
    uint32_t kv_layer = layer;
    uint32_t b, hh;
    size_t ple_stride = (e->n_ple > 0) ? (size_t)e->n_ple * h->n_blocks : 0;
    float* att_batch = e->pba; /* [B x q_dim] 暂存 attn 输出 */
    const float* rope_if = NULL;
    float inv_d = 1.0f;
    int swa;

    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    llf_gemma4_ext(h, &g4);
    theta = llf_gemma4_rope_theta(h, &g4, il);
    swa = llf_gemma4_is_swa(&g4, il);
    if (g4.n_kv_shared_layers > 0 && g4.n_kv_shared_layers < h->n_blocks) {
        uint32_t kv_from = h->n_blocks - g4.n_kv_shared_layers;
        if (il >= kv_from) {
            has_kv = 0;
            kv_layer = kv_from - (swa ? 2u : 1u) + 1u;
        }
    }
    if (mt[SLOT_K].size == 0) has_kv = 0;
    if (mt[SLOT_Q].ndim >= 2 && hidden > 0)
        q_dim = mt[SLOT_Q].shape[0] * mt[SLOT_Q].shape[1] / hidden;
    hd = h->n_heads ? q_dim / h->n_heads : hd;
    if (has_kv && mt[SLOT_K].ndim >= 2 && hidden > 0)
        kvd = mt[SLOT_K].shape[0] * mt[SLOT_K].shape[1] / hidden;

    /* 缓冲不够或尺寸异常时退回逐 token */
    if (q_dim > e->pbq_dim || inter > e->inter || B > e->pb_cap) {
        for (b = 0; b < B; b++) {
            memcpy(e->x, e->pb + (size_t)b * hidden, (size_t)hidden * 4);
            if (e->ple && e->ple_batch && ple_stride)
                memcpy(e->ple, e->ple_batch + (size_t)b * ple_stride, ple_stride * 4);
            if (e->ops->fwd_block(e, layer, pos_start + b) != 0) return -1;
            memcpy(e->pb + (size_t)b * hidden, e->x, (size_t)hidden * 4);
        }
        return 0;
    }

    if (swa && e->rope_if_swa && e->n_rope_if_swa == hd / 2)
        rope_if = e->rope_if_swa;
    else if (!swa && e->rope_ff && e->n_rope_ff == hd / 2)
        rope_if = e->rope_ff;

    for (b = 0; b < B; b++)
        rmsnorm(e->pb2 + (size_t)b * hidden, e->pb + (size_t)b * hidden,
                base + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
    matmul_batch(e->pbq, e->pb2, base + mt[SLOT_Q].offset, q_dim, hidden, mt[SLOT_Q].dtype, B);
    if (has_kv) {
        matmul_batch(e->pbk, e->pb2, base + mt[SLOT_K].offset, kvd, hidden, mt[SLOT_K].dtype, B);
        if (mt[SLOT_V].size > 0)
            matmul_batch(e->pbv, e->pb2, base + mt[SLOT_V].offset, kvd, hidden, mt[SLOT_V].dtype, B);
        else
            memcpy(e->pbv, e->pbk, (size_t)B * kvd * 4);
    }

    {
        uint16_t* kcache = e->kv + (size_t)kv_layer * e->max_seq * kv_dim;
        uint16_t* vcache = e->kv + (size_t)(h->n_blocks + kv_layer) * e->max_seq * kv_dim;
        for (b = 0; b < B; b++) {
            uint32_t pos = pos_start + b;
            float* q = e->pbq + (size_t)b * q_dim;
            float* k = e->pbk + (size_t)b * kvd;
            float* v = e->pbv + (size_t)b * kvd;
            if (mt[SLOT_QBIAS].size > 0) {
                const float* bq = (const float*)(base + mt[SLOT_QBIAS].offset);
                uint32_t j; for (j = 0; j < q_dim; j++) q[j] += bq[j];
            }
            if (has_kv && mt[SLOT_KBIAS].size > 0) {
                const float* bk = (const float*)(base + mt[SLOT_KBIAS].offset);
                uint32_t j; for (j = 0; j < kvd; j++) k[j] += bk[j];
            }
            if (has_kv && mt[SLOT_VBIAS].size > 0) {
                const float* bv = (const float*)(base + mt[SLOT_VBIAS].offset);
                uint32_t j; for (j = 0; j < kvd; j++) v[j] += bv[j];
            }
            if (mt[SLOT_QNORM].size > 0)
                for (hh = 0; hh < h->n_heads; hh++)
                    rmsnorm(q + (size_t)hh * hd, q + (size_t)hh * hd,
                            base + mt[SLOT_QNORM].offset, hd, eps, mt[SLOT_QNORM].dtype);
            if (has_kv && mt[SLOT_KNORM].size > 0)
                for (hh = 0; hh < h->n_kv_heads; hh++)
                    rmsnorm(k + (size_t)hh * hd, k + (size_t)hh * hd,
                            base + mt[SLOT_KNORM].offset, hd, eps, mt[SLOT_KNORM].dtype);
            if (has_kv)
                for (hh = 0; hh < h->n_kv_heads; hh++)
                    rmsnorm_unit(v + (size_t)hh * hd, v + (size_t)hh * hd, hd, eps);
            for (hh = 0; hh < h->n_heads; hh++) {
                if (rope_if) rope_inplace_neox_if(q + (size_t)hh * hd, hd, pos, rope_if);
                else rope_inplace_qwen_ff(q + (size_t)hh * hd, hd, pos, theta, NULL);
            }
            if (has_kv) {
                for (hh = 0; hh < h->n_kv_heads; hh++) {
                    if (rope_if) rope_inplace_neox_if(k + (size_t)hh * hd, hd, pos, rope_if);
                    else rope_inplace_qwen_ff(k + (size_t)hh * hd, hd, pos, theta, NULL);
                }
                f32_to_f16_buf(k, kcache + (uint64_t)pos * kv_dim, kvd);
                f32_to_f16_buf(v, vcache + (uint64_t)pos * kv_dim, kvd);
            }
        }
        #pragma omp parallel for schedule(static)
        for (b = 0; b < B; b++) {
            uint32_t pos = pos_start + b;
            float* q = e->pbq + (size_t)b * q_dim;
            float* att_out = att_batch + (size_t)b * q_dim;
            uint32_t s0 = 0;
            if (swa && g4.swa_window > 0 && pos + 1 > g4.swa_window)
                s0 = pos + 1 - g4.swa_window;
            attn_kv_f16(att_out, q, kcache, vcache, s0, pos,
                        h->n_heads, h->n_kv_heads, hd, kv_dim, inv_d, 0.0f);
        }
    }

    matmul_batch(e->pb2, att_batch, base + mt[SLOT_O].offset, hidden, q_dim, mt[SLOT_O].dtype, B);
    for (b = 0; b < B; b++) {
        float* x = e->pb + (size_t)b * hidden;
        float* ao = e->pb2 + (size_t)b * hidden;
        float* x2 = e->hb;
        rmsnorm(x2, ao, base + mt[SLOT_NORM3].offset, hidden, eps, mt[SLOT_NORM3].dtype);
        add_inplace(x, x2, hidden);
        rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
        memcpy(e->pb2 + (size_t)b * hidden, x2, (size_t)hidden * 4);
    }
    matmul_batch(e->pbg, e->pb2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype, B);
    matmul_batch(e->pbu, e->pb2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype, B);
    for (b = 0; b < B; b++)
        geglu(e->pbg + (size_t)b * inter, e->pbg + (size_t)b * inter, e->pbu + (size_t)b * inter, inter);
    matmul_batch(e->pb2, e->pbg, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype, B);
    for (b = 0; b < B; b++) {
        float* x = e->pb + (size_t)b * hidden;
        float* ao = e->pb2 + (size_t)b * hidden;
        float* x2 = e->hb;
        rmsnorm(x2, ao, base + mt[SLOT_NORM4].offset, hidden, eps, mt[SLOT_NORM4].dtype);
        add_inplace(x, x2, hidden);
    }
    if (e->n_ple > 0 && mt[SLOT_PLE_GATE].size > 0 && e->ple_batch && ple_stride) {
        uint32_t n_ple = e->n_ple;
        uint32_t j;
        matmul_batch(e->pbg, e->pb, base + mt[SLOT_PLE_GATE].offset, n_ple, hidden, mt[SLOT_PLE_GATE].dtype, B);
        for (b = 0; b < B; b++) {
            float* gate = e->pbg + (size_t)b * n_ple;
            const float* pe = e->ple_batch + (size_t)b * ple_stride + (size_t)il * n_ple;
            gelu_inplace(gate, n_ple);
            for (j = 0; j < n_ple; j++) gate[j] *= pe[j];
        }
        matmul_batch(e->pb2, e->pbg, base + mt[SLOT_PLE_PROJ].offset, hidden, n_ple, mt[SLOT_PLE_PROJ].dtype, B);
        for (b = 0; b < B; b++) {
            float* x = e->pb + (size_t)b * hidden;
            float* ao = e->pb2 + (size_t)b * hidden;
            if (mt[SLOT_PLE_POST].size > 0)
                rmsnorm(ao, ao, base + mt[SLOT_PLE_POST].offset, hidden, eps, mt[SLOT_PLE_POST].dtype);
            add_inplace(x, ao, hidden);
        }
    }
    for (b = 0; b < B; b++) {
        float* x = e->pb + (size_t)b * hidden;
        float ls = 1.0f;
        uint32_t j;
        if (mt[SLOT_LAYER_SCALE].size > 0) {
            const uint8_t* lsptr = base + mt[SLOT_LAYER_SCALE].offset;
            if (mt[SLOT_LAYER_SCALE].dtype == DT_F32) memcpy(&ls, lsptr, 4);
            else { uint16_t lsh; memcpy(&lsh, lsptr, 2); ls = f16_to_f32(lsh); }
            for (j = 0; j < hidden; j++) x[j] *= ls;
        }
    }
    return 0;
}

int arch_gemma4_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + ws->model.dir[layer].offset;
    uint16_t* kv = e->kv;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    float eps, theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    float* x = e->x;
    float* x2 = e->hb;
    float* q = e->hb2;
    float* k = e->hb2 + h->n_heads * h->head_dim;
    float* v = e->hb2 + h->n_heads * h->head_dim + kv_dim;
    float* att_out = e->hb2 + h->n_heads * h->head_dim + 2 * kv_dim;
    const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;
    uint32_t q_dim = h->n_heads * h->head_dim;
    uint32_t hd = h->head_dim;
    uint32_t kvd = kv_dim;
    LlfGemma4Ext g4;
    uint32_t il = layer > 0 ? layer - 1 : 0;
    int has_kv = 1;
    uint32_t kv_layer = layer;
    uint32_t hh;
    const float* rope_if = NULL;
    uint32_t s0 = 0;

    llf_gemma4_ext(h, &g4);
    theta = llf_gemma4_rope_theta(h, &g4, il);
    if (g4.n_kv_shared_layers > 0 && g4.n_kv_shared_layers < h->n_blocks) {
        uint32_t kv_from = h->n_blocks - g4.n_kv_shared_layers;
        if (il >= kv_from) {
            has_kv = 0;
            kv_layer = kv_from - (llf_gemma4_is_swa(&g4, il) ? 2u : 1u) + 1u;
        }
    }
    if (mt[SLOT_K].size == 0) has_kv = 0;
    if (hidden > 0) {
        if (mt[SLOT_Q].ndim >= 2)
            q_dim = mt[SLOT_Q].shape[0] * mt[SLOT_Q].shape[1] / hidden;
        hd = h->n_heads ? q_dim / h->n_heads : hd;
        if (has_kv && mt[SLOT_K].ndim >= 2)
            kvd = mt[SLOT_K].shape[0] * mt[SLOT_K].shape[1] / hidden;
    }

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
    if (mt[SLOT_QNORM].size > 0) {
        for (hh = 0; hh < h->n_heads; hh++)
            rmsnorm(q + (size_t)hh * hd, q + (size_t)hh * hd,
                    base + mt[SLOT_QNORM].offset, hd, eps, mt[SLOT_QNORM].dtype);
    }
    if (has_kv && mt[SLOT_KNORM].size > 0) {
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm(k + (size_t)hh * hd, k + (size_t)hh * hd,
                    base + mt[SLOT_KNORM].offset, hd, eps, mt[SLOT_KNORM].dtype);
    }
    if (has_kv) {
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm_unit(v + (size_t)hh * hd, v + (size_t)hh * hd, hd, eps);
    }

    {
        uint16_t* kcache = kv + (size_t)kv_layer * e->max_seq * kv_dim;
        uint16_t* vcache = kv + (size_t)(h->n_blocks + kv_layer) * e->max_seq * kv_dim;
        uint64_t kvp = (uint64_t)pos * kv_dim;
        int swa = llf_gemma4_is_swa(&g4, il);
        if (swa && e->rope_if_swa && e->n_rope_if_swa == hd / 2)
            rope_if = e->rope_if_swa;
        else if (!swa && e->rope_ff && e->n_rope_ff == hd / 2)
            rope_if = e->rope_ff;
        for (hh = 0; hh < h->n_heads; hh++) {
            if (rope_if)
                rope_inplace_neox_if(q + (size_t)hh * hd, hd, pos, rope_if);
            else
                rope_inplace_qwen_ff(q + (size_t)hh * hd, hd, pos, theta, NULL);
        }
        if (has_kv) {
            for (hh = 0; hh < h->n_kv_heads; hh++) {
                if (rope_if)
                    rope_inplace_neox_if(k + (size_t)hh * hd, hd, pos, rope_if);
                else
                    rope_inplace_qwen_ff(k + (size_t)hh * hd, hd, pos, theta, NULL);
            }
            f32_to_f16_buf(k, kcache + kvp, kvd);
            f32_to_f16_buf(v, vcache + kvp, kvd);
        }
        if (swa && g4.swa_window > 0 && pos + 1 > g4.swa_window)
            s0 = pos + 1 - g4.swa_window;
        attn_kv_f16(att_out, q, kcache, vcache, s0, pos,
                    h->n_heads, h->n_kv_heads, hd, kv_dim, 1.0f, 0.0f);
    }
    memcpy(x2, att_out, (size_t)q_dim * 4);
    matmul(att_out, x2, base + mt[SLOT_O].offset, hidden, q_dim, mt[SLOT_O].dtype);
    {
        float ls = 1.0f;
        uint32_t j;
        rmsnorm(x2, att_out, base + mt[SLOT_NORM3].offset, hidden, eps, mt[SLOT_NORM3].dtype);
        add_inplace(x, x2, hidden);
        rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
        {
            float* fg = e->ffn;
            float* fu = e->ffn + inter;
            if (mt[SLOT_GATE].dtype == DT_W4B64) {
                uint32_t nb = hidden / W4B64_BLK;
                int8_t* xq = (int8_t*)alloca((size_t)hidden);
                float* xs = (float*)alloca((size_t)nb * 4);
                int32_t* xsum = (int32_t*)alloca((size_t)nb * 4);
                w4b64_act_quant(x2, xq, xs, xsum, hidden);
                matmul_w4b64_xq(fg, xq, xs, xsum, base + mt[SLOT_GATE].offset, inter, hidden);
                matmul_w4b64_xq(fu, xq, xs, xsum, base + mt[SLOT_UP].offset, inter, hidden);
            } else {
                matmul(fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
                matmul(fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
            }
            geglu(x2, fg, fu, inter);
        }
        matmul(att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
        rmsnorm(x2, att_out, base + mt[SLOT_NORM4].offset, hidden, eps, mt[SLOT_NORM4].dtype);
        add_inplace(x, x2, hidden);
        if (e->ple && e->n_ple > 0 && mt[SLOT_PLE_GATE].size > 0) {
            uint32_t n_ple = e->n_ple;
            float* gate = e->ple_work;
            matmul(gate, x, base + mt[SLOT_PLE_GATE].offset, n_ple, hidden, mt[SLOT_PLE_GATE].dtype);
            gelu_inplace(gate, n_ple);
            for (j = 0; j < n_ple; j++)
                gate[j] *= e->ple[(size_t)il * n_ple + j];
            matmul(att_out, gate, base + mt[SLOT_PLE_PROJ].offset, hidden, n_ple, mt[SLOT_PLE_PROJ].dtype);
            if (mt[SLOT_PLE_POST].size > 0)
                rmsnorm(att_out, att_out, base + mt[SLOT_PLE_POST].offset, hidden, eps, mt[SLOT_PLE_POST].dtype);
            add_inplace(x, att_out, hidden);
        }
        if (mt[SLOT_LAYER_SCALE].size > 0) {
            const uint8_t* lsptr = base + mt[SLOT_LAYER_SCALE].offset;
            if (mt[SLOT_LAYER_SCALE].dtype == DT_F32) memcpy(&ls, lsptr, 4);
            else { uint16_t lsh; memcpy(&lsh, lsptr, 2); ls = f16_to_f32(lsh); }
            for (j = 0; j < hidden; j++) x[j] *= ls;
        }
    }
    return 0;
}
