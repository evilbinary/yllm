#include "arch.h"
#include "yllm.h"
#include "llf.h"
#include "matvec.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

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
    .fwd_block = arch_llama_fwd_block,
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
