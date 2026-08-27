#include "yllm.h"
#include "matvec.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if !defined(_WIN32)
#include <alloca.h>
#endif
#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>
#endif

/* 批量 prefill: 默认开启; 编译期关闭用 -DYLLM_BATCH_PREFILL=0 */
#ifndef YLLM_BATCH_PREFILL
#define YLLM_BATCH_PREFILL 1
#endif
/* 批量收益拐点(实测: tinyllama/qwen3 均在 B≈16 后才优于顺序;
 * gemma4 用层外×token 内循环吃权缓存, B≥2 即有收益) */
#define PREFILL_BATCH_MIN 16
#define PREFILL_BATCH_MIN_GEMMA4 2

typedef struct {
    Ws* ws;
    uint32_t next;
    uint32_t end;
} PFJob;

typedef struct { float prob; int index; } ProbIdx;

typedef struct {
    uint32_t attn_logit_cap_bits;
    uint32_t final_logit_cap_bits;
    uint32_t swa_window;
    uint32_t swa_pattern;
    uint32_t n_kv_shared_layers;
    uint32_t n_embd_per_layer;
    uint64_t swa_mask;
    uint32_t rope_theta_swa_bits; /* 0 = 默认 10000 */
} LlfGemma4Ext;

static void llf_gemma4_ext(const LlfHeader* h, LlfGemma4Ext* ext)
{
    memset(ext, 0, sizeof(*ext));
    if (h && h->arch == ARCH_GEMMA4)
        memcpy(ext, h->reserved, sizeof(*ext));
}

static int gemma4_is_swa(const LlfGemma4Ext* ext, uint32_t il)
{
    if (ext->swa_mask) return (int)((ext->swa_mask >> il) & 1ull);
    uint32_t p = ext->swa_pattern ? ext->swa_pattern : 6;
    return (il % p) < (p - 1);
}

/* llama.cpp: GEMMA4 = NEOX RoPE; SWA 层用 rope.freq_base_swa(默认 10000) */
static float gemma4_rope_theta(const LlfHeader* h, const LlfGemma4Ext* ext, uint32_t il)
{
    float theta;
    memcpy(&theta, &h->rope_theta_bits, 4);
    if (!gemma4_is_swa(ext, il)) return theta;
    float swa = 0.0f;
    memcpy(&swa, &ext->rope_theta_swa_bits, 4);
    if (!(swa > 0.0f)) swa = 10000.0f;
    return swa;
}

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

static float llf_gemma4_attn_cap(const LlfHeader* h)
{
    LlfGemma4Ext ext;
    float cap = 0.0f;
    llf_gemma4_ext(h, &ext);
    memcpy(&cap, &ext.attn_logit_cap_bits, 4);
    return cap;
}

static float llf_gemma4_final_cap(const LlfHeader* h)
{
    LlfGemma4Ext ext;
    float cap = 0.0f;
    llf_gemma4_ext(h, &ext);
    memcpy(&cap, &ext.final_logit_cap_bits, 4);
    return cap;
}

/* resid = embed 后残差(通常 e->x); PP 下游用临时缓冲装 scaled embed, 不碰激活 */
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

static void gemma4_prepare_ple(Engine* e, uint32_t token)
{
    gemma4_prepare_ple_resid(e, token, e->x);
}

/* PP 下游: 用 token 重算 scaled embed → PLE, 不改写 e->x(中段激活) */
static void gemma4_refresh_ple_pp(Engine* e, uint32_t token)
{
    if (!e->n_ple || !e->hb) return;
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    if (h->arch != ARCH_GEMMA4) return;
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

/* prefill: tok embed 仍按 token, mproj 对 B 个激活只反量化一次 */
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

void arch_gemma4_after_embed(Engine* e, uint32_t token)
{
    uint32_t hidden = e->ws.model.h.hidden;
    float scale = sqrtf((float)hidden);
    uint32_t j;
    for (j = 0; j < hidden; j++) e->x[j] *= scale;
    gemma4_prepare_ple(e, token);
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

void arch_gemma4_refresh_ple_pp(Engine* e, uint32_t token)
{
    gemma4_refresh_ple_pp(e, token);
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

static void engine_embed_into(Engine* e, float* dst, uint32_t token)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[0].offset;
    const LlfTensorMeta* tm = &m->metas[m->base_idx[0]];
    switch (tm->dtype) {
    case DT_F32: embed_f32(dst, base + tm->offset, token, h->hidden); break;
    case DT_Q4K: embed_q4k(dst, base + tm->offset, token, h->hidden); break;
    case DT_Q6K: embed_q6k(dst, base + tm->offset, token, h->hidden); break;
    case DT_Q5K: embed_q5k(dst, base + tm->offset, token, h->hidden); break;
    case DT_IQ4XS: embed_iq4xs(dst, base + tm->offset, token, h->hidden); break;
    default: embed_f16(dst, base + tm->offset, token, h->hidden); break;
    }
}

static int cmp_prob_desc(const void* a, const void* b)
{
    float pa = ((const ProbIdx*)a)->prob;
    float pb = ((const ProbIdx*)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

typedef struct {
    Ws* ws;
    PFJob queue[256];
    uint32_t head;
    uint32_t tail;
    volatile int stop;
    void* mtx;
} Worker;

static void worker_loop(void* arg)
{
    Worker* w = (Worker*)arg;
    for (;;) {
        if (w->stop) return;
        ymutex_lock(w->mtx);
        int has = 0;
        uint32_t idx = 0, end = 0;
        if (w->head != w->tail) {
            idx = w->queue[w->head & 255].next;
            end = w->queue[w->head & 255].end;
            w->head++;
            has = 1;
        }
        ymutex_unlock(w->mtx);
        if (!has) {
            ymsleep(1);
            continue;
        }
        uint32_t i;
        for (i = idx; i < end; i++) {
            if (w->ws->pstate[i] == 1) {
                ws_prefetch(w->ws, i);
                w->ws->pstate[i] = 2;
                w->ws->resident += w->ws->layer_size[i];
            }
        }
    }
}

static void sched_enqueue(Worker* w, uint32_t begin, uint32_t end)
{
    if (begin >= end) return;
    ymutex_lock(w->mtx);
    uint32_t t = w->tail;
    w->queue[t & 255].next = begin;
    w->queue[t & 255].end = end;
    w->tail = t + 1;
    ymutex_unlock(w->mtx);
}

static void sched_ensure(Ws* ws, uint32_t layer)
{
    if (ws->pstate[layer] == 4) ws->pstate[layer] = 0; /* 上轮已释放, 本轮重新武装 */
    if (ws->pstate[layer] != 0) return;
    if (ws->budget > 0 && ws->res[layer] == 1) {
        /* 页缓存已真实驻留 → 跳过预取 syscall(自适应跳过) */
        ws->pstate[layer] = 2;
        return;
    }
#if YLLM_TENSOR_STREAM
    /* 受限模式: 层大于预算时整层预取只会被立即释放(白读磁盘),
     * 改为按需缺页(张量级释放会把已算完的张量立即 DONTNEED)。 */
    if (ws->budget > 0 && ws->layer_size[layer] > ws->budget) {
        ws->pstate[layer] = 2;
        return;
    }
#endif
    ws_prefetch(ws, layer);
    ws->pstate[layer] = 2;
    ws->resident += ws->layer_size[layer];
}

/* 用 mincore 刷新每层真实驻留位图,并重算驻留字节(内存受限模式才调用) */
static void sched_refresh_resident(Ws* ws)
{
    uint32_t i;
    uint64_t tot = 0;
    for (i = 0; i < ws->model.n_layers; i++) {
        int r = wmap_resident(&ws->map, ws->model.dir[i].offset, ws->model.dir[i].size);
        if (r == 1) { ws->res[i] = 1; tot += ws->layer_size[i]; }
        else if (r == 0) { ws->res[i] = 0; }
    }
    ws->resident = tot;
}

/* 预算自适应: 缺页太多且内存富余 → 多驻留一层(减重读); 内存告急 → 缩驻留。
 * 只调层数上限 budget_layers; 字节上限由 sched_release_budget 的 bytes <= budget 硬约束保证。 */
static void sched_adapt_budget(Ws* ws)
{
#ifndef _WIN32
    if (ws->budget == 0) return;
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return;
    long pf = ru.ru_majflt;
    long delta = pf - ws->last_majflt;
    ws->last_majflt = pf;
    long avph = sysconf(_SC_AVPHYS_PAGES);
    long pgsz = sysconf(_SC_PAGESIZE);
    uint64_t mem_free = (avph > 0 && pgsz > 0) ? (uint64_t)avph * (uint64_t)pgsz : 0;
    uint64_t cap = ws->budget;
    if (delta > 0 && mem_free > cap + cap / 2) {
        if (ws->budget_layers < ws->model.n_layers) ws->budget_layers++;
    } else if (mem_free < cap) {
        if (ws->budget_layers > 1) ws->budget_layers--;
    }
#else
    (void)ws;
#endif
}

static void sched_release_budget(Ws* ws, uint32_t cur)
{
    if (ws->budget == 0) return;
    uint32_t i;
    for (;;) {
        uint32_t n = 0;
        uint64_t bytes = 0;
        for (i = 0; i < ws->model.n_layers; i++) {
            /* 已预取(pstate==2)或已算过(pstate==3)的层都算占用;
             * pstate==4 已释放, 不计入(否则统计永不下降 → 死循环) */
            if (ws->pstate[i] == 2 || ws->pstate[i] == 3) {
                n++; bytes += ws->layer_size[i];
            }
        }
        if (n <= ws->budget_layers && bytes <= ws->budget) return;
        int found = 0;
        for (i = 0; i < cur; i++) {
            if (ws->pstate[i] == 2 || ws->pstate[i] == 3) {
                ws_release(ws, i);
                ws->pstate[i] = 4;
                ws->res[i] = 0;
                ws->resident -= ws->layer_size[i];
                found = 1;
                break;
            }
        }
        if (!found) return;
    }
}

/* ---- 层前向: CPU 图在 ArchOps; GPU 覆盖在 Device.fwd_block ---- */
#if YLLM_TENSOR_STREAM
/* 页对齐的 lm_head 分块行数: 最大 C <= max_rows 使 C*行字节 % 4096 == 0。
 * 非量化(整行矩阵)返回 0 → 不分块整算。 */
static uint32_t head_chunk_rows(size_t rbytes, uint32_t max_rows)
{
    if (rbytes == 0) return 0;
    uint64_t g = 4096, r = rbytes;
    while (r) { uint64_t t = g % r; g = r; r = t; }
    uint64_t per = 4096 / g; /* 页对齐的最小行数 */
    uint64_t c = (uint64_t)max_rows / per * per;
    return c >= per ? (uint32_t)c : 0;
}

static void lm_head_chunked(Engine* e, const uint8_t* w, uint64_t w_file_off,
                            uint32_t hidden, uint32_t vocab, uint32_t dtype)
{
    Ws* ws = &e->ws;
    size_t rbytes = matmul_row_bytes(dtype, hidden);
    uint32_t chunk = head_chunk_rows(rbytes,
        ws->budget < 32u * 1024 * 1024 ? 2048u : 4096u);
    if (chunk == 0) {
        matmul(e->logits, e->x, w, vocab, hidden, dtype);
        return;
    }
    const float* xa = e->x;
    int pre = (dtype == DT_Q6K);
    if (pre) {
        matvec_q8k_quant(e->x, e->hb, hidden);
        xa = e->hb;
    }
    {
        uint32_t rows = 0;
        while (rows + chunk <= vocab) {
            if (pre)
                matmul_rows_preq(e->logits + rows, xa, w, rows, chunk, hidden, vocab, dtype);
            else
                matmul_rows(e->logits + rows, e->x, w, rows, chunk, hidden, vocab, dtype);
            ws_release_aligned(ws, w_file_off + (uint64_t)rows * rbytes,
                               (uint64_t)chunk * rbytes);
            rows += chunk;
        }
        if (rows < vocab) {
            uint32_t n = vocab - rows;
            if (pre)
                matmul_rows_preq(e->logits + rows, xa, w, rows, n, hidden, vocab, dtype);
            else
                matmul_rows(e->logits + rows, e->x, w, rows, n, hidden, vocab, dtype);
        }
    }
}
#endif

int engine_init(Engine* e, const char* model_path, uint64_t budget, int depth, char* err, size_t errlen)
{
    memset(e, 0, sizeof(*e));
    yllm_tune_cpu();
    if (wmap_open(model_path, &e->ws.map) != 0) {
        snprintf(err, errlen, "cannot open %s", model_path);
        return -1;
    }
    if (llf_read(&e->ws.map, &e->ws.model) != 0) {
        LlfHeader* h = (e->ws.map.base && e->ws.map.size >= sizeof(LlfHeader))
            ? (LlfHeader*)e->ws.map.base : NULL;
        if (h)
            snprintf(err, errlen, "bad llf file %s: magic=%.8s ver=%u (need %u)",
                     model_path, h->magic, h->version, (unsigned)YLLM_VERSION);
        else
            snprintf(err, errlen, "bad llf file %s", model_path);
        wmap_close(&e->ws.map);
        return -1;
    }
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    ws->budget = budget;
    
    ws->depth = depth > 0 ? depth : 2;
    ws->pstate = (uint8_t*)ycalloc(m->n_layers, 1);
    ws->res = (uint8_t*)ycalloc(m->n_layers, 1);
    ws->layer_size = (uint64_t*)ymalloc((size_t)m->n_layers * 8);
    uint32_t i;
    for (i = 0; i < m->n_layers; i++) {
        ws->layer_size[i] = m->dir[i].size;
    }
    /* 自适应层预算初值: 由字节预算按平均层大小折算 */
    ws->budget_layers = m->n_layers;
    if (budget > 0) {
        uint64_t avg = 0;
        for (i = 0; i < m->n_layers; i++) avg += ws->layer_size[i];
        avg = m->n_layers ? avg / m->n_layers : 0;
        uint32_t bl = avg ? (uint32_t)(budget / avg) : 1;
        if (bl < 1) bl = 1;
        if (bl > m->n_layers) bl = m->n_layers;
        ws->budget_layers = bl;
    }
    ws->last_majflt = 0;

    uint32_t hidden = m->h.hidden;
    uint32_t kv_dim = m->h.n_kv_heads * m->h.head_dim;
    uint32_t q_dim = m->h.n_heads * m->h.head_dim;
    uint32_t vocab = m->h.vocab;
    /* FFN 中间维度取各层最大值(如 gemma4 不同层 inter 可能不同) */
    uint32_t inter = hidden;
    if (m->n_layers > 1 && m->base_idx && m->metas) {
        uint32_t li;
        for (li = 1; li <= m->h.n_blocks; li++) {
            uint32_t bi = m->base_idx[li];
            const LlfTensorMeta* g;
            uint64_t prod;
            if (bi + SLOT_GATE >= m->n_layers * BLOCK_TENSORS_MTP) continue;
            g = &m->metas[bi + SLOT_GATE];
            if (g->ndim < 2 || !g->shape[0] || !g->shape[1]) continue;
            prod = (uint64_t)g->shape[0] * g->shape[1];
            if (hidden > 0 && prod % hidden == 0) {
                uint32_t cur = (uint32_t)(prod / hidden);
                if (cur > inter) inter = cur;
            }
        }
    }
    e->inter = inter;
    e->layer_begin = 0;
    e->layer_end = m->n_layers;
    e->kv_dim = kv_dim;
    e->max_seq = m->h.max_seq;
    e->kv = (uint16_t*)ycalloc((size_t)(2 * m->h.n_blocks + 1) * e->max_seq * kv_dim, 2);
    e->x = (float*)ycalloc(hidden, 4);
    e->hb = (float*)ycalloc(hidden, 4 * 9);
    /* hb2 需容纳 q(q_dim) + k(kv_dim) + v(kv_dim) + att_out(hidden); 至少 9*hidden */
    {
        uint32_t hb2_min = q_dim + 2 * kv_dim + hidden;
        uint32_t hb2_sz = hb2_min > hidden * 9 ? hb2_min : hidden * 9;
        e->hb2 = (float*)ycalloc(hb2_sz, 4);
    }
    e->ffn = (float*)ycalloc((size_t)2 * inter, 4);
    e->att = (float*)ymalloc((size_t)e->max_seq * m->h.n_heads * 4);
    e->logits = (float*)ymalloc((size_t)vocab * 4);
    e->n_ple = 0;
    e->ple = NULL;
    e->ple_work = NULL;
    e->ple_batch = NULL;
    e->rope_ff = NULL;
    e->n_rope_ff = 0;
    e->rope_if_swa = NULL;
    e->n_rope_if_swa = 0;
    if (m->h.arch == ARCH_GEMMA4) {
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
                if (!gemma4_is_swa(&g4, li - 1)) continue;
                if (mt[SLOT_Q].ndim >= 2 && hidden > 0 && m->h.n_heads) {
                    uint32_t qd = mt[SLOT_Q].shape[0] * mt[SLOT_Q].shape[1] / hidden;
                    hd_s = qd / m->h.n_heads;
                }
                swa_il = li - 1;
                break;
            }
            if (hd_s >= 2) {
                float ths = gemma4_rope_theta(&m->h, &g4, swa_il);
                e->n_rope_if_swa = hd_s / 2;
                e->rope_if_swa = (float*)ymalloc((size_t)e->n_rope_if_swa * 4);
                gemma4_fill_inv_freq(e->rope_if_swa, e->n_rope_if_swa, hd_s, ths, 0);
            }
        }
    }
    /* 批量 prefill 工作区 */
    {
        uint32_t PB_MAX = 64;
        uint32_t kv_dim2 = m->h.n_kv_heads * m->h.head_dim;
        uint32_t q_dim_max = hidden;
        if (m->h.arch == ARCH_GEMMA4 && hidden > 0) {
            uint32_t li;
            for (li = 1; li <= m->h.n_blocks; li++) {
                const LlfTensorMeta* mt = &m->metas[m->base_idx[li]];
                if (mt[SLOT_Q].ndim >= 2) {
                    uint32_t qd = mt[SLOT_Q].shape[0] * mt[SLOT_Q].shape[1] / hidden;
                    if (qd > q_dim_max) q_dim_max = qd;
                }
            }
        }
        e->pb_cap = PB_MAX;
        e->pbq_dim = q_dim_max;
        e->pb  = (float*)ycalloc((size_t)PB_MAX * hidden, 4);
        e->pb2 = (float*)ycalloc((size_t)PB_MAX * hidden, 4);
        e->pbq = (float*)ycalloc((size_t)PB_MAX * q_dim_max, 4);
        e->pbk = (float*)ycalloc((size_t)PB_MAX * kv_dim2, 4);
        e->pbv = (float*)ycalloc((size_t)PB_MAX * kv_dim2, 4);
        e->pbg = (float*)ycalloc((size_t)PB_MAX * inter, 4);
        e->pbu = (float*)ycalloc((size_t)PB_MAX * inter, 4);
        e->pba = (float*)ymalloc((size_t)PB_MAX * m->h.n_heads * e->max_seq * 4);
        if (m->h.arch == ARCH_GEMMA4 && e->n_ple > 0) {
            size_t psz = (size_t)e->n_ple * m->h.n_blocks;
            e->ple_batch = (float*)ycalloc((size_t)PB_MAX * psz, 4);
            free(e->ple_work);
            e->ple_work = (float*)ycalloc((size_t)PB_MAX * psz, 4);
        }
    }

    /* qwen35 混合架构: 分配 GDN 状态/conv 延迟线/临时工作区 */
    if (m->h.arch == ARCH_QWEN35) {
        uint32_t n_gdn = 0;
        uint32_t conv_chan = 0, kwidth = 0;
        uint32_t n_vheads = 0, hvd = 0;
        uint32_t li;
        for (li = 1; li <= m->h.n_blocks; li++) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[li]];
            if (mt[SLOT_SSM_CONV1D].size > 0) {
                n_gdn++;
                if (conv_chan == 0) {
                    conv_chan = mt[SLOT_SSM_CONV1D].shape[1];
                    kwidth = mt[SLOT_SSM_CONV1D].shape[0];
                    n_vheads = mt[SLOT_SSM_A].shape[0];
                    hvd = mt[SLOT_SSM_NORM].shape[0];
                }
            }
        }
        if (n_gdn > 0 && conv_chan > 0 && n_vheads > 0 && hvd > 0) {
            e->ssm_state = (float*)ycalloc((size_t)n_gdn * n_vheads * hvd * hvd, 4);
            e->ssm_conv = (float*)ycalloc((size_t)n_gdn * (kwidth > 0 ? kwidth : 1) * conv_chan, 4);
            e->scratch = (float*)ymalloc(65536 * 4);
        }
    }

    Worker* w = (Worker*)ycalloc(1, sizeof(Worker));
    w->ws = ws;
    e->ws.worker = w;
    {
        void* mt = NULL;
        ymutex_create(&mt);
        w->mtx = mt;
        void* th = NULL;
        if (ythread_create(&th, worker_loop, w) != 0) {
            w->stop = 1;
        } else {
            w->stop = 0;
            e->ws.worker_th = th;
        }
    }
    e->ws.pstate = ws->pstate;
    /* MTP 权重检测: output 层槽 24..27 是否存在(eh_proj 为标志) */
    {
        uint32_t ol = m->h.n_blocks + 2;
        const LlfTensorMeta* tm = &m->metas[m->base_idx[ol] + SLOT_MTP_EH];
        if (tm->size > 0 && tm->name[0]) {
            e->mtp_eh_slot = SLOT_MTP_EH;
            e->mtp_enorm_slot = SLOT_MTP_ENORM;
            e->mtp_hnorm_slot = SLOT_MTP_HNORM;
            e->mtp_headnorm_slot = SLOT_MTP_HEAD_NORM;
            e->mtp_layer = m->h.n_blocks;   /* blk.64 = MTP 块, 不进主干 */
            e->mtp_h = (float*)ycalloc(m->h.hidden, 4);
            e->mtp_logits = (float*)ymalloc((size_t)m->h.vocab * 4);
            ylog_info("engine: MTP weights detected (eh_proj=%s size=%llu, trunk=%u layers)",
                      tm->name, (unsigned long long)tm->size, m->h.n_blocks - 1);
        } else {
            e->mtp_eh_slot = 0;
            e->mtp_layer = 0;
        }
    }
    /* 层前向: 按 header.arch 绑 const 图 */
    e->arch = m->h.arch;
    e->ops = arch_lookup(m->h.arch);
    if (!e->ops) {
        snprintf(err, errlen, "unknown arch %u", m->h.arch);
        engine_free(e);
        return -1;
    }
#if defined(__aarch64__)
    {
        int i8mm = 0;
#if defined(__linux__)
        {
            unsigned long hw2 = 0;
#  if defined(__ANDROID__) || defined(__linux__)
            /* AT_HWCAP2: bit 13 = I8MM on aarch64 */
            extern unsigned long getauxval(unsigned long);
#    ifndef AT_HWCAP2
#      define AT_HWCAP2 26
#    endif
            hw2 = getauxval(AT_HWCAP2);
            i8mm = (hw2 & (1ul << 13)) != 0;
#  endif
        }
#elif defined(__ARM_FEATURE_MATMUL_INT8)
        i8mm = 1;
#endif
        w4b64_set_hw_i8mm(i8mm);
        {
            int arm86 = (m->h.reserved[LLF_W4_LAYOUT_OFF] == LLF_W4_LAYOUT_ARM86);
            w4b64_set_layout_arm86(arm86);
            /* GEMV 即使用 Arm86 无 zip, smmla 仍慢于 sdot; 仅 batch 走 smmla */
            w4b64_set_prefer_i8mm(0);
            if (i8mm && arm86)
                ylog_info("w4: Arm86: batch smmla (no zip), decode sdot+unz");
            else if (i8mm)
                ylog_info("w4: ARM i8mm: batch smmla, decode sdot");
            else if (arm86)
                ylog_info("w4: Arm86 file, decode sdot+unz");
            else
                ylog_info("w4: using Arm82 sdot");
        }
    }
#endif
    if (m->h.arch == ARCH_GEMMA4) {
        LlfGemma4Ext g4log;
        llf_gemma4_ext(&m->h, &g4log);
        ylog_info("gemma4: n_ple=%u shared_kv=%u swa_win=%u swa_pat=%u rope_freqs=%u",
                  e->n_ple, g4log.n_kv_shared_layers, g4log.swa_window, g4log.swa_pattern,
                  e->n_rope_ff);
    }
    /* 默认 CPU 设备: load_weights 空操作, 后续可 engine_bind_device(CUDA) */
    if (engine_bind_device(e, DEV_CPU, 0, err, errlen) != 0) {
        engine_free(e);
        return -1;
    }
    return 0;
}

int engine_bind_device(Engine* e, DeviceKind kind, int device_id, char* err, size_t errlen)
{
    if (!e) {
        if (err && errlen) snprintf(err, errlen, "null engine");
        return -1;
    }
    if (e->dev) {
        if (e->dev->free_dev) e->dev->free_dev(e);
        device_destroy(e->dev);
        e->dev = NULL;
    }
    e->weights_ready = 0;
    e->device_mode = DEV_MODE_CPU;
    e->w_dev = NULL;
    e->d_kv = NULL;
    e->dev = device_create(kind, device_id, err, errlen);
    if (!e->dev) return -1;
    if (!e->dev->load_weights) {
        if (err && errlen) snprintf(err, errlen, "device missing load_weights");
        device_destroy(e->dev);
        e->dev = NULL;
        return -1;
    }
    if (e->dev->load_weights(e, err, errlen) != 0) {
        if (e->dev->free_dev) e->dev->free_dev(e);
        device_destroy(e->dev);
        e->dev = NULL;
        return -1;
    }
    return 0;
}

int engine_load_weights(Engine* e, char* err, size_t errlen)
{
    if (!e || !e->dev || !e->dev->load_weights) {
        if (err && errlen) snprintf(err, errlen, "no device / load_weights");
        return -1;
    }
    e->weights_ready = 0;
    return e->dev->load_weights(e, err, errlen);
}

void engine_free(Engine* e)
{
    Worker* w;
    if (e->dev) {
        if (e->dev->free_dev) e->dev->free_dev(e);
        device_destroy(e->dev);
        e->dev = NULL;
    }
    w = (Worker*)e->ws.worker;
    if (w) {
        w->stop = 1;
        if (e->ws.worker_th) ythread_join(&e->ws.worker_th);
        ymutex_destroy(w->mtx);
    }
    wmap_close(&e->ws.map);
    free(e->kv);
    free(e->x);
    free(e->hb);
    free(e->hb2);
    free(e->ffn);
    free(e->att);
    free(e->logits);
    if (e->mtp_h) free(e->mtp_h);
    if (e->mtp_logits) free(e->mtp_logits);
    free(e->pb); free(e->pb2); free(e->pbq);
    free(e->pbk); free(e->pbv); free(e->pbg); free(e->pbu);
    free(e->pba);
    if (e->ssm_state) free(e->ssm_state);
    if (e->ssm_conv) free(e->ssm_conv);
    if (e->scratch) free(e->scratch);
    if (e->ple) free(e->ple);
    if (e->ple_work) free(e->ple_work);
    if (e->ple_batch) free(e->ple_batch);
    if (e->rope_ff) free(e->rope_ff);
    if (e->rope_if_swa) free(e->rope_if_swa);
    if (e->ws.model.base_idx) free(e->ws.model.base_idx);
    if (e->ws.pstate) free(e->ws.pstate);
    if (e->ws.res) free(e->ws.res);
    if (e->ws.layer_size) free(e->ws.layer_size);
    free(w);
    memset(e, 0, sizeof(*e));
}

/* 批量前向一层: 同时处理 B 个 token(pos 连续: pos_start..pos_start+B-1) */
int arch_llama_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
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
            if (h->arch == ARCH_QWEN)
                rope_inplace_qwen(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
            else
                rope_inplace(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        }
        for (hh = 0; hh < h->n_kv_heads; hh++) {
            if (h->arch == ARCH_QWEN)
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

/* 批量 prefill: 一次处理 n 个 prompt token(start_pos 起), 结果 logits 为最后 token */
int engine_forward_prefill(Engine* e, const uint32_t* tokens, int n, int start_pos)
{
    if (e->dev && e->dev->prefill) {
        if (e->dev->prefill(e, tokens, n, start_pos) == 0)
            return 0;
        /* GPU 失败: gemma/qwen35 走 CPU 批; 其余退回逐 token */
        if (!e->ops || !e->ops->cpu_batch_prefill) {
            int i;
            for (i = 0; i < n; i++) {
                if (engine_forward(e, tokens[i], (uint32_t)(start_pos + i)) != 0)
                    return -1;
            }
            return 0;
        }
    }
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t B = e->pb_cap ? e->pb_cap : 16;
    uint32_t batch_min = (e->ops && e->ops->id == ARCH_GEMMA4) ? PREFILL_BATCH_MIN_GEMMA4 : PREFILL_BATCH_MIN;
    int off = 0;
    while (off < n) {
        uint32_t nb = (uint32_t)(n - off);
        if (nb > B) nb = B;
        if (nb < batch_min) {
            uint32_t i;
            for (i = 0; i < nb; i++)
                engine_forward(e, tokens[off + i], (uint32_t)(start_pos + off + i));
            off += (int)nb;
            continue;
        }
        uint32_t b;
        for (b = 0; b < nb; b++)
            engine_embed_into(e, e->pb + (size_t)b * hidden, tokens[off + b]);
        if (e->ops && e->ops->after_embed_batch)
            e->ops->after_embed_batch(e, tokens + off, nb);
        uint32_t i;
        uint32_t trunk = h->n_blocks - (e->mtp_layer ? 1u : 0u);
        for (i = 1; i <= trunk; i++) {
            if (ws->budget > 0) sched_ensure(ws, i);
            engine_call_fwd_block_batch(e, i, (uint32_t)(start_pos + off), nb);
            if (ws->budget > 0) sched_release_budget(ws, i);
        }
        if (e->mtp_h) {
            memcpy(e->mtp_h, e->pb + (size_t)(nb - 1) * hidden, (size_t)hidden * 4);
            e->mtp_h_ready = 1;
        }
        {
            const uint8_t* base = (const uint8_t*)ws->map.base;
            const LlfTensorMeta* fn = &m->metas[m->base_idx[h->n_blocks + 1]];
            const LlfTensorMeta* out = &m->metas[m->base_idx[h->n_blocks + 2]];
            float eps;
            memcpy(&eps, &h->norm_eps_bits, 4);
            if (off + (int)nb == n) {
                float* xlast = e->pb + (size_t)(nb - 1) * hidden;
                rmsnorm(e->x, xlast, base + m->dir[h->n_blocks + 1].offset + fn->offset,
                        hidden, eps, fn->dtype);
#if YLLM_TENSOR_STREAM
                if (ws->budget > 0) {
                    size_t rbytes = matmul_row_bytes(out->dtype, h->hidden);
                    uint32_t dl = h->n_blocks + 2;
                    if (rbytes > 0)
                        lm_head_chunked(e, base + m->dir[dl].offset + out->offset,
                                        m->dir[dl].offset + out->offset,
                                        h->hidden, h->vocab, out->dtype);
                    else
                        matmul(e->logits, e->x, base + m->dir[dl].offset + out->offset,
                               h->vocab, hidden, out->dtype);
                } else {
                    matmul(e->logits, e->x, base + m->dir[h->n_blocks + 2].offset + out->offset,
                           h->vocab, hidden, out->dtype);
                }
#else
                matmul(e->logits, e->x, base + m->dir[h->n_blocks + 2].offset + out->offset,
                       h->vocab, hidden, out->dtype);
#endif
                if (e->ops && e->ops->post_logits)
                    e->ops->post_logits(e);
            }
        }
        off += (int)nb;
    }
    return 0;
}

/* qwen35: 返回该 gdn 层在所有 GDN 层中的序号(0-based); 非 gdn 层返回下一 gdn 序号 */
static uint32_t gdn_index_of(const LlModel* m, uint32_t layer)
{
    uint32_t cnt = 0, i;
    for (i = 1; i <= layer; i++) {
        const LlfTensorMeta* mt = &m->metas[m->base_idx[i]];
        if (mt[SLOT_SSM_CONV1D].size > 0) cnt++;
    }
    return cnt ? cnt - 1 : 0;
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
    theta = gemma4_rope_theta(h, &g4, il);
    swa = gemma4_is_swa(&g4, il);
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

/* qwen35 混合架构批量前向: GDN 状态在层内连续, 退化为逐 token */
int arch_qwen35_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
{
    /* 批量输入/输出在 e->pb, 而 arch_qwen35_fwd_block 就地读写 e->x: 逐 token 拷贝进出 */
    uint32_t hidden = e->ws.model.h.hidden;
    uint32_t b;
    for (b = 0; b < B; b++) {
        memcpy(e->x, e->pb + (size_t)b * hidden, (size_t)hidden * 4);
        if (e->ops->fwd_block(e, layer, pos_start + b) != 0) return -1;
        memcpy(e->pb + (size_t)b * hidden, e->x, (size_t)hidden * 4);
    }
    return 0;
}

/* qwen35 混合架构单层前向: Gated Attention 层 + GDN 层 */
int arch_qwen35_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[layer].offset;
    uint32_t hidden = h->hidden;
    float* x = e->x;
    float eps, theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    if (getenv("YLLM_QDBG")) {
        float xn = 0; uint32_t kk;
        for (kk = 0; kk < hidden; kk++) if (x[kk] != x[kk]) xn++;
        fprintf(stderr, "[qdbg] layer=%u pos=%u x_nan=%g x[0..9]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                layer, pos, (double)xn, (double)x[0], (double)x[1], (double)x[2], (double)x[3],
                (double)x[4], (double)x[5], (double)x[6], (double)x[7], (double)x[8], (double)x[9]);
    }
    uint32_t hidx = m->base_idx[layer];
    const LlfTensorMeta* mt = &m->metas[hidx];
    float* x2 = e->hb;
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;

    if (mt[SLOT_SSM_CONV1D].size > 0) {
        /* ===== GDN 层 ===== */
        uint32_t conv_chan = mt[SLOT_SSM_CONV1D].shape[1];   /* 10240 */
        uint32_t kwidth = mt[SLOT_SSM_CONV1D].shape[0];      /* 4 */
        uint32_t n_vheads = mt[SLOT_SSM_A].shape[0];         /* 48 */
        uint32_t hvd = mt[SLOT_SSM_NORM].shape[0];           /* 128 */
        uint32_t value_dim = hvd * n_vheads;                 /* 6144 */
        uint32_t key_dim = (conv_chan - value_dim) / 2;      /* 2048 */
        uint32_t num_k_heads = key_dim / hvd;                /* 16 */

        float* scratch = e->scratch;
        float* qkv = scratch;              /* [conv_chan] conv 输入/输出 */
        float* z = scratch + 16384;        /* [value_dim] */
        float* alpha = scratch + 32768;    /* [n_vheads] */
        float* beta = scratch + 32816;     /* [n_vheads] */
        float* gate = scratch + 32864;     /* [n_vheads] */
        float* attn_out = scratch + 33000; /* [value_dim] */
        float* ssm_o = scratch + 40000;    /* [hidden] ssm_out 输出缓冲 */

        rmsnorm(x2, x, base + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
        matmul(qkv, x2, base + mt[SLOT_QKV].offset, conv_chan, hidden, mt[SLOT_QKV].dtype);
        matmul(z, x2, base + mt[SLOT_GATE_ATTN].offset, value_dim, hidden, mt[SLOT_GATE_ATTN].dtype);
        matmul(alpha, x2, base + mt[SLOT_SSM_ALPHA].offset, n_vheads, hidden, mt[SLOT_SSM_ALPHA].dtype);
        matmul(beta, x2, base + mt[SLOT_SSM_BETA].offset, n_vheads, hidden, mt[SLOT_SSM_BETA].dtype);

        const float* ssm_dt = (const float*)(base + mt[SLOT_SSM_DT].offset);
        const float* ssm_a = (const float*)(base + mt[SLOT_SSM_A].offset);
        uint32_t vh;
        for (vh = 0; vh < n_vheads; vh++) {
            beta[vh] = 1.0f / (1.0f + expf(-beta[vh]));
            float ab = alpha[vh] + ssm_dt[vh];
            float sp = log1pf(expf(ab));
            gate[vh] = sp * ssm_a[vh];
        }
        if (getenv("YLLM_QDBG")) {
            fprintf(stderr, "[qdbg] layer=%u gate[0..2]=%g %g %g alpha0=%g dt0=%g a0=%g beta0=%g\n", layer,
                    (double)gate[0], (double)gate[1], (double)gate[2],
                    (double)alpha[0], (double)ssm_dt[0], (double)ssm_a[0], (double)beta[0]);
        }

        /* conv1d: 延迟线滑入当前输入, 卷积+silu 就地到 qkv */
        uint32_t gdn_idx = gdn_index_of(m, layer);
        float* conv_state = e->ssm_conv + (size_t)gdn_idx * kwidth * conv_chan;
        const uint8_t* conv_w = base + mt[SLOT_SSM_CONV1D].offset;
        if (getenv("YLLM_QDBG") && layer == 1) {
            fprintf(stderr, "[qdbg] layer=1 conv_pre[0..2]=%g %g %g wf3=%g\n",
                    (double)qkv[0], (double)qkv[1], (double)qkv[2],
                    (double)((const float*)conv_w)[0 * kwidth + 3]);
        }
        conv1d_update(conv_state, qkv, conv_w, conv_chan, kwidth);
        if (getenv("YLLM_QDBG") && layer == 1) {
            fprintf(stderr, "[qdbg] layer=1 conv_post[0..2]=%g %g %g\n",
                    (double)qkv[0], (double)qkv[1], (double)qkv[2]);
        }
        uint32_t ch;
        for (ch = 0; ch < conv_chan; ch++) qkv[ch] = qkv[ch] / (1.0f + expf(-qkv[ch]));  /* silu */
        if (getenv("YLLM_QDBG") && layer == 1) {
            fprintf(stderr, "[qdbg] layer=1 conv_silu[0..2]=%g %g %g\n",
                    (double)qkv[0], (double)qkv[1], (double)qkv[2]);
        }

        /* 切 q/k/v 并 L2 归一化 q/k 每 k_head */
        float* q = qkv;
        float* k = qkv + key_dim;
        float* v = qkv + 2 * key_dim;
        uint32_t i;
        for (i = 0; i < num_k_heads; i++) l2norm_inplace(q + (size_t)i * hvd, hvd, eps);
        for (i = 0; i < num_k_heads; i++) l2norm_inplace(k + (size_t)i * hvd, hvd, eps);

        /* 递归状态更新 + 注意力输出 */
        float* state_base = e->ssm_state + (size_t)gdn_idx * n_vheads * hvd * hvd;
        gdn_state_update(state_base, attn_out, q, k, v, gate, beta, num_k_heads, hvd, n_vheads, hvd);
        if (getenv("YLLM_QDBG") && layer == 1) {
            fprintf(stderr, "[qdbg] layer=1 state_out[0..2]=%g %g %g z[0..2]=%g %g %g q0=%g k0=%g v0=%g\n",
                    (double)attn_out[0], (double)attn_out[1], (double)attn_out[2],
                    (double)z[0], (double)z[1], (double)z[2],
                    (double)q[0], (double)k[0], (double)v[0]);
        }

        /* rmsnorm_gated: out = rmsnorm(attn_out, ssm_norm) * silu(z) */
        const uint8_t* ssm_norm_w = base + mt[SLOT_SSM_NORM].offset;
        uint32_t snd = mt[SLOT_SSM_NORM].dtype;
        if (getenv("YLLM_QDBG") && layer == 1) {
            float s0 = 0.0f; uint32_t kk;
            for (kk = 0; kk < hvd; kk++) s0 += attn_out[kk]*attn_out[kk];
            float inv0 = 1.0f/sqrtf(s0/(float)hvd + eps);
            float w0 = (snd == DT_F32) ? ((const float*)ssm_norm_w)[0] : f16_to_f32(((const uint16_t*)ssm_norm_w)[0]);
            fprintf(stderr, "[qdbg] layer=1 rg_pre: s=%g inv=%g w0=%g x0=%g z0=%g snd=%u\n",
                    (double)s0, (double)inv0, (double)w0, (double)attn_out[0], (double)z[0], snd);
        }
        for (vh = 0; vh < n_vheads; vh++)
            rmsnorm_gated(attn_out + (size_t)vh * hvd, attn_out + (size_t)vh * hvd,
                          ssm_norm_w, z + (size_t)vh * hvd, hvd, eps, snd);
        if (getenv("YLLM_QDBG") && layer == 1) {
            fprintf(stderr, "[qdbg] layer=1 rg_post[0..2]=%g %g %g\n",
                    (double)attn_out[0], (double)attn_out[1], (double)attn_out[2]);
        }

        /* 输出投影 + 残差 + FFN */
        if (getenv("YLLM_QDBG") && layer == 1 && pos == 0)
            fprintf(stderr, "[qdbg] layer1 pre-ssmo attn_out[0]=%g ssm_o=%p attn_out=%p dtype=%u off=%llu\n", (double)attn_out[0], (void*)ssm_o, (void*)attn_out, mt[SLOT_SSM_OUT].dtype, (unsigned long long)mt[SLOT_SSM_OUT].offset);
        matmul(ssm_o, attn_out, base + mt[SLOT_SSM_OUT].offset, hidden, value_dim, mt[SLOT_SSM_OUT].dtype);
        if (getenv("YLLM_QDBG") && layer == 1 && pos == 0)
            fprintf(stderr, "[qdbg] layer1 post-ssmo attn_out[0]=%g\n", (double)attn_out[0]);
        uint32_t j;
        for (j = 0; j < hidden; j++) x[j] += ssm_o[j];
        if (getenv("YLLM_QDBG")) {
            fprintf(stderr, "[qdbg] gdn layer=%u ssm_o[0..2]=%g %g %g x[0..2]=%g %g %g\n",
                    layer, (double)ssm_o[0], (double)ssm_o[1], (double)ssm_o[2],
                    (double)x[0], (double)x[1], (double)x[2]);
        }
        rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
        if (getenv("YLLM_QDBG")) {
            fprintf(stderr, "[qdbg] gdn layer=%u x2[0..2]=%g %g %g\n", layer, (double)x2[0], (double)x2[1], (double)x2[2]);
        }
        float* fg = e->ffn;
        float* fu = e->ffn + inter;
        matmul(fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
        matmul(fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
        swiglu(x2, fg, fu, inter);
        matmul(attn_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
        for (j = 0; j < hidden; j++) x[j] += attn_out[j];
        return 0;
    }

    /* ===== Gated Attention 层 ===== */
    {
        uint32_t n_heads = h->n_heads;        /* 24 */
        uint32_t n_kv = h->n_kv_heads;        /* 4 */
        uint32_t hd = h->head_dim;            /* 256 */
        uint32_t qdim = n_heads * hd;         /* 6144 */
        uint32_t kv_dim = n_kv * hd;          /* 1024 */
        uint32_t n_rot = 64;                  /* rope.dimension_count */

        float* scratch = e->scratch;
        float* q = scratch;                   /* [6144] q 部分(转换期已拆出) */
        float* gate = scratch + qdim;         /* [6144] gate 部分 */
        float* k = scratch + 2 * qdim;        /* [1024] */
        float* v = scratch + 2 * qdim + kv_dim;/* [1024] */
        float* att_out = e->hb2;              /* [6144] 复用 */

        rmsnorm(x2, x, base + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
        matmul(q, x2, base + mt[SLOT_Q].offset, qdim, hidden, mt[SLOT_Q].dtype);
        matmul(gate, x2, base + mt[SLOT_QGATE].offset, qdim, hidden, mt[SLOT_QGATE].dtype);
        matmul(k, x2, base + mt[SLOT_K].offset, kv_dim, hidden, mt[SLOT_K].dtype);
        matmul(v, x2, base + mt[SLOT_V].offset, kv_dim, hidden, mt[SLOT_V].dtype);

        uint32_t hh, ii;
        if (getenv("YLLM_QDBG")) {
            float qn = 0, kn = 0, vn = 0; uint32_t kk;
            for (kk = 0; kk < qdim; kk++) if (q[kk] != q[kk]) qn++;
            for (kk = 0; kk < kv_dim; kk++) { if (k[kk] != k[kk]) kn++; if (v[kk] != v[kk]) vn++; }
            fprintf(stderr, "[qdbg] attn layer=%u q_nan=%g q0=%g k0=%g v0=%g\n", layer, (double)qn, (double)q[0], (double)k[0], (double)v[0]);
        }

        /* q/k QK-norm 每头 ×256 */
        for (hh = 0; hh < n_heads; hh++)
            rmsnorm(q + (size_t)hh * hd, q + (size_t)hh * hd, base + mt[SLOT_QNORM].offset, hd, eps, mt[SLOT_QNORM].dtype);
        for (hh = 0; hh < n_kv; hh++)
            rmsnorm(k + (size_t)hh * hd, k + (size_t)hh * hd, base + mt[SLOT_KNORM].offset, hd, eps, mt[SLOT_KNORM].dtype);

        /* M-RoPE: 前 n_rot 维 interleaved */
        for (hh = 0; hh < n_heads; hh++)
            rope_inplace_mrope(q + (size_t)hh * hd, hd, n_rot, pos, theta);
        for (hh = 0; hh < n_kv; hh++)
            rope_inplace_mrope(k + (size_t)hh * hd, hd, n_rot, pos, theta);

        /* 写 KV cache + attention(复用现有逻辑) */
        uint16_t* kcache = e->kv + (size_t)layer * e->max_seq * kv_dim;
        uint16_t* vcache = e->kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
        uint64_t kvp = (uint64_t)pos * kv_dim;
        f32_to_f16_buf(k, kcache + kvp, kv_dim);
        f32_to_f16_buf(v, vcache + kvp, kv_dim);
        float inv_d = 1.0f / sqrtf((float)hd);
        attn_kv_f16(att_out, q, kcache, vcache, 0, pos,
                    n_heads, n_kv, hd, kv_dim, inv_d, 0.0f);
        /* gate 门控: att_out *= sigmoid(gate) */
        for (ii = 0; ii < qdim; ii++)
            att_out[ii] *= 1.0f / (1.0f + expf(-gate[ii]));
        if (getenv("YLLM_QDBG")) {
            float an = 0, gn = 0; uint32_t kk;
            for (kk = 0; kk < qdim; kk++) { if (att_out[kk] != att_out[kk]) an++; if (gate[kk] != gate[kk]) gn++; }
            fprintf(stderr, "[qdbg] attn layer=%u after_attn nan=%g gate_nan=%g gate0=%g att0=%g gate[0..7]=%g %g %g %g %g %g %g %g\n",
                    layer, (double)an, (double)gn, (double)gate[0], (double)att_out[0],
                    (double)gate[0], (double)gate[1], (double)gate[2], (double)gate[3],
                    (double)gate[4], (double)gate[5], (double)gate[6], (double)gate[7]);
        }

        memcpy(x2, att_out, (size_t)qdim * 4);
        matmul(att_out, x2, base + mt[SLOT_O].offset, hidden, qdim, mt[SLOT_O].dtype);
        add_inplace(x, att_out, hidden);
        rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
        float* fg = e->ffn;
        float* fu = e->ffn + inter;
        matmul(fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
        matmul(fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
        swiglu(x2, fg, fu, inter);
        matmul(att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
        add_inplace(x, att_out, hidden);
    }
    return 0;
}

int arch_llama_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    const uint8_t* base = (const uint8_t*)e->ws.map.base + e->ws.model.dir[layer].offset;
    return engine_fwd_block_at(e, layer, pos, base, e->kv);
}

int engine_fwd_block_at(Engine* e, uint32_t layer, uint32_t pos,
                        const uint8_t* base, uint16_t* kv)
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
    LlfGemma4Ext g4;
    uint32_t il = layer > 0 ? layer - 1 : 0;
    int has_kv = 1;
    uint32_t kv_layer = layer;
    llf_gemma4_ext(h, &g4);
    if (h->arch == ARCH_GEMMA4)
        theta = gemma4_rope_theta(h, &g4, il);
    if (h->arch == ARCH_GEMMA4 && g4.n_kv_shared_layers > 0 &&
        g4.n_kv_shared_layers < h->n_blocks) {
        uint32_t kv_from = h->n_blocks - g4.n_kv_shared_layers;
        if (il >= kv_from) {
            has_kv = 0;
            kv_layer = kv_from - (gemma4_is_swa(&g4, il) ? 2u : 1u) + 1u;
        }
    }
    if (mt[SLOT_K].size == 0) has_kv = 0;
    if (h->arch == ARCH_GEMMA4 && hidden > 0) {
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
    if (has_kv && h->arch == ARCH_GEMMA4) {
        uint32_t hh;
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm_unit(v + (size_t)hh * hd, v + (size_t)hh * hd, hd, eps);
    }

    uint16_t* kcache = kv + (size_t)kv_layer * e->max_seq * kv_dim;
    uint16_t* vcache = kv + (size_t)(h->n_blocks + kv_layer) * e->max_seq * kv_dim;
    uint64_t kvp = (uint64_t)pos * kv_dim;
    uint32_t hh;
    const float* rope_if = NULL;
    if (h->arch == ARCH_GEMMA4) {
        int swa = gemma4_is_swa(&g4, il);
        if (swa && e->rope_if_swa && e->n_rope_if_swa == hd / 2)
            rope_if = e->rope_if_swa;
        else if (!swa && e->rope_ff && e->n_rope_ff == hd / 2)
            rope_if = e->rope_ff;
    }
    for (hh = 0; hh < h->n_heads; hh++) {
        if (h->arch == ARCH_QWEN)
            rope_inplace_qwen(q + (size_t)hh * hd, hd, pos, theta);
        else if (h->arch == ARCH_GEMMA4 && rope_if)
            rope_inplace_neox_if(q + (size_t)hh * hd, hd, pos, rope_if);
        else if (h->arch == ARCH_GEMMA4)
            rope_inplace_qwen_ff(q + (size_t)hh * hd, hd, pos, theta, NULL);
        else
            rope_inplace(q + (size_t)hh * hd, hd, pos, theta);
    }
    if (has_kv) {
        for (hh = 0; hh < h->n_kv_heads; hh++) {
            if (h->arch == ARCH_QWEN)
                rope_inplace_qwen(k + (size_t)hh * hd, hd, pos, theta);
            else if (h->arch == ARCH_GEMMA4 && rope_if)
                rope_inplace_neox_if(k + (size_t)hh * hd, hd, pos, rope_if);
            else if (h->arch == ARCH_GEMMA4)
                rope_inplace_qwen_ff(k + (size_t)hh * hd, hd, pos, theta, NULL);
            else
                rope_inplace(k + (size_t)hh * hd, hd, pos, theta);
        }
        f32_to_f16_buf(k, kcache + kvp, kvd);
        f32_to_f16_buf(v, vcache + kvp, kvd);
    }

    float inv_d = (h->arch == ARCH_GEMMA4) ? 1.0f : 1.0f / sqrtf((float)hd);
    float attn_cap = 0.0f;
    uint32_t s0 = 0;
    if (h->arch == ARCH_GEMMA4) {
        /* llama.cpp gemma4: f_attention_scale=1, 无 attention logit soft-cap */
        attn_cap = 0.0f;
        if (gemma4_is_swa(&g4, il) && g4.swa_window > 0 && pos + 1 > g4.swa_window)
            s0 = pos + 1 - g4.swa_window;
    }
    attn_kv_f16(att_out, q, kcache, vcache, s0, pos,
                h->n_heads, h->n_kv_heads, hd, kv_dim, inv_d, attn_cap);
    memcpy(x2, att_out, (size_t)q_dim * 4);
    matmul(att_out, x2, base + mt[SLOT_O].offset, hidden, q_dim, mt[SLOT_O].dtype);
    if (h->arch == ARCH_GEMMA4) {
        float ls = 1.0f;
        uint32_t j;
        /* post-attn residual, then FFN, then PLE, then layer_scale — 对齐 llama.cpp gemma4.cpp */
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
        return 0;
    }
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

void engine_attach_cpu_fwd(Engine* e)
{
    if (e && e->dev) {
        e->dev->fwd_block = NULL;
        e->dev->fwd_block_batch = NULL;
    }
}

/* 单层前向(含 embed / block / final norm / head 分派) */
static void forward_layer(Engine* e, uint32_t i, uint32_t token, uint32_t pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[i].offset;
    int on_dev = layer_in_gpu_range(e, i);
    if (i == 0) {
        int got = 0;
        if (on_dev && e->dev && e->dev->embed && e->dev->embed(e, token) == 0)
            got = 1;
        if (!got) {
            engine_embed_into(e, e->x, token);
            engine_dev_mark_x_host(e);
        }
        if (e->ops && e->ops->after_embed)
            e->ops->after_embed(e, token);
    } else if (i <= (h->n_blocks - (e->mtp_layer ? 1u : 0u))) {
        if (!layer_on_device(e, i) && e->dev && e->dev->fwd_block) {
            /* 混合切点: GPU 激活 → host, 后续 Arch CPU */
            engine_dev_sync_x(e);
            engine_dev_mark_x_host(e);
        }
        engine_call_fwd_block(e, i, pos);
        if (e->mtp_h && i == (h->n_blocks - (e->mtp_layer ? 1u : 0u))) {
            engine_dev_sync_x(e);
            memcpy(e->mtp_h, e->x, (size_t)h->hidden * 4);
            e->mtp_h_ready = 1;
        }
    } else if (e->mtp_layer && i == e->mtp_layer) {
        /* MTP 块不进主干, 仅 MTP 预测时使用 */
    } else if (i == h->n_blocks + 1) {
        int got = 0;
        if (on_dev && e->dev && e->dev->final_norm && e->dev->final_norm(e) == 0)
            got = 1;
        if (!got) {
            engine_dev_sync_x(e);
            engine_dev_mark_x_host(e);
            float eps;
            memcpy(&eps, &h->norm_eps_bits, 4);
            const LlfTensorMeta* tm = &m->metas[m->base_idx[i]];
            rmsnorm(e->x, e->x, base + tm->offset, h->hidden, eps, tm->dtype);
        }
    } else {
        int got = 0;
        if (on_dev && e->dev && e->dev->lm_head && e->dev->lm_head(e) == 0)
            got = 1;
        if (!got) {
            engine_dev_sync_x(e);
            engine_dev_mark_x_host(e);
            const LlfTensorMeta* tm = &m->metas[m->base_idx[i]];
            if (tm->ndim == 2 && tm->size >= (uint64_t)h->hidden * 4) {
#if YLLM_TENSOR_STREAM
                size_t rbytes = matmul_row_bytes(tm->dtype, h->hidden);
                if (ws->budget > 0 && rbytes > 0) {
                    lm_head_chunked(e, base + tm->offset,
                                    m->dir[i].offset + tm->offset,
                                    h->hidden, h->vocab, tm->dtype);
                    if (e->ops && e->ops->post_logits)
                        e->ops->post_logits(e);
                    return;
                }
                matmul(e->logits, e->x, base + tm->offset, h->vocab, h->hidden, tm->dtype);
                if (ws->budget > 0) ws_release(ws, i);
#else
                matmul(e->logits, e->x, base + tm->offset, h->vocab, h->hidden, tm->dtype);
#endif
            } else {
                switch (tm->dtype) {
                case DT_F32: matmul_f32_t(e->logits, e->x, base + tm->offset, h->hidden, h->vocab); break;
                case DT_Q4K: matmul_q4k(e->logits, e->x, base + tm->offset, h->vocab, h->hidden); break;
                case DT_Q6K: matmul_q6k(e->logits, e->x, base + tm->offset, h->vocab, h->hidden); break;
                case DT_IQ4XS: matmul_iq4xs(e->logits, e->x, base + tm->offset, h->vocab, h->hidden); break;
                default: matmul_f16_t(e->logits, e->x, base + tm->offset, h->hidden, h->vocab); break;
                }
            }
        }
        if (e->ops && e->ops->post_logits)
            e->ops->post_logits(e);
    }
}

int engine_forward(Engine* e, uint32_t token, uint32_t pos)
{
    return engine_forward_range(e, token, 1, pos, NULL, NULL);
}

void engine_set_layers(Engine* e, uint32_t begin, uint32_t end)
{
    e->layer_begin = begin;
    e->layer_end = end;
}

void engine_set_gpu_layers(Engine* e, int n_blocks)
{
    if (!e || n_blocks < 0) return;
    uint32_t nb = e->ws.model.h.n_blocks;
    if ((uint32_t)n_blocks >= nb)
        e->gpu_layer_end = 0; /* 全段走 device */
    else
        e->gpu_layer_end = 1u + (uint32_t)n_blocks; /* embed + n_blocks 个块 */
}

/* 分布式分片前向: 只处理 [layer_begin, layer_end) 区间。
 * need_embed=1 时先用 token 做 embed(rank0); e->x 须在调用前含输入(中段 rank 由
 * 上层 recv 填充)。x_out/logits_out 非空时分别拷贝输出激活/最终 logits。 */
int engine_forward_range(Engine* e, uint32_t token, int need_embed, uint32_t pos,
                         float* x_out, float* logits_out)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    Worker* w = (Worker*)ws->worker;
    const LlfHeader* h = &m->h;
    uint32_t i;
    static int prof_init;
    static uint64_t prof_blk_ns, prof_head_ns, prof_n;
    int do_prof = 0;
    if (!prof_init) {
        prof_init = 1;
        do_prof = getenv("YLLM_PROF") != NULL;
        if (!do_prof) prof_init = 2; /* off */
    } else if (prof_init == 1)
        do_prof = 1;
    if (ws->budget > 0) {
        sched_refresh_resident(ws);
        sched_adapt_budget(ws);
    }
    if (need_embed && e->layer_begin == 0) {
        forward_layer(e, 0, token, pos);
#if YLLM_TENSOR_STREAM
        /* embed 只在 prefill 用一次, 受限模式下算完即释放 */
        if (ws->budget > 0) ws_release(ws, 0);
#endif
    } else if (!need_embed) {
        /* PP 中段等: 调用方已把激活写入 e->x, 设备侧可能仍是旧 d_x */
        engine_dev_mark_x_host(e);
        if (e->ops && e->ops->refresh_ple_pp)
            e->ops->refresh_ple_pp(e, token);
    }
    for (i = e->layer_begin; i < e->layer_end; i++) {
        uint64_t t0 = 0;
        if (i == 0) continue; /* embed 已在上方处理 */
        if (do_prof) t0 = ynow_ns();
        sched_ensure(ws, i);
        forward_layer(e, i, token, pos);
        if (do_prof) {
            uint64_t dt = ynow_ns() - t0;
            if (i <= h->n_blocks) prof_blk_ns += dt;
            else                   prof_head_ns += dt;
        }
        if (getenv("YLLM_NANDBG") && i == 33)
            fprintf(stderr, "[nandbg] x after layer33 pos %u: %g %g %g %g\n", pos,
                    (double)e->x[0], (double)e->x[1], (double)e->x[2], (double)e->x[3]);
        if (getenv("YLLM_NANDBG") && i == 59)
            fprintf(stderr, "[nandbg] x after layer59 pos %u: %g %g %g %g\n", pos,
                    (double)e->x[0], (double)e->x[1], (double)e->x[2], (double)e->x[3]);
        if (getenv("YLLM_NANDBG") && (i == 34 || i == 40 || i == 50))
            fprintf(stderr, "[nandbg] x after layer%u pos %u: %g %g %g %g\n", i, pos,
                    (double)e->x[0], (double)e->x[1], (double)e->x[2], (double)e->x[3]);
        if (getenv("YLLM_NANDBG") && (i == 35 || i == 36 || i == 37))
            fprintf(stderr, "[nandbg] x after layer%u pos %u: %g %g %g %g\n", i, pos,
                    (double)e->x[0], (double)e->x[1], (double)e->x[2], (double)e->x[3]);
        if (getenv("YLLM_NANDBG") && (i == 33 || i == 35 || i == 36 || i == 59)) {
            float l2 = 0, s = 0; unsigned h = 0; uint32_t j, hd = e->ws.model.h.hidden;
            for (j = 0; j < hd; j++) { l2 += e->x[j]*e->x[j]; s += e->x[j]; h = h*131 + (unsigned)floorf(e->x[j]*1000.0f); }
            fprintf(stderr, "[nandbg] chk layer%u pos %u: l2=%.5g sum=%.5g hash=%u\n", i, pos, (double)l2, (double)s, h);
        }
        if (getenv("YLLM_NANDBG") && e->x[0] != e->x[0])
            fprintf(stderr, "[nandbg] rank rng NaN after layer %u pos %u\n", i, pos);
        if (w) {
            uint32_t d = (uint32_t)ws->depth;
            uint32_t nb = i + d;
            uint32_t ne = nb + d;
            if (nb < m->n_layers) {
                if (ne > m->n_layers) ne = m->n_layers;
                while (nb < ne && ws->pstate[nb] != 0) nb++;
                if (nb < ne) {
#if YLLM_TENSOR_STREAM
                    if (ws->budget > 0) {
                        /* 受限模式: 预取窗口累计字节 <= 预算(当前层已占预算)。
                         * 放不下的层标 2 走按需缺页, 防止预取即被释放的白读。 */
                        uint64_t acc = ws->layer_size[i];
                        uint32_t nb2 = nb;
                        while (nb2 < ne) {
                            if (ws->layer_size[nb2] > ws->budget ||
                                acc + ws->layer_size[nb2] > ws->budget) {
                                ws->pstate[nb2] = 2;
                                break;
                            }
                            acc += ws->layer_size[nb2];
                            ws->pstate[nb2] = 1;
                            nb2++;
                        }
                        if (nb2 > nb) sched_enqueue(w, nb, nb2);
                    } else {
                        ws->pstate[nb] = 1;
                        sched_enqueue(w, nb, ne);
                    }
#else
                    ws->pstate[nb] = 1;
                    sched_enqueue(w, nb, ne);
#endif
                }
            }
        }
        sched_release_budget(ws, i);
    }
    if (do_prof) {
        prof_n++;
        /* decode 多 token 后打印一次平均拆分 */
        if (prof_n == 8 || (prof_n > 8 && (prof_n % 16) == 0)) {
            double inv = 1e-6 / (double)prof_n;
            fprintf(stderr,
                    "[prof] n=%llu blocks=%.2fms head=%.2fms (%.0f%% head)\n",
                    (unsigned long long)prof_n,
                    (double)prof_blk_ns * inv,
                    (double)prof_head_ns * inv,
                    100.0 * (double)prof_head_ns /
                        (double)(prof_blk_ns + prof_head_ns + 1));
        }
    }
    if (x_out) {
        engine_dev_sync_x(e);
        memcpy(x_out, e->x, (size_t)h->hidden * 4);
    }
    if (logits_out) memcpy(logits_out, e->logits, (size_t)h->vocab * 4);
    return 0;
}

/* 批量前向(distributed): tokens → 本 rank 层段 → 每 token 激活。
 * 仅用于 PP 首段(rank 0): 内部 embed 全部 token, 层段批量计算。
 * x_out 非空时填充 n*hidden; 返回 0/-1。 */
int engine_forward_batch_tokens(Engine* e, const uint32_t* tokens, int n, uint32_t pos,
                                float* x_out)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t B = e->pb_cap ? e->pb_cap : 16;
    if (n < 1 || (uint32_t)n > B) return -1;
    /* CUDA/Vulkan 批路径读设备激活; 走逐 token range 保正确 */
    if (e->dev && e->dev->fwd_block && e->device_mode != DEV_MODE_CPU) {
        int i;
        for (i = 0; i < n; i++) {
            if (engine_forward_range(e, tokens[i], 1, pos + (uint32_t)i,
                                     x_out ? x_out + (size_t)i * hidden : NULL,
                                     NULL) != 0)
                return -1;
        }
        return 0;
    }
    if (e->layer_begin == 0) {
        uint32_t b;
        for (b = 0; b < (uint32_t)n; b++)
            engine_embed_into(e, e->pb + (size_t)b * hidden, tokens[b]);
        if (e->ops && e->ops->after_embed_batch)
            e->ops->after_embed_batch(e, tokens, (uint32_t)n);
    }
    uint32_t i;
    for (i = e->layer_begin; i < e->layer_end; i++) {
        if (i == 0 || i > h->n_blocks) continue;
        engine_call_fwd_block_batch(e, i, pos, (uint32_t)n);
    }
    if (x_out) {
        uint32_t b;
        for (b = 0; b < (uint32_t)n; b++)
            memcpy(x_out + (size_t)b * hidden, e->pb + (size_t)b * hidden, (size_t)hidden * 4);
    }
    return 0;
}

/* 批量前向(distributed): 输入激活 → 本 rank 层段 → 输出激活或 logits。
 * 中段: x_out 填 n*hidden(转发给下一段);
 * 末段: logits_out 填最后 token 的 norm+output 结果。
 * 返回 0/-1。 */
int engine_forward_batch_x(Engine* e, const float* xin, int n, uint32_t pos,
                           float* x_out, float* logits_out, const uint32_t* tokens)
{
    if (e->dev && e->dev->forward_batch_x) {
        if (!tokens && e->dev->forward_batch_x(e, xin, n, pos, x_out, logits_out) == 0)
            return 0;
        /* 回退: 逐 token(带 token 时刷新 gemma4 PLE) */
        uint32_t hidden = e->ws.model.h.hidden;
        int i;
        for (i = 0; i < n; i++) {
            memcpy(e->x, xin + (size_t)i * hidden, (size_t)hidden * 4);
            engine_dev_mark_x_host(e);
            uint32_t tok = tokens ? tokens[i] : 0;
            if (i + 1 == n && logits_out) {
                if (engine_forward_range(e, tok, 0, pos + (uint32_t)i, NULL, logits_out) != 0)
                    return -1;
            } else if (x_out) {
                if (engine_forward_range(e, tok, 0, pos + (uint32_t)i,
                                        x_out + (size_t)i * hidden, NULL) != 0)
                    return -1;
            } else {
                if (engine_forward_range(e, tok, 0, pos + (uint32_t)i, NULL, NULL) != 0)
                    return -1;
            }
        }
        return 0;
    }
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t B = e->pb_cap ? e->pb_cap : 16;
    if (n < 1 || (uint32_t)n > B) return -1;
    /* gemma4 PP: 先用 token embed 填 pb 算 PLE, 再覆盖为上游激活 */
    if (tokens && e->n_ple && e->ops && e->ops->after_embed_batch && e->ple_batch) {
        uint32_t b;
        for (b = 0; b < (uint32_t)n; b++)
            engine_embed_into(e, e->pb + (size_t)b * hidden, tokens[b]);
        e->ops->after_embed_batch(e, tokens, (uint32_t)n);
    }
    uint32_t b;
    for (b = 0; b < (uint32_t)n; b++)
        memcpy(e->pb + (size_t)b * hidden, xin + (size_t)b * hidden, (size_t)hidden * 4);
    const uint8_t* base = (const uint8_t*)ws->map.base;
    float eps;
    memcpy(&eps, &h->norm_eps_bits, 4);
    uint32_t i;
    for (i = e->layer_begin; i < e->layer_end; i++) {
        if (i == 0) continue;
        if (i <= (h->n_blocks - (e->mtp_layer ? 1u : 0u))) {
            engine_call_fwd_block_batch(e, i, pos, (uint32_t)n);
            if (getenv("YLLM_NANDBG") && e->pb[0] != e->pb[0])
                fprintf(stderr, "[nandbg] batch NaN after layer %u pos %u\n", i, pos);
        } else if (e->mtp_layer && i == e->mtp_layer) {
            /* MTP 块不进主干, 仅 MTP 预测时使用 */
        } else if (i == h->n_blocks + 1) {
            const LlfTensorMeta* fn = &m->metas[m->base_idx[i]];
            rmsnorm(e->x, e->pb + (size_t)(n - 1) * hidden,
                    base + m->dir[i].offset + fn->offset, hidden, eps, fn->dtype);
        } else {
            const LlfTensorMeta* out = &m->metas[m->base_idx[i]];
            matmul(e->logits, e->x, base + m->dir[i].offset + out->offset,
                   h->vocab, hidden, out->dtype);
            if (e->ops && e->ops->post_logits)
                e->ops->post_logits(e);
        }
    }
    if (x_out) {
        for (b = 0; b < (uint32_t)n; b++)
            memcpy(x_out + (size_t)b * hidden, e->pb + (size_t)b * hidden, (size_t)hidden * 4);
    }
    if (logits_out) {
        if (e->layer_end <= h->n_blocks) return -1;   /* 非末段无 logits */
        memcpy(logits_out, e->logits, (size_t)h->vocab * 4);
    }
    return 0;
}

/* MTP(Multi-Token Prediction)前向: 与 llama.cpp qwen35 graph_mtp 等价。
 * 输入: h_main = 主干最后一层输出(norm 前)的 hidden, token = 下一个 token,
 *       pos = token 的位置(rope)。
 * 公式:
 *   e_norm = rmsnorm(embed(token), enorm)
 *   h_norm = rmsnorm(h_main, hnorm)
 *   cur    = eh_proj([e_norm | h_norm])           # [2*hidden → hidden], F16
 *   cur   += attn_block(cur, q/k/v/gate, 单 token self-attention, 无历史 KV)
 *   cur   += ffn(cur)
 *   head   = rmsnorm(cur, shared_head_norm)
 *   logits = lm_head(head)
 * 返回 0 成功, -1 无 MTP。 */
int engine_mtp_predict(Engine* e, const float* h_main, uint32_t token, uint32_t pos, float* logits_out)
{
    if (!e->mtp_eh_slot) return -1;
    LlModel* m = &e->ws.model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)e->ws.map.base;
    uint32_t hidden = h->hidden;
    uint32_t mb = h->n_blocks;                  /* MTP 块 llf 层 */
    uint32_t ol = h->n_blocks + 2;              /* output 层(nextn 权重) */
    const LlfTensorMeta* mt = &m->metas[m->base_idx[mb]];
    const LlfTensorMeta* nt = &m->metas[m->base_idx[ol]];
    const LlfTensorMeta* emb = &m->metas[m->base_idx[0] + SLOT_EMBED];
    uint32_t n_heads = h->n_heads;
    uint32_t n_kv = h->n_kv_heads;
    uint32_t hd = h->head_dim;
    uint32_t qdim = n_heads * hd;
    uint32_t kv_dim = n_kv * hd;
    uint32_t n_rot = 64;
    float eps, theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;

    /* 缓冲布局(scratch 65536 floats, 与 GDN/attention 不并发):
     * concat[2*hidden] | cur[hidden] | q[qdim] | gate[qdim] | k[kv_dim] | v[kv_dim] | attn_out[qdim] */
    float* scratch = e->scratch;
    float* concat = scratch;
    float* cur = scratch + 2 * hidden;
    float* q = scratch + 3 * hidden;
    float* gate = scratch + 3 * hidden + qdim;
    float* k = scratch + 3 * hidden + 2 * qdim;
    float* v = scratch + 3 * hidden + 2 * qdim + kv_dim;
    float* attn_out = scratch + 3 * hidden + 2 * qdim + 2 * kv_dim;
    float* x2 = e->hb;
    uint32_t j;

    /* ① embed + e_norm(就地到 concat 前半) */
    switch (emb->dtype) {
    case DT_F32: embed_f32(concat, base + m->dir[0].offset + emb->offset, token, hidden); break;
    case DT_Q4K: embed_q4k(concat, base + m->dir[0].offset + emb->offset, token, hidden); break;
    case DT_Q6K: embed_q6k(concat, base + m->dir[0].offset + emb->offset, token, hidden); break;
    case DT_Q5K: embed_q5k(concat, base + m->dir[0].offset + emb->offset, token, hidden); break;
    default:     embed_f16(concat, base + m->dir[0].offset + emb->offset, token, hidden); break;
    }
    rmsnorm(concat, concat, base + m->dir[ol].offset + nt[e->mtp_enorm_slot].offset,
            hidden, eps, nt[e->mtp_enorm_slot].dtype);
    /* ② h_norm 到 concat 后半 */
    rmsnorm(concat + hidden, h_main, base + m->dir[ol].offset + nt[e->mtp_hnorm_slot].offset,
            hidden, eps, nt[e->mtp_hnorm_slot].dtype);
    /* ③ eh_proj: [2*hidden → hidden] F16 */
    matmul_f16_t(cur, concat, base + m->dir[ol].offset + nt[e->mtp_eh_slot].offset,
                 2 * hidden, hidden);
    /* ④ attn_norm + q/k/v/gate */
    rmsnorm(x2, cur, base + m->dir[mb].offset + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
    matmul(q, x2, base + m->dir[mb].offset + mt[SLOT_Q].offset, qdim, hidden, mt[SLOT_Q].dtype);
    matmul(gate, x2, base + m->dir[mb].offset + mt[SLOT_QGATE].offset, qdim, hidden, mt[SLOT_QGATE].dtype);
    matmul(k, x2, base + m->dir[mb].offset + mt[SLOT_K].offset, kv_dim, hidden, mt[SLOT_K].dtype);
    matmul(v, x2, base + m->dir[mb].offset + mt[SLOT_V].offset, kv_dim, hidden, mt[SLOT_V].dtype);
    /* ⑤ q/k QK-norm + M-RoPE */
    {
        uint32_t hh;
        for (hh = 0; hh < n_heads; hh++)
            rmsnorm(q + (size_t)hh * hd, q + (size_t)hh * hd,
                    base + m->dir[mb].offset + mt[SLOT_QNORM].offset, hd, eps, mt[SLOT_QNORM].dtype);
        for (hh = 0; hh < n_kv; hh++)
            rmsnorm(k + (size_t)hh * hd, k + (size_t)hh * hd,
                    base + m->dir[mb].offset + mt[SLOT_KNORM].offset, hd, eps, mt[SLOT_KNORM].dtype);
        for (hh = 0; hh < n_heads; hh++)
            rope_inplace_mrope(q + (size_t)hh * hd, hd, n_rot, pos, theta);
        for (hh = 0; hh < n_kv; hh++)
            rope_inplace_mrope(k + (size_t)hh * hd, hd, n_rot, pos, theta);
    }
    /* ⑥ 单 token self-attention: softmax(q·k)=1, 输出 = v(按 q 头组复制), ×sigmoid(gate) */
    {
        uint32_t hh;
        for (hh = 0; hh < n_heads; hh++) {
            uint32_t kv_head = hh * n_kv / n_heads;
            memcpy(attn_out + (size_t)hh * hd, v + (size_t)kv_head * hd, (size_t)hd * 4);
        }
        for (j = 0; j < qdim; j++)
            attn_out[j] *= 1.0f / (1.0f + expf(-gate[j]));
    }
    /* ⑦ o_proj + 残差 */
    matmul(x2, attn_out, base + m->dir[mb].offset + mt[SLOT_O].offset, hidden, qdim, mt[SLOT_O].dtype);
    for (j = 0; j < hidden; j++) cur[j] += x2[j];
    /* ⑧ post norm + FFN */
    rmsnorm(x2, cur, base + m->dir[mb].offset + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
    {
        float* fg = e->ffn;
        float* fu = e->ffn + inter;
        matmul(fg, x2, base + m->dir[mb].offset + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
        matmul(fu, x2, base + m->dir[mb].offset + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
        swiglu(x2, fg, fu, inter);
        matmul(attn_out, x2, base + m->dir[mb].offset + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
    }
    for (j = 0; j < hidden; j++) cur[j] += attn_out[j];
    /* ⑨ shared head norm + lm_head */
    rmsnorm(x2, cur, base + m->dir[ol].offset + nt[e->mtp_headnorm_slot].offset,
            hidden, eps, nt[e->mtp_headnorm_slot].dtype);
    matmul_rows(logits_out, x2, base + m->dir[ol].offset + nt[0].offset,
                0, h->vocab, hidden, h->vocab, nt[0].dtype);
    return 0;
}

int engine_sample(Engine* e, uint32_t vocab, float temp, float top_p, uint64_t* rng, uint32_t* out)
{
    float* logits = e->logits;
    uint32_t i;
    if (temp <= 0.0f) {
        /* 贪婪: 严格取 argmax */
        uint32_t best = 0;
        for (i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        *out = best;
        return 0;
    }
    if (temp > 0 && temp != 1.0f) {
        for (i = 0; i < vocab; i++) logits[i] /= temp;
    }
    float m = logits[0];
    for (i = 1; i < vocab; i++) if (logits[i] > m) m = logits[i];
    /* gemma4 等 26 万词表: 全量 expf + qsort 每 token 能吃掉数十毫秒。
     * 只对 max 附近 (~e^-20) 的候选做 softmax / nucleus, 分布几乎不变。 */
    if (vocab >= 32768u) {
        const float cut = 20.0f;
        uint32_t n = 0, cap = 1024, j, keep;
        float s = 0.0f, cum, r, acc;
        ProbIdx* cand = (ProbIdx*)ymalloc((size_t)cap * sizeof(ProbIdx));
        if (!cand) return -1;
        for (i = 0; i < vocab; i++) {
            float z = logits[i] - m;
            if (z < -cut) continue;
            if (n == cap) {
                uint32_t ncap = cap * 2u;
                ProbIdx* p = (ProbIdx*)realloc(cand, (size_t)ncap * sizeof(ProbIdx));
                if (!p) { free(cand); return -1; }
                cand = p;
                cap = ncap;
            }
            cand[n].prob = expf(z);
            cand[n].index = (int)i;
            n++;
        }
        if (n == 0) { *out = 0; free(cand); return 0; }
        for (j = 0; j < n; j++) s += cand[j].prob;
        for (j = 0; j < n; j++) cand[j].prob /= s;
        keep = n;
        cum = 1.0f;
        if (top_p < 1.0f && n > 1) {
            qsort(cand, (size_t)n, sizeof(ProbIdx), cmp_prob_desc);
            cum = 0.0f;
            keep = n;
            for (j = 0; j < n; j++) {
                cum += cand[j].prob;
                if (cum >= top_p) { keep = j + 1; break; }
            }
        }
        r = ((float)(yrng(rng) >> 40) / 16777216.0f) * cum;
        acc = 0.0f;
        for (j = 0; j < keep; j++) {
            acc += cand[j].prob;
            if (r < acc) { *out = (uint32_t)cand[j].index; free(cand); return 0; }
        }
        *out = (uint32_t)cand[keep - 1].index;
        free(cand);
        return 0;
    }
    {
    float s = 0.0f;
    for (i = 0; i < vocab; i++) { logits[i] = expf(logits[i] - m); s += logits[i]; }
    for (i = 0; i < vocab; i++) logits[i] /= s;
    }
    if (top_p < 1.0f) {
        ProbIdx* sorted = (ProbIdx*)ymalloc((size_t)vocab * sizeof(ProbIdx));
        uint32_t j;
        for (j = 0; j < vocab; j++) {
            sorted[j].prob = logits[j];
            sorted[j].index = (int)j;
        }
        qsort(sorted, (size_t)vocab, sizeof(ProbIdx), cmp_prob_desc);
        float cum = 0.0f;
        uint32_t keep = vocab;
        for (j = 0; j < vocab; j++) {
            cum += sorted[j].prob;
            if (cum >= top_p) { keep = j + 1; break; }
        }
        /* 在截断分布内采样: r ∈ [0, cum), 与累加概率对齐
         * (此前 r ∈ [0,1) 在 r ≥ cum 时回退到概率最低的 top-p token, 产生垃圾) */
        float r = ((float)(yrng(rng) >> 40) / 16777216.0f) * cum;
        float acc = 0.0f;
        uint32_t k;
        for (k = 0; k < keep; k++) {
            acc += sorted[k].prob;
            if (r < acc) { *out = (uint32_t)sorted[k].index; free(sorted); return 0; }
        }
        *out = (uint32_t)sorted[keep - 1].index;
        free(sorted);
        return 0;
    }
    {
        float r = (float)(yrng(rng) >> 40) / 16777216.0f;
        float acc = 0.0f;
        for (i = 0; i < vocab; i++) {
            acc += logits[i];
            if (r < acc) { *out = i; return 0; }
        }
        *out = vocab - 1;
    }
    return 0;
}

/* 返回 0 成功;-1 失败。timings 非空时填充 prefill/decode 分别的耗时与 token 数 */
int engine_generate(Engine* e, const uint32_t* prompt, int nprompt, int ntokens,
                    float temp, float top_p, uint64_t seed, int eos_stop,
                    int (*on_token)(uint32_t id, void* ctx), void* ctx,
                    EngineTimings* timings, char* err, size_t errlen)
{
    uint64_t rng = ysrand(seed);
    uint32_t pos = 0;
    int i;
    uint64_t t0 = 0, t1 = 0;
    if (timings) { memset(timings, 0, sizeof(*timings)); t0 = ynow_ms(); }
#if YLLM_BATCH_PREFILL
    if (nprompt > 0) {
        if ((uint32_t)nprompt > e->max_seq) {
            if (err) snprintf(err, errlen, "prompt too long");
            return -1;
        }
        engine_forward_prefill(e, prompt, nprompt, 0);
        engine_dev_sync_x(e);
        pos = (uint32_t)nprompt;
    }
#else
    for (i = 0; i < nprompt; i++) {
        if (pos >= e->max_seq) {
            if (err) snprintf(err, errlen, "prompt too long");
            return -1;
        }
        engine_forward(e, prompt[i], pos);
        pos++;
    }
#endif
    if (timings) {
        timings->n_prefill = nprompt > 0 ? (uint32_t)nprompt : 0;
        t1 = ynow_ms();
        timings->prefill_ms = t1 - t0;
    }
    uint32_t ngen = 0;
    uint32_t n_accept = 0, n_try = 0;
    int pending = 0;              /* 上一轮 MTP 接受: 本轮跳过采样直接 forward */
    uint32_t pending_tok = 0;
    uint32_t vocab = e->ws.model.h.vocab;
    for (i = 0; i < ntokens; i++) {
        if (pos >= e->max_seq) break;
        uint32_t nxt;
        if (pending) { nxt = pending_tok; pending = 0; }
        else {
            if (engine_sample(e, vocab, temp, top_p, &rng, &nxt) != 0) return -1;
            if (eos_stop >= 0 && (int)nxt == eos_stop) break;
            if (on_token && on_token(nxt, ctx) != 0) break;     /* 回调可中止(如对端断开) */
        }
        /* MTP draft: 输入 hidden(pos-1 主干输出) + embed(nxt=pos) → 预测 pos+1 */
        uint32_t draft = 0xFFFFFFFFu;
        if (e->mtp_enable && e->mtp_eh_slot && e->mtp_h_ready) {
            if (engine_mtp_predict(e, e->mtp_h, nxt, pos, e->mtp_logits) == 0) {
                uint32_t b2, best = 0;
                for (b2 = 1; b2 < vocab; b2++)
                    if (e->mtp_logits[b2] > e->mtp_logits[best]) best = b2;
                draft = best;
            }
        }
        if (pos >= e->max_seq) break;
        engine_forward(e, nxt, pos);
        pos++;
        ngen++;
        if (draft != 0xFFFFFFFFu) {
            n_try++;
            uint32_t real = 0, b3;
            for (b3 = 1; b3 < vocab; b3++)
                if (e->logits[b3] > e->logits[real]) real = b3;
            if (getenv("YLLM_MTDBG")) {
                fprintf(stderr, "[mtdbg] nxt=%u pos=%u draft=%u real=%u %s\n", nxt, pos, draft, real,
                        draft == real ? "ACCEPT" : "reject");
            }
            if (real == draft) {
                if (eos_stop >= 0 && (int)draft == eos_stop) { /* eos 不入列 */ }
                else if (on_token && on_token(draft, ctx) != 0) { break; }
                else {
                    pending = 1;
                    pending_tok = draft;
                    ngen++;
                    n_accept++;
                    pos++;          /* draft 占位; KV 由下轮 forward 写入 */
                    if (pos >= e->max_seq) break;
                }
            }
        }
    }
    if (e->mtp_enable && e->mtp_eh_slot && n_try > 0) {
        uint32_t nfwd = ngen > n_accept ? ngen - n_accept : 1;
        ylog_info("mtp: draft accepted %u/%u (%.0f%%) -> %.2f tok/forward", n_accept, n_try,
                  (double)n_accept * 100.0 / (double)n_try, (double)ngen / (double)nfwd);
    }
    if (timings) {
        timings->n_decode = ngen;
        timings->decode_ms = ynow_ms() - t1;
    }
    return 0;
}

uint64_t engine_resident(const Engine* e)
{
    uint64_t rss = yproc_rss();
    if (rss > 0) return rss;
    return e->ws.resident;
}