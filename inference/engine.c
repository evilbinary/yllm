#include "yllm.h"
#include "matvec.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 批量 prefill: 默认开启; 编译期关闭用 -DYLLM_BATCH_PREFILL=0 */
#ifndef YLLM_BATCH_PREFILL
#define YLLM_BATCH_PREFILL 1
#endif
/* 批量收益拐点(实测: tinyllama/qwen3 均在 B≈16 后才优于顺序) */
#define PREFILL_BATCH_MIN 16
#include <math.h>
#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>
#endif
typedef struct {
    Ws* ws;
    uint32_t next;
    uint32_t end;
} PFJob;

typedef struct { float prob; int index; } ProbIdx;

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
#endif

int engine_init(Engine* e, const char* model_path, uint64_t budget, int depth, char* err, size_t errlen)
{
    memset(e, 0, sizeof(*e));
    if (wmap_open(model_path, &e->ws.map) != 0) {
        snprintf(err, errlen, "cannot open %s", model_path);
        return -1;
    }
    if (llf_read(&e->ws.map, &e->ws.model) != 0) {
        wmap_close(&e->ws.map);
        snprintf(err, errlen, "bad llf file %s", model_path);
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
    uint32_t vocab = m->h.vocab;
    /* FFN 中间维度(inter)从首个 block 的 gate 张量推出;gate+up 共需 2*inter 个 float */
    uint32_t inter = hidden;
    if (m->n_layers > 1 && m->base_idx && m->metas &&
        (m->base_idx[1] + SLOT_GATE) < m->n_layers * BLOCK_TENSORS) {
        const LlfTensorMeta* g = &m->metas[m->base_idx[1] + SLOT_GATE];
        if (g->ndim >= 2 && g->shape[0] && g->shape[1]) {
            uint64_t prod = (uint64_t)g->shape[0] * g->shape[1];
            if (prod % hidden == 0) inter = (uint32_t)(prod / hidden);
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
    e->hb2 = (float*)ycalloc(hidden, 4 * 9);
    e->ffn = (float*)ycalloc((size_t)2 * inter, 4);
    e->att = (float*)ymalloc((size_t)e->max_seq * m->h.n_heads * 4);
    e->logits = (float*)ymalloc((size_t)vocab * 4);
    /* 批量 prefill 工作区 */
    {
        uint32_t PB_MAX = 64;
        uint32_t kv_dim2 = m->h.n_kv_heads * m->h.head_dim;
        e->pb_cap = PB_MAX;
        e->pb  = (float*)ycalloc((size_t)PB_MAX * hidden, 4);
        e->pb2 = (float*)ycalloc((size_t)PB_MAX * hidden, 4);
        e->pbq = (float*)ycalloc((size_t)PB_MAX * hidden, 4);
        e->pbk = (float*)ycalloc((size_t)PB_MAX * kv_dim2, 4);
        e->pbv = (float*)ycalloc((size_t)PB_MAX * kv_dim2, 4);
        e->pbg = (float*)ycalloc((size_t)PB_MAX * inter, 4);
        e->pbu = (float*)ycalloc((size_t)PB_MAX * inter, 4);
        e->pba = (float*)ymalloc((size_t)PB_MAX * m->h.n_heads * e->max_seq * 4);
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
    return 0;
}

void engine_free(Engine* e)
{
    Worker* w = (Worker*)e->ws.worker;
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
    free(e->pb); free(e->pb2); free(e->pbq);
    free(e->pbk); free(e->pbv); free(e->pbg); free(e->pbu);
    free(e->pba);
    if (e->ws.model.base_idx) free(e->ws.model.base_idx);
    if (e->ws.pstate) free(e->ws.pstate);
    if (e->ws.res) free(e->ws.res);
    if (e->ws.layer_size) free(e->ws.layer_size);
    free(w);
    memset(e, 0, sizeof(*e));
}

/* 批量前向一层: 同时处理 B 个 token(pos 连续: pos_start..pos_start+B-1) */
static int forward_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
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
            uint32_t j;
            for (j = 0; j < kv_dim; j++) {
                kcache[kvp + j] = f32_to_f16(kvk[j]);
                vcache[kvp + j] = f32_to_f16(kvv[j]);
            }
        }
    }

    /* 4) 注意力: 每 token 因果关注 0..pos_b; 并行于 head */
    {
        uint32_t hh;
        #pragma omp parallel for schedule(static)
        for (hh = 0; hh < h->n_heads; hh++) {
            uint32_t kv_head = hh * h->n_kv_heads / h->n_heads;
            uint32_t bb;
            for (bb = 0; bb < B; bb++) {
                uint32_t pos = pos_start + bb;
                const float* qh = e->pbq + (size_t)bb * hidden + (size_t)hh * h->head_dim;
                float* att_h = e->pba + ((size_t)bb * h->n_heads + hh) * e->max_seq;
                float inv_d = 1.0f / sqrtf((float)h->head_dim);
                uint32_t s, jj;
                for (s = 0; s <= pos; s++) {
                    const uint16_t* kh = e->kv + (size_t)layer * e->max_seq * kv_dim
                                         + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
                    float acc = 0.0f;
                    for (jj = 0; jj < h->head_dim; jj++) acc += qh[jj] * f16_to_f32(kh[jj]);
                    att_h[s] = acc * inv_d;
                }
                softmax(att_h, pos + 1);
                float* out = e->pb2 + (size_t)bb * hidden + (size_t)hh * h->head_dim;
                memset(out, 0, (size_t)h->head_dim * 4);
                for (s = 0; s <= pos; s++) {
                    const uint16_t* vh = e->kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim
                                         + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
                    float a = att_h[s];
                    for (jj = 0; jj < h->head_dim; jj++) out[jj] += a * f16_to_f32(vh[jj]);
                }
            }
        }
    }

    /* 5) o_proj + 残差, norm2, FFN */
    matmul_batch(e->pbq, e->pb2, base + mt[SLOT_O].offset, hidden, hidden, mt[SLOT_O].dtype, B);
    for (b = 0; b < B; b++) {
        float* xb = e->pb + (size_t)b * hidden;
        float* ob = e->pbq + (size_t)b * hidden;
        uint32_t j;
        for (j = 0; j < hidden; j++) xb[j] += ob[j];
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
        uint32_t j;
        for (j = 0; j < hidden; j++) xb[j] += e->pbq[(size_t)b * hidden + j];
    }
    return 0;
}

/* 批量 prefill: 一次处理 n 个 prompt token(start_pos 起), 结果 logits 为最后 token */
int engine_forward_prefill(Engine* e, const uint32_t* tokens, int n, int start_pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t hidx0 = m->base_idx[0];
    const LlfTensorMeta* tm = &m->metas[hidx0];
    const uint8_t* base = (const uint8_t*)ws->map.base;
    uint32_t B = e->pb_cap ? e->pb_cap : 16;
    int off = 0;
    while (off < n) {
        uint32_t nb = (uint32_t)(n - off);
        if (nb > B) nb = B;
        /* 小批(< 16)批量收益不抵反量化/调度开销, 回退顺序逐 token */
        if (nb < PREFILL_BATCH_MIN) {
            uint32_t i;
            for (i = 0; i < nb; i++)
                engine_forward(e, tokens[off + i], (uint32_t)(start_pos + off + i));
            off += (int)nb;
            continue;
        }
        /* embed 全部 batch token */
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* emb = base + m->dir[0].offset;
            switch (tm->dtype) {
            case DT_F32: embed_f32(e->pb + (size_t)b * hidden, emb, tokens[off + b], hidden); break;
            case DT_Q4K: embed_q4k(e->pb + (size_t)b * hidden, emb, tokens[off + b], hidden); break;
            case DT_Q6K: embed_q6k(e->pb + (size_t)b * hidden, emb, tokens[off + b], hidden); break;
            case DT_IQ4XS: embed_iq4xs(e->pb + (size_t)b * hidden, emb, tokens[off + b], hidden); break;
            default: embed_f16(e->pb + (size_t)b * hidden, emb, tokens[off + b], hidden); break;
            }
        }
        uint32_t i;
        for (i = 1; i <= h->n_blocks; i++) {
            if (ws->budget > 0) sched_ensure(ws, i);
            forward_block_batch(e, i, (uint32_t)(start_pos + off), nb);
            if (ws->budget > 0) sched_release_budget(ws, i);
        }
        /* 最后一层: final norm + output(只算最后 token) */
        {
            const LlfTensorMeta* fn = &m->metas[m->base_idx[h->n_blocks + 1]];
            const LlfTensorMeta* out = &m->metas[m->base_idx[h->n_blocks + 2]];
            float eps;
            memcpy(&eps, &h->norm_eps_bits, 4);
            if (off + (int)nb == n) {
                float* xlast = e->pb + (size_t)(nb - 1) * hidden;
                rmsnorm(e->x, xlast, base + m->dir[h->n_blocks + 1].offset + fn->offset,
                        hidden, eps, fn->dtype);
#if YLLM_TENSOR_STREAM
                /* 受限模式: 量化 lm_head 按页对齐行分块, 每块算完 munmap 释放
                 * (页缓存保留, 下个 token 重读是 minor fault); 尾块保留驻留。 */
                if (ws->budget > 0) {
                    size_t rbytes = matmul_row_bytes(out->dtype, h->hidden);
                    uint32_t dl = h->n_blocks + 2;
                    uint32_t chunk = head_chunk_rows(rbytes,
                        ws->budget < 32u * 1024 * 1024 ? 2048u : 4096u);
                    if (chunk > 0) {
                        uint32_t rows = 0;
                        while (rows + chunk <= h->vocab) {
                            matmul_rows(e->logits + rows, e->x,
                                        base + m->dir[dl].offset + out->offset,
                                        rows, chunk, h->hidden, h->vocab, out->dtype);
                            ws_release_aligned(ws, m->dir[dl].offset + out->offset +
                                                   (uint64_t)rows * rbytes,
                                               (uint64_t)chunk * rbytes);
                            rows += chunk;
                        }
                        if (rows < h->vocab)
                            matmul_rows(e->logits + rows, e->x,
                                        base + m->dir[dl].offset + out->offset,
                                        rows, h->vocab - rows, h->hidden, h->vocab, out->dtype);
                    } else {
                        matmul(e->logits, e->x, base + m->dir[dl].offset + out->offset,
                               h->vocab, hidden, out->dtype);
                    }
                } else {
                    matmul(e->logits, e->x, base + m->dir[h->n_blocks + 2].offset + out->offset,
                           h->vocab, hidden, out->dtype);
                }
#else
                matmul(e->logits, e->x, base + m->dir[h->n_blocks + 2].offset + out->offset,
                       h->vocab, hidden, out->dtype);
#endif
            }
        }
        off += (int)nb;
    }
    return 0;
}

static int forward_block(Engine* e, uint32_t layer, uint32_t pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[layer].offset;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    float eps;
    float theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    float* x = e->x;
    float* x2 = e->hb;
    float* q = e->hb2;
    float* k = e->hb2 + hidden;
    float* v = e->hb2 + hidden + kv_dim;
    float* att_out = e->hb2 + hidden + 2 * kv_dim;
    uint32_t hidx = m->base_idx[layer];

    const LlfTensorMeta* mt = &m->metas[hidx];
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;

    rmsnorm(x2, x, base + mt[SLOT_NORM1].offset, hidden, eps, mt[SLOT_NORM1].dtype);
    matmul(q, x2, base + mt[SLOT_Q].offset, hidden, hidden, mt[SLOT_Q].dtype);
    matmul(k, x2, base + mt[SLOT_K].offset, kv_dim, hidden, mt[SLOT_K].dtype);
    matmul(v, x2, base + mt[SLOT_V].offset, kv_dim, hidden, mt[SLOT_V].dtype);
    /* 注意力 bias(qwen2.5 gguf 带有非零 bias) */
    if (mt[SLOT_QBIAS].size > 0) {
        const float* bq = (const float*)(base + mt[SLOT_QBIAS].offset);
        uint32_t j;
        for (j = 0; j < hidden; j++) q[j] += bq[j];
    }
    if (mt[SLOT_KBIAS].size > 0) {
        const float* bk = (const float*)(base + mt[SLOT_KBIAS].offset);
        uint32_t j;
        for (j = 0; j < kv_dim; j++) k[j] += bk[j];
    }
    if (mt[SLOT_VBIAS].size > 0) {
        const float* bv = (const float*)(base + mt[SLOT_VBIAS].offset);
        uint32_t j;
        for (j = 0; j < kv_dim; j++) v[j] += bv[j];
    }

    /* qwen3 QK-norm: 对 q 每头 / k 每 kv 头做 RMSNorm(共享 head_dim 权重), 在 rope 之前 */
    if (mt[SLOT_QNORM].size > 0) {
        uint32_t hh;
        for (hh = 0; hh < h->n_heads; hh++)
            rmsnorm(q + (size_t)hh * h->head_dim, q + (size_t)hh * h->head_dim,
                    base + mt[SLOT_QNORM].offset, h->head_dim, eps, mt[SLOT_QNORM].dtype);
    }
    if (mt[SLOT_KNORM].size > 0) {
        uint32_t hh;
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm(k + (size_t)hh * h->head_dim, k + (size_t)hh * h->head_dim,
                    base + mt[SLOT_KNORM].offset, h->head_dim, eps, mt[SLOT_KNORM].dtype);
    }

    uint16_t* kcache = e->kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = e->kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
    uint64_t kvp = (uint64_t)pos * kv_dim;
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


    uint32_t j;
    for (j = 0; j < kv_dim; j++) {
        kcache[kvp + j] = f32_to_f16(k[j]);
        vcache[kvp + j] = f32_to_f16(v[j]);
    }

    float* att = e->att;
    float inv_d = 1.0f / sqrtf((float)h->head_dim);
    #pragma omp parallel for schedule(static)
    for (hh = 0; hh < h->n_heads; hh++) {
        float* att_h = att + (size_t)hh * e->max_seq;
        uint32_t kv_head = hh * h->n_kv_heads / h->n_heads;
        const float* qh = q + (size_t)hh * h->head_dim;
        uint32_t s, jj;
        for (s = 0; s <= pos; s++) {
            const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
            float acc = 0.0f;
            for (jj = 0; jj < h->head_dim; jj++) acc += qh[jj] * f16_to_f32(kh[jj]);
            att_h[s] = acc * inv_d;
        }
        softmax(att_h, pos + 1);
        float* out = att_out + (size_t)hh * h->head_dim;
        memset(out, 0, (size_t)h->head_dim * 4);
        for (s = 0; s <= pos; s++) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
            float a = att_h[s];
            for (jj = 0; jj < h->head_dim; jj++) out[jj] += a * f16_to_f32(vh[jj]);
        }
    }
    memcpy(x2, att_out, (size_t)hidden * 4);
    matmul(att_out, x2, base + mt[SLOT_O].offset, hidden, hidden, mt[SLOT_O].dtype);
    for (j = 0; j < hidden; j++) x[j] += att_out[j];
    rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
    float* fg = e->ffn;
    float* fu = e->ffn + inter;
    matmul(fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
    matmul(fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
    swiglu(x2, fg, fu, inter);
    matmul(att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
    for (j = 0; j < hidden; j++) x[j] += att_out[j];
    return 0;
}

/* 单层前向(含 embed / block / final norm / head 分派) */
static void forward_layer(Engine* e, uint32_t i, uint32_t token, uint32_t pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    const uint8_t* base = (const uint8_t*)ws->map.base + m->dir[i].offset;
    if (i == 0) {
        const LlfTensorMeta* tm = &m->metas[m->base_idx[0]];
        switch (tm->dtype) {
        case DT_F32: embed_f32(e->x, base + tm->offset, token, h->hidden); break;
        case DT_Q4K: embed_q4k(e->x, base + tm->offset, token, h->hidden); break;
        case DT_Q6K: embed_q6k(e->x, base + tm->offset, token, h->hidden); break;
        case DT_IQ4XS: embed_iq4xs(e->x, base + tm->offset, token, h->hidden); break;
        default: embed_f16(e->x, base + tm->offset, token, h->hidden); break;
        }
    } else if (i <= h->n_blocks) {
        forward_block(e, i, pos);
    } else if (i == h->n_blocks + 1) {
        float eps;
        memcpy(&eps, &h->norm_eps_bits, 4);
        const LlfTensorMeta* tm = &m->metas[m->base_idx[i]];
        rmsnorm(e->x, e->x, base + tm->offset, h->hidden, eps, tm->dtype);
    } else {
        const LlfTensorMeta* tm = &m->metas[m->base_idx[i]];
        if (tm->ndim == 2 && tm->size >= (uint64_t)h->hidden * 4) {
#if YLLM_TENSOR_STREAM
            /* 受限模式: 量化 lm_head 按页对齐行分块, 每块算完 munmap 释放
             * (页缓存保留, 下个 token 重读是 minor fault); 尾块保留驻留。 */
            size_t rbytes = matmul_row_bytes(tm->dtype, h->hidden);
            if (ws->budget > 0 && rbytes > 0) {
                uint32_t chunk = head_chunk_rows(rbytes,
                    ws->budget < 32u * 1024 * 1024 ? 2048u : 4096u);
                if (chunk > 0) {
                    uint32_t rows = 0;
                    while (rows + chunk <= h->vocab) {
                        matmul_rows(e->logits + rows, e->x, base + tm->offset,
                                    rows, chunk, h->hidden, h->vocab, tm->dtype);
                        ws_release_aligned(ws, m->dir[i].offset + tm->offset +
                                                (uint64_t)rows * rbytes,
                                            (uint64_t)chunk * rbytes);
                        rows += chunk;
                    }
                    if (rows < h->vocab)
                        matmul_rows(e->logits + rows, e->x, base + tm->offset,
                                    rows, h->vocab - rows, h->hidden, h->vocab, tm->dtype);
                } else {
                    matmul(e->logits, e->x, base + tm->offset, h->vocab, h->hidden, tm->dtype);
                }
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
    }
    for (i = e->layer_begin; i < e->layer_end; i++) {
        if (i == 0) continue; /* embed 已在上方处理 */
        sched_ensure(ws, i);
        forward_layer(e, i, token, pos);
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
    if (x_out) memcpy(x_out, e->x, (size_t)h->hidden * 4);
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
    if (e->layer_begin == 0) {
        uint32_t hidx0 = m->base_idx[0];
        const LlfTensorMeta* tm = &m->metas[hidx0];
        const uint8_t* base = (const uint8_t*)ws->map.base;
        const uint8_t* emb = base + m->dir[0].offset;
        uint32_t b;
        for (b = 0; b < (uint32_t)n; b++) {
            switch (tm->dtype) {
            case DT_F32: embed_f32(e->pb + (size_t)b * hidden, emb, tokens[b], hidden); break;
            case DT_Q4K: embed_q4k(e->pb + (size_t)b * hidden, emb, tokens[b], hidden); break;
            case DT_Q6K: embed_q6k(e->pb + (size_t)b * hidden, emb, tokens[b], hidden); break;
            case DT_IQ4XS: embed_iq4xs(e->pb + (size_t)b * hidden, emb, tokens[b], hidden); break;
            default: embed_f16(e->pb + (size_t)b * hidden, emb, tokens[b], hidden); break;
            }
        }
    }
    uint32_t i;
    for (i = e->layer_begin; i < e->layer_end; i++) {
        if (i == 0 || i > h->n_blocks) continue;
        forward_block_batch(e, i, pos, (uint32_t)n);
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
                           float* x_out, float* logits_out)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t B = e->pb_cap ? e->pb_cap : 16;
    if (n < 1 || (uint32_t)n > B) return -1;
    uint32_t b;
    for (b = 0; b < (uint32_t)n; b++)
        memcpy(e->pb + (size_t)b * hidden, xin + (size_t)b * hidden, (size_t)hidden * 4);
    const uint8_t* base = (const uint8_t*)ws->map.base;
    float eps;
    memcpy(&eps, &h->norm_eps_bits, 4);
    uint32_t i;
    for (i = e->layer_begin; i < e->layer_end; i++) {
        if (i == 0) continue;
        if (i <= h->n_blocks) {
            forward_block_batch(e, i, pos, (uint32_t)n);
        } else if (i == h->n_blocks + 1) {
            const LlfTensorMeta* fn = &m->metas[m->base_idx[i]];
            rmsnorm(e->x, e->pb + (size_t)(n - 1) * hidden,
                    base + m->dir[i].offset + fn->offset, hidden, eps, fn->dtype);
        } else {
            const LlfTensorMeta* out = &m->metas[m->base_idx[i]];
            matmul(e->logits, e->x, base + m->dir[i].offset + out->offset,
                   h->vocab, hidden, out->dtype);
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
    float s = 0.0f;
    for (i = 0; i < vocab; i++) { logits[i] = expf(logits[i] - m); s += logits[i]; }
    for (i = 0; i < vocab; i++) logits[i] /= s;
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
    for (i = 0; i < ntokens; i++) {
        if (pos >= e->max_seq) break;
        uint32_t nxt;
        if (engine_sample(e, e->ws.model.h.vocab, temp, top_p, &rng, &nxt) != 0) return -1;
        if (eos_stop >= 0 && (int)nxt == eos_stop) break;   /* eos 不发给对端 */
        if (on_token && on_token(nxt, ctx) != 0) break;     /* 回调可中止(如对端断开) */
        engine_forward(e, nxt, pos);
        pos++;
        ngen++;
    }
    if (timings) {
        timings->n_decode = ngen;
        timings->decode_ms = ynow_ms() - t1;
    }
    return 0;
}

uint64_t engine_resident(const Engine* e)
{
#ifdef __linux__
    /* mincore 对 MAP_SHARED 页缓存恒为 present(假象), 直接读真实 RSS */
    long v = 0;
    FILE* f = fopen("/proc/self/status", "r");
    char l[256];
    if (f) {
        while (fgets(l, sizeof l, f))
            if (!strncmp(l, "VmRSS:", 6)) { sscanf(l + 6, "%ld", &v); break; }
        fclose(f);
    }
    if (v > 0) return (uint64_t)v * 1024;
#endif
    return e->ws.resident;
}