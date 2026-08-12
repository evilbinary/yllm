#include "yllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    Ws* ws;
    uint32_t next;
    uint32_t end;
} PFJob;

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
    if (ws->pstate[layer] != 0) return;
    ws_prefetch(ws, layer);
    ws->pstate[layer] = 2;
    ws->resident += ws->layer_size[layer];
}

static void sched_release_budget(Ws* ws, uint32_t cur)
{
    if (ws->budget == 0) return;
    uint32_t i;
    for (;;) {
        if (ws->resident <= ws->budget) return;
        int found = 0;
        for (i = 0; i < cur; i++) {
            if (ws->pstate[i] == 2 && !ws->hot[i]) {
                ws_release(ws, i);
                ws->pstate[i] = 3;
                ws->resident -= ws->layer_size[i];
                found = 1;
                break;
            }
        }
        if (!found) return;
    }
}

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
    ws->hot = (uint8_t*)ycalloc(m->n_layers, 1);
    ws->layer_size = (uint64_t*)ymalloc((size_t)m->n_layers * 8);
    uint32_t i;
    for (i = 0; i < m->n_layers; i++) {
        ws->layer_size[i] = m->dir[i].size;
        if (i == 0 || i == m->h.n_blocks + 1 || i == m->h.n_blocks + 2) ws->hot[i] = 1;
    }

    uint32_t hidden = m->h.hidden;
    uint32_t kv_dim = m->h.n_kv_heads * m->h.head_dim;
    uint32_t vocab = m->h.vocab;
    e->kv_dim = kv_dim;
    e->max_seq = m->h.max_seq;
    e->kv = (uint16_t*)ycalloc((size_t)(2 * m->h.n_blocks + 1) * e->max_seq * kv_dim, 2);
    e->x = (float*)ycalloc(hidden, 4);
    e->hb = (float*)ycalloc(hidden, 4 * 9);
    e->hb2 = (float*)ycalloc(hidden, 4 * 9);
    e->att = (float*)ymalloc((size_t)e->max_seq * 4);
    e->logits = (float*)ymalloc((size_t)vocab * 4);

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
    free(e->att);
    free(e->logits);
    if (e->ws.model.base_idx) free(e->ws.model.base_idx);
    if (e->ws.pstate) free(e->ws.pstate);
    if (e->ws.hot) free(e->ws.hot);
    if (e->ws.layer_size) free(e->ws.layer_size);
    free(w);
    memset(e, 0, sizeof(*e));
}

static void gsm_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 8] & 63;
    } else {
        *d = (q[j + 4] >> 6) | ((q[j] >> 4) << 2);
        *m = (q[j + 12] >> 6) | ((q[j + 8] >> 4) << 2);
    }
}

static float q6k_val(const uint8_t* blk, uint32_t e)
{
    float d = f16_to_f32(((const uint16_t*)blk)[104]);
    uint32_t half = e >> 7;
    uint32_t k = e & 0x7f;
    uint32_t quad = k >> 5;
    uint32_t ll = k & 31;
    uint8_t ql = blk[half * 64 + (quad & 1) * 32 + ll];
    uint8_t qh = blk[128 + half * 32 + ll];
    uint32_t bits = ((quad & 2) ? (ql >> 4) : (ql & 0xF)) | (((qh >> (quad * 2)) & 3) << 4);
    int8_t q = (int8_t)bits - 32;
    int8_t s = ((const int8_t*)(blk + 192))[half * 8 + quad * 2 + (ll >> 4)];
    return d * ((float)s + 32.0f) * (float)q;
}

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static void iq4xs_block(float* y, const uint8_t* blk, float d)
{
    const uint8_t* qs = blk + 2;
    const uint8_t* sc_h = blk + 130;
    const uint8_t* sc_l = blk + 134;
    uint32_t ib;
    for (ib = 0; ib < 8; ib++) {
        int ls = ((sc_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sc_h[ib / 2] >> 2 * (ib % 2)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        uint32_t j;
        for (j = 0; j < 16; j++) {
            y[ib * 32 + j] = dl * (float)kvalues_iq4nl[qs[ib * 16 + j] & 0xF];
            y[ib * 32 + j + 16] = dl * (float)kvalues_iq4nl[qs[ib * 16 + j] >> 4];
        }
    }
}

static void embed_q4k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 144;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        const uint8_t* blk = r + (size_t)b * 144;
        float d = f16_to_f32(((const uint16_t*)blk)[0]);
        float min = f16_to_f32(((const uint16_t*)blk)[1]);
        float d1[8], m1[8];
        uint32_t g;
        for (g = 0; g < 8; g++) {
            uint8_t sc, m;
            gsm_k4((int)g, blk + 4, &sc, &m);
            d1[g] = d * (float)sc;
            m1[g] = min * (float)m;
        }
        for (g = 0; g < 8; g++) {
            uint32_t e;
            for (e = 0; e < 32; e++) {
                uint8_t nib = blk[16 + g * 16 + (e >> 1)];
                nib = (e & 1) ? (nib >> 4) : (nib & 0xF);
                y[b * 256 + g * 32 + e] = d1[g] * nib - m1[g];
            }
        }
    }
}

static void embed_q6k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 210;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        const uint8_t* blk = r + (size_t)b * 210;
        uint32_t e;
        for (e = 0; e < 256; e++) y[b * 256 + e] = q6k_val(blk, e);
    }
}

static void embed_f32(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    const float* wp = (const float*)w + (size_t)row * hidden;
    memcpy(y, wp, (size_t)hidden * 4);
}

static void rmsnorm(float* y, const float* x, const uint8_t* w, uint32_t n, float eps, uint32_t dtype)
{
    float s = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    float inv = 1.0f / sqrtf(s / (float)n + eps);
    if (dtype == DT_F32) {
        const float* wp = (const float*)w;
        for (i = 0; i < n; i++) y[i] = x[i] * wp[i] * inv;
    } else {
        const uint16_t* wp = (const uint16_t*)w;
        for (i = 0; i < n; i++) y[i] = x[i] * f16_to_f32(wp[i]) * inv;
    }
}

static void matmul_f32(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    const float* wp = (const float*)w;
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * wp[(size_t)oo * in + ii];
        y[oo] = acc;
    }
}

static void matmul_f16(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t oo;
    const uint16_t* wp = (const uint16_t*)w;
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * f16_to_f32(wp[(size_t)oo * in + ii]);
        y[oo] = acc;
    }
}

static void matmul_iq4xs(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    float tmp[256];
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            float d = f16_to_f32(((const uint16_t*)row)[0]);
            iq4xs_block(tmp, row + (size_t)b * 144, d);
            const float* xb = x + (size_t)b * 256;
            uint32_t e;
            for (e = 0; e < 256; e++) acc += xb[e] * tmp[e];
        }
        y[oo] = acc;
    }
}

static void embed_iq4xs(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 144;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        float d = f16_to_f32(((const uint16_t*)r)[0]);
        iq4xs_block(y + (size_t)b * 256, r + (size_t)b * 144, d);
    }
}

static void matmul_f32_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out)
{
    const float* wp = (const float*)w;
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * wp[(size_t)ii * out + oo];
        y[oo] = acc;
    }
}

static void matmul_q4k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float* xb = x + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[0]);
            float min = f16_to_f32(((const uint16_t*)blk)[1]);
            float d1[8], m1[8];
            uint32_t g;
            for (g = 0; g < 8; g++) {
                uint8_t sc, m;
                gsm_k4((int)g, blk + 4, &sc, &m);
                d1[g] = d * (float)sc;
                m1[g] = min * (float)m;
            }
            for (g = 0; g < 8; g++) {
                uint32_t e;
                for (e = 0; e < 32; e++) {
                    uint8_t nib = blk[16 + g * 16 + (e >> 1)];
                    nib = (e & 1) ? (nib >> 4) : (nib & 0xF);
                    acc += xb[g * 32 + e] * (d1[g] * nib - m1[g]);
                }
            }
        }
        y[oo] = acc;
    }
}

static void matmul_q6k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 210;
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const float* xb = x + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[104]);
            uint32_t e;
            for (e = 0; e < 256; e++) {
                uint32_t half = e >> 7;
                uint32_t k = e & 0x7f;
                uint32_t quad = k >> 5;
                uint32_t ll = k & 31;
                uint8_t ql = blk[half * 64 + (quad & 1) * 32 + ll];
                uint8_t qh = blk[128 + half * 32 + ll];
                uint32_t bits = ((quad & 2) ? (ql >> 4) : (ql & 0xF)) | (((qh >> (quad * 2)) & 3) << 4);
                int8_t q = (int8_t)bits - 32;
                int8_t s = ((const int8_t*)(blk + 192))[half * 8 + quad * 2 + (ll >> 4)];
                acc += xb[e] * d * (float)s * (float)q;
            }
        }
        y[oo] = acc;
    }
}

static void matmul(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype)
{
    switch (dtype) {
    case DT_F32: matmul_f32(y, x, w, out, in); break;
    case DT_Q4K: matmul_q4k(y, x, w, out, in); break;
    case DT_Q6K: matmul_q6k(y, x, w, out, in); break;
    case DT_IQ4XS: matmul_iq4xs(y, x, w, out, in); break;
    default: matmul_f16(y, x, w, out, in); break;
    }
}

static void matmul_f16_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out)
{
    uint32_t oo;
    const uint16_t* wp = (const uint16_t*)w;
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * f16_to_f32(wp[(size_t)ii * out + oo]);
        y[oo] = acc;
    }
}

static void embed_f16(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    const uint16_t* wp = (const uint16_t*)w + (size_t)row * hidden;
    uint32_t i;
    for (i = 0; i < hidden; i++) y[i] = f16_to_f32(wp[i]);
}

static void rope_inplace(float* v, uint32_t d, uint32_t pos, float theta)
{
    uint32_t half = d / 2;
    uint32_t j;
    for (j = 0; j < half; j++) {
        float freq = powf(theta, -2.0f * (float)j / (float)d);
        float ang = freq * (float)pos;
        float c = cosf(ang);
        float s = sinf(ang);
        float a = v[j];
        float b = v[j + half];
        v[j] = a * c - b * s;
        v[j + half] = a * s + b * c;
    }
}

static void softmax(float* v, uint32_t n)
{
    float m = v[0];
    uint32_t i;
    for (i = 1; i < n; i++) if (v[i] > m) m = v[i];
    float s = 0.0f;
    for (i = 0; i < n; i++) { v[i] = expf(v[i] - m); s += v[i]; }
    for (i = 0; i < n; i++) v[i] /= s;
}

static void swiglu(float* y, const float* gate, const float* up, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) {
        float g = gate[i];
        y[i] = g / (1.0f + expf(-g)) * up[i];
    }
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

    uint16_t* kcache = e->kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = e->kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
    uint64_t kvp = (uint64_t)pos * kv_dim;
    uint32_t j;
    for (j = 0; j < kv_dim; j++) {
        kcache[kvp + j] = f32_to_f16(k[j]);
        vcache[kvp + j] = f32_to_f16(v[j]);
    }
    uint32_t hh;
    for (hh = 0; hh < h->n_heads; hh++) {
        rope_inplace(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
    }
    for (hh = 0; hh < h->n_kv_heads; hh++) {
        rope_inplace(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
    }

    float* att = e->att;
    float inv_d = 1.0f / sqrtf((float)h->head_dim);
    for (hh = 0; hh < h->n_heads; hh++) {
        uint32_t kv_head = hh * h->n_kv_heads / h->n_heads;
        const float* qh = q + (size_t)hh * h->head_dim;
        uint32_t s;
        for (s = 0; s <= pos; s++) {
            const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
            float acc = 0.0f;
            for (j = 0; j < h->head_dim; j++) acc += qh[j] * f16_to_f32(kh[j]);
            att[s] = acc * inv_d;
        }
        softmax(att, pos + 1);
        float* out = att_out + (size_t)hh * h->head_dim;
        memset(out, 0, (size_t)h->head_dim * 4);
        for (s = 0; s <= pos; s++) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
            float a = att[s];
            for (j = 0; j < h->head_dim; j++) out[j] += a * f16_to_f32(vh[j]);
        }
    }
    matmul(att_out, att_out, base + mt[SLOT_O].offset, hidden, hidden, mt[SLOT_O].dtype);
    for (j = 0; j < hidden; j++) x[j] += att_out[j];

    rmsnorm(x2, x, base + mt[SLOT_NORM2].offset, hidden, eps, mt[SLOT_NORM2].dtype);
    matmul(q, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
    matmul(k, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
    swiglu(x2, q, k, inter);
    matmul(att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
    for (j = 0; j < hidden; j++) x[j] += att_out[j];
    return 0;
}

int engine_forward(Engine* e, uint32_t token, uint32_t pos)
{
    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    Worker* w = (Worker*)ws->worker;
    const LlfHeader* h = &m->h;
    uint32_t i;
    for (i = 0; i < m->n_layers; i++) {
        sched_ensure(ws, i);
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
            if (tm->shape[0] == h->vocab) {
                matmul(e->logits, e->x, base + tm->offset, h->vocab, h->hidden, tm->dtype);
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
        if (w) {
            uint32_t d = (uint32_t)ws->depth;
            uint32_t nb = i + d;
            uint32_t ne = nb + d;
            if (nb < m->n_layers) {
                if (ne > m->n_layers) ne = m->n_layers;
                while (nb < ne && ws->pstate[nb] != 0) nb++;
                if (nb < ne) {
                    ws->pstate[nb] = 1;
                    sched_enqueue(w, nb, ne);
                }
            }
        }
        sched_release_budget(ws, i);
    }
    return 0;
}

int engine_sample(Engine* e, uint32_t vocab, float temp, float top_p, uint64_t* rng, uint32_t* out)
{
    float* logits = e->logits;
    uint32_t i;
    if (temp > 0 && temp != 1.0f) {
        for (i = 0; i < vocab; i++) logits[i] /= temp;
    }
    float m = logits[0];
    for (i = 1; i < vocab; i++) if (logits[i] > m) m = logits[i];
    float s = 0.0f;
    for (i = 0; i < vocab; i++) { logits[i] = expf(logits[i] - m); s += logits[i]; }
    for (i = 0; i < vocab; i++) logits[i] /= s;
    if (top_p < 1.0f) {
        uint32_t* idx = (uint32_t*)ymalloc((size_t)vocab * 4);
        uint32_t j;
        for (j = 0; j < vocab; j++) idx[j] = j;
        uint32_t k;
        for (k = 0; k < vocab; k++) {
            uint32_t best = k;
            for (j = k + 1; j < vocab; j++) if (logits[idx[j]] > logits[idx[best]]) best = j;
            uint32_t t2 = idx[k];
            idx[k] = idx[best];
            idx[best] = t2;
        }
        float cum = 0.0f;
        uint32_t keep = vocab;
        for (k = 0; k < vocab; k++) {
            cum += logits[idx[k]];
            if (cum >= top_p) { keep = k + 1; break; }
        }
        float c2 = 0.0f;
        for (k = 0; k < keep; k++) c2 += logits[idx[k]];
        for (k = 0; k < keep; k++) logits[idx[k]] /= c2;
        float r = (float)(yrng(rng) >> 40) / 16777216.0f;
        float acc = 0.0f;
        for (k = 0; k < keep; k++) {
            acc += logits[idx[k]];
            if (r < acc) { *out = idx[k]; free(idx); return 0; }
        }
        *out = idx[keep - 1];
        free(idx);
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

int engine_generate(Engine* e, const uint32_t* prompt, int nprompt, int ntokens,
                    float temp, float top_p, uint64_t seed,
                    void (*on_token)(uint32_t id, void* ctx), void* ctx, char* err, size_t errlen)
{
    uint64_t rng = ysrand(seed);
    uint32_t pos = 0;
    int i;
    {
        int bi = 0;
        uint32_t bj;
        for (bj = 1; bj < e->ws.model.h.vocab; bj++) if (e->logits[bj] > e->logits[bi]) bi = (int)bj;
        printf("DEBUG logits[0..5]=%.3g %.3g %.3g %.3g %.3g %.3g argmax=%d=%.3g finite=%d\n",
            (double)e->logits[0], (double)e->logits[1], (double)e->logits[2], (double)e->logits[3],
            (double)e->logits[4], (double)e->logits[5], bi, (double)e->logits[bi],
            (int)(e->logits[0] == e->logits[0]));
    }
    for (i = 0; i < nprompt; i++) {
        if (pos >= e->max_seq) {
            if (err) snprintf(err, errlen, "prompt too long");
            return -1;
        }
        engine_forward(e, prompt[i], pos);
        pos++;
    }
    for (i = 0; i < ntokens; i++) {
        if (pos >= e->max_seq) break;
        uint32_t nxt;
        if (engine_sample(e, e->ws.model.h.vocab, temp, top_p, &rng, &nxt) != 0) return -1;
        if (on_token) on_token(nxt, ctx);
        engine_forward(e, nxt, pos);
        pos++;
    }
    return 0;
}

uint64_t engine_resident(const Engine* e)
{
    return e->ws.resident;
}