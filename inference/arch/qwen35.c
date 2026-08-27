#include "arch.h"
#include "yllm.h"
#include "llf.h"
#include "matvec.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


typedef struct {
    float *ssm_state, *ssm_conv, *scratch;
} Qwen35Ctx;

static Qwen35Ctx* q35c(Engine* e)
{
    return (e && e->arch_ctx) ? (Qwen35Ctx*)e->arch_ctx : NULL;
}

int arch_qwen35_alloc(Engine* e)
{
    LlModel* m = &e->ws.model;
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
    Qwen35Ctx* c = (Qwen35Ctx*)ycalloc(1, sizeof(*c));
    if (!c) return -1;
    e->arch_ctx = c;
    c->scratch = (float*)ymalloc(65536 * 4);
    if (n_gdn > 0 && conv_chan > 0 && n_vheads > 0 && hvd > 0) {
        c->ssm_state = (float*)ycalloc((size_t)n_gdn * n_vheads * hvd * hvd, 4);
        c->ssm_conv = (float*)ycalloc((size_t)n_gdn * (kwidth > 0 ? kwidth : 1) * conv_chan, 4);
    }
    return 0;
}

void arch_qwen35_free(Engine* e)
{
    Qwen35Ctx* c = q35c(e);
    if (!c) return;
    free(c->ssm_state);
    free(c->ssm_conv);
    free(c->scratch);
    free(c);
    e->arch_ctx = NULL;
}

const ArchOps arch_qwen35_ops = {
    .name = "qwen35",
    .id = ARCH_QWEN35,
    .cpu_batch_prefill = 1,
    .prefill_batch_min = 16,
    .gpu_fused = 0,
    .qwen_rope = 0,
    .alloc = arch_qwen35_alloc,
    .free = arch_qwen35_free,
    .fwd_block = arch_qwen35_fwd_block,
    .fwd_block_batch = arch_qwen35_fwd_block_batch,
};

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
    Qwen35Ctx* c = q35c(e);
    if (!c || !c->scratch) return -1;
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

        float* scratch = c->scratch;
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
        float* conv_state = c->ssm_conv + (size_t)gdn_idx * kwidth * conv_chan;
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
        float* state_base = c->ssm_state + (size_t)gdn_idx * n_vheads * hvd * hvd;
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

        float* scratch = c->scratch;
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
