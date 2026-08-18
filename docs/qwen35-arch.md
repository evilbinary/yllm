# Qwen35(Qwen3.8)引擎实现设计

## 目标
在现有引擎(纯 attention Transformer)上新增 `ARCH_QWEN35`,支持 Qwen3.8-27B
的 **Gated Attention + GDN(SSM)混合架构**。**不改动 llf 文件格式**。

## 架构事实(从 Qwen3.8-27B-Q4_K_M.gguf 解析确认)

- `general.architecture = qwen35`,65 层,`block_count=65`
- `embedding_length = 5120`, `head_count = 24`, `head_count_kv = 4`(GQA 24:4)
- **`attention.key_length = 256`** → head_dim = 256(非 `hidden/heads`=213,故需独立存)
- `rope.freq_base = 1e7`, `rope.dimension_count = 64`
- **`rope.dimension_sections = [11,11,10,0]`**(M-RoPE;文本推理折叠为 1-D RoPE)
- vocab = 248320
- **混合层**:17 个 Gated Attention 层 + 48 个 GDN 层

### 两类层的张量

**GDN 层**(0,1,2,4,5,6,8,...,62 — 无 ssm 的才是 attention):
```
attn_norm[5120] attn_qkv[5120,10240] attn_gate[5120,6144]
post_attention_norm[5120]
ssm_a[48] ssm_alpha[5120,48] ssm_beta[5120,48]
ssm_conv1d[4,10240] ssm_dt.bias[48] ssm_norm[128] ssm_out[6144,5120]
ffn_gate[5120,17408] ffn_up[5120,17408] ffn_down[17408,5120]
```

**Gated Attention 层**(3,7,11,15,19,...,63,64 — 无 ssm/qkv):
```
attn_norm[5120] attn_q[5120,12288] attn_q_norm[256]
attn_k[5120,1024] attn_k_norm[256] attn_v[5120,1024]
attn_o[6144,5120]
post_attention_norm[5120]
ffn_gate[5120,17408] ffn_up[5120,17408] ffn_down[17408,5120]
```

## 关键理解

- **attn_qkv[5120,10240] 属于 GDN 层**,不是 attention 层。它等宽 2×hidden。
- **Gated Attention 层的 attn_q[5120,12288] = q|gate 拼接**:
  q 部分 24 heads × 256 = 6144, gate 部分 24 × 256 = 6144。attn_o 输入宽 6144(仅 q 输出)。
- **head_dim = 256**(非 5120/24)。q/k/v per-head 都是 256。
  - q 投影后 6144 = 24 heads × 256(去 gate)
  - k/v 投影后 1024 = 4 kv heads × 256
- **M-RoPE 折叠**:文本推理所有 section 用同一位置 → 等价标准 1-D RoPE,head_dim=64 段。

## 引擎改动(不改 llf 格式)

### 1. llf.h — 仅加枚举与 slot(格式不变)
```c
#define ARCH_QWEN35 2
/* BLOCK_TENSORS 14 → 25: 复用 SLOT_NORM1/NORM2/GATE/UP/DOWN/QNORM/KNORM, 新增: */
#define SLOT_QKV       14  /* GDN attn_qkv [in, 2*hidden] */
#define SLOT_GATE_ATTN 15  /* attn_gate [in, hidden] */
#define SLOT_QGATE     16  /* attention 层 attn_q 拆分后的 gate 半部(转换期写) */
#define SLOT_SSM_CONV1D 17 /* [conv_dim, 2*hidden] */
#define SLOT_SSM_A     18   /* [n_vheads] */
#define SLOT_SSM_DT    19   /* [n_vheads] bias */
#define SLOT_SSM_ALPHA 20   /* [in, n_vheads] */
#define SLOT_SSM_BETA  21   /* [in, n_vheads] */
#define SLOT_SSM_NORM  22   /* [head_v_dim] */
#define SLOT_SSM_OUT   23   /* [in, hidden] */
#define BLOCK_TENSORS  24
```
> `LlfHeader.arch` 已有,加枚举值即支持;head_dim 用现有 `LlfHeader.head_dim` 存 256(转换期从 `attention.key_length` 读)。

### 2. convert_gguf.c — 张量映射 + head_dim
`gg_slot_for` 的 `blk.` 分支加:
```c
{ "attn_qkv.weight", SLOT_QKV },
{ "attn_gate.weight", SLOT_GATE_ATTN },
{ "post_attention_norm.weight", SLOT_FFN_NORM },  /* 与 llama.cpp 一致: 放 norm2 槽 */
{ "ssm_conv1d.weight", SLOT_SSM_CONV1D },
{ "ssm_a", SLOT_SSM_A },
{ "ssm_dt.bias", SLOT_SSM_DT },
{ "ssm_alpha.weight", SLOT_SSM_ALPHA },
{ "ssm_beta.weight", SLOT_SSM_BETA },
{ "ssm_norm.weight", SLOT_SSM_NORM },
{ "ssm_out.weight", SLOT_SSM_OUT },
```
head_dim 处理:convert_gguf.c 解析 `attention.key_length`(256)写入 `h.head_dim`,不再用 `hidden/heads`。

**GDN 层的 attn_qkv 处理**:10240 = 6144(q) + 4096(k+v)。转换期写入 `SLOT_QKV` 一个张量,引擎运行时按 [q | k+v] 切分(非对称 q vs kv)。

### 3. engine.c — Engine 加架构成员 + 分派

`Engine` 结构体加(模型级身份):
```c
uint32_t arch;   /* 从 header 读入; 混合层类型靠 SSM slot 存在性判断, 不新增字段 */
```

分派函数(架构级):
```c
static int fwd_block_qwen35(Engine* e, uint32_t layer, uint32_t pos);
```
`forward_layer`/`forward_block_batch` 里按 arch 走不同函数。沿用方案:`Engine` 挂 `fwd_block`/`fwd_block_batch` 指针,`engine_init` 填一次。

### 4. engine.c — 每层类型判定(不改 llf)
```c
/* 该层是否有 ssm 张量 → GDN 层, 否则 Gated Attention 层 */
static int layer_is_gdn(const LlModel* m, uint32_t layer) {
    const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
    return mt[SLOT_SSM_CONV1D].size > 0;
}
```

### 5. matvec.c — 新内核
```c
void gdn_state_update(float* S, const float* K, const float* V,
                      const float* beta, const float* alpha, const float* dtb,
                      const float* a, uint32_t n_vheads, uint32_t hk, uint32_t hv);
void rmsnorm_gated(float* y, const float* x, const uint8_t* w, const float* z,
                   uint32_t n, float eps, uint32_t dtype);
void conv1d_update(float* state, const float* x, const uint8_t* w,
                   uint32_t dim, uint32_t kwidth);
```

### 6. Engine 加 SSM 状态缓存(每层固定 O(1))
```c
float* ssm_state;  /* [n_gdn_layers × n_vheads × hk × hv] */
uint8_t* ssm_conv; /* [n_gdn_layers × 2×2×hidden] conv1d 延迟线 */
```

## 层前向逻辑

### Gated Attention 层(`layer_is_gdn==0`)
```
1. x2 = rmsnorm(x, attn_norm)                    [SLOT_NORM1]
2. q_raw = matmul(x2, attn_q)                    [SLOT_Q, in=12288]
   q = q_raw[0:6144]  (24 heads × 256)
   gate = q_raw[6144:12288]                       (24 × 256)
3. k = matmul(x2, attn_k)  [SLOT_K, 1024]
   v = matmul(x2, attn_v)  [SLOT_V, 1024]
4. q/k QK-norm 每头 ×256                          [SLOT_QNORM/KNORM]
5. rope(1-D 折叠, head_dim=256, theta=1e7)
6. 写 KV cache(和现有 attention 相同)
7. softmax attention(现有 attn_score_out 逻辑)  → attn_out[6144]
8. attn_out *= silu(gate)  (门控)                ← qwen35 特有
9. out = matmul(attn_out, attn_o)  [SLOT_O, in=6144]
10. x += out
11. ffn_swiglu (现有)                              [SLOT_NORM2/GATE/UP/DOWN]
```

### GDN 层(`layer_is_gdn==1`)
```
1. x2 = rmsnorm(x, attn_norm)
2. qkv = matmul(x2, attn_qkv)  [SLOT_QKV, 10240]
   拆: q[6144] + k[2048] + v[2048]?  (需按参考实现确认切分)
   实际参考: GDN 的 attn_qkv 拆分为 Q|K|V(见 llama.cpp qwen35 实现)
3. alpha = matmul(x2, ssm_alpha)  [SLOT_SSM_ALPHA, 48]
   beta  = matmul(x2, ssm_beta)   [SLOT_SSM_BETA, 48]
4. dt = alpha + ssm_dt.bias
5. conv1d 更新(ssm_conv1d, 宽 10240)
6. gate = softplus(dt) * ssm_a                   (ssm_a 转换期 = -exp(A_log))
7. gdn_state_update(S)  递归状态更新
8. out = rmsnorm_gated(state_out, ssm_norm, z)
9. out = matmul(out, ssm_out)  [SLOT_SSM_OUT, in=6144]
10. gated_attn = matmul(x2, attn_gate)  [SLOT_GATE_ATTN, 6144]
11. x += silu(gated_attn) * out                  (门控融合)
12. ffn_swiglu
```

## 参考实现确认的精确公式(对照 llama.cpp `src/models/qwen35.cpp` + `ggml/src/ggml-cpu/ops.cpp`)

**SSM 超参(从 gguf 元数据读到)**:
```
ssm.conv_kernel=4  ssm.group_count=16  ssm.inner_size=6144
ssm.state_size=128 ssm.time_step_rank=48   full_attention_interval=4
```
推导维度:
- num_k_heads=group_count=16, head_k_dim=state_size=128 → key_dim=16×128=2048
- num_v_heads=time_step_rank=48, head_v_dim=inner/num_v_heads=6144/48=128 → value_dim=48×128=6144
- qkv_dim = key_dim×2 + value_dim = 2048×2+6144 = 10240 ✓(attn_qkv[10240])
- conv_channels = inner + 2×group×state = 6144+4096 = 10240 ✓(ssm_conv1d[4,10240])

**GDN 层 attn_qkv[10240] 切分(顺序 Q|K|V)**: q[0:2048] | k[2048:4096] | v[4096:10240]。
注意:**qkv_mixed 先整体过 conv1d,再从卷积输出切分**。

**GDN 层完整前向(单 token, AR 递归)**:
```
1. x2 = rmsnorm(x, attn_norm)
2. qkv_mixed = matmul(x2, attn_qkv)            [10240]
   z          = matmul(x2, attn_gate)          [6144]   (SLOT_GATE_ATTN)
   alpha      = matmul(x2, ssm_alpha)          [48]
   beta       = matmul(x2, ssm_beta)           [48]
3. beta = sigmoid(beta)
   gate = softplus(alpha + ssm_dt.bias) * ssm_a   [48 每 v_head 标量]
4. conv1d: conv_input=[4 历史qkv_mixed, 当前qkv_mixed], 无bias
   conv_out = silu(conv1d(qkv_mixed))            [10240]
   切分 q=conv[0:2048] k=conv[2048:4096] v=conv[4096:10240]
   q reshape[128,16] k reshape[128,16] v reshape[128,48]
5. l2_norm q/k 每 k_head(128 维): scale=1/max(sqrt(Σx²), eps)
   v_head h 复用 k_head = h%16 的 q/k(repeat 16→48)
6. 递归状态更新(每 v_head h, S 为 [128,128] 存储转置 S[j*128+i]=Smat[i][j]):
   g = exp(gate[h]);  S *= g
   delta[j] = (v[j] - dot(S列j, k)) * beta[h]   for j in 0..127
   S列j += k * delta[j]
   attn_out[j] = dot(S列j, q) * (1/sqrt(128))   for j in 0..127
7. out = rmsnorm(attn_out, ssm_norm[128]) * silu(z reshape[128,48])   [128,48→6144]
8. out = matmul(out, ssm_out)                   [SLOT_SSM_OUT, 6144→5120]
9. x += out;  ffn_swiglu
```

**Gated Attention 层(确认无 ssm 张量,含 gate)**:
```
1. x2 = rmsnorm(x, attn_norm)
2. q_raw = matmul(x2, attn_q)[12288] = 24heads × 512
   q    = 每 head 前 256 [24×256=6144]
   gate = 每 head 后 256 [24×256=6144]   ← q|gate 拼接
3. k = matmul(x2, attn_k)[1024], v = matmul(x2, attn_v)[1024]   (4 kv heads × 256)
4. q/k QK-norm 每头 ×256(attn_q_norm/attn_k_norm)
5. M-RoPE(见下) → 写 KV cache
6. softmax attention(24:4 GQA, kq_scale=1/sqrt(256)) → attn_out[24×256]
7. attn_out *= sigmoid(gate)
8. out = matmul(attn_out, attn_o)[6144→5120];  x += out;  ffn_swiglu
```

**M-RoPE 折叠(纯文本退化,已确认)**:
- 只作用前 n_dims=64 维(head_dim=256 其余不动)
- **interleaved 对**:pair i = (v[i], v[i+32])  for i=0..31(步长 head_dim/2)
- 每对角度 = pos × freq_base^(-2i/64)(freq_base=1e7)
- 与现有引擎 `rope_inplace`(adjacent 对 (v[2i],v[2i+1]), 全维)不同 → 需新内核
- 现有 `rope_inplace_qwen` 若为 interleaved 且参数化 n_dims,可复用/泛化

**其他确认**:
- ssm_a 转换期**不取负**:gate = softplus(alpha+bias) × ssm_a(ssm_a 直接存 gguf 值,ggml 里 `ggml_mul(alpha_softplus, ssm_a)`)
- conv1d 无 bias;因果:输出[ch] = Σ_{i=0}^{3} history[i][ch]×kernel[i][ch](当前 qkv 是窗口末位)
- beta/alpha/gate 均为每 v_head 标量(非 KDA 全维),state 为 [128,128,48]

## 实施步骤
1. llf.h 加 ARCH_QWEN35 + slot(格式不变)
2. convert_gguf.c 加张量映射 + head_dim 读取 → 成功转出 qwen3.8-27b.llf
3. Engine 加 arch 成员 + fwd_block 分派 + 每层类型判定
4. matvec.c 加 gdn/conv1d/rmsnorm_gated 内核
5. 实现 fwd_block_qwen35(两层分支)+ SSM 状态缓存
6. 对照参考实现确认切分/公式
7. 实测