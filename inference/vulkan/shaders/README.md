# inference/vulkan/shaders — Vulkan SPIR-V（预编译，三端共用）

构建示例（需 glslc）:

```bash
glslc -fshader-stage=compute rmsnorm.comp -o rmsnorm.spv
glslc -fshader-stage=compute gemv_q4k.comp -o gemv_q4k.spv
glslc -fshader-stage=compute gemv_q6k.comp -o gemv_q6k.spv
glslc -fshader-stage=compute q8k_quant.comp -o q8k_quant.spv
glslc -fshader-stage=compute embed_q4k.comp -o embed_q4k.spv
glslc -fshader-stage=compute swiglu.comp -o swiglu.spv
glslc -fshader-stage=compute attn_decode.comp -o attn_decode.spv
```

当前:

- `rmsnorm.comp` — 单 WG RMSNorm
- `gemv_q4k.comp` — 每 WG 一行 Q4_K·x（144B/block）
- `gemv_q6k.comp` — 每 WG 一行 Q6_K·x（210B/block, lm_head）
- `q8k_quant.comp` — Q8_K 激活量化 in-place（lm_head 前）
- `embed_q4k.comp` — token embedding 行 Q4_K 解包 → buf_x
- `swiglu.comp` — silu(gate)*up
- `attn_decode.comp` — 每 head 一个 WG、64 线程；score 共享缓存；可与 O gemv 同 submit
- `rope.comp` — llama/qwen RoPE in-place（可与 QKV 同 submit）

运行时搜 `inference/vulkan/shaders/*.spv`，或设 `YLLM_SHADER_DIR`。
