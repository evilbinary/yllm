# inference/vulkan/shaders — Vulkan SPIR-V（预编译，三端共用）

构建示例（需 glslc）:

```bash
glslc -fshader-stage=compute rmsnorm.comp -o rmsnorm.spv
glslc -fshader-stage=compute gemv_q4k.comp -o gemv_q4k.spv
glslc -fshader-stage=compute swiglu.comp -o swiglu.spv
glslc -fshader-stage=compute attn_decode.comp -o attn_decode.spv
```

当前:

- `rmsnorm.comp` — 单 WG RMSNorm
- `gemv_q4k.comp` — 每 WG 一行 Q4_K·x（144B/block）
- `swiglu.comp` — silu(gate)*up
- `attn_decode.comp` — 每 head 一个 WG 的 decode attn（f32 KV）

运行时搜 `inference/vulkan/shaders/*.spv`，或设 `YLLM_SHADER_DIR`。
