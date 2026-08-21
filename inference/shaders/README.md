# inference/shaders — Vulkan SPIR-V（预编译，三端共用）
#
# 构建示例（需 glslc）:
#   glslc -fshader-stage=compute rmsnorm.comp -o rmsnorm.spv
#   glslc -fshader-stage=compute gemv_q4k.comp -o gemv_q4k.spv
#
# 当前:
#   rmsnorm.comp  — 单 WG RMSNorm
#   gemv_q4k.comp — 每 WG 一行 Q4_K·x（144B/block）
#   swiglu.comp   — silu(gate)*up
# 运行时搜 inference/shaders/*.spv，或设 YLLM_SHADER_DIR。

# 计划: rope / attn_decode；权常驻已做
