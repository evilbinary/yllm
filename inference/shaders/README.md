# inference/shaders — Vulkan SPIR-V（预编译，三端共用）
#
# 构建示例（需 glslc）:
#   glslc -fshader-stage=compute rmsnorm.comp -o rmsnorm.spv
#
# 当前:
#   rmsnorm.comp — 单 workgroup RMSNorm；由 vulkan_compute 在 load 时加载。
#   运行时搜 inference/shaders/rmsnorm.spv，或设 YLLM_SHADER_DIR。

# 计划中的 compute:
#   rope.comp / attn_decode.comp / swiglu.comp / gemv_q4k.comp
