# inference/shaders — Vulkan SPIR-V（预编译，三端共用）
#
# 构建示例（需 glslc）:
#   glslc -fshader-stage=compute rmsnorm.comp -o rmsnorm.spv
#
# 当前: 占位；device_vulkan P0 仅创建 VkDevice，前向仍 CPU。

# 计划中的 compute:
#   rmsnorm.comp / rope.comp / attn_decode.comp / swiglu.comp / gemv_q4k.comp
