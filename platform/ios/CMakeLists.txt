cmake_minimum_required(VERSION 3.22)
project(yllm_ios C)

set(YLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(INFER "${YLLM_ROOT}/inference")
set(INFER_INC
  ${INFER}/include
  ${INFER}/cuda
  ${INFER}/vulkan
)

option(YLLM_VULKAN "Vulkan via MoltenVK" OFF)

set(YLLM_SRCS
  ${INFER}/core/platform.c
  ${INFER}/core/log.c
  ${INFER}/core/llf.c
  ${INFER}/convert/convert.c
  ${INFER}/convert/convert_safetensors.c
  ${INFER}/convert/convert_gguf.c
  ${INFER}/convert/dump.c
  ${INFER}/core/tokenizer.c
  ${INFER}/core/matvec.c
  ${INFER}/core/engine.c
  ${INFER}/arch/arch.c
  ${INFER}/arch/llama.c
  ${INFER}/arch/qwen.c
  ${INFER}/arch/gemma4.c
  ${INFER}/arch/qwen35.c
  ${INFER}/core/cache.c
  ${INFER}/core/dist.c
  ${INFER}/device/device_cpu.c
)
if(YLLM_VULKAN)
  list(APPEND YLLM_SRCS
    ${INFER}/device/device_vulkan.c
    ${INFER}/vulkan/vulkan_load.c
    ${INFER}/vulkan/vulkan_fwd.c
    ${INFER}/vulkan/vulkan_compute.c)
endif()

add_library(yllm STATIC ${YLLM_SRCS})
target_include_directories(yllm PUBLIC ${INFER_INC})
target_compile_definitions(yllm PRIVATE YLLM_BATCH_PREFILL=1 YLLM_SESS_DEBUG=0)
if(YLLM_VULKAN)
  target_compile_definitions(yllm PRIVATE YLLM_VULKAN=1)
  # 链 MoltenVK: target_link_libraries(yllm "-framework Metal" moltenvk)
endif()
