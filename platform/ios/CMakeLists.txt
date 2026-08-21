cmake_minimum_required(VERSION 3.22)
project(yllm_ios C)

set(YLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(INFER "${YLLM_ROOT}/inference")

option(YLLM_VULKAN "Vulkan via MoltenVK" OFF)

set(YLLM_SRCS
  ${INFER}/platform.c
  ${INFER}/log.c
  ${INFER}/llf.c
  ${INFER}/convert.c
  ${INFER}/convert_safetensors.c
  ${INFER}/convert_gguf.c
  ${INFER}/tokenizer.c
  ${INFER}/matvec.c
  ${INFER}/engine.c
  ${INFER}/cache.c
  ${INFER}/dist.c
  ${INFER}/device_cpu.c
)
if(YLLM_VULKAN)
  list(APPEND YLLM_SRCS
    ${INFER}/device_vulkan.c
    ${INFER}/vulkan_load.c
    ${INFER}/vulkan_fwd.c)
endif()

add_library(yllm STATIC ${YLLM_SRCS})
target_include_directories(yllm PUBLIC ${INFER})
target_compile_definitions(yllm PRIVATE YLLM_BATCH_PREFILL=1 YLLM_SESS_DEBUG=0)
if(YLLM_VULKAN)
  target_compile_definitions(yllm PRIVATE YLLM_VULKAN=1)
  # 链 MoltenVK: target_link_libraries(yllm "-framework Metal" moltenvk)
endif()
