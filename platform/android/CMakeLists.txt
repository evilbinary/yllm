# Android NDK — libyllm (CPU) + yllm_gen
cmake_minimum_required(VERSION 3.22)
project(yllm_android C)

set(YLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(INFER "${YLLM_ROOT}/inference")

option(YLLM_VULKAN "Build Vulkan device backend" OFF)

set(YLLM_CORE_SRCS
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
  list(APPEND YLLM_CORE_SRCS
    ${INFER}/device_vulkan.c
    ${INFER}/vulkan_load.c
    ${INFER}/vulkan_fwd.c
    ${INFER}/vulkan_compute.c)
endif()

add_library(yllm SHARED ${YLLM_CORE_SRCS})
target_include_directories(yllm PUBLIC ${INFER})
target_compile_definitions(yllm PRIVATE
  YLLM_BATCH_PREFILL=1
  YLLM_SESS_DEBUG=0
  _DEFAULT_SOURCE
)
if(YLLM_VULKAN)
  target_compile_definitions(yllm PRIVATE YLLM_VULKAN=1)
  find_library(VULKAN_LIB vulkan)
  # 动态 dlopen; 仍链 NDK vulkan 以便包内有符号可选
  if(VULKAN_LIB)
    target_link_libraries(yllm ${VULKAN_LIB})
  endif()
endif()

# arm64: 允许编译器自动向量化; 显式 NEON 路径后续补 matvec
if(ANDROID_ABI STREQUAL "arm64-v8a")
  target_compile_options(yllm PRIVATE -O2 -ffast-math)
endif()

find_library(LOG_LIB log)
target_link_libraries(yllm ${LOG_LIB} m)

add_executable(yllm_gen
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_main.c
)
target_include_directories(yllm_gen PRIVATE ${INFER})
target_link_libraries(yllm_gen yllm ${LOG_LIB})
