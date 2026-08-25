# Android NDK — libyllm.so + yllm (同 PC main.c 命令)
cmake_minimum_required(VERSION 3.22)
project(yllm_android C)

set(YLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(INFER "${YLLM_ROOT}/inference")
set(SERVE "${YLLM_ROOT}/serve")
set(INFER_INC
  ${INFER}/include
  ${INFER}/cuda
  ${INFER}/vulkan
)

option(YLLM_VULKAN "Build Vulkan device backend" OFF)
option(YLLM_W4 "Convert/prefer W4B64 for linear Q4_K weights" ON)

set(YLLM_CORE_SRCS
  ${INFER}/core/platform.c
  ${INFER}/core/log.c
  ${INFER}/core/llf.c
  ${INFER}/convert/convert.c
  ${INFER}/convert/convert_safetensors.c
  ${INFER}/convert/convert_gguf.c
  ${INFER}/core/tokenizer.c
  ${INFER}/core/matvec.c
  ${INFER}/core/engine.c
  ${INFER}/core/cache.c
  ${INFER}/core/dist.c
  ${INFER}/device/device_cpu.c
)

if(YLLM_VULKAN)
  list(APPEND YLLM_CORE_SRCS
    ${INFER}/device/device_vulkan.c
    ${INFER}/vulkan/vulkan_load.c
    ${INFER}/vulkan/vulkan_fwd.c
    ${INFER}/vulkan/vulkan_compute.c)
endif()

add_library(yllm SHARED ${YLLM_CORE_SRCS})
target_include_directories(yllm PUBLIC ${INFER_INC})
target_compile_definitions(yllm PRIVATE
  YLLM_BATCH_PREFILL=1
  YLLM_SESS_DEBUG=0
  _DEFAULT_SOURCE
)
if(YLLM_W4)
  target_compile_definitions(yllm PRIVATE YLLM_W4=1)
else()
  target_compile_definitions(yllm PRIVATE YLLM_W4=0)
endif()
if(YLLM_VULKAN)
  target_compile_definitions(yllm PRIVATE YLLM_VULKAN=1)
  find_library(VULKAN_LIB vulkan)
  if(VULKAN_LIB)
    target_link_libraries(yllm ${VULKAN_LIB})
  endif()
endif()

# arm64: Q4_K NEON(+dotprod) + OpenMP。-static-openmp 避免设备缺 libomp.so
if(ANDROID_ABI STREQUAL "arm64-v8a")
  target_compile_options(yllm PRIVATE
    -O3 -ffast-math -fno-finite-math-only
    -march=armv8.2-a+dotprod
    -fopenmp)
  target_link_options(yllm PRIVATE -fopenmp -static-openmp)
endif()

find_library(LOG_LIB log)
target_link_libraries(yllm ${LOG_LIB} m)

set(YLLM_CLI_SRCS
  ${YLLM_ROOT}/main.c
  ${SERVE}/rank.c
  ${SERVE}/server.c
  ${SERVE}/router.c
  ${SERVE}/supervisor.c
  ${SERVE}/hub.c
  ${SERVE}/router_http.c
  ${SERVE}/status.c
  ${SERVE}/ctl.c
  ${SERVE}/sync.c
)

add_executable(yllm_cli ${YLLM_CLI_SRCS})
set_target_properties(yllm_cli PROPERTIES OUTPUT_NAME yllm)
target_include_directories(yllm_cli PRIVATE ${INFER_INC} ${SERVE} ${YLLM_ROOT})
target_compile_definitions(yllm_cli PRIVATE
  YLLM_BATCH_PREFILL=1
  YLLM_SESS_DEBUG=0
  _DEFAULT_SOURCE
)
if(YLLM_W4)
  target_compile_definitions(yllm_cli PRIVATE YLLM_W4=1)
else()
  target_compile_definitions(yllm_cli PRIVATE YLLM_W4=0)
endif()
if(YLLM_VULKAN)
  target_compile_definitions(yllm_cli PRIVATE YLLM_VULKAN=1)
endif()
target_link_libraries(yllm_cli yllm ${LOG_LIB})

# 兼容旧入口
add_executable(yllm_gen ${CMAKE_CURRENT_SOURCE_DIR}/gen_main.c)
target_include_directories(yllm_gen PRIVATE ${INFER_INC})
target_link_libraries(yllm_gen yllm ${LOG_LIB})

if(YLLM_VULKAN)
  file(GLOB YLLM_SPV "${INFER}/vulkan/shaders/*.spv")
  if(YLLM_SPV)
    file(COPY ${YLLM_SPV} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)
  endif()
endif()
