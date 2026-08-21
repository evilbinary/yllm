# yllm 构建 (Linux / macOS / Windows/MinGW)
#
#   make            标量版本
#   make avx2       x86_64 上的 AVX2 版本(其他架构退化为标量)
#   make cuda       AVX2 + CUDA 后端(产物在 build/avx2-cuda/, 与 avx2 隔离)
#   make vulkan     AVX2 + Vulkan 后端(产物在 build/avx2-vulkan/; P0 host-shim)
#   make gen-cuda / chat-cuda   用 --device cuda 冒烟
#   make gen-vulkan / chat-vulkan / chat-*-avx2-vulkan
#   make test       运行测试
#   make clean
#
# 跨平台打包见 platform/{android,ios,pc,web}/ 与 docs/design-mobile.md
# Windows 下请在 MSYS2 / MinGW 环境执行本 Makefile;
# 若使用 MSVC, 请改用 CMake:
#   cmake -S . -B build-msvc && cmake --build build-msvc --config Release

CC         ?= cc
LDFLAGS    ?=
LIBS       :=

# 公共头: inference/include; 后端私有头: cuda/ vulkan/
INFER_INC := -Iinference/include -Iinference/cuda -Iinference/vulkan
HDR_PUBLIC := inference/include/yllm.h inference/include/llf.h inference/include/convert.h \
	inference/include/matvec.h inference/include/dist.h inference/include/device.h \
	inference/include/log.h inference/include/cache.h

SRC := \
	inference/core/platform.c inference/core/log.c inference/core/llf.c \
	inference/convert/convert.c inference/convert/convert_safetensors.c inference/convert/convert_gguf.c \
	inference/core/tokenizer.c inference/core/matvec.c inference/core/engine.c \
	inference/core/cache.c inference/core/dist.c inference/device/device_cpu.c
TEST_ENGINE_CORE := \
	inference/core/platform.c inference/core/log.c inference/core/llf.c \
	inference/convert/convert.c inference/convert/convert_safetensors.c inference/convert/convert_gguf.c \
	inference/core/tokenizer.c inference/core/matvec.c inference/core/engine.c \
	inference/device/device_cpu.c

# ---- CUDA 后端(可选) ----
#   make cuda                              # 推荐: 独立目录 build/avx2-cuda
#   make gen-cuda / chat-cuda              # 构建并 --device cuda 跑 TinyLlama
#   make avx2 YLLM_CUDA=1                  # 旧写法(写入 build/avx2, 易与纯 CPU 产物混用)
#   make cuda YLLM_CUDA_HOST=1             # 无 toolkit / 强制 host-shim
#   make cuda YLLM_CUDA_HOST=0             # 强制真 CUDA runtime(需 libcudart)
YLLM_CUDA ?= 0
YLLM_CUDA_HOST ?=
YLLM_VULKAN ?= 0
GPU ?= 0
GPU_WEIGHTS ?= auto
NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(YLLM_CUDA),1)
  SRC += inference/device/device_cuda.c inference/cuda/cuda_fwd.c
  TEST_ENGINE_CORE += inference/device/device_cuda.c inference/cuda/cuda_fwd.c
  ifeq ($(YLLM_CUDA_HOST),)
    ifeq ($(NVCC),)
      YLLM_CUDA_HOST := 1
    else
      YLLM_CUDA_HOST := 0
    endif
  endif
endif
ifeq ($(YLLM_VULKAN),1)
  SRC += inference/device/device_vulkan.c inference/vulkan/vulkan_load.c \
	inference/vulkan/vulkan_fwd.c inference/vulkan/vulkan_compute.c
  TEST_ENGINE_CORE += inference/device/device_vulkan.c inference/vulkan/vulkan_load.c \
	inference/vulkan/vulkan_fwd.c inference/vulkan/vulkan_compute.c
endif

# 真 CUDA 时由 nvcc 编译的 .cu(host-shim 不编)
CUDA_CU_OBJ :=
CUDA_CU_OBJ_AVX2 :=
ifeq ($(YLLM_CUDA),1)
  ifneq ($(YLLM_CUDA_HOST),1)
    CUDA_CU_SRC := inference/cuda/cuda_kernels.cu
  endif
endif

# ---- OS 检测 (Windows: MSYS2/MinGW 的 uname 会带 MINGW/MSYS, 也归为 Windows) ----
ifneq ($(OS),Windows_NT)
UNAME_S := $(shell uname -s 2>/dev/null)
else
UNAME_S := Windows
endif
ifneq ($(findstring MINGW,$(UNAME_S)),)
UNAME_S := Windows
endif
ifneq ($(findstring MSYS,$(UNAME_S)),)
UNAME_S := Windows
endif

# ---- 架构检测 (AVX2 仅 x86_64) ----
ifneq ($(OS),Windows_NT)
ARCH := $(shell uname -m 2>/dev/null)
else
ARCH := x86_64
endif

# ---- 平台相关: 可执行后缀 / feature macros / 链接库 ----
ifeq ($(UNAME_S),Windows)
    EXE      := .exe
    PLATDEF  :=
    PLATLIBS := -lm -lws2_32
    OMPFLAG  := -fopenmp
else ifeq ($(UNAME_S),Darwin)
    EXE      :=
    PLATDEF  := -D_DARWIN_C_SOURCE
    PLATLIBS := -lm -pthread
    OMPFLAG  :=
else
    EXE      :=
    PLATDEF  := -D_DEFAULT_SOURCE
    PLATLIBS := -lm -pthread
    # Linux 默认启用 OpenMP 多核加速(可用 make OMPFLAG= 关闭)
    OMPFLAG  := -fopenmp
endif
LIBS += $(PLATLIBS)

# ---- 标量版本(默认) ----
# 批量 prefill(编译期开关): 默认开启, 用 make BATCH_PREFILL=0 关闭
BATCH_PREFILL ?= 1
# 会话数据包调试: make YLLM_SESS_DEBUG=1 开启(0 默认关闭)
YLLM_SESS_DEBUG ?= 0
CFLAGS_BASE   := -O2 -std=c99 -Wall -Wextra -DYLLM_BATCH_PREFILL=$(BATCH_PREFILL) -DYLLM_SESS_DEBUG=$(YLLM_SESS_DEBUG) $(PLATDEF) $(OMPFLAG)
ifeq ($(YLLM_CUDA),1)
  ifeq ($(YLLM_CUDA_HOST),1)
    CFLAGS_BASE += -DYLLM_CUDA=1 -DYLLM_CUDA_HOST=1
  else
    CFLAGS_BASE += -DYLLM_CUDA=1
    LIBS += -lcudart -lcublas
  endif
endif
ifeq ($(YLLM_VULKAN),1)
  CFLAGS_BASE += -DYLLM_VULKAN=1
  ifeq ($(YLLM_VULKAN_HOST),1)
    CFLAGS_BASE += -DYLLM_VULKAN_HOST=1
  endif
  # 头文件: VULKAN_SDK 或系统路径; 运行时动态加载 loader(无需链 vulkan-1.lib)
  ifneq ($(VULKAN_SDK),)
    CFLAGS_BASE += -I"$(VULKAN_SDK)/Include"
  endif
  ifneq ($(UNAME_S),Windows)
    LIBS += -ldl
  endif
endif
OBJDIR        := build
BIN           := build/yllm$(EXE)
OBJ           := $(SRC:inference/%.c=$(OBJDIR)/%.o) $(OBJDIR)/main.o $(OBJDIR)/rank.o $(OBJDIR)/server.o $(OBJDIR)/router.o $(OBJDIR)/supervisor.o $(OBJDIR)/hub.o $(OBJDIR)/router_http.o $(OBJDIR)/status.o $(OBJDIR)/ctl.o $(OBJDIR)/sync.o
ifneq ($(CUDA_CU_SRC),)
  CUDA_CU_OBJ := $(CUDA_CU_SRC:inference/%.cu=$(OBJDIR)/%.o)
  OBJ += $(CUDA_CU_OBJ)
endif
OBJ           := $(sort $(OBJ))

# ---- AVX2 版本(仅 x86_64; 其余架构退化为标量, 保持 target 可用) ----
CFLAGS_AVX2   := $(CFLAGS_BASE)
LDFLAGS_AVX2  :=
OBJDIR_AVX2   := build/avx2
BIN_AVX2      := build/avx2/yllm$(EXE)
OBJ_AVX2      := $(SRC:inference/%.c=$(OBJDIR_AVX2)/%.o) $(OBJDIR_AVX2)/main.o $(OBJDIR_AVX2)/rank.o $(OBJDIR_AVX2)/server.o $(OBJDIR_AVX2)/router.o $(OBJDIR_AVX2)/supervisor.o $(OBJDIR_AVX2)/hub.o $(OBJDIR_AVX2)/router_http.o $(OBJDIR_AVX2)/status.o $(OBJDIR_AVX2)/ctl.o $(OBJDIR_AVX2)/sync.o
ifneq ($(CUDA_CU_SRC),)
  CUDA_CU_OBJ_AVX2 := $(CUDA_CU_SRC:inference/%.cu=$(OBJDIR_AVX2)/%.o)
  OBJ_AVX2 += $(CUDA_CU_OBJ_AVX2)
endif
OBJ_AVX2      := $(sort $(OBJ_AVX2))
ifeq ($(ARCH),x86_64)
CFLAGS_AVX2   := $(CFLAGS_BASE) -mavx2 -mfma -mf16c
endif

# CUDA 独立产物(与 build/avx2 隔离, 避免 CPU/CUDA 目标混用同一 .o)
OBJDIR_CUDA := build/avx2-cuda
BIN_CUDA    := $(OBJDIR_CUDA)/yllm$(EXE)
OBJDIR_VULKAN := build/avx2-vulkan
BIN_VULKAN    := $(OBJDIR_VULKAN)/yllm$(EXE)

all: $(BIN)

avx2: $(BIN_AVX2)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS_BASE) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

$(BIN_AVX2): $(OBJ_AVX2)
	$(CC) $(CFLAGS_AVX2) -o $@ $(OBJ_AVX2) $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR)/%.o: inference/%.c $(HDR_PUBLIC) | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -c -o $@ $<

ifneq ($(CUDA_CU_SRC),)
$(OBJDIR)/%.o: inference/%.cu inference/cuda/cuda_kernels.h | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O2 -std=c++14 -Xcompiler -fPIC $(INFER_INC) -c -o $@ $<
endif

$(OBJDIR)/main.o: main.c $(HDR_PUBLIC) serve/rank.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/rank.o: serve/rank.c serve/protocol.h serve/rank.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/server.o: serve/server.c serve/protocol.h serve/server.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/router.o: serve/router.c serve/protocol.h serve/router.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/supervisor.o: serve/supervisor.c serve/protocol.h serve/supervisor.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/hub.o: serve/hub.c serve/hub.h serve/supervisor.h serve/router.h serve/server.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/status.o: serve/status.c serve/status.h serve/protocol.h serve/frame.h serve/sock.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/ctl.o: serve/ctl.c serve/ctl.h serve/protocol.h serve/frame.h serve/sock.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<
$(OBJDIR)/sync.o: serve/sync.c serve/sync.h serve/frame.h serve/sock.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR)/router_http.o: serve/router_http.c serve/router_http.h serve/router.h serve/json.h serve/http.h serve/sock.h $(HDR_PUBLIC) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/%.o: inference/%.c $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -c -o $@ $<

ifneq ($(CUDA_CU_SRC),)
$(OBJDIR_AVX2)/%.o: inference/%.cu inference/cuda/cuda_kernels.h | $(OBJDIR_AVX2)
	@mkdir -p $(dir $@)
	$(NVCC) -O2 -std=c++14 -Xcompiler "-fPIC $(filter -mavx2 -mfma -mf16c,$(CFLAGS_AVX2))" $(INFER_INC) -c -o $@ $<
endif

$(OBJDIR_AVX2)/main.o: main.c $(HDR_PUBLIC) serve/rank.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/rank.o: serve/rank.c serve/protocol.h serve/rank.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/server.o: serve/server.c serve/protocol.h serve/server.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/router.o: serve/router.c serve/protocol.h serve/router.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/supervisor.o: serve/supervisor.c serve/protocol.h serve/supervisor.h serve/sock.h serve/frame.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/hub.o: serve/hub.c serve/hub.h serve/supervisor.h serve/router.h serve/server.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/status.o: serve/status.c serve/status.h serve/protocol.h serve/frame.h serve/sock.h serve/node.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/ctl.o: serve/ctl.c serve/ctl.h serve/protocol.h serve/frame.h serve/sock.h serve/config.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<
$(OBJDIR_AVX2)/sync.o: serve/sync.c serve/sync.h serve/frame.h serve/sock.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/router_http.o: serve/router_http.c serve/router_http.h serve/router.h serve/json.h serve/http.h serve/sock.h $(HDR_PUBLIC) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Iserve -c -o $@ $<

# ---- 测试(标量 + AVX2 两套) ----
TEST_SRC := tests/test_matvec.c tests/test_tokenizer.c tests/test_llf.c tests/test_engine.c tests/test_prefill_batch.c tests/test_cache.c

$(OBJDIR)/test_matvec.exe: tests/test_matvec.c tests/ref_data.h inference/core/platform.c inference/core/llf.c inference/core/matvec.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -Itests -o $@ $< inference/core/platform.c inference/core/llf.c inference/core/matvec.c $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_tokenizer.exe: tests/test_tokenizer.c inference/core/platform.c inference/core/llf.c inference/core/tokenizer.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_matvec.exe: tests/test_matvec.c tests/ref_data.h inference/core/platform.c inference/core/llf.c inference/core/matvec.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -Itests -o $@ $< inference/core/platform.c inference/core/llf.c inference/core/matvec.c $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_tokenizer.exe: tests/test_tokenizer.c inference/core/platform.c inference/core/llf.c inference/core/tokenizer.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR)/test_prefill_batch.exe: tests/test_prefill_batch.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_cache.exe: tests/test_cache.c inference/core/cache.c inference/core/platform.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_prefill_batch.exe: tests/test_prefill_batch.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_cache.exe: tests/test_cache.c inference/core/cache.c inference/core/platform.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

TEST_BIN     := $(TEST_SRC:tests/%.c=$(OBJDIR)/%.exe)
TEST_BIN_AVX := $(TEST_SRC:tests/%.c=$(OBJDIR_AVX2)/%.exe)

# 默认 test 用 AVX2 版本(生产构建; golden 与 SIMD/fma 结果一致)。
# 标量版可用 make test-base 运行(其数值因无 fma 与 golden 存在预期差异)。
test: $(TEST_BIN_AVX)
	@echo "=== test_matvec (avx2) ==="
	./$(OBJDIR_AVX2)/test_matvec.exe
	@echo "=== test_tokenizer (avx2) ==="
	./$(OBJDIR_AVX2)/test_tokenizer.exe
	@echo "=== test_llf (avx2) ==="
	./$(OBJDIR_AVX2)/test_llf.exe
	@echo "=== test_engine (avx2) ==="
	./$(OBJDIR_AVX2)/test_engine.exe
	@echo "=== test_prefill_batch (avx2) ==="
	./$(OBJDIR_AVX2)/test_prefill_batch.exe
	@echo "=== test_cache (avx2) ==="
	./$(OBJDIR_AVX2)/test_cache.exe

test-avx2: test

test-base: $(TEST_BIN)
	@echo "=== test_matvec (scalar) ==="
	./$(OBJDIR)/test_matvec.exe
	@echo "=== test_tokenizer (scalar) ==="
	./$(OBJDIR)/test_tokenizer.exe
	@echo "=== test_llf (scalar) ==="
	./$(OBJDIR)/test_llf.exe
	@echo "=== test_engine (scalar, golden 差异预期) ==="
	./$(OBJDIR)/test_engine.exe
	@echo "=== test_prefill_batch (scalar) ==="
	./$(OBJDIR)/test_prefill_batch.exe
	@echo "=== test_cache (scalar) ==="
	./$(OBJDIR)/test_cache.exe

# ---- PP 会话缓存集成测试(2 段) ----
# tools/test_pp_sess.sh: 临时 ranks:2 → 启动 → tests/test_pp_sess.py → 恢复 serve.yaml
test-pp-sess: $(BIN_AVX2) $(MODEL_LLF)
	@bash tools/test_pp_sess.sh $(BIN_AVX2)

# ---- 集成: 转换模型 + 运行 chat/gen ----
MODEL_GGUF  ?= models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
MODEL_LLF   ?= models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf
MODEL_VOCAB ?= models/tinyllama.vocab.txt
CHAT_PROMPT ?= "Once upon a time"
CHAT_TOKENS ?= 30

# 推理线程数(OpenMP)。默认使用本机全部核心, 可用 NTHREADS=N 覆盖。
NTHREADS ?= $(shell nproc 2>/dev/null || echo 4)
RUN = OMP_NUM_THREADS=$(NTHREADS) $(BIN)
RUN_AVX2 = OMP_NUM_THREADS=$(NTHREADS) $(BIN_AVX2)

# llf 转换规则: 仅当 gguf 比 llf 新才转换(bin 用 order-only 依赖, 不触发重转)
$(MODEL_LLF): $(MODEL_GGUF) | $(BIN)
	@mkdir -p $(dir $@)
	$(BIN) convert --gguf $(MODEL_GGUF) --out $(MODEL_LLF) --vocab $(MODEL_VOCAB) --seq 2048

chat: $(BIN) $(MODEL_LLF)
	$(RUN) chat --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

gen: $(BIN) $(MODEL_LLF)
	$(RUN) gen --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-avx2: $(BIN_AVX2) $(MODEL_LLF)
	$(RUN_AVX2) chat --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

gen-avx2: $(BIN_AVX2) $(MODEL_LLF)
	$(RUN_AVX2) gen --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

# ---- CUDA: 独立目录构建 + 设备冒烟 ----
# 有 nvcc → FP16 权 + cublas decode; 无 nvcc → host-shim
cuda:
	$(MAKE) avx2 YLLM_CUDA=1 OBJDIR_AVX2=$(OBJDIR_CUDA) BIN_AVX2=$(BIN_CUDA)

RUN_CUDA = OMP_NUM_THREADS=$(NTHREADS) $(BIN_CUDA)

gen-cuda: cuda $(MODEL_LLF)
	$(RUN_CUDA) gen --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device cuda --gpu $(GPU) --gpu-weights $(GPU_WEIGHTS)

chat-cuda: cuda $(MODEL_LLF)
	$(RUN_CUDA) chat --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device cuda --gpu $(GPU) --gpu-weights $(GPU_WEIGHTS)

# ---- Vulkan: 独立目录; 默认尝试原生 VkDevice, YLLM_VULKAN_HOST=1 强制 shim ----
vulkan:
	$(MAKE) avx2 YLLM_VULKAN=1 OBJDIR_AVX2=$(OBJDIR_VULKAN) BIN_AVX2=$(BIN_VULKAN)

RUN_VULKAN = OMP_NUM_THREADS=$(NTHREADS) $(BIN_VULKAN)

gen-vulkan: vulkan $(MODEL_LLF)
	$(RUN_VULKAN) gen --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

chat-vulkan: vulkan $(MODEL_LLF)
	$(RUN_VULKAN) chat --model $(MODEL_LLF) --vocab $(MODEL_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

# 与 chat-avx2 / gen-avx2 对称的命名(产物仍为 build/avx2-vulkan)
chat-avx2-vulkan: chat-vulkan
gen-avx2-vulkan: gen-vulkan

# ---- 指定模型的 chat 快捷目标(qwen2.5 / qwen3) ----
Q25_GGUF  ?= models/qwen2.5-1.5b-instruct-q4_k_m.gguf
Q25_LLF   ?= models/qwen2.5-1.5b.llf
Q25_VOCAB ?= models/qwen2.5.vocab.txt
Q25_7B_GGUF  ?= models/qwen2.5-7b-instruct-q4_k_m.gguf
Q25_7B_LLF   ?= models/qwen2.5-7b.llf
Q25_7B_VOCAB ?= models/qwen2.5-7b.vocab.txt
Q3_8B_GGUF   ?= models/Qwen3-8B-Q4_K_M.gguf
Q3_8B_LLF    ?= models/qwen3-8b.llf
Q3_8B_VOCAB  ?= models/qwen3-8b.vocab.txt

$(Q25_LLF): $(Q25_GGUF) | $(BIN)
	@mkdir -p $(dir $@)
	$(BIN) convert --gguf $(Q25_GGUF) --out $(Q25_LLF) --vocab $(Q25_VOCAB) --seq 2048

$(Q25_7B_LLF): $(Q25_7B_GGUF) | $(BIN)
	@mkdir -p $(dir $@)
	$(BIN) convert --gguf $(Q25_7B_GGUF) --out $(Q25_7B_LLF) --vocab $(Q25_7B_VOCAB) --seq 4096

$(Q3_8B_LLF): $(Q3_8B_GGUF) | $(BIN)
	@mkdir -p $(dir $@)
	$(BIN) convert --gguf $(Q3_8B_GGUF) --out $(Q3_8B_LLF) --vocab $(Q3_8B_VOCAB) --seq 2048

chat-qwen2.5-1.5b: $(BIN) $(Q25_LLF)
	$(RUN) chat --model $(Q25_LLF) --vocab $(Q25_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen2.5-1.5b-avx2: $(BIN_AVX2) $(Q25_LLF)
	$(RUN_AVX2) chat --model $(Q25_LLF) --vocab $(Q25_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen2.5-1.5b-avx2-vulkan: vulkan $(Q25_LLF)
	$(RUN_VULKAN) chat --model $(Q25_LLF) --vocab $(Q25_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

chat-qwen2.5-7b: $(BIN) $(Q25_7B_LLF)
	$(RUN) chat --model $(Q25_7B_LLF) --vocab $(Q25_7B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen2.5-7b-avx2: $(BIN_AVX2) $(Q25_7B_LLF)
	$(RUN_AVX2) chat --model $(Q25_7B_LLF) --vocab $(Q25_7B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen2.5-7b-avx2-vulkan: vulkan $(Q25_7B_LLF)
	$(RUN_VULKAN) chat --model $(Q25_7B_LLF) --vocab $(Q25_7B_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

chat-qwen3-8b: $(BIN) $(Q3_8B_LLF)
	$(RUN) chat --model $(Q3_8B_LLF) --vocab $(Q3_8B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen3-8b-avx2: $(BIN_AVX2) $(Q3_8B_LLF)
	$(RUN_AVX2) chat --model $(Q3_8B_LLF) --vocab $(Q3_8B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen3-8b-avx2-vulkan: vulkan $(Q3_8B_LLF)
	$(RUN_VULKAN) chat --model $(Q3_8B_LLF) --vocab $(Q3_8B_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

# ---- qwen3.8-27b(Gated Attention + GDN 混合架构) ----
Q3_27B_GGUF  ?= models/Qwen3.8-27B-Q4_K_M.gguf
Q3_27B_LLF   ?= models/qwen3.8-27b.llf
Q3_27B_VOCAB ?= models/qwen3.vocab.txt

$(Q3_27B_LLF): $(Q3_27B_GGUF) | $(BIN)
	@mkdir -p $(dir $@)
	$(BIN) convert --gguf $(Q3_27B_GGUF) --out $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --seq 2048

gen-qwen3.8-27b: $(BIN) $(Q3_27B_LLF)
	$(RUN) gen --model $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

gen-qwen3.8-27b-avx2: $(BIN_AVX2) $(Q3_27B_LLF)
	$(RUN_AVX2) gen --model $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

gen-qwen3.8-27b-avx2-vulkan: vulkan $(Q3_27B_LLF)
	$(RUN_VULKAN) gen --model $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

chat-qwen3.8-27b: $(BIN) $(Q3_27B_LLF)
	$(RUN) chat --model $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen3.8-27b-avx2: $(BIN_AVX2) $(Q3_27B_LLF)
	$(RUN_AVX2) chat --model $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --prompt $(CHAT_PROMPT) --tokens $(CHAT_TOKENS)

chat-qwen3.8-27b-avx2-vulkan: vulkan $(Q3_27B_LLF)
	$(RUN_VULKAN) chat --model $(Q3_27B_LLF) --vocab $(Q3_27B_VOCAB) --prompt $(CHAT_PROMPT) \
		--tokens $(CHAT_TOKENS) --device vulkan --gpu $(GPU)

# ---- 指定模型的 serve 快捷目标(serve.yaml 多模型, 用 --model <名字> 只拉起对应模型) ----
#   make server-qwen2.5-7b   # 只起 qwen2.5(本机 rank)
#   make server-qwen38       # 只起 qwen38(远程 rank, 需先远程手工起)
#   make server-tinyllama    # 只起 tinyllama(本机 rank)
server-qwen2.5-7b: $(BIN_AVX2) $(Q25_7B_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup env OMP_NUM_THREADS=$(NTHREADS) $(BIN_AVX2) hub --config serve.yaml --model qwen2.5 > $(SERVE_LOGDIR)/hub.out 2>&1 &
	@echo "hub started (serve.yaml, model=qwen2.5); 用 make infer-qwen2.5-7b 发请求 (HTTP 127.0.0.1:8000)"

server-qwen38: $(BIN_AVX2) $(Q3_27B_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup env OMP_NUM_THREADS=$(NTHREADS) $(BIN_AVX2) hub --config serve.yaml --model qwen38 > $(SERVE_LOGDIR)/hub.out 2>&1 &
	@echo "hub started (serve.yaml, model=qwen38); 用 make infer-qwen38 发请求 (HTTP 127.0.0.1:8000)"

server-tinyllama: $(BIN_AVX2) $(MODEL_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup env OMP_NUM_THREADS=$(NTHREADS) $(BIN_AVX2) hub --config serve.yaml --model tinyllama > $(SERVE_LOGDIR)/hub.out 2>&1 &
	@echo "hub started (serve.yaml, model=tinyllama); 用 make infer-tinyllama 发请求 (HTTP 127.0.0.1:8000)"

# 对应模型的 infer 快捷目标(模型名需匹配 serve.yaml 的 model-name)
infer-qwen2.5-7b: $(BIN)
	$(BIN) router --config serve.yaml --send "qwen2.5 $(CHAT_TOKENS) $(SERVE_PROMPT)"

infer-qwen38: $(BIN)
	$(BIN) router --config serve.yaml --send "qwen38 $(CHAT_TOKENS) $(SERVE_PROMPT)"

infer-tinyllama: $(BIN)
	$(BIN) router --config serve.yaml --send "tinyllama $(CHAT_TOKENS) $(SERVE_PROMPT)"

# ---- 常驻推理服务(serve 层) ----
# 统一配置: serve.yaml(所有角色共用)
# 用法:
#   make serve            # supervisor --config serve.yaml 一键拉起 rank+server
#   make hub              # 合并模式: supervisor+router+server 同进程(推荐)
#   make serve-avx2       # 同上, avx2 版本
#   make serve-stop       # 停掉 serve 相关进程
#   make infer            # 客户端经 router 发请求
# 分开模式(各自独立进程, 同一份 config):
#   make supervisor / make router / make server / make rank

SERVE_CONFIG ?= serve.yaml
SERVE_LOGDIR ?= logs

serve: $(BIN) $(MODEL_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@echo "== supervisor --config $(SERVE_CONFIG) =="
	@nohup $(BIN) supervisor --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/serve.out 2>&1 &
	@echo "serve started (supervisor auto-spawns rank+server)"

serve-avx2: $(BIN_AVX2) $(MODEL_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@echo "== supervisor --config $(SERVE_CONFIG) (avx2) =="
	@nohup $(BIN_AVX2) supervisor --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/serve.out 2>&1 &
	@echo "serve started (avx2)"

# 合并模式: supervisor+router+server 同进程, 自动拉起 rank; 之后 make infer 即可
hub: $(BIN_AVX2) $(MODEL_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@echo "== hub --config $(SERVE_CONFIG) (avx2, 三合一) =="
	@nohup $(BIN_AVX2) hub --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/hub.out 2>&1 &
	@echo "hub started (--config $(SERVE_CONFIG)); 用 make infer 发请求"

# 分开模式(独立进程, 同一份 config)
supervisor: $(BIN)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup $(BIN) supervisor --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/supervisor.out 2>&1 &
	@echo "supervisor started (--config $(SERVE_CONFIG))"

router: $(BIN)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup $(BIN) router --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/router.out 2>&1 &
	@echo "router started (--config $(SERVE_CONFIG))"

server: $(BIN)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup $(BIN) server --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/server.out 2>&1 &
	@echo "server started (--config $(SERVE_CONFIG))"

rank: $(BIN) $(MODEL_LLF)
	@mkdir -p $(SERVE_LOGDIR)
	@nohup $(BIN) rank --config $(SERVE_CONFIG) > $(SERVE_LOGDIR)/rank.out 2>&1 &
	@echo "rank started (--config $(SERVE_CONFIG))"

# 客户端: 经 router 发请求(prompt 不含引号)
# SERVER_MODEL 需匹配 serve.yaml 的 model-name
SERVER_MODEL ?= tinyllama
SERVE_PROMPT ?= Once upon a time
# 支持 `make infer hello world` 把多余参数当 prompt
ifneq ($(filter-out infer,$(MAKECMDGOALS)),)
SERVE_PROMPT := $(filter-out infer,$(MAKECMDGOALS))
endif
status: $(BIN)
	$(BIN) ctl --config $(SERVE_CONFIG) status

# 管理命令: make ctl <target> <cmd> [need]  (目标用短别名, 避免与 make 目标名冲突)
#   r0=rank-0  s0=server-0  sv=supervisor  rt=router
#   例: make ctl status              # 完整状态查看
#       make ctl r0 PING / r0 STAT / r0 DRAIN / r0 QUIT
#       make ctl sv SCALE 2          # 扩一组 rank
#       make ctl sv QUERY_SERVERS
#       make ctl s0 DRAIN            # server 优雅下线
CTL_GOALS := $(filter-out ctl,$(MAKECMDGOALS))
CTL_T0 := $(word 1,$(CTL_GOALS))
CTL_TGT := $(if $(filter r0,$(CTL_T0)),rank-0,$(if $(filter s0,$(CTL_T0)),server-0,$(if $(filter sv,$(CTL_T0)),supervisor,$(if $(filter rt,$(CTL_T0)),router,$(CTL_T0)))))
ifeq ($(words $(CTL_GOALS)),1)
CTL_ARGS := --cmd $(CTL_GOALS)
else ifeq ($(words $(CTL_GOALS)),3)
CTL_ARGS := --target $(CTL_TGT) --cmd $(word 2,$(CTL_GOALS)) --need-groups $(word 3,$(CTL_GOALS))
else
CTL_ARGS := --target $(CTL_TGT) --cmd $(word 2,$(CTL_GOALS))
endif
ctl: $(BIN)
	$(BIN) ctl --config $(SERVE_CONFIG) $(CTL_ARGS)

# 文件分发: 接收端 make sync-serve PORT=9600; 发送端 make sync-push FILE=... TO=host:9600 DEST=...
sync-serve: $(BIN)
	$(BIN) sync --serve --port $(SYNC_PORT) --dir $(SYNC_DIR)

sync-push: $(BIN)
	$(BIN) sync --push $(SYNC_FILE) --to $(SYNC_TO) --dest $(SYNC_DEST)

infer: $(BIN)
	$(BIN) router --config $(SERVE_CONFIG) --send "$(SERVER_MODEL) $(CHAT_TOKENS) $(SERVE_PROMPT)"

# 捕获 `make infer <prompt...>` 的多余目标词, 避免 "没有规则" 报错
%:
	@:

serve-stop:
ifeq ($(UNAME_S),Windows)
	@taskkill //F //IM yllm.exe 2>/dev/null; echo "serve processes stopped"
else
	@-pkill -f "yllm hub" 2>/dev/null; \
	pkill -f "yllm rank" 2>/dev/null; \
	pkill -f "yllm supervisor" 2>/dev/null; \
	pkill -f "yllm router" 2>/dev/null; \
	pkill -f "yllm server" 2>/dev/null; \
	echo "serve processes stopped"
endif

# ---- 模型文件 dump 工具(LLF / GGUF / Safetensors) ----
DUMP_BIN := $(OBJDIR)/llfdump

$(DUMP_BIN): tools/dump.c inference/core/llf.c inference/core/platform.c inference/include/llf.h inference/include/yllm.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $^ $(LDFLAGS) $(LIBS)

dump: $(DUMP_BIN)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR_AVX2):
	mkdir -p $(OBJDIR_AVX2)

clean:
	rm -rf build

# ---- 分布式远程测试(局域网多节点, worker 常驻) ----
# 用法:
#   make dist-serve  DIST_SERVE_PORT=9360                        # 管理节点起文件服务(大模型/日志中转)
#   make dist-deploy NODES="192.168.1.10 192.168.1.11" USER=root  # 构建+SSH分发(不含大模型)+各节点起 worker
#   make dist        NODES="192.168.1.10 192.168.1.11"            # 各节点 sync 拉模型(vocab 已随 rsync)+ 下发 run
#   make dist-stop   NODES="192.168.1.10 192.168.1.11"            # 终止已下发进程
# 变量: NODES(节点IP列表, 按 rank 顺序) USER(SSH 用户) DIST_PORT(worker 控制端口)
#       DIST_DIR(远端工作目录) DIST_RANK_N(rank 数, 默认=节点数) DIST_PORT_BASE(推理数据端口基数)
#       DIST_TOKENS DIST_PROMPT DIST_SEED
#       DIST_SERVE_HOST DIST_SERVE_PORT(管理节点文件服务地址; 各节点 sync 用)
# 文件分发: 模型 models/*.llf 不随 rsync 上传, 由各节点经 sync 从管理节点文件服务拉取(私有 TCP 帧)。
DIST_WORKER := $(OBJDIR)/dist-worker
DIST_WORKER_AVX2 := $(OBJDIR_AVX2)/dist-worker$(EXE)
NODES      ?= 127.0.0.1 127.0.0.1
USER       ?= $(USERNAME)
DIST_PORT  ?= 9100
DIST_DIR   ?= /tmp/yllm
DIST_PORT_BASE ?= 8900
DIST_TOKENS ?= 8
DIST_PROMPT ?= "Once upon a time"
DIST_SEED  ?= 42
DIST_RANK_N ?= $(words $(NODES))
DIST_SERVE_HOST ?= 127.0.0.1
DIST_SERVE_PORT ?= 9360
# 远端模型/vocab 路径(相对 DIST_DIR); 大模型经 sync 拉到同路径
DIST_MODEL  ?= models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf
DIST_VOCAB  ?= models/tinyllama.vocab.txt

$(DIST_WORKER): serve/dist_worker.c serve/protocol.h inference/core/log.c inference/include/log.h inference/core/platform.c inference/include/yllm.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) $(INFER_INC) -o $@ $< inference/core/log.c inference/core/platform.c $(LDFLAGS) $(LIBS)

$(DIST_WORKER_AVX2): serve/dist_worker.c inference/core/log.c inference/include/log.h inference/core/platform.c inference/include/yllm.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) $(INFER_INC) -o $@ $< inference/core/log.c inference/core/platform.c $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

dist-worker: $(DIST_WORKER)
dist-worker-avx2: $(DIST_WORKER_AVX2)

# 管理节点起文件服务: 供各节点 sync 拉模型/日志
dist-serve: $(DIST_WORKER_AVX2)
	@set -e; echo "== serve on $(DIST_SERVE_PORT) =="; \
	./$(DIST_WORKER_AVX2) --serve --root $(abspath .) --port $(DIST_SERVE_PORT) > logs/serve.log 2>&1 &

# 上传源码(排除大模型/日志/build), 远端编译并启动 worker
dist-deploy: $(DIST_WORKER) $(BIN) $(BIN_AVX2)
	@set -e; for h in $(NODES); do \
	  echo "== deploy to $$h =="; \
	  ssh $(USER)@$$h "mkdir -p $(DIST_DIR)" || exit 1; \
	  rsync -a --exclude build --exclude logs --exclude 'models/*.llf' ./ $(USER)@$$h:$(DIST_DIR)/ || exit 1; \
	  ssh $(USER)@$$h "cd $(DIST_DIR) && make avx2 dist-worker" || exit 1; \
	  ssh $(USER)@$$h "cd $(DIST_DIR) && nohup ./build/avx2/dist-worker --port $(DIST_PORT) --bin ./build/avx2/yllm --logdir logs > logs/worker.log 2>&1 &" || exit 1; \
	  ssh $(USER)@$$h "cd $(DIST_DIR) && ./build/avx2/dist-worker --host 127.0.0.1 --port $(DIST_PORT) --send ping" || exit 1; \
	done; echo "dist-deploy OK"

# 一键: 各节点 sync 拉模型(已有且 size+mtime 一致则跳过)后, 向各节点 worker 下发 run(rank 按 NODES 顺序)
dist: dist-deploy
	@set -e; r=0; for h in $(NODES); do \
	  echo "== sync + rank $$r @ $$h =="; \
	  ssh $(USER)@$$h "cd $(DIST_DIR) && ./build/avx2/dist-worker --host 127.0.0.1 --port $(DIST_PORT) --send \"sync $(DIST_SERVE_HOST) $(DIST_SERVE_PORT) $(DIST_MODEL) $(DIST_MODEL)\"" || exit 1; \
	  ssh $(USER)@$$h "cd $(DIST_DIR) && ./build/avx2/dist-worker --host 127.0.0.1 --port $(DIST_PORT) --send \"run $$r $(DIST_RANK_N) $(DIST_PORT_BASE) $(DIST_TOKENS) $(DIST_MODEL) $(DIST_VOCAB) $(DIST_PROMPT)\"" || exit 1; \
	  r=$$((r+1)); \
	done; echo "dist dispatched (see rank*.log on each node)"

# 终止各节点已下发的推理进程
dist-stop:
	@for h in $(NODES); do \
	  echo "== stop $$h =="; \
	  ssh $(USER)@$$h "cd $(DIST_DIR) && ./build/avx2/dist-worker --host 127.0.0.1 --port $(DIST_PORT) --send stop" || echo "stop $$h failed"; \
	done

.PHONY: all avx2 cuda vulkan clean test test-base test-avx2 test-pp-sess chat gen chat-avx2 gen-avx2 gen-cuda chat-cuda gen-vulkan chat-vulkan chat-avx2-vulkan gen-avx2-vulkan chat-qwen2.5-1.5b chat-qwen2.5-1.5b-avx2 chat-qwen2.5-1.5b-avx2-vulkan chat-qwen2.5-7b chat-qwen2.5-7b-avx2 chat-qwen2.5-7b-avx2-vulkan chat-qwen3-8b-avx2-vulkan chat-qwen3.8-27b-avx2-vulkan gen-qwen3.8-27b-avx2-vulkan dump dist dist-deploy dist-serve dist-stop serve serve-avx2 hub supervisor router server rank infer status ctl sync-serve sync-push serve-stop server-qwen2.5-7b server-qwen38 server-tinyllama infer-qwen2.5-7b infer-qwen38 infer-tinyllama