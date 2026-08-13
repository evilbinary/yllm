# yllm 构建 (Linux / macOS / Windows/MinGW)
#
#   make            标量版本
#   make avx2       x86_64 上的 AVX2 版本(其他架构退化为标量)
#   make test       运行测试
#   make clean
#
# Windows 下请在 MSYS2 / MinGW 环境执行本 Makefile;
# 若使用 MSVC, 请改用 CMake:
#   cmake -S . -B build-msvc && cmake --build build-msvc --config Release

CC         ?= cc
LDFLAGS    ?=
LIBS       :=

SRC      := src/platform.c src/log.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c src/dist.c src/main.c
TEST_ENGINE_CORE := src/platform.c src/log.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c

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
CFLAGS_BASE   := -O2 -std=c99 -Wall -Wextra $(PLATDEF) $(OMPFLAG)
OBJDIR        := build
BIN           := build/yllm$(EXE)
OBJ           := $(SRC:src/%.c=$(OBJDIR)/%.o)

# ---- AVX2 版本(仅 x86_64; 其余架构退化为标量, 保持 target 可用) ----
CFLAGS_AVX2   := $(CFLAGS_BASE)
LDFLAGS_AVX2  :=
OBJDIR_AVX2   := build/avx2
BIN_AVX2      := build/avx2/yllm$(EXE)
OBJ_AVX2      := $(SRC:src/%.c=$(OBJDIR_AVX2)/%.o)
ifeq ($(ARCH),x86_64)
CFLAGS_AVX2   := $(CFLAGS_BASE) -mavx2 -mfma
endif

all: $(BIN)

avx2: $(BIN_AVX2)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS_BASE) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

$(BIN_AVX2): $(OBJ_AVX2)
	$(CC) $(CFLAGS_AVX2) -o $@ $(OBJ_AVX2) $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR)/%.o: src/%.c src/yllm.h src/llf.h src/convert.h src/matvec.h src/dist.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -c -o $@ $<

$(OBJDIR_AVX2)/%.o: src/%.c src/yllm.h src/llf.h src/convert.h src/matvec.h src/dist.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -c -o $@ $<

# ---- 测试(标量 + AVX2 两套) ----
TEST_SRC := tests/test_matvec.c tests/test_tokenizer.c tests/test_llf.c tests/test_engine.c

$(OBJDIR)/test_matvec.exe: tests/test_matvec.c tests/ref_data.h src/platform.c src/llf.c src/matvec.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Isrc -Itests -o $@ $< src/platform.c src/llf.c src/matvec.c $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_tokenizer.exe: tests/test_tokenizer.c src/platform.c src/llf.c src/tokenizer.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_matvec.exe: tests/test_matvec.c tests/ref_data.h src/platform.c src/llf.c src/matvec.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -Itests -o $@ $< src/platform.c src/llf.c src/matvec.c $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_tokenizer.exe: tests/test_tokenizer.c src/platform.c src/llf.c src/tokenizer.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

TEST_BIN     := $(TEST_SRC:tests/%.c=$(OBJDIR)/%.exe)
TEST_BIN_AVX := $(TEST_SRC:tests/%.c=$(OBJDIR_AVX2)/%.exe)

test: $(TEST_BIN)
	@echo "=== test_matvec ==="
	./$(OBJDIR)/test_matvec.exe
	@echo "=== test_tokenizer ==="
	./$(OBJDIR)/test_tokenizer.exe
	@echo "=== test_llf ==="
	./$(OBJDIR)/test_llf.exe
	@echo "=== test_engine ==="
	./$(OBJDIR)/test_engine.exe

test-avx2: $(TEST_BIN_AVX)
	@echo "=== test_matvec (avx2) ==="
	./$(OBJDIR_AVX2)/test_matvec.exe
	@echo "=== test_tokenizer (avx2) ==="
	./$(OBJDIR_AVX2)/test_tokenizer.exe
	@echo "=== test_llf (avx2) ==="
	./$(OBJDIR_AVX2)/test_llf.exe
	@echo "=== test_engine (avx2) ==="
	./$(OBJDIR_AVX2)/test_engine.exe

# ---- 集成: 转换模型 + 运行 chat/gen ----
MODEL_GGUF  ?= tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
MODEL_LLF   ?= test/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf
MODEL_VOCAB ?= test/tinyllama.vocab.txt
CHAT_PROMPT ?= "Once upon a time"
CHAT_TOKENS ?= 30

# 推理线程数(OpenMP)。默认使用本机全部核心, 可用 NTHREADS=N 覆盖。
NTHREADS ?= $(shell nproc 2>/dev/null || echo 4)
RUN = OMP_NUM_THREADS=$(NTHREADS) $(BIN)
RUN_AVX2 = OMP_NUM_THREADS=$(NTHREADS) $(BIN_AVX2)

$(MODEL_LLF): $(MODEL_GGUF) $(BIN)
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

# ---- 模型文件 dump 工具(LLF / GGUF / Safetensors) ----
DUMP_BIN := $(OBJDIR)/llfdump

$(DUMP_BIN): src/dump.c src/llf.c src/platform.c src/llf.h src/yllm.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

dump: $(DUMP_BIN)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR_AVX2):
	mkdir -p $(OBJDIR_AVX2)

clean:
	rm -rf build

.PHONY: all avx2 clean test test-avx2 chat gen chat-avx2 gen-avx2 dump