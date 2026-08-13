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

SRC      := inference/platform.c inference/log.c inference/llf.c inference/convert.c inference/convert_safetensors.c inference/convert_gguf.c inference/tokenizer.c inference/matvec.c inference/engine.c inference/dist.c
TEST_ENGINE_CORE := inference/platform.c inference/log.c inference/llf.c inference/convert.c inference/convert_safetensors.c inference/convert_gguf.c inference/tokenizer.c inference/matvec.c inference/engine.c

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
OBJ           := $(SRC:inference/%.c=$(OBJDIR)/%.o) $(OBJDIR)/main.o $(OBJDIR)/rank.o
OBJ           := $(sort $(OBJ))

# ---- AVX2 版本(仅 x86_64; 其余架构退化为标量, 保持 target 可用) ----
CFLAGS_AVX2   := $(CFLAGS_BASE)
LDFLAGS_AVX2  :=
OBJDIR_AVX2   := build/avx2
BIN_AVX2      := build/avx2/yllm$(EXE)
OBJ_AVX2      := $(SRC:inference/%.c=$(OBJDIR_AVX2)/%.o) $(OBJDIR_AVX2)/main.o $(OBJDIR_AVX2)/rank.o
OBJ_AVX2      := $(sort $(OBJ_AVX2))
ifeq ($(ARCH),x86_64)
CFLAGS_AVX2   := $(CFLAGS_BASE) -mavx2 -mfma
endif

all: $(BIN)

avx2: $(BIN_AVX2)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS_BASE) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

$(BIN_AVX2): $(OBJ_AVX2)
	$(CC) $(CFLAGS_AVX2) -o $@ $(OBJ_AVX2) $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR)/%.o: inference/%.c inference/yllm.h inference/llf.h inference/convert.h inference/matvec.h inference/dist.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -c -o $@ $<

$(OBJDIR)/main.o: main.c inference/yllm.h inference/dist.h inference/log.h serve/rank.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -Iserve -c -o $@ $<

$(OBJDIR)/rank.o: serve/rank.c serve/protocol.h serve/rank.h inference/yllm.h inference/log.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/%.o: inference/%.c inference/yllm.h inference/llf.h inference/convert.h inference/matvec.h inference/dist.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -c -o $@ $<

$(OBJDIR_AVX2)/main.o: main.c inference/yllm.h inference/dist.h inference/log.h serve/rank.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -Iserve -c -o $@ $<

$(OBJDIR_AVX2)/rank.o: serve/rank.c serve/protocol.h serve/rank.h inference/yllm.h inference/log.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -Iserve -c -o $@ $<

# ---- 测试(标量 + AVX2 两套) ----
TEST_SRC := tests/test_matvec.c tests/test_tokenizer.c tests/test_llf.c tests/test_engine.c

$(OBJDIR)/test_matvec.exe: tests/test_matvec.c tests/ref_data.h inference/platform.c inference/llf.c inference/matvec.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -Itests -o $@ $< inference/platform.c inference/llf.c inference/matvec.c $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_tokenizer.exe: tests/test_tokenizer.c inference/platform.c inference/llf.c inference/tokenizer.c | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_matvec.exe: tests/test_matvec.c tests/ref_data.h inference/platform.c inference/llf.c inference/matvec.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -Itests -o $@ $< inference/platform.c inference/llf.c inference/matvec.c $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_tokenizer.exe: tests/test_tokenizer.c inference/platform.c inference/llf.c inference/tokenizer.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

$(OBJDIR_AVX2)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -o $@ $^ $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

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

$(DUMP_BIN): tools/dump.c inference/llf.c inference/platform.c inference/llf.h inference/yllm.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -o $@ $^ $(LDFLAGS) $(LIBS)

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
# 文件分发: 模型 test/*.llf 不随 rsync 上传, 由各节点经 sync 从管理节点文件服务拉取(私有 TCP 帧)。
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
DIST_MODEL  ?= test/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf
DIST_VOCAB  ?= test/tinyllama.vocab.txt

$(DIST_WORKER): serve/dist_worker.c serve/protocol.h inference/log.c inference/log.h inference/platform.c inference/yllm.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -Iinference -o $@ $< inference/log.c inference/platform.c $(LDFLAGS) $(LIBS)

$(DIST_WORKER_AVX2): serve/dist_worker.c inference/log.c inference/log.h inference/platform.c inference/yllm.h | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Iinference -o $@ $< inference/log.c inference/platform.c $(LDFLAGS) $(LDFLAGS_AVX2) $(LIBS)

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
	  rsync -a --exclude build --exclude logs --exclude 'test/*.llf' ./ $(USER)@$$h:$(DIST_DIR)/ || exit 1; \
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

.PHONY: all avx2 clean test test-avx2 chat gen chat-avx2 gen-avx2 dump dist dist-deploy dist-serve dist-stop