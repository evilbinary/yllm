CC         ?= cc
LDFLAGS    ?=
LIBS       :=

SRC      := src/platform.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c src/main.c
TEST_ENGINE_CORE := src/platform.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifneq ($(filter Linux Darwin,$(UNAME_S)),)
    LIBS += -lm -pthread
endif

# ---- 标量版本(默认,无 SIMD) ----
CFLAGS_BASE   := -O2 -std=c99 -Wall -Wextra
OBJDIR        := build
BIN           := build/yllm
OBJ           := $(SRC:src/%.c=$(OBJDIR)/%.o)

# ---- AVX2 版本(加 -mavx2 -mfma) ----
CFLAGS_AVX2   := $(CFLAGS_BASE) -mavx2 -mfma
OBJDIR_AVX2   := build/avx2
BIN_AVX2      := build/avx2/yllm
OBJ_AVX2      := $(SRC:src/%.c=$(OBJDIR_AVX2)/%.o)

all: $(BIN)

avx2: $(BIN_AVX2)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS_BASE) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

$(BIN_AVX2): $(OBJ_AVX2)
	$(CC) $(CFLAGS_AVX2) -o $@ $(OBJ_AVX2) $(LDFLAGS) $(LIBS)

$(OBJDIR)/%.o: src/%.c src/yllm.h src/llf.h src/convert.h src/matvec.h | $(OBJDIR)
	$(CC) $(CFLAGS_BASE) -c -o $@ $<

$(OBJDIR_AVX2)/%.o: src/%.c src/yllm.h src/llf.h src/convert.h src/matvec.h | $(OBJDIR_AVX2)
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
	$(CC) $(CFLAGS_AVX2) -Isrc -Itests -o $@ $< src/platform.c src/llf.c src/matvec.c $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_tokenizer.exe: tests/test_tokenizer.c src/platform.c src/llf.c src/tokenizer.c | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

$(OBJDIR_AVX2)/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | $(OBJDIR_AVX2)
	$(CC) $(CFLAGS_AVX2) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

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

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR_AVX2):
	mkdir -p $(OBJDIR_AVX2)

clean:
	rm -rf build

.PHONY: all avx2 clean test test-avx2
