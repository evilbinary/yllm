CC       ?= cc
CFLAGS   ?= -O2 -std=c99 -Wall -Wextra
LDFLAGS  ?=
LIBS     :=
BIN      := build/yllm
SRC      := src/platform.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c src/main.c
OBJ      := $(SRC:src/%.c=build/%.o)

# 测试:每个 test_*.c 独立可执行,链接除 main.c 外的全部源码
TEST_SRC := tests/test_matvec.c tests/test_tokenizer.c tests/test_llf.c tests/test_engine.c
TEST_BIN := $(TEST_SRC:tests/%.c=build/%.exe)
TEST_ENGINE_CORE := src/platform.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifneq ($(filter Linux Darwin,$(UNAME_S)),)
    LIBS += -lm -pthread
endif

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

build/%.o: src/%.c src/yllm.h src/llf.h src/convert.h src/matvec.h | build
	$(CC) $(CFLAGS) -c -o $@ $<

build/test_matvec.exe: tests/test_matvec.c src/platform.c src/llf.c src/matvec.c | build
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

build/test_tokenizer.exe: tests/test_tokenizer.c src/platform.c src/llf.c src/tokenizer.c | build
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

build/test_llf.exe: tests/test_llf.c $(TEST_ENGINE_CORE) | build
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

build/test_engine.exe: tests/test_engine.c $(TEST_ENGINE_CORE) | build
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDFLAGS) $(LIBS)

test: build/test_matvec.exe build/test_tokenizer.exe build/test_llf.exe build/test_engine.exe
	@echo "=== test_matvec ==="
	./build/test_matvec.exe
	@echo "=== test_tokenizer ==="
	./build/test_tokenizer.exe
	@echo "=== test_llf ==="
	./build/test_llf.exe
	@echo "=== test_engine ==="
	./build/test_engine.exe

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean test
