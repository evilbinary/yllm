CC       ?= cc
CFLAGS   ?= -O2 -std=c99 -Wall -Wextra
LDFLAGS  ?=
LIBS     :=
BIN      := build/yllm
SRC      := src/platform.c src/llf.c src/convert.c src/convert_safetensors.c src/convert_gguf.c src/tokenizer.c src/matvec.c src/engine.c src/main.c
OBJ      := $(SRC:src/%.c=build/%.o)

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifneq ($(filter Linux Darwin,$(UNAME_S)),)
    LIBS += -lm -pthread
endif

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(LIBS)

build/%.o: src/%.c src/yllm.h src/llf.h src/convert.h src/matvec.h | build
	$(CC) $(CFLAGS) -c -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean