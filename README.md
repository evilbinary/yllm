# yllm

轻量级 C99 大语言模型推理/转换工具。支持从 GGUF/Safetensors 转换为自研 `.llf` 格式,并做
`convert / check / gen / chat` 等操作。

## 构建

```sh
make            # 标量版本 -> build/yllm
make avx2       # AVX2 优化版本(仅 x86_64)-> build/avx2/yllm
```

Linux 默认开启 OpenMP 多核加速;macOS / Windows(MinGW)也可构建(不带 OpenMP)。
如需关闭 OpenMP:`make OMPFLAG=`。

## 测试

```sh
make test        # 标量测试
make test-avx2   # AVX2 测试(matmul / tokenizer / llf / engine 回归)
```

## 转换模型

```sh
# 从 GGUF 转换(会顺带导出 vocab)
build/yllm convert --gguf model.Q4_K_M.gguf --out model.llf --vocab vocab.txt --seq 2048

# 生成 dummy 模型
build/yllm convert --out dummy.llf --blocks 2 --hidden 64 --heads 4 --vocab-size 32000
```

## 运行 chat / gen(推荐用 make 目标)

```sh
make chat-avx2                                        # 默认参数
make chat-avx2 NTHREADS=16 CHAT_PROMPT=你好啊 CHAT_TOKENS=80
make gen NTHREADS=4 CHAT_PROMPT="Once upon a time"    # 标量版
```

- `model` 未转换时会先自动从 `MODEL_GGUF` 转成 `MODEL_LLF` 再运行。
- `NTHREADS` 默认取本机核心数,可指定推理使用的 OpenMP 线程数。
- 相关变量均可覆盖:`MODEL_GGUF` / `MODEL_LLF` / `MODEL_VOCAB` / `CHAT_PROMPT` / `CHAT_TOKENS`。

## 直接调用二进制

```sh
build/avx2/yllm check --model model.llf
build/avx2/yllm gen   --model model.llf --vocab vocab.txt --prompt "Hello" --tokens 64
build/avx2/yllm chat  --model model.llf --vocab vocab.txt --prompt "你好啊" --tokens 80 --temp 0.8 --top-p 0.9
```

## 目录

```
src/         核心代码(platform / convert / tokenizer / matvec / engine / main)
tests/       单元与回归测试
docs/        设计文档
```
