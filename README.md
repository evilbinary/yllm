# yllm

轻量级大语言模型推理工具:最小 49M 内存,支持 GGUF/Safetensors 转换、常驻推理服务与分布式层流水线。

## 快速开始

```sh
make avx2                        # 构建 AVX2 优化版 -> build/avx2/yllm
make chat-avx2 CHAT_PROMPT=你好啊 # 转换模型 + 聊天(未转换时自动从 GGUF 转)
make test-avx2                   # 跑测试
```

构建标量版:`make`(不带 OpenMP:`make OMPFLAG=`)。

## 推理服务(serve)

OpenAI 兼容 HTTP 默认在 `127.0.0.1:8000`。角色、部署、ctl、HTTP 等全部命令见 **`docs/serve-cli.md`**,架构见 **`docs/serving-architecture.md`**。

模型按层切分到多台机器,每台只 mmap/计算自己的层段;机器间每 token 只传激活向量。用法、参数与吞吐模型见 **`docs/distributed-cpu-inference.md`**。

## 目录

```
inference/   推理内核(platform / log / llf / convert / tokenizer / matvec / engine / dist)
serve/       服务层(frame / node / sock 抽象 + rank / server / router / supervisor / hub)
tools/       命令行工具(dump)
main.c       入口: yllm <convert|check|gen|chat|rank|server|router|supervisor|hub|...>
tests/       单元与回归测试
docs/        设计文档
```

## 相关文档

- `docs/serving-architecture.md`       常驻推理服务架构
- `docs/serve-cli.md`                  serve 命令速查(转换/部署/ctl/HTTP)
- `docs/distributed-cpu-inference.md`  分布式层流水线
- `docs/design-mmap-layer-streaming.md` mmap 层流式加载
- `docs/llf-vs-gguf.md` / `docs/memory-vs-picolm.md` 格式与内存对比
- `docs/benchmark.md` / `docs/qwen35-arch.md`
