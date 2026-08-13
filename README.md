# yllm

轻量级 C99 大语言模型推理/转换工具。支持从 GGUF/Safetensors 转换为自研 `.llf` 格式,并做
`convert / check / gen / chat` 等操作;内置**分布式层流水线**与**常驻推理服务**(rank/server/router/supervisor/hub)。

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

## 常驻推理服务(serve 层)

模型权重常驻、进程活等请求,调度层只做毫秒级路由决策;进程只在该重启时重启
(部署更新 / 扩缩容 / 故障自愈 / 显存整理)。详见 `docs/serving-architecture.md`。

**最小部署(make 一键,合并模式):**

```sh
make serve       # hub(supervisor+router+server 同一进程三线程)+ rank
make infer       # 客户端经 router 发请求: yllm router --send "tinyllama 30 Once upon a time"
make serve-stop  # 停掉所有 serve 进程
```

**分开模式(各自独立进程):**

```sh
make supervisor            # 管理节点: 收全部节点心跳, 驱动 router 注册表
make router                # 调度层: 路由决策(least-inflight/round-robin), 收客户端请求
make server                # 业务逻辑组: 租用 rank 组, 转发请求给 leader
make rank                  # 公用推理单元: 模型层段权重常驻, 收 INFER 流式推理
```

**两机部署示例(本机 hub + 远端 rank):**

```sh
# 本机(管理+控制面)
build/avx2/yllm hub --port 9500 --router-port 9400 --server-port 9420 \
    --server-model tinyllama --server-leader 192.168.0.23:9410 --log logs/hub.log

# 远端(推理节点, 心跳发本机 supervisor)
build/avx2/yllm rank --model model.llf --vocab vocab.txt --port 9410 \
    --supervisor 192.168.1.161:9500 --id rank-r0 --log logs/rank.log

# 客户端(本机)
build/avx2/yllm router --port 9400 --send "tinyllama 30 Once upon a time"
```

**角色说明:**

| 命令 | 角色 | 职责 |
|---|---|---|
| `yllm rank` | 公用推理单元(常驻) | 加载模型层段权重,收 PING/STAT/INFER/DRAIN 帧,流式推理 |
| `yllm server` | 业务逻辑组(常驻) | 从 rank 池租用 rank 组,请求队列/会话亲和,转发给 leader |
| `yllm router` | 调度层(常驻,可多实例) | server 注册表(由 supervisor 通知)、路由决策、请求转发 |
| `yllm supervisor` | 管理节点(常驻) | 收全部节点心跳、汇总、判死、驱动 router(ADD/DEL/UPDATE) |
| `yllm hub` | 合并模式 | supervisor+router+server 同一进程不同线程,逻辑/端口不变 |

**协议:** 所有节点统一 `frame`(文本命令行 + 二进制 payload)+ `Node`(身份/心跳/状态)抽象,
心跳统一发 supervisor(数据面),注册表增删由 supervisor 通知 router(生命周期面)。

## 分布式推理(层流水线)

多台机器各跑一个进程,模型按层切分,每台只 mmap/计算自己的层段;机器间每 token 只传
激活向量(hidden fp32)。所有 rank 用同一条命令:

```sh
# 2 台机器(机器 A 和 B 分别执行, 端口一致即可)
yllm gen --model model.llf --vocab vocab.txt --prompt "Hello" --tokens 64 \
         --ranks 2 --rank 0 --port-base 8900     # 机器 A
yllm gen --model model.llf --vocab vocab.txt --prompt "Hello" --tokens 64 \
         --ranks 2 --rank 1 --port-base 8900     # 机器 B
```

- `--ranks N --rank R`:进程总数与自身编号;rank 0 同时是 master(采样/输出)。
- `--port-base P`:TCP 端口基址(默认 8900,每 rank 用 P+rank)。
- rank 0 的 stdout 是生成结果;其余 rank 只打印框架信息。
- 单机验证:`--ranks 2 --rank 0` 与 `--rank 1` 同机跑即可(正确性应等于单机输出)。
- 吞吐模型与分片方案见 `docs/distributed-cpu-inference.md`。

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

- `docs/serving-architecture.md`     常驻推理服务架构(router/server/公用 rank 池/supervisor)
- `docs/distributed-cpu-inference.md` 分布式层流水线
- `docs/design-mmap-layer-streaming.md` mmap 层流式加载
- `docs/llf-vs-gguf.md` / `docs/memory-vs-picolm.md` 格式与内存对比
