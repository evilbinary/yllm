# Bench 使用文档

`tools/` 下有两个 PP 基准脚本:

| 脚本 | 场景 | 适用 |
|------|------|------|
| `bench_threads.py` | 单机 本地 PP | 一台机器上多段进程, 对比 gen 直跑 vs 分布式 |
| `bench_cross.py` | 跨机 PP | rank0 跑 Windows(hub), rank1..N-1 跑 Linux(ssh 拉起) |

## 通用指标

- **pre tok/s**: 整条流水线 prefill 吞吐(50-token prompt)
- **dec tok/s**: 整条流水线 decode 吞吐(16 个生成 token)
- 计时取 rank0 的 `logs/tinyllama-rank-0.log`(rank0 恒为本机/leader, 含全流水线)

## 环境变量(两个脚本通用)

| 变量 | 默认 | 说明 |
|------|------|------|
| `THREADS` | bench_threads: `1 4 8 16`; bench_cross: `8 16` | 每 rank 的 OMP 线程数 |
| `RANKS` | bench_threads: `1 2 4`; bench_cross: `2 3` | PP 总段数 |
| `RUNS` | bench_threads: `2`; bench_cross: `1` | 每组重复次数(取平均) |
| `PROMPT` / `NTOKENS` | 50-token / 16 | prompt 文本 / 生成 token 数 |

## 一、单机基准 bench_threads.py

```
python3 tools/bench_threads.py
THREADS="8" RANKS="2" RUNS="1" python3 tools/bench_threads.py
```

- 在 Windows 需 MSYS2/MinGW 环境执行(内部用 bash 语法, 会备份/恢复 `serve.yaml`)。
- 输出形如:
  ```
  线程T    | pre tok/s | dec tok/s
  8 ranks=2 | 34.9      | 11.4
  ```

## 二、跨机基准 bench_cross.py

拓扑: **rank0 在 Windows(本机 `yllm hub`, supervisor+router+server+rank0)**, 
rank1..N-1 在 Linux 后台起 `yllm rank`, 通过 ssh 拉起。

```
python3 tools/bench_cross.py
```

### 前置要求

1. 配好本机到 Linux 的免密 ssh(`ssh -o BatchMode=yes <LIN_IP>`)。
2. Linux 端源码在 `~/develop/yllm`, 且已构建 **avx2** 版 `build/avx2/yllm`
   (标量 `build/yllm` 缺 `--peers`, 会拒当前 llf, 不能用)。
3. Windows 端已构建 `build/avx2/yllm.exe`。

### 可配置常量(脚本顶部)

| 常量 | 默认 | 说明 |
|------|------|------|
| `WIN_IP` | `192.168.1.161` | Windows 局域网 IP(hub 绑 INADDR_ANY) |
| `LIN_IP` | `192.168.0.23` | Linux 主机 IP(经 ssh) |
| `LIN_DIR` | `~/develop/yllm` | Linux 远程工作目录 |
| `LIN_BIN` | `./build/avx2/yllm` | Linux 端二进制(相对 LIN_DIR) |
| `WIN_BIN` | `./build/avx2/yllm.exe` | Windows 端二进制 |

启动命令示例(Linux 端, rank1):
```
OMP_NUM_THREADS=T setsid sh -c '<cd> && ./build/avx2/yllm rank \
  --model ... --vocab ... --model-name tinyllama \
  --port 9410+r --rank r --ranks N \
  --supervisor 192.168.1.161:9500 --id rank-r \
  --peers <win_ip>,<lin_ip>... --cache-dir sessions \
  --log logs/tinyllama-rank-r.log' </dev/null
```
`setsid sh -c '... &' </dev/null` 用于让 ssh 立即返回(rank 常驻进程不会挂住 ssh 通道)。

### 关键点(已排过的坑)

- **租约按 model-name 分组**: 远程 rank 必须传 `--model-name tinyllama`, 否则进不了该模型租约池。
- **peers 自动生成**: 本机段(supervisor 拉起 rank0)默认 `127.0.0.1,127.0.0.1`;
  跨机时 rank0 的 peers 由 server 下发的 LEASE(`peers=192.168.1.161,192.168.0.23`)在每次 INFER 中携带覆盖。
- **会话模式必须带 seg/segs/peers**: `server.c` 的 `forward_infer_sess` 现在会把
  `seg=0 segs=%d peers=%s` 拼进 INFER 参数;此前缺失导致 rank0 一直用启动时的 `127.0.0.1`
  peers 去连远端 rank1 而挂死(已修复, 需要重新编译 Windows 二进制)。
- **pipe 端口**: `pipe_base = rank_port_base - dist_rank + 100`。
  rank0 起 9510, rank1 起 9511, rank2 起 9512 …;master 连下段, 末段连回 rank0。
- **Windows 必须能连到 Linux 的 pipe 端口**: master(rank0)要主动 connect 到远端段
  `9510+r`(脚本已通: Windows→Linux TCP 可用)。
- **OMP_NUM_THREADS 传递**: `spawn_proc`(CreateProcess, env=NULL)继承父进程环境,
  hub 启动时注入 `OMP_NUM_THREADS` 即可传给 rank0。

### 输出示例

```
线程T    | pre tok/s | dec tok/s
8 ranks=2 | 34.9      | 11.4
8 ranks=3 | 43.5      | 13.0
16 ranks=2 | 35.8      | 11.0
16 ranks=3 | 38.6      | 11.5
```

### 注意

- 脚本结束会 `kill_all` 清理两端进程(Windows `taskkill /F /IM yllm.exe`,
  Linux `pkill -9 -f build/avx2/yllm`)。中途 Ctrl-C 后请手动执行
  `taskkill /F /IM yllm.exe` 和远端 `pkill -9 -f build/avx2/yllm` 清理。
- 3-rank 拓扑 = rank0 Windows + rank1/rank2 Linux。

### 网络统计

`bench_cross.py` 读取环境变量 `YLLM_DIST_STATS`,设 1 时自动传给两端
(rank0 + 远端 rank),rank0 每 8 个 token 打印一次 `dist@tokN`,结束打印汇总
(`X frames`/`block`/`bw`)。可用于观察段间传输阻塞与带宽。

## 性能对比与结论

跨机 vs 单机的实测数据、prefill/decode 时序图、decode 网络开销分析,
见 [`docs/benchmark.md`](../docs/benchmark.md) 的「跨机 PP」小节。

