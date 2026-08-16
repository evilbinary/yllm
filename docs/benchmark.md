# 性能对比: yllm vs picolm (TinyLlama-1.1B Q4_K_M)

同模型、同 prompt("Once upon a time")、生成 64 token(greedy, 预热后计时)。
脚本:`tools/compare.sh`(可复跑, `THREADS=... NTOKENS=...` 可覆盖),
每配置输出 **prefill / decode** 两组 tok/s。

- yllm 读数: 二进制 `gen` 输出的 `prefill:` / `decode:` 行(纯引擎计时)
- picolm 读数: `Prefill:` / `Generation` 行
- picolm 构建: 标量 `make picolm`(基础 CFLAGS, 无 SIMD);AVX2 `make native-avx2`(`-O3 -mavx2 -mfma -mpopcnt`)

> ⚠️ 注意: picolm 的 Makefile 中 `native` 是**第一个 target**,裸 `make` 会自动带
> `-march=native`(本机即 AVX2)。要得到真标量版必须显式 `make picolm`。
> 早期文档里"picolm 标量 5.0 tok/s"是误把 AVX2 版当成了标量。

## 无 AVX2(标量)

| 线程数 | yllm prefill | yllm decode | picolm prefill | picolm decode |
|---|---|---|---|---|
| 1 | 1.5 | 1.5 | 1.5 | 1.4 |
| 4 | 5.3 | 5.0 | 5.0 | 4.8 |
| 8 | **8.1** | **8.1** | 8.5 | 5.6 |
| 16 | 6.4 | 6.7 | 8.4 | 6.7 |

## 有 AVX2

| 线程数 | yllm prefill | yllm decode | picolm prefill | picolm decode |
|---|---|---|---|---|
| 1 | 5.3 | 5.4 | 5.6 | 4.9 |
| 4 | 17.4 | 17.0 | 19.0 | 13.8 |
| 8 | 19.1 | **26.9** | 22.1 | 16.2 |
| 16 | 23.4 | 21.3 | 28.9 | 17.5 |

## 直跑(gen) vs 服务(serve)差距

`gen` 是单进程直跑(引擎最优路径); serve 模式无论 ranks=1(无分布式)
还是 ranks=2/3/4 都走 `dist_gen` 主分支, 每步多了框架开销:

| 路径 | decode | 说明 |
|---|---|---|
| `gen` 直跑(8 线程) | **26.9 tok/s**(37 ms/step) | 单进程, 无网络/服务/会话 |
| serve ranks=1(16 线程) | 16.8 tok/s(59 ms/step) | 服务 + 会话管理 + 日志 |
| serve ranks=2(每 rank 8 线程) | 17.0 tok/s(58 ms/step) | 上述 + PP 分段串行 + 传输 |

差距 ~1.6x, 主要来自:
- **PP 分段串行**: decode 每步 = master 段 + 传输 + worker 段(ranks=2 时
  24+27 ms)串行相加, 且各段 OMP 线程减半(16/ranks), 层内并行度下降;
  单进程 32 层一次算完反而更快(37 ms)
- **传输 + 同步**: fp32 激活每跳 256KB, 每步一次往返 + 采样等待
- **服务框架**: HTTP 解析、会话管理、日志(实测 KV 落盘/恢复开销可忽略:
  cache-dir 开/关端到端仅差 ~0.01s, 2.22 vs 2.21s)

即: 分布式 PP 的目标是**跨机/大模型内存扩展**, 在单机小模型上分段只会更慢;
这是架构取舍, 不是实现缺陷。

## 同等线程对比(每 rank 线程 T, 总线程 T×ranks)

50-token prompt + 16 decode; `gen` 用 OMP_NUM_THREADS=T, `serve` 用
OMP_NUM_THREADS=T + ranks(环境变量优先, 覆盖 rank.c 的 nproc/ranks 分配)。
脚本: `tools/bench_threads.py`(可复跑, THREADS/RANKS/RUNS 覆盖)。

| T | gen prefill | gen decode | r1 prefill | r1 decode | r2 prefill | r2 decode | r4 prefill | r4 decode |
|---|---|---|---|---|---|---|---|---|
| 1 | 9.4 | 5.4 | 9.9 | 5.0 | 11.0 | 4.4 | 13.3 | 5.0 |
| 4 | 33.5 | 18.4 | 32.4 | 14.8 | 31.6 | 14.8 | 29.8 | 13.9 |
| 8 | 46.3 | **26.8** | 49.9 | 23.2 | 35.7 | 17.6 | 24.3 | 16.6 |
| 16 | 41.0 | 19.8 | 44.6 | 18.8 | 11.3 | 12.0 | 8.6 | 7.8 |

- **总线程 ≤ 16 核**(T=1/4/8): gen ≈ r1 > r2 ≈ r4。服务单机(r1)比 gen 慢
  ~10-20%(HTTP/会话/日志框架); 分布式(r2/r4)因 PP 分段串行 + 传输再慢 ~20%。
- **T=16 超订**: r2/r4 总线程 32/64 > 16 核, 线程争抢使性能崩溃(r2 prefill
  11.3, r4 8.6 tok/s)。
- decode 峰值在 T=8(核数的一半, 每层并行 + 带宽平衡)。
- 同等线程下分布式对 1.1B 小模型无性能收益; 收益仅在跨机内存扩展场景。

## 分布式 PP(ranks=2, 批量 prefill 开/关)

场景: 50-token prompt + 16 decode, serve 模式 curl 端到端耗时;
每轮重启服务 + 清 sessions 保证全量 prefill, ABAB 交替取均值。

- 开(默认): `make chat-avx2`(`YLLM_BATCH_PREFILL=1`, 批量前向 + XB 帧批量传输)
- 关: `make BATCH_PREFILL=0 chat-avx2`(逐 token prefill + logits 回执软同步)

| 配置 | 端到端(50-token prompt + 16 decode) |
|---|---|
| 批量 prefill(默认) | **2.24s**(7.1 tok/s) |
| BATCH_PREFILL=0(逐 token) | 2.79s(5.7 tok/s) |

收益 ~20%, 全部来自 prefill 段: 批量前向(块反量化共享一次、多 token 点积
分摊)+ XB 帧批量传输, 消除逐 token logits 回执软同步; decode 两版相同
(~51 ms/step, master 24 + worker 27, 受自回归依赖链限制)。
小批(<16 token)自动回退逐 token(`PREFILL_BATCH_MIN`), 避免反量化/调度开销。

## 分布式 PP: 无分布式 vs worker 数量(单机, ranks=1/2/3/4)

场景同上(50-token prompt + 16 decode, 全量 prefill, 预热 + 交错轮次均值)。
脚本: `tools/bench_pp.sh`(可复跑, `RANKS_LIST=... RUNS=...` 覆盖)。

| 配置 | 层分割(32 层) | 总耗时 | prefill tok/s | decode tok/s | 总 tok/s |
|---|---|---|---|---|---|
| 无分布式(单 rank 直跑) | 32 | 2.38s | 34.9 | 16.8 | 27.7 |
| rank0 + 1 worker(ranks=2) | 16 / 16 | **2.30s** | 36.7 | 17.0 | 28.6 |
| rank0 + 2 worker(ranks=3) | 11 / 11 / 10 | 2.35s | 35.7 | 16.8 | 28.0 |
| rank0 + 3 worker(ranks=4) | 8 / 8 / 8 / 8 | 2.35s | 35.7 | 17.0 | 28.0 |

无分布式(单 rank)与 1 worker 几乎持平, 2/3 worker 略慢(~+2%)。
prefill 受批量前向控制(50 tokens ≈ 4 批, ~35 tok/s); decode 固定 ~17 tok/s
(58 ms/step, 含 master 前向 + 传输 + 采样)。decode 单步延迟 = 最慢段 + 传输
跳数: 段数增多时每段层数变少(计算降)但 fp32 激活每跳多传 256KB(1.1B 模型
带宽主导), 净效果无收益; 自回归依赖链(无跨 token 并行空间)限制扩展性。
更大模型(层/计算密度更高)分段收益会更明显。

## 分析

**1) 为什么 yllm 标量曾被 picolm"标量"甩开 6 倍?**
picolm 裸 `make` 走 `native` target(`-march=native`)= AVX2 版,不是标量;
两者对比从一开始就不公平。真正标量对标的差距只有 ~1.4x(见上表)。

**2) 标量内核的优化**
yllm 标量 Q4K/Q6K 原实现是单累加器串行依赖链 + 逐元素分支解 nibble,
gcc 在 `-O2` 下不向量化 fp 归约(`-ffast-math` 缺失),单线程仅 ~0.5 tok/s。
三轮优化:
- **4 独立累加器 + 同字节解低/高 nibble**(消除分支、给足 ILP):0.5 → 1.0 tok/s
- **qx 与 x 分开累加、组末合并**(Q4K 每元素 4 op→2 op):8 线程 5.0 → 5.5
- **Q6K 块结构重写**:原来每元素算 `half/quad/ll` 索引(4 次移位),改为每次
  3 个 load 解出 4 个 6-bit 权重、无索引运算。内核 10.0ms → **2.0ms**(2048×2048,
  1 线程,反超 picolm 的 2.7ms)。这是标量场景的主要瓶颈(注意力投影是 Q6_K):
  1 线程 1.0 → 1.4,8 线程 5.5 → 8.1

**3) 为什么 picolm 标量 ≈ picolm AVX2(旧困惑)**
两个"版本"其实都是 AVX2 构建(见 ⚠️),所以性能相同。

**4) 有 AVX2 时 yllm 8 线程 decode 26.9 vs picolm 16.2**
yllm 的 AVX2 Q4K 内核对权重行连续流式读取 + 寄存器内反量化累加,FLOP/字节 效率更高;
两者 16 线程都被 DRAM 带宽封顶(~10-13GB/s)。

## 结论

- 有 AVX2:yllm 8 线程 decode 明显领先(+66%),是主力配置。
- 无 AVX2:标量已追平甚至反超(1/16 线程持平,4/8 线程领先)。
- decode 最终都被 DRAM 带宽封顶:单核 ~3-4GB/s,整机共享 ~10-13GB/s。
- 分布式 PP(ranks=2):批量 prefill(默认开)比逐 token 快 ~20%,`make BATCH_PREFILL=0` 可关闭。
- 直跑(gen)比服务(serve)decode 快 ~1.6x:PP 分段串行 + 传输 + 服务/会话开销;
  分布式目标是跨机/大模型内存扩展, 单机小模型上分段无性能收益。
- 单机多 worker(ranks=1/2/3/4):单 rank 直跑与 1 worker 持平(2.38 vs 2.30s),
  2/3 worker 略慢(~+2%); 受传输带宽与自回归依赖链限制, 小模型分段无扩展收益。
  `tools/bench_pp.sh` 可复跑(prefill/decode/总 tok/s 一并输出)。