# 性能对比: yllm vs picolm (TinyLlama-1.1B Q4_K_M)

同模型、同 prompt("Once upon a time")、纯 decode(生成 64 token, greedy, 预热后计时)。
脚本:`tools/compare.sh`(可复跑, `THREADS=... NTOKENS=...` 可覆盖)。

- yllm 读数: 二进制 `gen` 输出的 `decode:` 行(纯 decode tok/s)
- picolm 读数: `Generation` 行(纯 decode tok/s)
- picolm 构建: 标量 `make picolm`(基础 CFLAGS, 无 SIMD);AVX2 `make native-avx2`(`-O3 -mavx2 -mfma -mpopcnt`)

> ⚠️ 注意: picolm 的 Makefile 中 `native` 是**第一个 target**,裸 `make` 会自动带
> `-march=native`(本机即 AVX2)。要得到真标量版必须显式 `make picolm`。
> 早期文档里"picolm 标量 5.0 tok/s"是误把 AVX2 版当成了标量。

## 无 AVX2(标量)

| 线程数 | yllm-scalar | picolm-scalar |
|---|---|---|
| 1 | 1.4 | 1.5 |
| 4 | 5.1 | 4.7 |
| 8 | **8.4** | 5.3 |
| 16 | 6.7 | 6.7 |

## 有 AVX2

| 线程数 | yllm-avx2 | picolm-avx2 |
|---|---|---|
| 1 | 5.1 | 4.9 |
| 4 | 17.2 | 14.3 |
| 8 | **27.2** | 15.4 |
| 16 | 22.5 | 17.5 |

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

## 分布式 PP: worker 数量对比(单机, ranks=2/3/4)

场景同上(50-token prompt + 16 decode, 全量 prefill, 预热 + 交错轮次均值)。
脚本: `tools/bench_pp.sh`(可复跑, `RANKS_LIST=... RUNS=...` 覆盖)。

| 配置 | 层分割(32 层) | 端到端 |
|---|---|---|
| rank0 + 1 worker(ranks=2) | 16 / 16 | **3.21s** |
| rank0 + 2 worker(ranks=3) | 11 / 11 / 10 | 3.22s |
| rank0 + 3 worker(ranks=4) | 8 / 8 / 8 / 8 | 3.32s |

worker 1→2 个几乎持平, 3 个略慢(+3%)。decode 单步延迟 = 最慢段 + 传输跳数:
段数增多时每段层数变少(计算降)但 fp32 激活每跳多传 256KB(1.1B 模型带宽
主导), 净效果无收益; 分段均衡性(master 更重)与自回归依赖链(无跨 token
并行空间)限制了扩展性。更大模型(层/计算密度更高)分段收益会更明显。

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

**4) 有 AVX2 时 yllm 8 线程 27.2 vs picolm 15.4**
yllm 的 AVX2 Q4K 内核对权重行连续流式读取 + 寄存器内反量化累加,FLOP/字节 效率更高;
两者 16 线程都被 DRAM 带宽封顶(~10-13GB/s)。

## 结论

- 有 AVX2:yllm 在 8 线程明显领先(+77%),是主力配置。
- 无 AVX2:标量已追平甚至反超(1/16 线程持平,4/8 线程领先,8 线程 +58%)。
- decode 最终都被 DRAM 带宽封顶:单核 ~3-4GB/s,整机共享 ~10-13GB/s。
- 分布式 PP(ranks=2):批量 prefill(默认开)比逐 token 快 ~20%,`make BATCH_PREFILL=0` 可关闭。
- 单机多 worker(ranks=2/3/4):worker 1/2 个持平、3 个略慢(+3%),受传输带宽与
  自回归依赖链限制,小模型分段无扩展收益。