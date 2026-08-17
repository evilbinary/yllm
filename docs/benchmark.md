# 性能对比: yllm vs picolm (TinyLlama-1.1B Q4\_K\_M)

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
| --- | ------------ | ----------- | -------------- | ------------- |
| 1   | 1.5          | 1.5         | 1.5            | 1.4           |
| 4   | 5.3          | 5.0         | 5.0            | 4.8           |
| 8   | **8.1**      | **8.1**     | 8.5            | 5.6           |
| 16  | 6.4          | 6.7         | 8.4            | 6.7           |

## 有 AVX2

| 线程数 | yllm prefill | yllm decode | picolm prefill | picolm decode |
| --- | ------------ | ----------- | -------------- | ------------- |
| 1   | 5.3          | 5.4         | 5.6            | 4.9           |
| 4   | 17.4         | 17.0        | 19.0           | 13.8          |
| 8   | 19.1         | **26.9**    | 22.1           | 16.2          |
| 16  | 23.4         | 21.3        | 28.9           | 17.5          |

## 直跑(gen) vs 服务(serve)差距

`gen` 是单进程直跑(引擎最优路径); serve 模式无论 ranks=1(无分布式)
还是 ranks=2/3/4 都走 `dist_gen` 主分支, 每步多了框架开销:

| 路径                         | decode                     | 说明                |
| -------------------------- | -------------------------- | ----------------- |
| `gen` 直跑(8 线程)             | **26.9 tok/s**(37 ms/step) | 单进程, 无网络/服务/会话    |
| serve ranks=1(16 线程)       | 16.8 tok/s(59 ms/step)     | 服务 + 会话管理 + 日志    |
| serve ranks=2(每 rank 8 线程) | 17.0 tok/s(58 ms/step)     | 上述 + PP 分段串行 + 传输 |

差距 \~1.6x, 主要来自:

- **PP 分段串行**: decode 每步 = master 段 + 传输 + worker 段(ranks=2 时
  24+27 ms)串行相加, 且各段 OMP 线程减半(16/ranks), 层内并行度下降;
  单进程 32 层一次算完反而更快(37 ms)
- **传输 + 同步**: fp32 激活每跳 256KB, 每步一次往返 + 采样等待
- **服务框架**: HTTP 解析、会话管理、日志(实测 KV 落盘/恢复开销可忽略:
  cache-dir 开/关端到端仅差 \~0.01s, 2.22 vs 2.21s)

即: 分布式 PP 的目标是**跨机/大模型内存扩展**, 在单机小模型上分段只会更慢;
这是架构取舍, 不是实现缺陷。

## 同等线程对比(每 rank 线程 T, 总线程 T×ranks)

50-token prompt + 16 decode; `gen` 用 OMP\_NUM\_THREADS=T, `serve` 用
OMP\_NUM\_THREADS=T + ranks(环境变量优先, 覆盖 rank.c 的 nproc/ranks 分配)。
脚本: `tools/bench_threads.py`(可复跑, THREADS/RANKS/RUNS 覆盖)。

| T  | gen prefill | gen decode | r1 prefill | r1 decode | r2 prefill | r2 decode | r4 prefill | r4 decode |
| -- | ----------- | ---------- | ---------- | --------- | ---------- | --------- | ---------- | --------- |
| 1  | 9.4         | 5.4        | 9.9        | 5.0       | 11.0       | 4.4       | 13.3       | 5.0       |
| 4  | 33.5        | 18.4       | 32.4       | 14.8      | 31.6       | 14.8      | 29.8       | 13.9      |
| 8  | 46.3        | **26.8**   | 49.9       | 23.2      | 35.7       | 17.6      | 24.3       | 16.6      |
| 16 | 41.0        | 19.8       | 44.6       | 18.8      | 11.3       | 12.0      | 8.6        | 7.8       |

- **总线程 ≤ 16 核**(T=1/4/8): gen ≈ r1 > r2 ≈ r4。服务单机(r1)比 gen 慢
  \~10-20%(HTTP/会话/日志框架); 分布式(r2/r4)因 PP 分段串行 + 传输再慢 \~20%。
- **T=16 超订**: r2/r4 总线程 32/64 > 16 核, 线程争抢使性能崩溃(r2 prefill
  11.3, r4 8.6 tok/s)。
- decode 峰值在 T=8(核数的一半, 每层并行 + 带宽平衡)。
- 同等线程下分布式对 1.1B 小模型无性能收益; 收益仅在跨机内存扩展场景。

## 分布式 PP(ranks=2, 批量 prefill 开/关)

场景: 50-token prompt + 16 decode, serve 模式 curl 端到端耗时;
每轮重启服务 + 清 sessions 保证全量 prefill, ABAB 交替取均值。

- 开(默认): `make chat-avx2`(`YLLM_BATCH_PREFILL=1`, 批量前向 + XB 帧批量传输)
- 关: `make BATCH_PREFILL=0 chat-avx2`(逐 token prefill + logits 回执软同步)

| 配置                        | 端到端(50-token prompt + 16 decode) |
| ------------------------- | -------------------------------- |
| 批量 prefill(默认)            | **2.24s**(7.1 tok/s)             |
| BATCH\_PREFILL=0(逐 token) | 2.79s(5.7 tok/s)                 |

收益 \~20%, 全部来自 prefill 段: 批量前向(块反量化共享一次、多 token 点积
分摊)+ XB 帧批量传输, 消除逐 token logits 回执软同步; decode 两版相同
(\~51 ms/step, master 24 + worker 27, 受自回归依赖链限制)。
小批(<16 token)自动回退逐 token(`PREFILL_BATCH_MIN`), 避免反量化/调度开销。

## 分布式 PP: 无分布式 vs worker 数量(单机, ranks=1/2/3/4)

场景同上(50-token prompt + 16 decode, 全量 prefill, 预热 + 交错轮次均值)。
脚本: `tools/bench_pp.sh`(可复跑, `RANKS_LIST=... RUNS=...` 覆盖)。

| 配置                        | 层分割(32 层)     | 总耗时       | prefill tok/s | decode tok/s | 总 tok/s  |
| ------------------------- | ------------- | --------- | ------------- | ------------ | -------- |
| 无分布式(单 rank 直跑)           | 32            | **2.10s** | **42.9**      | 16.6         | **30.9** |
| rank0 + 1 worker(ranks=2) | 16 / 16       | 2.35s     | 35.7          | 17.0         | 28.0     |
| rank0 + 2 worker(ranks=3) | 11 / 11 / 10  | 2.80s     | 30.3          | 13.9         | 23.5     |
| rank0 + 3 worker(ranks=4) | 8 / 8 / 8 / 8 | 2.92s     | 28.5          | 13.6         | 22.6     |

单 rank 直跑最快, 每加一个 worker 端到端递增 \~+20%(2.10 → 2.35 → 2.80 → 2.92s)。
prefill 随 worker 数下降(42.9 → 28.5 tok/s): 层段切分后各段批量反量化效率下降 +
每段 XB 帧传输; decode 类似(17.0 → 13.6): 每段 OMP 线程自动减半(nproc/ranks)
且传输跳数增多。自回归依赖链(无跨 token 并行空间)限制扩展性;
更大模型(层/计算密度更高)分段收益才会出现。

## 分析

**1) 为什么 yllm 标量曾被 picolm"标量"甩开 6 倍?**
picolm 裸 `make` 走 `native` target(`-march=native`)= AVX2 版,不是标量;
两者对比从一开始就不公平。真正标量对标的差距只有 \~1.4x(见上表)。

**2) 标量内核的优化**
yllm 标量 Q4K/Q6K 原实现是单累加器串行依赖链 + 逐元素分支解 nibble,
gcc 在 `-O2` 下不向量化 fp 归约(`-ffast-math` 缺失),单线程仅 \~0.5 tok/s。
三轮优化:

- **4 独立累加器 + 同字节解低/高 nibble**(消除分支、给足 ILP):0.5 → 1.0 tok/s
- **qx 与 x 分开累加、组末合并**(Q4K 每元素 4 op→2 op):8 线程 5.0 → 5.5
- **Q6K 块结构重写**:原来每元素算 `half/quad/ll` 索引(4 次移位),改为每次
  3 个 load 解出 4 个 6-bit 权重、无索引运算。内核 10.0ms → **2.0ms**(2048×2048,
  1 线程,反超 picolm 的 2.7ms)。这是标量场景的主要瓶颈(注意力投影是 Q6\_K):
  1 线程 1.0 → 1.4,8 线程 5.5 → 8.1

**3) 为什么 picolm 标量 ≈ picolm AVX2(旧困惑)**
两个"版本"其实都是 AVX2 构建(见 ⚠️),所以性能相同。

**4) 有 AVX2 时 yllm 8 线程 decode 26.9 vs picolm 16.2**
yllm 的 AVX2 Q4K 内核对权重行连续流式读取 + 寄存器内反量化累加,FLOP/字节 效率更高;
两者 16 线程都被 DRAM 带宽封顶(\~10-13GB/s)。

## 结论

- 有 AVX2:yllm 8 线程 decode 明显领先(+66%),是主力配置。
- 无 AVX2:标量已追平甚至反超(1/16 线程持平,4/8 线程领先)。
- decode 最终都被 DRAM 带宽封顶:单核 \~3-4GB/s,整机共享 \~10-13GB/s。
- 分布式 PP(ranks=2):批量 prefill(默认开)比逐 token 快 \~20%,`make BATCH_PREFILL=0` 可关闭。
- 直跑(gen)比服务(serve)decode 快 \~1.6x:PP 分段串行 + 传输 + 服务/会话开销;
  分布式目标是跨机/大模型内存扩展, 单机小模型上分段无性能收益。
- 单机多 worker(ranks=1/2/3/4):单 rank 直跑最快(2.10s),每加一个 worker
  端到端 \~+20%(2.35/2.80/2.92s); 分段使 prefill/decode 双双下降。
  `tools/bench_pp.sh` 可复跑(prefill/decode/总 tok/s 一并输出)。

## 跨机 PP(rank0 Windows + rank1/2 Linux)

拓扑: rank0 跑本机 `yllm hub`(supervisor+router+server+rank0), rank1..N-1 经 ssh
在 Linux(28 核)后台起 `yllm rank`。脚本: `tools/bench_cross.py`
(THREADS/RANKS/RUNS 覆盖)。场景: 50-token prompt + 16 decode。

| 线程×段数 | prefill tok/s | decode tok/s |
| ----- | ------------- | ------------ |
| 8×2   | 34.9          | 11.4         |
| 8×3   | 43.5          | 13.0         |
| 16×2  | 35.8          | 11.0         |
| 16×3  | 38.6          | 11.5         |

对比纯 Windows 单机(12 逻辑核):

- **prefill**: 跨机 2 段比单机 2 段快约 30%,3 段接近翻倍(43.5 vs 21.6)
  —— Linux 端核更多, 且各段并行算各自层段, prefill 天然并行加速。
- **decode**: 持平或略降(跨机 2 段 11.4 vs 单机 12.3)—— decode 是串行流水线,
  受跨机网络往返延迟拖累, 抵消 Linux 更多核的优势。

**一句话: 跨机对 prefill 帮助大, decode 无明显收益(受网络延迟限制)。**

### prefill / decode 关系(时序图)

模型按**层切分**(PP),每个 rank 只持有部分层,激活 X 必须跨机流动。

```
一次请求 = prefill(处理整个输入) → decode(逐个生成 token)
```

```
═══ 一次推理请求(rank0 Win 主控, rank1 Linux worker)═══
                 rank0(Windows)                rank1(Linux)
                 │ 层0..N/2                    │ 层N/2..N
─────────────────┼─────────────────────────────┼─────────────
 ① prefill(51token 分批)                        │
   [16tok 前向]   │───X帧(128KB,16tok)──────────→│ 前半层→后半层前向
   [16tok 前向]   │───X帧(128KB,16tok)──────────→│
   [16tok 前向]   │───X帧(128KB,16tok)──────────→│
   [3tok 前向]    │───X帧(128KB,3tok)───────────→│
                 │←────logits(最后token)─────────│ 采样出第1个token
 ② decode(逐 token ×16)                         │
   采样t1→前向    │───X帧(8KB,1tok)────────────→│ 后半层前向
                 │←────logits─────────────────│
   采样t2→前向    │───X帧(8KB,1tok)────────────→│
                 │←────logits─────────────────│
      ... (每 token = 发1帧 → 等logits 返回)      ...
   采样t16        │─(最后冲刷 logits)──────────→│
                 │←DONE───────────────────────│
```

每 decode token 的周期(串行累加 = 1 token 时延):

```
  ┌─前向计算─┐ ┌─网络发X─┐ ┌─worker计算─┐ ┌─网络回logits─┐ ┌─采样─┐
  │  rank0   │ │  send   │ │   rank1    │ │     recv     │ │ rank0 │
  └──────────┘ └─────────┘ └────────────┘ └──────────────┘ └───────┘
                          ↑ recv 等待约占整个周期的 ~57%
```

要点:

- **prefill**: X 帧批量发(每帧 ≤16 token, \~128KB), 一次管道只走 4 次传输, 吞吐高。
- **decode**: 16 个 token 逐帧, 每 token = rank0前向→发X→rank1前向→回logits→采样,
  串行 16 次, 每次都吃一次跨机网络 RTT。
- 同一请求 prefill 只 1 段、decode 16 段;decode 每段都吃网络往返延迟
  → 这是 decode 跨机无收益的根因。

### decode 的网络开销(实测)

用 `YLLM_DIST_STATS=1` 采集 rank0(2段, 16 token decode @ 11.7 tok/s):

| 项                   | 值                                       |
| ------------------- | --------------------------------------- |
| recv 阻塞(等远端 logits) | **778.2 ms**(占 decode 总时 1.37s 的 \~57%) |
| send 阻塞             | 仅 1.7 ms(send 只写 TCP 缓冲, 异步不占时间)        |
| recv 带宽             | 3.3 MB/s                                |
| send 带宽             | 309 MB/s(瞬时写入速率, 非链路速率)                 |

结论: decode 变慢的主因是 **recv 等待**(master 等 worker 算完并回传 logits),
纯网络传输本身不是瓶颈(send 几乎不阻塞)。`bench_cross.py` 读取环境变量
`YLLM_DIST_STATS`,设 1 时自动传给两端;rank0 每 8 token 打印一次 `dist@tokN`,
结束打印汇总(`X frames`/`block`/`bw`)。

### X 帧构成: 一次请求发了多少数据

本次请求 prefill 51 token + decode 16 token, rank0 共发 20 个 X 帧:

- **prefill**: `ceil(51/16)=4` 个批量帧(每帧 ≤16 token, \~128KB, `PP_XB=16`)
- **decode**: 16 个单帧(每 token 1 帧, \~8KB)
- 单帧大小 = `4 + hidden×(fp16?2:4)`;tinyllama `hidden=2048`, 默认 fp32(4B) →
  decode 单帧 \~8KB, prefill 批量帧 \~128KB。

### decode 每 token 周期拆解(实测, YLLM\_DISTTIMING)

用 `YLLM_DISTTIMING=1` 采集 2 段跨机, rank0(master)/rank1(worker)每 token 各阶段耗时:

**master(rank0, 6 线程)**:

| 阶段             | 耗时         | 说明                |
| -------------- | ---------- | ----------------- |
| wait           | \~38ms     | 等 worker 回 logits |
| sample         | \~4ms      | 采样                |
| fwd            | \~33ms     | 算前半层              |
| send           | \~0ms      | 发 X(异步)           |
| **单 token 周期** | **\~75ms** | <br />            |

**worker(rank1, 8 线程)**:

| 阶段             | 耗时         | 说明           |
| -------------- | ---------- | ------------ |
| wait           | \~49ms     | 等 master 发 X |
| fwd            | \~29ms     | 算后半层         |
| **单 token 周期** | **\~78ms** | <br />       |

两端对得上: master `wait`= 等 worker 算完, worker `wait`= 等 master 算完。

**结论: 瓶颈 = 分段串行 + 并行度不足, 不是网络。**

- 真实算力 = master fwd(33) + worker fwd(29) ≈ **62ms, 占单 token 周期 75ms 的 \~83%**
- 当前 master/worker 前向是**加法不是重叠**: 75 ≈ 33(master) + 29(worker) + 13(RTT+调度)
- 网络 RTT 实际只有 \~13ms(其中真跨机往返更少), **不是瓶颈**
- 每段前向慢的原因: rank0 线程减半到 6, 且小模型 32/2=16 层分段后层内并行度不足

**优化方向(按收益排序)**:

1. **流水化(master/worker 前向重叠)**: 让 master 算前向的同时 worker 也在算,
   单 token 周期 75 → max(33,29)+RTT ≈ **40ms**, decode 11 → **\~25 tok/s, 接近翻倍**。
   这是 PP 架构标准优化(micro-batch 交错), 但对单序列受自回归依赖链限制,
   需多请求 batch 才能真正把等待隐藏。
2. **提高单段并行度**: 不要按 nproc/ranks 减半线程, 缓解每段层内并行不足。
3. **dist-fp16 / 降 RTT**: 次要, 因为网络只占 \~17%。

**一句话: 跨机 decode 卡在"master 前向 + worker 前向 串行相加", 流水化重叠前向
可翻倍到 \~25 tok/s; 网络不是当前瓶颈。**

### 与单机逐段对比(同 8 线程, decode)

把跨机 2 段的每 token 周期与单机对比, 分离"分段串行"与"网络"各自的代价:

| 配置          | 每段线程  | 每 token 前向               | 网络 RTT  | 单 token 周期 | decode |
| ----------- | ----- | ------------------------ | ------- | ---------- | ------ |
| 单机 r1(直跑)   | 8     | 一段 \~59ms                | 无       | \~59ms     | \~17   |
| 单机 r2(本地分段) | 8     | master+worker 串行 \~62ms  | 本地 pipe | \~62ms     | \~16   |
| 跨机 r2       | 6 / 8 | master 33 + worker 29 串行 | \~13ms  | \~75ms     | \~11.4 |

要点:

- **单机 r1 vs r2**: 本地分段后前向相加(\~62ms)略增, 分段串行的代价很小(\~+5%)。
  单机 r1/r2 都 \~16-17 tok/s, 差别来自框架开销, 网络为零。
- **单机 r2 vs 跨机 r2**: 前向相同(\~62ms), 差别只在**网络 \~13ms** →
  单 token 周期 62 → 75ms, decode 16 → 11.4, **跨网络净降 \~30%**。
- **跨机增加的 13ms 里**, 真·网络 RTT 只是其中一部分, 另一部分是
  master/worker 在本地 pipe → TCP 之间的调度抖动。其余 62ms 前向在跨机下
  master 因线程减半(6 线程)反而比单机更慢。

**结论**:

- 单机 r2 的瓶颈 = 分段串行(master+worker 前向相加), 但本地无网络, 代价小。
- 跨机 r2 额外多付的只是 \~13ms 网络, 真正拉开差距的是 rank0 线程减半(6 线程)
  使 master 前向变慢 + 网络调度。
- 无论单机/跨机, **共同瓶颈都是"两段前向串行相加"**——流水化重叠是唯一能同时
  解放两者的优化, 单机也能从 16 提到 \~25 tok/s, 跨机同理。

## HTTP / 会话框架对性能的影响(gen vs serve)

`gen` 直跑与 `serve`(HTTP + 会话 + PP 分布式)在 **decode 热路径**上的差异,
是 gen 26.8 而 serve 慢 \~13% 的直接原因。

### 完整请求链路(serve 分布式)

```
curl POST → [router HTTP: 解析JSON] → [server: 词表渲染+sess_commit]
→ [server→rank TCP] → [rank handle_infer_cache] → [dist_gen master循环 每token]:
     dist_recv_logits → 采样 → emit(on_token_rank) → forward → send_x
→ [server收 TS 行: sess_append加锁 + sock_send_n回HTTP]
→ [HTTP collect: append到内存]
```

### 每 token 的框架开销(decode 热路径, 相对 gen 的增量)

| 操作                                 | 频次     | 成本    | 位置                                        |
| ---------------------------------- | ------ | ----- | ----------------------------------------- |
| `vocab_decode`(65536 大缓冲)          | 每token | 高     | rank.c:200 `on_token_rank`                |
| `xsend_rank` ×2                    | 每token | 中     | rank.c:206-207 (TCP 写回)                   |
| `ylog_raw_log`                     | 每token | **高** | rank.c:208 → log.c:130 `fflush`(同步刷盘+全局锁) |
| server `sess_append` + `sess_lock` | 每token | 中     | server.c:257-260                          |
| server `sock_send_n` 回 HTTP        | 每token | 中     | server.c:261                              |

**`ylog_raw_log`** **是最大杀手**: 每次 `fprintf + fflush` 同步刷磁盘, 且取全局 mutex。
gen 直跑 decode 热路径只有 `engine_sample + engine_forward`, **零 IO/日志/锁/TCP**。

### 量化

| 配置           | decode | 每步   | 框架开销                  |
| ------------ | ------ | ---- | --------------------- |
| gen(8T)      | 26.8   | 37ms | 无                     |
| serve r1(8T) | 23.2   | 43ms | **+6ms/步 (日志+TCP+锁)** |
| serve r2 跨机  | 11.4   | 88ms | +6ms(叠加在分段串行+网络上)     |

- 6ms/步 框架开销在 r1 占 **\~14%**, 是 gen vs serve 的核心差距。
- 跨机下这 6ms **不在网络等待里**, 而是加长 master 本地周期
  (recv\_logits 后 → emit → forward), 让跨机从 82 也拖到 88ms。

### 验证实验: 关日志后的实测(2026-08)

临时注释掉 `on_token_rank` 里的 `ylog_raw_log`(消除每 token 同步刷盘 + 全局锁),
两端重编后实测:

| 场景        | 开日志       | 关日志  | 提升    |
| --------- | --------- | ---- | ----- |
| 跨机 r2(8T) | 11.4      | 11.8 | +3.5% |
| 单机 r1(8T) | 14.1(同机测) | —    | —     |

**结论: 日志刷盘 NOT 主要开销**(跨机仅 +3.5%)。原因:

- `fflush` 虽同步刷盘, 但一次只刷几字节文本, 文件在内存缓存, 实际耗时不显著。
- 全局 mutex 在 rank 单线程 decode 循环内无竞争, 锁开销可忽略。
- decode 热路径真正的大头仍是"两段前向串行 + 网络", 框架 emit 链占比小。

### 低风险优化建议(部分已验证, 未实施)

1. ~~关掉~~ ~~`ylog_raw_log`/`fflush`~~ — **已验证, 仅 +3.5%, 不值得改**(破坏日志)。
2. **`vocab_decode`** **改用小栈缓冲**(65536 太大, 每 token 分配/清零)——未验证, 潜在微收益。
3. **合并** **`xsend`** **次数**(TS 头 + payload 一次 send)——未验证, 潜在微收益。
4. 非 stream(collect)模式在 decode 热路径其实**无 HTTP 逐 token 转发**,
   只有一次最终 JSON; stream 模式才有逐 token SSE 开销。

**修正后结论: serve 比 gen 慢主要来自框架 emit 链, 但其中日志刷盘占比很小
(+3.5%)。gen 与 serve 的差距更多是"分段串行 + 网络", 而非框架本身。
框架优化的收益有限(\~几个 %), 不值得为日志改动破坏可观测性。**

### 优化实验: emit 移到 send\_x 之后(2026-08, 已实施)

**改动**: dist.c master decode 循环中, `emit` 回调从"forward 之前"移到
"`send_x` 之后"。让 worker 先收到 X 开工, master 再同步输出 token,
避免 emit 的同步 send/日志阻塞推迟 worker 启动。

```
优化前: recv_logits → 采样 → emit → forward → send_x
优化后: recv_logits → 采样 → forward → send_x → emit
```

**实测(跨机 r2, 8T, YLLM\_DISTTIMING)**:

| 指标             | 优化前    | 优化后             |
| -------------- | ------ | --------------- |
| decode         | 11.1   | **11.8(+7%)**   |
| master fwd 总和  | 529ms  | **472ms(-11%)** |
| master wait 总和 | 673ms  | 667ms           |
| worker wait    | \~49ms | \~46ms          |

**结论**:

- emit 后移让 worker 在 master 输出期间并行计算, decode +7%。
- 风险场景缓解: stream/慢客户端下, emit 阻塞不再推迟 worker 开工。
- 其余 emit 链优化(块读替代逐字节 recv、合并 send、SSE 异步转发)
  对当前 collect 模式收益有限, 留作 stream 高并发场景再评估。

### 对照实验: 完全去掉 emit 链(2026-08)

临时注释 `if (emit) emit(nxt, ctx)` 后两端重编实测(跨机 r2, 8T):

| 方案                  | decode   | master wait | master fwd |
| ------------------- | -------- | ----------- | ---------- |
| emit 在 forward 前(原) | 11.1     | 673ms       | 529ms      |
| emit 后移(已实施)        | 11.8     | 667ms       | 472ms      |
| **完全去掉 emit**       | **11.9** | 670ms       | 472ms      |

**结论: 去掉 emit 仅比后移多 \~1%(0.1 tok/s)** → emit 链本身成本极低
(collect 模式下), +7% 提升来自**排列顺序**(先发 X 让 worker 开工),
而非省掉 emit 的工作。**emit 链不是瓶颈**, 保留后移版本即可。

## 多 rank 段数对比(跨机, 8T, 2026-08)

### 层分配(25 层, 按字节均衡切)

| ranks | 拓扑        | rank0       | rank1        | rank2       | rank3       |
| ----- | --------- | ----------- | ------------ | ----------- | ----------- |
| 2     | Win1+Lin1 | \[0,12) 12层 | \[12,25) 13层 | —           | —           |
| 3     | Win1+Lin2 | \[0,9) 9层   | \[9,16) 7层   | \[16,25) 9层 | —           |
| 4     | Win2+Lin2 | \[0,7) 7层   | \[7,12) 5层   | \[12,18) 6层 | \[18,25) 7层 |

### 性能对比

| 配置      | prefill | decode   | rank0 fwd/token | rank0 wait/token |
| ------- | ------- | -------- | --------------- | ---------------- |
| ranks=2 | 35.8    | 11.4     | 33ms            | 38ms             |
| ranks=3 | 43.6    | **13.0** | 21.5ms          | 43ms             |
| ranks=4 | 39.1    | 11.9     | 17ms            | 52ms             |

### 关键发现

1. **每段层数越少 → fwd 越短**(33→21.5→17ms), 这是分段收益。
2. **但 wait 随段数增加**(38→43→52ms): 跨机串行跳数增多,
   master 要等全部下游算完 + 网络逐跳往返。
3. **ranks=3 是甜点**: 每段够小(7-9 层)且跳数不多, decode 13.0 最佳。
4. **ranks=4 反而降**: 多一跳的串行等待盖过了 fwd 减少 → 11.9。
5. **结论: 小模型(1.1B)跨机 PP decode 收益在 3 段封顶**。等待主导后,
   加段无益——这是串行链 + 网络跳数的物理规律。

### 线程调优(多 rank 下)

| 配置                  | decode | 结论                          |
| ------------------- | ------ | --------------------------- |
| ranks=3, 全 8T       | 13.0   | 基线                          |
| ranks=3, rank0=12T  | 12.5   | Windows 12核跑9层, 12线程争抢, 反而降 |
| ranks=3, worker=16T | 11.5   | 算完更早=更早等待, 降                |

**结论: 多 rank 下 8 线程最优, 加线程有害**——decode 是等待主导,
线程多只是让每段更快算完并更早进入等待, 总周期由最慢段+跳数决定。

### 工具: bench\_cross.py 支持 Windows 多 rank

`WIN_RANKS` 环境变量控制 Windows 本地段数(默认 1):

```
WIN_RANKS=2 RANKS=4 THREADS=8 python3 tools/bench_cross.py
```

- supervisor 只自动拉 rank0(local=1)
- rank1..win\_ranks-1 在 Windows 手动拉起(带正确 peers 才能连远端)
- rank win\_ranks..ranks-1 在 Linux ssh 拉起
- 全链路依赖: LEASE 按端口排序生成 peers → INFER 下发 → rank0 用正确 peers 连下游

## Bug 修复: HTTP 采样参数未透传导致单 rank 慢 \~10%(2026-08)

### 症状

单机 serve r1(13.7-14.3 tok/s)比 gen(15.5-16.5)慢 \~10%, 且逐项计时显示
**r1 的 sample 耗时 64ms/16tok(4ms/token), 而 gen 仅 0ms**。

### 根因

HTTP 请求的 `temperature=0` **没有透传链路**:

```
router_http: 只解析 max_tokens, 忽略 temperature/top_p  ✗
  → router_infer_sess: args 不含 temp
  → server forward_infer_sess: fargs 不含 temp
  → rank: 用启动默认 temp=1.0, top_p=0.9(config.h:105)
    → engine_sample 走 softmax+expf+qsort(32000 元素)慢路径
```

gen 直跑用 `--temp 0` → argmax 快路径, 采样近乎零成本。
r1 的 rank 用默认 temp=1.0 → 每 token 4ms 做 softmax+qsort。

### 修复(已实施)

沿 HTTP→router→server→rank 四层透传 temp/top\_p:

1. **router\_http.c**: 解析 body 的 `temperature`/`top_p`
2. **router.c**: `router_infer`/`router_infer_sess` 签名加 temp/top\_p, args 带 `temp=... top_p=...`
3. **server.c**: `forward_infer_sess` 从 args 提取 temp/top\_p 追加到 fargs
   (`forward_infer` 原本透传 args, 无需改)
4. **rank.c**: `handle_infer` 用 `proto_get("temp"/"top_p")` 覆盖 `r->temp`/`r->top_p`

### 实测(单机 r1, 8T, 同 50-token prompt + 16 decode)

| 指标         | 修复前       | <br /> | 修复后      |
| ---------- | --------- | :----- | -------- |
| r1 decode  | 13.7-14.3 | <br /> | **14.4** |
| sample 总耗时 | 64ms      | <br /> | **0ms**  |
| fwd 总耗时    | 1104ms    | <br /> | 1034ms   |

- sample 归零(走 argmax 快路径), decode +3\~5%。
- 剩余 \~4% 差距(fwd 1034 vs gen 967)为框架逐 token 转发/会话记账开销。
- **此前所有 emit/日志优化无效的原因就是它: 瓶颈根本不是框架, 而是采样参数 bug。**

### 附带发现

- gen 与 r1 的引擎 forward 性能一致(fwd 967 vs 1034, 差 \~7% 为框架开销)。
- emit 链本身只有 1ms/16tok, 不是瓶颈。
- 合并 send、关日志、小缓冲均无效(+1%), 已回退或保留无害优化。

