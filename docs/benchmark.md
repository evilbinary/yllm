# 性能对比: yllm vs picolm (TinyLlama-1.1B Q4_K_M)

同模型、同 prompt("Once upon a time")、纯 decode(生成 64 token, greedy, 预热后计时)。
脚本:`tests/compare.sh`(可复跑, `THREADS=... NTOKENS=...` 可覆盖)。

- yllm 读数: 二进制 `gen` 输出的 `decode:` 行(纯 decode tok/s)
- picolm 读数: `Generation` 行(纯 decode tok/s)
- picolm 构建: 标量 `make picolm`(基础 CFLAGS, 无 SIMD);AVX2 `make native-avx2`(`-O3 -mavx2 -mfma -mpopcnt`)

> ⚠️ 注意: picolm 的 Makefile 中 `native` 是**第一个 target**,裸 `make` 会自动带
> `-march=native`(本机即 AVX2)。要得到真标量版必须显式 `make picolm`。
> 早期文档里"picolm 标量 5.0 tok/s"是误把 AVX2 版当成了标量。

## 无 AVX2(标量)

| 线程数 | yllm-scalar | picolm-scalar |
|---|---|---|
| 1 | 1.0 | 1.5 |
| 4 | 3.2 | 4.7 |
| 8 | 5.5 | 5.0 |
| 16 | 4.9 | 6.6 |

## 有 AVX2

| 线程数 | yllm-avx2 | picolm-avx2 |
|---|---|---|
| 1 | 4.6 | 4.8 |
| 4 | 14.2 | 14.2 |
| 8 | **22.8** | 15.2 |
| 16 | 16.2 | 17.2 |

## 分析

**1) 为什么 yllm 标量曾被 picolm"标量"甩开 6 倍?**
picolm 裸 `make` 走 `native` target(`-march=native`)= AVX2 版,不是标量;
两者对比从一开始就不公平。真正标量对标的差距只有 ~1.4x(见上表)。

**2) 标量内核的优化(本次改动)**
yllm 标量 Q4K/Q6K 原实现是单累加器串行依赖链 + 逐元素分支解 nibble,
gcc 在 `-O2` 下不向量化 fp 归约(`-ffast-math` 缺失),单线程仅 ~0.5 tok/s。
两轮优化:
- **4 独立累加器 + 同字节解低/高 nibble**(消除分支、给足 ILP):0.5 → 1.0 tok/s
- **qx 与 x 分开累加、组末合并**(Q4K 每元素 4 op→2 op;Q6K 按 16 子组累加再乘 d·s):
  8 线程 5.0 → 5.5,16 线程 4.2 → 4.9(追平并反超 picolm 标量 8 线程)

**3) 为什么 picolm 标量 ≈ picolm AVX2(旧困惑)**
两个"版本"其实都是 AVX2 构建(见 ⚠️),所以性能相同。

**4) 有 AVX2 时 yllm 8 线程 21.9 vs picolm 15.4**
yllm 的 AVX2 Q4K 内核对权重行连续流式读取 + 寄存器内反量化累加,FLOP/字节 效率更高;
两者 16 线程都被 DRAM 带宽封顶(~10-13GB/s)。

## 结论

- 有 AVX2:yllm 在 8 线程明显领先(+42%),是主力配置。
- 无 AVX2:两者标量接近(1 线程差 ~30%,8 线程持平);如需进一步拉开可用 SSE2 intrinsics。
- decode 最终都被 DRAM 带宽封顶:单核 ~3-4GB/s,整机共享 ~10-13GB/s。
