# 并行共享权重解码（--parallel N）

## 思想

decode 阶段每出一个 token 都要把全部权重从内存搬进计算单元一次，
batch=1 时算力严重闲置——慢的根源不是算得慢，是搬得慢。
把 N 条互相独立的序列堆成 batch 做同一次前向：

```
[W] × [h₁ h₂ h₃ h₄]ᵀ → [y₁ y₂ y₃ y₄]
```

一次权重搬运产出 N 个 token，搬运成本被 N 摊薄。
本实现是"请求级/序列级凑无依赖 h 批次"，与投机解码（猜测级）、
MTP（解码头级）正交，可叠加。

## 使用

```sh
# 同一 prompt 复制 N 份(便于测加速比/一致性)
build/avx2/yllm.exe gen --model m.llf --vocab vocab.txt \
    --prompt "..." --tokens 48 --temp 0 --parallel 4

# N 条独立 prompt(每行一条)
build/avx2/yllm.exe gen --model m.llf --vocab vocab.txt \
    --parallel-prompt prompts.txt --parallel 4 --tokens 48
```

输出按分支打印，并给出 prefill/decode 耗时与聚合吞吐。

## 实现要点

| 组件 | 位置 | 说明 |
|---|---|---|
| 开关解析 | `main.c cmd_gen` | `--parallel N` / `--parallel-prompt FILE`；限制 CPU + llama/qwen |
| KV 槽位 | `engine.c engine_set_parallel_slots` | KV 扩容 N 份，`kv_slot_stride` 为槽位跨度；单槽位布局与原版完全一致 |
| 多槽位批量前向 | `arch/llama.c arch_llama_fwd_block_batch_slots` | 复用 `matmul_batch`（量化块反量化一次、B 序列共享——内核级摊薄）；RoPE 按各自 pos、KV 读写/注意力按各自 slot；qwen 经 `e->ops->qwen_rope` 复用 |
| 多序列生成循环 | `engine.c engine_generate_parallel` | 逐槽位 prefill（期间 e->kv 指向该 slot，复用现有全部路径）→ batch decode（活跃序列压缩、每序列独立采样/计数） |
| API | `yllm.h` | `engine_set_parallel_slots` / `engine_generate_parallel`（供 serve 层后续接入） |

## 实测（tinyllama-1.1B Q4K，AVX2 CPU，4 分支 × 48 tok，贪心）

```
串行:  48 tok / 2.4-2.9 s   ≈ 16-20 tok/s（单流）
并行: 192 tok / 7.5 s       ≈ 25.7 tok/s 聚合（1.3-1.6× 吞吐）
```

- 4 分支之间**逐 token 完全一致**（同 batch，确定性一致）
- CPU 上收益受算力上限约束（compute-bound），GPU 上差距更大；
  叠加 KV prefix 共享/早释放后收益更高（未实现，见下）

## 数值口径（重要）

batch 路径（`matmul_batch`，反量化块 + 浮点乘累加）与单流 GEMV 路径
（激活 int8 量化 + 整块点积）**舍入路径不同**，近平局 token 可能翻转，
长文本会与串行输出分叉。两者都是合法近似（int8 激活量化本身也是近似），
与 vLLM/llama.cpp 的 continuous batching 行为同理。需要严格逐位一致时
应使用同一 batch 布局做对比。

## 已知限制 / 后续工作

- 仅 llama/qwen 架构 + `--device cpu`（gemma4/qwen35 的 batch 算子未加
  槽位变体；GPU 路径 `Device.fwd_block_batch` 未接 slots）
- 与 `--ranks` 分布式、`--mtp` 互斥（显式检查）
- serve 层接入：`ServeConfig` 加字段 → rank 持多会话 slot 池，
  并发请求按 least-loaded 分槽（当前 rank 一次只服务一条序列）
- KV prefix 共享（相同前缀只存一份）与分支完成早释放尚未做
