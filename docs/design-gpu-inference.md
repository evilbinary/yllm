# yllm GPU 推理设计方案

版本：v0.4 ｜ 状态：P3 核心项已落地 ｜ 关联：`design-mmap-layer-streaming.md`

## 1. 目标与边界

| 要做 | 暂不做 |
|------|--------|
| 单卡 CUDA 跑 TinyLlama / Qwen2.5-7B 量级 | 连续多请求 batch（vLLM 式） |
| 保留 LLF 量化权重（Q4_K 等） | Tensor Parallel |
| Prefill B≤64 / Decode B=1 与现网一致 | OpenCL / Vulkan 双栈 |
| 多卡复用现有 Pipeline Parallel（`dist.c`） | 立刻改 router / 会话协议 |

**原则：** serve / dist / sample 仍只调 `engine_forward*`；设备细节关在 `Device` + `fwd_block(_batch)` 里。

## 2. 总体架构

```text
┌─────────────────────────────────────────────────────────┐
│  serve (rank / server / router)  ← 协议面不变            │
│    engine_forward / prefill / sample / dist_gen         │
└───────────────────────┬─────────────────────────────────┘
                        │ Engine API
┌───────────────────────▼─────────────────────────────────┐
│  Device (cpu | cuda)                                    │
│  - load_weights：本 rank 权重上设备（常驻或建流式通道）   │
│  - prefetch_layer / release_layer（大模型可选）          │
│  - 挂接 fwd_block / fwd_block_batch                     │
│  - KV + activations 在设备上（CUDA）；CPU 即现有堆缓冲   │
└─────────────────────────────────────────────────────────┘
│  host: LLF mmap（只读视图仍保留）                        │
```

插入点优先级：

1. `Engine.fwd_block` / `fwd_block_batch` — 整层图上 GPU  
2. `matvec` 内核族 — 先 `matmul` / `matmul_batch`  
3. KV → 设备分配（与 mmap 流式文档一致：KV 不跟 mmap）

## 3. 命名约定

| 符号 | 职责 |
|------|------|
| **`load_weights`** | 把本 Engine 要用的权准备到设备（常驻拷贝，或建双缓冲通道）。**不用** `upload_weights` |
| `prefetch_layer` | （可选）算前预取下一层 |
| `release_layer` | （可选）流式时释放上一层设备侧权重 |
| `fwd_block` / `fwd_block_batch` | 算，不负责拷权重 |
| `engine_bind_device` | 创建/切换 Device，并调用 `load_weights` |

## 4. 重点数据结构

```c
typedef enum { DEV_CPU = 0, DEV_CUDA = 1 } DeviceKind;
typedef enum {
    DEV_MODE_CPU = 0,        /* 主机 mmap + CPU */
    DEV_MODE_CUDA_HOST = 1,  /* --device cuda 但 host-shim */
    DEV_MODE_CUDA = 2        /* 真 CUDA kernel / cublas */
} DeviceMode;

/* Engine 增量 */
Device*    dev;
void*      d_kv;
void*      w_dev;
int        weights_ready;
DeviceMode device_mode;   /* load_weights / bind 置位; 前向用此判断路径 */
```

`DeviceKind` = 绑定的后端；`device_mode` = 实际怎么算。

## 5. `load_weights` 语义

- **时机：** `engine_init` 末尾默认绑 CPU 并调用；`dist_split_layers` 之后再调一次（本 rank 层区间已定）。  
- **范围：** 只准备 `[layer_begin, layer_end)`。  
- **CPU：** 空操作（或别名 host 指针），`weights_ready = 1`。  
- **CUDA（后续）：** 按层 H2D；小模型全常驻；大模型可只建双缓冲，真正拷贝进 `prefetch_layer`。

## 6. 初始化流水线

```text
engine_init
  └─ wmap_open / 分配 host kv·x·pb …
  └─ 按 arch 挂 CPU fwd_block
  └─ engine_bind_device(DEV_CPU) → load_weights

[可选] dist_split_layers → load_weights 再入（刷新本段）

[可选] engine_bind_device(DEV_CUDA, gpu_id) → load_weights
        └─ 替换 fwd_block 为 CUDA 实现
```

CLI / 配置：

- `--device cpu|cuda`（默认 `cpu`）  
- `--gpu N`（默认 `0`）  
- `serve.yaml`：顶层或模型级 `device` / `gpu`

## 7. 前向与采样边界

- 控制流：`engine_forward` / `engine_forward_prefill` **不改**；只换函数指针实现。  
- 采样：lm_head 后 **logits D2H 一次**，仍走现有 `engine_sample`。  
- 会话 KV 落盘：`.rN.kv` 格式不变；CUDA 路径 save/load 时 D2H / H2D。

## 8. 多卡

沿用 PP：一 rank 一进程一 GPU；激活帧 transport 后期由 TCP 换 peer copy。不上 TP。

## 9. 文件布局

```text
inference/device.h
inference/device_cpu.c       # CPU: load_weights 空操作
inference/device_cuda.c      # CUDA: shim / GPU FP32 解量化上卡
inference/cuda_fwd.c         # fwd_block + embed/norm/head; 激活常驻设备
inference/cuda_kernels.cu    # rmsnorm/rope/attn/swiglu + cublasSgemv
inference/cuda_kernels.h
inference/cuda_ctx.h
docs/design-gpu-inference.md
```

## 10. 分阶段

| 阶段 | 内容 |
|------|------|
| **P0** | `Device` + CPU `load_weights`；`engine_bind_device`；rank/config `--device` |
| **P1** | `device_cuda` + raw blob `load_weights`；host-shim 可测通路径 |
| **P2** | Decode：线性权上卡 + cublas + 小 kernel；`device_mode`；激活常驻 `d_x`；GPU 逐 token prefill |
| **P3（进行中）** | ✅ FP16 权；✅ GPU batch prefill；✅ 原生 Q4_K；✅ **Flash 风格 attn**（online-softmax，无 O(seq) score 缓冲） |

### 构建

```bash
make cuda
make gen-cuda CHAT_PROMPT=Hi CHAT_TOKENS=16
# 权格式: GPU_WEIGHTS=fp16|q4k|auto (默认 auto=原生 Q4_K)
# make gen-cuda GPU_WEIGHTS=fp16
# 数值对齐: --temp 0 对比 --device cpu
# 无 nvcc: 自动 host-shim; 强制 shim: make cuda YLLM_CUDA_HOST=1
```

真 CUDA 产物：`build/avx2-cuda/yllm`（链 `-lcudart -lcublas`，编 `cuda_kernels.cu`）。

`load_weights`（GPU）：`--gpu-weights auto|q4k|fp16`（默认 auto）。`auto`/`q4k`：`DT_Q4K` 原生上卡，其余解 FP16；`fp16`：全部线性权解量化（更快、更费显存）。ARCH_QWEN35 暂拒。

## 11. 与 mmap 流式文档的关系

- Host mmap 仍是权重权威视图与冷启动手段。  
- GPU 路径：`load_weights` 从 mmap 视图拷到设备（或双缓冲）；**KV 始终独立分配**（见 `design-mmap-layer-streaming.md` §3.4）。  
- CPU `budget` / `ws_release` 逻辑在 `DEV_CPU` 下不变；`DEV_CUDA` 的显存水位另计（P1+）。
