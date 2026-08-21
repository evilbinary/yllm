# yllm GPU 推理设计方案

版本：v0.5 ｜ 状态：P3 + 混合路径已落地 ｜ 关联：`design-mmap-layer-streaming.md`

## 1. 目标与边界

| 要做 | 暂不做 |
|------|--------|
| 单卡 CUDA 跑 TinyLlama / Qwen2.5-7B 量级 | 连续多请求 batch（vLLM 式） |
| 保留 LLF 量化权重（Q4_K 等） | Tensor Parallel |
| Prefill B≤64 / Decode B=1 与现网一致 | 立刻改 router / 会话协议 |
| 多卡复用现有 Pipeline Parallel（`dist.c`） | OpenCL 第二栈 |
| GPU↔CPU 混合（PP / 单进程层切 / 权流式） | 双缓冲异步 prefetch |
| 跨端 Vulkan（见 `design-mobile.md`） | Web/WASM（`platform/web` 预留） |

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
│  - prefetch_layer / release_layer（--gpu-stream）       │
│  - 挂接 fwd_block / fwd_block_batch                     │
│  - KV + activations 在设备上（CUDA）；CPU 即现有堆缓冲   │
└─────────────────────────────────────────────────────────┘
│  host: LLF mmap（只读视图仍保留）                        │
```

## 3. 命名约定

| 符号 | 职责 |
|------|------|
| **`load_weights`** | 把本 Engine 要用的权准备到设备（常驻拷贝，或建单层缓冲+host 打包）。**不用** `upload_weights` |
| `prefetch_layer` | 流式：把一层权 H2D 到单层缓冲 |
| `release_layer` | 流式：作废当前驻留层设备偏移 |
| `fwd_block` / `fwd_block_batch` | 算，不负责拷权重（流式时入口先 prefetch） |
| `engine_bind_device` | 创建/切换 Device，并调用 `load_weights` |
| `cuda_mark_x_host` | host 已写 `e->x`：清 `x_on_dev`，下次 fwd 再 H2D（PP 收包必用） |

## 4. 重点数据结构

```c
typedef enum { DEV_CPU = 0, DEV_CUDA = 1 } DeviceKind;
typedef enum {
    DEV_MODE_CPU = 0,
    DEV_MODE_CUDA_HOST = 1,
    DEV_MODE_CUDA = 2
} DeviceMode;

Device*    dev;
DeviceMode device_mode;
CudaWeightMode cuda_wmode;   /* auto|q4k|fp16 */
uint32_t   gpu_layer_end;    /* 0=全段 device; >0 → [begin,end) 中仅 < gpu_layer_end 走 CUDA */
int        cuda_stream_w;    /* 1=权 host 打包 + 按层 prefetch */
```

## 5. 混合路径

### 5.1 PP：GPU rank + CPU rank

- 各 rank 独立 `--device`；`engine_forward_range` 末尾 `x_out` 经 `cuda_sync_x_to_host`。
- **收激活后** `cuda_mark_x_host`（`dist` 写 `e->x` / `need_embed=0`），避免旧 `d_x` 被当成权威。
- GPU 中/末段批路径：`cuda_forward_batch_x`（失败则逐 token 回退）。

### 5.2 单进程层切

- `--gpu-layers N`：embed + 前 N 个 transformer block 在 GPU，其后层 CPU（KV 按层分设备）。
- `N >= n_blocks` 或省略 → 全 GPU（`gpu_layer_end=0`）。
- `load_weights` 只上卡 `[layer_begin, gpu_layer_end)`；切点处 `sync_x` 再进 CPU 块。

### 5.3 权流式（CPU 常驻 / GPU 算）

- `--gpu-stream 1`：host 保留打包 Q4/F16/F32；设备只开**单层峰值**缓冲。
- 每层 fwd 前 `prefetch_layer`；显存权 ≈ 一层，代价是层间同步 H2D。

## 6. CLI / 配置

- `--device cpu|cuda`、`--gpu N`、`--gpu-weights auto|q4k|fp16`
- `--gpu-layers N`、`--gpu-stream 0|1`
- `serve.yaml`：同名键

## 7. 分阶段

| 阶段 | 内容 |
|------|------|
| **P0–P2** | Device / host-shim / decode FP16·激活常驻 |
| **P3** | ✅ Q4_K；✅ batch prefill；✅ flash-style attn |
| **混合** | ✅ PP `x_on_dev`；✅ `--gpu-layers`；✅ `--gpu-stream` |

### 构建

```bash
make cuda
make gen-cuda CHAT_PROMPT=Hi CHAT_TOKENS=16
# GPU_WEIGHTS=fp16|q4k|auto
# 混合例: --device cuda --gpu-layers 8
# 流式例: --device cuda --gpu-stream 1
```

真 CUDA：`build/avx2-cuda/yllm`。ARCH_QWEN35 暂拒。
