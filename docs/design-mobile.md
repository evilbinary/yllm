# yllm 跨平台与 Vulkan

版本：v0.1 ｜ 关联：`design-gpu-inference.md`、`platform/`

## 1. 目录

| 路径 | 职责 |
|------|------|
| `inference/` | 可移植引擎核 + `device_cpu` / `device_cuda` / `device_vulkan` |
| `platform/android/` | NDK CMake → `libyllm.so` + `yllm_gen` |
| `platform/ios/` | 静态库 + MoltenVK 说明（骨架） |
| `platform/pc/` | 桌面入口说明（根 `Makefile`） |
| `platform/web/` | 预留 WASM/WebGPU |

## 2. Device

```text
--device cpu | cuda | vulkan
```

- **cpu**：全平台。
- **cuda**：仅 PC + NVIDIA（现有路径）。
- **vulkan**：Android / iOS(MoltenVK) / PC 共用 `DEV_VULKAN`。
  - **P0（当前）**：host-shim，绑定成功后仍走 CPU `fwd_block`（验证打包与 CLI）。
  - **后续**：VkBuffer + SPIR-V compute（Q4/FP16 gemv 等），镜像 CUDA 分层。

## 3. 构建

### PC

```bash
make avx2                    # CPU
make cuda                    # CUDA → build/avx2-cuda/
make vulkan                  # Vulkan P0 → build/avx2-vulkan/
./build/avx2-vulkan/yllm gen --device vulkan ... --temp 0
```

### Android

见 [`platform/android/README.md`](../platform/android/README.md)（需 `ANDROID_NDK`）。

### iOS

见 [`platform/ios/README.md`](../platform/ios/README.md)。

## 4. 原则

- `engine_forward*` 不感知 Vulkan/CUDA；只换 `Device` + `fwd_block`。
- SPIR-V 预编译放 `inference/shaders/`（待补），三端共用。
- 移动端默认不上 router/supervisor；PP 多机为后续。
