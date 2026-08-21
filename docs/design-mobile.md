# yllm 跨平台与 Vulkan

版本：v0.7 ｜ 关联：`design-gpu-inference.md`、`platform/`

## 1. 目录

| 路径 | 职责 |
|------|------|
| `inference/` | 引擎核 + `device_cpu` / `device_cuda` / `device_vulkan` + `vulkan_load` |
| `inference/shaders/` | GLSL → SPIR-V（预编译进仓库） |
| `platform/android/` | NDK CMake → `libyllm.so` + `yllm_gen` |
| `platform/ios/` | 静态库 CMake + MoltenVK 说明 |
| `platform/pc/` | 桌面入口（根 `Makefile`） |
| `platform/web/` | 预留 WASM/WebGPU |

## 2. Device

```text
--device cpu | cuda | vulkan
```

- **vulkan**：动态加载 `vulkan-1.dll` / `libvulkan.so` / MoltenVK；创建 **compute 队列** `VkDevice`。
  - 成功 → `DEV_MODE_VULKAN`（`mode=native`）：
    - `rmsnorm.spv`：块内 F32/F16 RMSNorm
    - `gemv_q4k.spv`：块内 Q4_K；**load 时整包常驻**（`resident=1`）
    - **fused**：`rmsnorm+QKV`；整段 FFN `rmsnorm+gate+up+swiglu+down`（`swi=1`）；O 仍单次 gemv；attn/rope 仍 CPU
  - 失败 → `DEV_MODE_VULKAN_HOST`
  - 强制 shim：`make vulkan YLLM_VULKAN_HOST=1`
  - SPIR-V：`YLLM_SHADER_DIR` 或 `inference/shaders/`

## 3. 构建

### PC

```bash
# 需 VULKAN_SDK（头文件）；运行时动态加载，无需链 vulkan-1.lib
set VULKAN_SDK=...   # Windows
# 改 .comp 后: glslc -fshader-stage=compute inference/shaders/rmsnorm.comp -o inference/shaders/rmsnorm.spv
make vulkan
./build/avx2-vulkan/yllm gen --device vulkan ... --temp 0
```

### Android（本机已验证 Ninja + NDK r27）

```bash
cmake -G Ninja -S platform/android -B build/android \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DANDROID_NDK=$ANDROID_NDK \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=$ANDROID_SDK/cmake/*/bin/ninja
cmake --build build/android -j
# 产物: build/android/libyllm.so  build/android/yllm_gen
```

可选 `-DYLLM_VULKAN=ON`（需把 `vulkan_compute.c` / `.spv` 打进 APK assets）。

### iOS

见 `platform/ios/`（CMake 骨架已加）。

## 4. 下一步

1. rope / attn_decode compute；lm_head  
2. 激活跨层常驻  
3. `adb` 冒烟；MoltenVK iOS
