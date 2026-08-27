# platform/pc

桌面构建仍以仓库根目录 [`Makefile`](../../Makefile) 为主：

```bash
# CPU
make avx2

# CUDA（NVIDIA）
make cuda

# Vulkan（跨端 Device；无 loader 时为 host-shim）
make vulkan
./build/vulkan/yllm gen --device vulkan --model ... --temp 0
# AVX2 + Vulkan: make vulkan-avx2 → build/avx2-vulkan/yllm
```

`YLLM_CUDA` 与 `YLLM_VULKAN` 可分别编译进不同产物目录，避免混用 `.o`。
