# platform/pc

桌面构建仍以仓库根目录 [`Makefile`](../../Makefile) 为主：

```bash
# CPU
make avx2

# CUDA（NVIDIA）
make cuda

# Vulkan（跨端 Device；无 loader 时为 host-shim）
make vulkan
# 或: make avx2 YLLM_VULKAN=1
./build/avx2-vulkan/yllm gen --device vulkan --model ... --temp 0
```

`YLLM_CUDA` 与 `YLLM_VULKAN` 可分别编译进不同产物目录，避免混用 `.o`。
