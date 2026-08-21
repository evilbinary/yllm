# inference/

LLM 引擎核与设备后端（与 `serve/` 服务层分离）。

```text
inference/
├── include/     # 公共 API：yllm.h, device.h, matvec.h, llf.h, convert.h, …
├── core/        # engine, matvec, tokenizer, llf, cache, dist, platform, log
├── convert/     # GGUF / Safetensors → LLF
├── device/      # device_cpu / device_cuda / device_vulkan 入口
├── cuda/        # CUDA fwd + kernels（私有头）
└── vulkan/      # Vulkan load/compute/fwd + shaders/
```

构建：根目录 `Makefile` 使用 `-Iinference/include -Iinference/cuda -Iinference/vulkan`。
SPIR-V：`YLLM_SHADER_DIR` 或 `inference/vulkan/shaders/`。
