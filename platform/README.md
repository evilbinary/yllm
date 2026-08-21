# platform/

跨平台打包与工具链目录。引擎核在仓库根目录 `inference/`。

| 目录 | 状态 | 说明 |
|------|------|------|
| [android/](android/) | 进行中 | NDK CMake → `libyllm` + `yllm_gen`（CPU；可选 Vulkan） |
| [ios/](ios/) | 骨架 | 静态库 + MoltenVK 说明 |
| [pc/](pc/) | 骨架 | 桌面构建入口（根 `Makefile` + Vulkan 开关） |
| [web/](web/) | 预留 | 未来 WASM / WebGPU，本阶段无代码 |

详见 [docs/design-mobile.md](../docs/design-mobile.md)。
