# platform/ios

## 目标

- 同一套 `inference/*.c` → `libyllm.a`（先 **CPU + NEON**）
- GPU：`--device vulkan` 经 **MoltenVK**（Vulkan → Metal），不另写 Metal 引擎

## 构建（草案）

需 macOS + Xcode / CMake iOS toolchain。示例：

```bash
cmake -S platform/ios -B build/ios \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build build/ios
```

MoltenVK：通过 CocoaPods / XCFramework 链入，定义 `YLLM_VULKAN=1`。

当前仓库仅保留说明与占位；完整 `CMakeLists.txt` 随 Android CPU 冒烟稳定后补齐。
