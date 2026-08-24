# platform/android

NDK 构建 yllm 引擎。仓库根目录：

```bash
make android          # arm64 + Vulkan → build/android/yllm + libyllm.so
make android-cpu      # 不编 Vulkan → build/android-cpu/
```

命令入口与 PC 相同：`./yllm gen|chat|convert|check|...`（`main.c` + `serve/`）。

未设 `ANDROID_NDK` 时会尝试 `E:/soft/android-ndk-*` 和 `%LOCALAPPDATA%/Android/Sdk/ndk`。

## 依赖

- Android NDK r25+
- CMake 3.22+（可用 SDK 自带 `cmake/3.22.1`）

## 手动 CMake

```bash
cmake -G Ninja -S platform/android -B build/android \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_NDK=$ANDROID_NDK \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DYLLM_VULKAN=ON
cmake --build build/android -j
```

## 设备冒烟

```bash
# Termux 示例
scp -P 8022 build/android/yllm build/android/libyllm.so user@phone:~/yllm-android/
scp -P 8022 build/android/shaders/*.spv user@phone:~/yllm-android/shaders/
# 设备上:
#   export LD_LIBRARY_PATH=. YLLM_SHADER_DIR=$PWD/shaders
#   ./yllm gen --model ... --vocab ... --prompt "Once upon a time" --tokens 16 --temp 0 --device vulkan
```

CPU 会自动绑到高频核(big.LITTLE)，`YLLM_NO_AFFINITY=1` 可关闭。
