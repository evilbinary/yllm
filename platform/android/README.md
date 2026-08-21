# platform/android

NDK 构建 yllm 引擎（默认 **CPU**；可选 `YLLM_VULKAN=1`）。

## 依赖

- Android NDK r25+（`ANDROID_NDK` 环境变量）
- CMake 3.22+

## 构建

```bash
# 在仓库根目录
cmake -S platform/android -B build/android \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DANDROID_NDK=$ANDROID_NDK \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
cmake --build build/android -j

# 产物
#   build/android/libyllm.so
#   build/android/yllm_gen
```

## 设备冒烟

```bash
adb push build/android/yllm_gen /data/local/tmp/
adb push models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf /data/local/tmp/
adb push models/tinyllama.vocab.txt /data/local/tmp/
adb shell "cd /data/local/tmp && ./yllm_gen --model tinyllama-1.1b-chat-v1.0.Q4_K_M.llf \
  --vocab tinyllama.vocab.txt --prompt Hi --tokens 16 --temp 0 --device cpu --budget 512MB"
```
