#!/usr/bin/env python3
"""用 vLLM 运行由 gguf2safetensors.py 转换的模型目录。

用法:
    python3 tools/run_vllm.py models/Qwen3-8B-st
    python3 tools/run_vllm.py models/Qwen3-8B-st --prompt "Hello! What is 2+2?"
    python3 tools/run_vllm.py models/Qwen3-8B-st -p "你好" -n 128 -t 0.7

依赖: pip install vllm
说明:
    - 默认 CPU 后端(gpu_memory_utilization 实为内存占用比例, 机器内存不足时调小)
    - 目录需含: safetensors + config.json + tokenizer.json
      (由 tools/gguf2safetensors.py 生成, 已有权重时加 --config-only)
"""
import argparse

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser(description="vLLM 运行 yllm 转换的 safetensors 模型")
    ap.add_argument("model", help="模型目录(含 safetensors + config.json + tokenizer.json)")
    ap.add_argument("-p", "--prompt", default="Hello! What is 2+2? Please answer briefly.",
                    help="输入 prompt(默认 'Hello! What is 2+2? ...')")
    ap.add_argument("-n", "--max-tokens", type=int, default=64, help="最大生成 token 数(默认 64)")
    ap.add_argument("-t", "--temperature", type=float, default=0.0, help="采样温度(默认 0 = 贪心)")
    ap.add_argument("-m", "--max-model-len", type=int, default=4096, help="max_model_len(默认 4096)")
    ap.add_argument("-g", "--gpu-memory-utilization", type=float, default=0.6,
                    help="内存占用比例, CPU 后端为物理内存比例(默认 0.6, 不足时调小)")
    ap.add_argument("-d", "--dtype", default="bfloat16", help="权重精度(默认 bfloat16, 老硬件可换 float32)")
    args = ap.parse_args()

    llm = LLM(model=args.model,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len,
              enforce_eager=True,
              dtype=args.dtype)
    outs = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens,
                                                      temperature=args.temperature))
    print(f"\n>>> {args.model}\n>>> prompt: {args.prompt}\n")
    print(outs[0].outputs[0].text)


if __name__ == "__main__":
    main()
