#!/usr/bin/env python3
"""HuggingFace 模型下载工具(国内镜像)。

用法:
    python3 tools/download_model.py <repo_id> [local_dir]

内置:
    HF_ENDPOINT=https://hf-mirror.com(国内镜像, 可用 HF_MIRROR 覆盖)
    HF_HUB_DISABLE_XET=1(mirror 不支持 xet 协议, 强制走普通 http)

示例:
    python3 tools/download_model.py Qwen/Qwen3-8B models/Qwen3-8B-hf
    python3 tools/download_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/tinyllama-1.1b-chat-v1.0-hf
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_MIRROR", "https://hf-mirror.com"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    repo = sys.argv[1]
    local_dir = sys.argv[2] if len(sys.argv) > 2 else None
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo, local_dir=local_dir)
    print("done:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
