#!/bin/bash
set -e
cd ~/yllm
mkdir -p models logs
cd models
echo "[$(date)] download GGUF..."
curl -L --fail --retry 5 --retry-delay 5 -C - \
  -o gemma-4-E2B-it-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
ls -lh gemma-4-E2B-it-Q4_K_M.gguf

echo "[$(date)] convert to LLF..."
cd ~/yllm
./build/avx2/yllm convert --gguf models/gemma-4-E2B-it-Q4_K_M.gguf \
  --out models/gemma-4-E2B-it-Q4_K_M.llf \
  --vocab models/gemma4.vocab.txt --seq 2048
ls -lh models/gemma-4-E2B-it-Q4_K_M.llf

echo "[$(date)] remove GGUF to free disk..."
rm -f models/gemma-4-E2B-it-Q4_K_M.gguf

echo "[$(date)] download mmproj..."
curl -L --fail --retry 5 --retry-delay 5 -C - \
  -o models/mmproj-gemma-4-E2B-F16.gguf \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf"
ls -lh models/

echo "[$(date)] DONE"
df -h . | tail -1
