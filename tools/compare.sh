#!/usr/bin/env bash
# yllm vs picolm 性能对比基准
#
# 场景:
#   1) 无 AVX2(标量):  yllm-scalar      vs picolm-scalar
#   2) 有 AVX2:        yllm-avx2        vs picolm-avx2
#
# 用法:
#   bash bench/compare.sh                 # 用下方默认路径
#   NTOKENS=128 THREADS="1 4 8" bash bench/compare.sh
#
# 说明:
#   - yllm 与 picolm 均取"纯 decode" tok/s
#   - picolm 标量版必须显式 `make picolm` 构建(裸 `make` 会走 native=-march=native,
#     变成 AVX2 版);AVX2 版用 `make native-avx2`
#   - 每个配置先跑一次 warm-up 再计时

set -u
cd "$(dirname "$0")/.."   # 仓库根目录

# ---- 配置(可按需覆盖) ----
GGUF=${GGUF:-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}
LLF=${LLF:-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf}
VOCAB=${VOCAB:-models/tinyllama.vocab.txt}
PROMPT=${PROMPT:-"Once upon a time"}
NTOKENS=${NTOKENS:-64}
THREADS=${THREADS:-"1 4 8 16"}

YLLM_SCALAR=${YLLM_SCALAR:-./build/yllm}
YLLM_AVX2=${YLLM_AVX2:-./build/avx2/yllm}
PICOLM_SCALAR=${PICOLM_SCALAR:-/tmp/opencode/picolm_scalar}
PICOLM_AVX2=${PICOLM_AVX2:-/tmp/opencode/picolm_avx2}

# ---- 单次测量 ----
# yllm: 返回纯 decode tok/s
y_tok() {  # $1=bin  $2=threads
    OMP_NUM_THREADS=$2 "$1" gen --model "$LLF" --vocab "$VOCAB" \
        --prompt "$PROMPT" --tokens "$NTOKENS" --temp 0 2>&1 \
        | sed -n 's/.*decode:[[:space:]]*[0-9]* tokens in .* (\([0-9.]*\) tok.*/\1/p'
}
# picolm: 返回纯 decode tok/s
p_tok() {  # $1=bin  $2=threads
    "$1" "$GGUF" -p "$PROMPT" -n "$NTOKENS" -t 0 -j "$2" 2>&1 \
        | sed -n 's/.*Generation: .* (\([0-9.]*\) tok.*/\1/p'
}

run_pair() {  # $1=标题  $2=yllm bin  $3=picolm bin
    echo "=== $1 ==="
    printf '%-7s | %-12s | %-12s\n' threads yllm picolm
    for t in $THREADS; do
        # warm-up
        OMP_NUM_THREADS=$t "$2" gen --model "$LLF" --vocab "$VOCAB" \
            --prompt "$PROMPT" --tokens 8 --temp 0 >/dev/null 2>&1
        "$3" "$GGUF" -p "$PROMPT" -n 8 -t 0 -j "$t" >/dev/null 2>&1
        # measure
        local y p
        y=$(y_tok "$2" "$t")
        p=$(p_tok "$3" "$t")
        printf '%-7s | %-12s | %-12s\n' "$t" "$y" "$p"
    done
    echo
}

# ---- 两个对比场景 ----
run_pair "无 AVX2 (标量)" "$YLLM_SCALAR" "$PICOLM_SCALAR"
run_pair "有 AVX2"          "$YLLM_AVX2"  "$PICOLM_AVX2"
