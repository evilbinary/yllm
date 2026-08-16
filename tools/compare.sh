#!/usr/bin/env bash
# yllm vs picolm 性能对比基准(prefill + decode)
#
# 场景:
#   1) 无 AVX2(标量):  yllm-scalar      vs picolm-scalar
#   2) 有 AVX2:        yllm-avx2        vs picolm-avx2
#
# 用法:
#   bash tools/compare.sh                 # 用下方默认路径
#   NTOKENS=128 THREADS="1 4 8" bash tools/compare.sh
#
# 说明:
#   - yllm 取 `gen` 输出的 prefill/decode 两行; picolm 取 Prefill/Generation 两行
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
# yllm: $1=bin  $2=threads; 输出 "prefill_tok decode_tok" 两个值
y_vals() {  # $1=bin  $2=threads
    OMP_NUM_THREADS=$2 "$1" gen --model "$LLF" --vocab "$VOCAB" \
        --prompt "$PROMPT" --tokens "$NTOKENS" --temp 0 2>&1 \
        | sed -n 's/.*prefill:[[:space:]]*[0-9]* tokens in .* (\([0-9.]*\) tok.*/\1/p; s/.*decode:[[:space:]]*[0-9]* tokens in .* (\([0-9.]*\) tok.*/\1/p'
}
# picolm: $1=bin  $2=threads; 输出 "prefill_tok decode_tok" 两个值
p_vals() {  # $1=bin  $2=threads
    "$1" "$GGUF" -p "$PROMPT" -n "$NTOKENS" -t 0 -j "$2" 2>&1 \
        | sed -n 's/.*Prefill:[[:space:]]*[0-9]* tokens in .* (\([0-9.]*\) tok.*/\1/p; s/.*Generation: .* (\([0-9.]*\) tok.*/\1/p'
}

run_pair() {  # $1=标题  $2=yllm bin  $3=picolm bin
    echo "=== $1 ==="
    printf '%-7s | %-13s | %-13s | %-13s | %-13s\n' threads yllm-prefill yllm-decode picolm-prefill picolm-decode
    for t in $THREADS; do
        # warm-up
        OMP_NUM_THREADS=$t "$2" gen --model "$LLF" --vocab "$VOCAB" \
            --prompt "$PROMPT" --tokens 8 --temp 0 >/dev/null 2>&1
        "$3" "$GGUF" -p "$PROMPT" -n 8 -t 0 -j "$t" >/dev/null 2>&1
        # measure
        local yp yd pp pd
        yp=$(y_vals "$2" "$t" | sed -n 1p)
        yd=$(y_vals "$2" "$t" | sed -n 2p)
        pp=$(p_vals "$3" "$t" | sed -n 1p)
        pd=$(p_vals "$3" "$t" | sed -n 2p)
        printf '%-7s | %-13s | %-13s | %-13s | %-13s\n' "$t" "$yp" "$yd" "$pp" "$pd"
    done
    echo
}

# ---- 两个对比场景 ----
run_pair "无 AVX2 (标量)" "$YLLM_SCALAR" "$PICOLM_SCALAR"
run_pair "有 AVX2"          "$YLLM_AVX2"  "$PICOLM_AVX2"
