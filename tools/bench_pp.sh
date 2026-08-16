#!/usr/bin/env bash
# PP 分布式单机基准: 无分布式(ranks=1) vs worker rank 数(ranks=2/3/4)
#
# 场景: 50-token prompt + 16 decode, serve 模式 curl 端到端耗时;
#       每轮重启服务 + 清 sessions 保证全量 prefill; 预热后交错轮次。
# 指标: 总耗时 / prefill tok/s / decode tok/s / 总 tok/s
#
# 用法:
#   bash tools/bench_pp.sh [RANKS_LIST] [RUNS]
#   RANKS_LIST 默认 "1 2 3 4"(1 = 无分布式, 单 rank 全层直跑)
#   PROMPT / NTOKENS / MODEL 可环境变量覆盖
#
# 输出: 表格行(耗时 + prefill/decode/总 tok/s), 便于填入 docs/benchmark.md

set -u
cd "$(dirname "$0")/.."   # 仓库根目录

RANKS_LIST=${RANKS_LIST:-"1 2 3 4"}
RUNS=${RUNS:-3}
PROMPT=${PROMPT:-"A merchant named Chen walked through the market square at dawn, carrying two baskets of ripe peaches and humming a song he learned from his grandfather."}
NTOKENS=${NTOKENS:-16}
MODEL=${MODEL:-tinyllama}
CFG=serve.yaml
CFG_BAK=/tmp/opencode/serve.yaml.bench.bak
WARMUP=${WARMUP:-1}
LOG0=logs/tinyllama-rank-0.log

pkill -9 -f 'build/avx2/yllm (rank|serve|hub)' 2>/dev/null
sleep 1
cp "$CFG" "$CFG_BAK"

cleanup() {
    pkill -9 -f 'build/avx2/yllm (rank|serve|hub)' 2>/dev/null
    sleep 1
    mv "$CFG_BAK" "$CFG"
}
trap cleanup EXIT

# 交错轮次: 1 2 3 4 1 2 3 4 ...
SEQ=""
for i in $(seq 1 "$RUNS"); do
    for R in $RANKS_LIST; do SEQ="$SEQ $R"; done
done

declare -A TOT DECT PRET PRE_CNT DEC_CNT
for R in $SEQ; do
    sed -i "s/^ranks: [0-9]* .*本模型 rank 进程数/ranks: $R             # 本模型 rank 进程数/" "$CFG"
    pkill -9 -f 'build/avx2/yllm (rank|serve|hub)' 2>/dev/null
    sleep 1
    rm -rf sessions && mkdir sessions
    ./build/avx2/yllm ctl start >/dev/null 2>&1
    sleep 7
    if [ "$WARMUP" = "1" ]; then
        # 预热用独立短 prompt: 避免与正式请求共享 session(否则正式请求变增量)
        curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":0}" >/dev/null
    fi
    mark=$(wc -l < "$LOG0")
    t0=$(date +%s.%N)
    curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$NTOKENS,\"temperature\":0}" >/dev/null
    t1=$(date +%s.%N)
    tot=$(echo "$t1 - $t0" | bc)
    sleep 2   # rank 日志有缓冲延迟, 等刷盘后再解析
    LOGNEW="sed -n '$((mark + 1)),\$p' $LOG0"

    # 解析日志: decode 行 "decode:  N tokens in X s (Y tok/s)"(纯 decode 时间);
    # pp done 行 resume=0 end=N → prefill tokens = N - decode tokens
    dec=$(eval "$LOGNEW" | grep -oE "decode:[[:space:]]*[0-9]+ tokens in [0-9.]+ s" | tail -1)
    dec_tok=$(echo "$dec" | grep -oE "[0-9]+ tokens" | grep -oE "^[0-9]+")
    dec_s=$(echo "$dec" | grep -oE "[0-9.]+ s" | grep -oE "^[0-9.]+")
    end=$(eval "$LOGNEW" | grep -oE "pp done rc=0 \(resume=0 end=[0-9]+" | tail -1 | grep -oE "end=[0-9]+" | grep -oE "[0-9]+")
    [ -z "$dec_tok" ] && dec_tok=$NTOKENS
    [ -z "$dec_s" ] && dec_s=0
    [ -z "$end" ] && end=$((NTOKENS + 1))
    pret=$((end - dec_tok))
    pre_s=$(echo "$tot - $dec_s" | bc)

    TOT[$R]=$(echo "${TOT[$R]:-0} + $tot" | bc)
    DEC_CNT[$R]=$((${DEC_CNT[$R]:-0} + 1))
    DECT[$R]=$(echo "${DECT[$R]:-0} + $dec_s" | bc)
    PRET[$R]=$(echo "${PRET[$R]:-0} + $pre_s" | bc)
    PRE_CNT[$R]=$((${PRE_CNT[$R]:-0} + $pret))
done

printf "%-28s | %-8s | %-12s | %-12s | %-10s\n" "配置" "总耗时" "prefill tok/s" "decode tok/s" "总 tok/s"
for R in $RANKS_LIST; do
    n=${CNT:-${DEC_CNT[$R]}}
    tot_avg=$(echo "scale=2; ${TOT[$R]} / ${DEC_CNT[$R]}" | bc)
    dec_avg=$(echo "scale=2; ${DECT[$R]} / ${DEC_CNT[$R]}" | bc)
    pre_avg=$(echo "scale=2; ${PRET[$R]} / ${DEC_CNT[$R]}" | bc)
    pre_tok=$(echo "${PRE_CNT[$R]} / ${DEC_CNT[$R]}" | bc)
    pre_tps=$(echo "scale=1; $pre_tok / $pre_avg" | bc)
    dec_tps=$(echo "scale=1; $NTOKENS / $dec_avg" | bc)
    tot_tps=$(echo "scale=1; ($pre_tok + $NTOKENS) / $tot_avg" | bc)
    label="无分布式(单 rank)" ; [ "$R" != "1" ] && label="rank0 + $((R-1)) worker"
    printf "%-28s | %-8s | %-12s | %-12s | %-10s\n" \
        "$label" "${tot_avg}s" "$pre_tps" "$dec_tps" "$tot_tps"
done