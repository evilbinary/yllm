#!/usr/bin/env bash
# PP 分布式单机基准: 对比 worker rank 数(总 ranks=2/3/4, 即 rank0 + 1/2/3 个 worker)
#
# 场景: 50-token prompt + 16 decode, serve 模式 curl 端到端耗时;
#       每轮重启服务 + 清 sessions 保证全量 prefill。
#
# 用法:
#   bash tools/bench_pp.sh [RANKS_LIST] [RUNS]
#   RANKS_LIST 默认 "2 3 4"(总 rank 数); RUNS 默认 3(每配置轮数)
#   PROMPT / NTOKENS / MODEL 可环境变量覆盖
#
# 输出: 每配置各轮耗时 + 均值, 便于填入 docs/benchmark.md

set -u
cd "$(dirname "$0")/.."   # 仓库根目录

RANKS_LIST=${RANKS_LIST:-"2 3 4"}
RUNS=${RUNS:-3}
PROMPT=${PROMPT:-"A merchant named Chen walked through the market square at dawn, carrying two baskets of ripe peaches and humming a song he learned from his grandfather."}
NTOKENS=${NTOKENS:-16}
MODEL=${MODEL:-tinyllama}
CFG=serve.yaml
CFG_BAK=/tmp/opencode/serve.yaml.bench.bak
WARMUP=${WARMUP:-1}

pkill -9 -f 'build/avx2/yllm (rank|serve|hub)' 2>/dev/null
sleep 1
cp "$CFG" "$CFG_BAK"

cleanup() {
    pkill -9 -f 'build/avx2/yllm (rank|serve|hub)' 2>/dev/null
    sleep 1
    mv "$CFG_BAK" "$CFG"
}
trap cleanup EXIT

# 交错轮次: 2 3 4 2 3 4 ...
SEQ=""
for i in $(seq 1 "$RUNS"); do
    for R in $RANKS_LIST; do SEQ="$SEQ $R"; done
done

declare -A SUM CNT
for R in $SEQ; do
    sed -i "s/^ranks: [0-9]* .*本模型 rank 进程数/ranks: $R             # 本模型 rank 进程数/" "$CFG"
    pkill -9 -f 'build/avx2/yllm (rank|serve|hub)' 2>/dev/null
    sleep 1
    rm -rf sessions && mkdir sessions
    ./build/avx2/yllm ctl start >/dev/null 2>&1
    sleep 7
    if [ "$WARMUP" = "1" ]; then
        curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":1,\"temperature\":0}" >/dev/null
    fi
    t0=$(date +%s.%N)
    curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$NTOKENS,\"temperature\":0}" >/dev/null
    t1=$(date +%s.%N)
    dt=$(echo "$t1 - $t0" | bc)
    SUM[$R]=$(echo "${SUM[$R]:-0} + $dt" | bc)
    CNT[$R]=$((${CNT[$R]:-0} + 1))
done

for R in $RANKS_LIST; do
    avg=$(echo "scale=2; ${SUM[$R]} / ${CNT[$R]}" | bc)
    printf "ranks=%d (rank0 + %d worker): avg %.2fs (%d runs)\n" "$R" "$((R - 1))" "$avg" "${CNT[$R]}"
done