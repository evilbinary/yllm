#!/bin/bash
# PP 会话缓存集成测试驱动:
#   临时把 serve.yaml 的 tinyllama 改为 ranks:2 → 启动 → tests/test_pp_sess.py → 恢复 serve.yaml
# 用法: bash tools/test_pp_sess.sh [BIN]
set -u
cd "$(dirname "$0")/.."
BIN="${1:-./build/avx2/yllm}"

if [ ! -x "$BIN" ]; then
    echo "二进制不存在: $BIN (先 make avx2)"; exit 1
fi

python3 - <<'PYEOF'
import re
p = 'serve.yaml'
src = open(p).read()
open(p + '.bak', 'w').write(src)
src = re.sub(r'(?m)^(\s+)ranks: \d+', r'\1ranks: 1', src)
src = re.sub(r'(?s)(- name: tinyllama.*?)(?=\n\s*- name: |\Z)',
             lambda m: m.group(0).replace('ranks: 1', 'ranks: 2'), src)
open(p, 'w').write(src)
print("serve.yaml: tinyllama -> ranks: 2 (备份 serve.yaml.bak)")
PYEOF

cleanup() {
    "$BIN" ctl exit >/dev/null 2>&1
    sleep 2
    for pid in $(pgrep -f "rank --model" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
    if [ -f serve.yaml.bak ]; then mv serve.yaml.bak serve.yaml; fi
}
trap cleanup EXIT

"$BIN" ctl stop >/dev/null 2>&1 || true
rm -rf sessions; mkdir -p sessions
"$BIN" ctl start >/dev/null 2>&1

if python3 tests/test_pp_sess.py --bin "$BIN"; then
    echo "test-pp-sess: passed (serve.yaml restored)"
    exit 0
else
    echo "test-pp-sess: FAILED (serve.yaml restored)"
    exit 1
fi