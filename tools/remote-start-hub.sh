#!/bin/bash
cd /home/evil/yllm
mkdir -p logs sessions
pkill -f '/home/evil/yllm/build/avx2/yllm' 2>/dev/null || true
sleep 1
export OMP_NUM_THREADS=1
export YLLM_SRV_TIMEOUT=300
export YLLM_BIND_HOST=127.0.0.1
nohup ./build/avx2/yllm hub --config serve.yaml --model gemma4-e2b > logs/hub.out 2>&1 &
echo "started pid=$!"
sleep 12
pgrep -a yllm || echo no_yllm
ss -lntp 2>/dev/null | grep -E '8000|9410|9500|9400' || true
curl -s -m 5 -H 'Authorization: Bearer sk-yllm-123' http://127.0.0.1:8000/v1/models || echo CURL_FAIL
echo
