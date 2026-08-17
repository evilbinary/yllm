#!/usr/bin/env python3
# PP 同等线程对比: gen 直跑 vs 分布式(每 rank 线程 T, 总线程 T×ranks)
#
# 场景: 50-token prompt + 16 decode
#   - gen:  OMP_NUM_THREADS=T 直跑(参考基准)
#   - serve: OMP_NUM_THREADS=T + ranks=N(每 rank T 线程)
# 指标: prefill / decode / 端到端 tok/s
#
# 用法:
#   python3 tools/bench_threads.py
#   环境变量 THREADS / RANKS / RUNS 可覆盖

import os
import re
import subprocess
import sys
import time
import urllib.request
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

THREADS = [int(x) for x in os.environ.get("THREADS", "1 4 8 16").split()]
RANKS = [int(x) for x in os.environ.get("RANKS", "1 2 4").split()]
RUNS = int(os.environ.get("RUNS", "2"))
PROMPT = os.environ.get(
    "PROMPT",
    "A merchant named Chen walked through the market square at dawn, "
    "carrying two baskets of ripe peaches and humming a song he learned "
    "from his grandfather.",
)
NTOKENS = int(os.environ.get("NTOKENS", "16"))
MODEL = "tinyllama"
CFG = "serve.yaml"
CFG_BAK = "/tmp/opencode/serve.yaml.threads.bak"
LOG0 = "logs/tinyllama-rank-0.log"
LLF = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf"
VOCAB = "models/tinyllama.vocab.txt"
URL = "http://127.0.0.1:8000/v1/chat/completions"

PAT_PREF = re.compile(r"prefill:\s+(\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)")
PAT_DEC = re.compile(r"decode:\s+(\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)")
PAT_PPDONE = re.compile(r"pp done rc=0 \(resume=0 end=(\d+)")
PAT_GENOK = re.compile(r"generate ok \((\d+) delta")


MSYS_BASH = r"E:\soft\msys2\usr\bin\bash.exe"


def sh(cmd, env=None, timeout=120):
    e = dict(os.environ)
    if env:
        e.update(env)
    if os.name == "nt":
        e.setdefault("MSYSTEM", "MINGW64")
        e.setdefault("CHERE_INVOKING", "1")
        e.setdefault("MSYS2_PATH_TYPE", "inherit")
        return subprocess.run([MSYS_BASH, "--login", "-c", cmd],
                              env=e, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(cmd, env=e, shell=isinstance(cmd, str),
                          capture_output=True, text=True, timeout=timeout)


def kill_all():
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/IM", "yllm.exe"],
                       capture_output=True, text=True)
    else:
        sh("pkill -9 -f 'build/avx2/yllm (rank|serve|hub)'")
    time.sleep(1)


def set_ranks(r):
    with open(CFG) as f:
        lines = f.readlines()
    out, in_tiny, done = [], False, False
    for ln in lines:
        if ln.strip().startswith("- name: tinyllama"):
            in_tiny = True
        elif in_tiny and ln.strip().startswith("- name:"):
            in_tiny = False
        if in_tiny and not done and ln.strip().startswith("ranks:"):
            ln = f"    ranks: {r}             # 本模型 rank 进程数\n"
            done = True
        if os.name == "nt" and ln.startswith("bin:") and ".exe" not in ln:
            ln = ln.rstrip("\n").rstrip() + ".exe\n"
        out.append(ln)
    with open(CFG, "w") as f:
        f.writelines(out)


def curl_payload(content, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status


def gen_vals(t):
    out = sh(f"./build/avx2/yllm gen --model {LLF} --vocab {VOCAB} "
             f"--prompt '{PROMPT}' --tokens {NTOKENS} --temp 0",
             env={"OMP_NUM_THREADS": str(t)}).stderr + \
          sh(f"./build/avx2/yllm gen --model {LLF} --vocab {VOCAB} "
             f"--prompt '{PROMPT}' --tokens {NTOKENS} --temp 0",
             env={"OMP_NUM_THREADS": str(t)}).stdout
    pm = PAT_PREF.search(out)
    dm = PAT_DEC.search(out)
    if not pm or not dm:
        return None
    return float(pm.group(3)), float(dm.group(3))


def serve_vals(t, r):
    set_ranks(r)
    kill_all()
    sh("rm -rf sessions && mkdir sessions")
    sh("./build/avx2/yllm ctl start", env={"OMP_NUM_THREADS": str(t)})
    time.sleep(7)
    # 预热用独立短 prompt, 避免与正式请求共享 session
    try:
        curl_payload("Hi", 1)
    except Exception:
        pass
    with open(LOG0) as f:
        mark = len(f.readlines())
    t0 = time.time()
    curl_payload(PROMPT, NTOKENS)
    total = time.time() - t0
    time.sleep(2)  # rank 日志缓冲延迟
    with open(LOG0) as f:
        lines = f.readlines()[mark:]
    log = "".join(lines)
    dms = PAT_DEC.findall(log)
    pps = PAT_PPDONE.findall(log)
    gos = PAT_GENOK.findall(log)
    dm = dms[-1] if dms else None
    if dm:
        dec_s = float(dm[1])
        dec_tps = float(dm[2])
        if pps:
            end = int(pps[-1])
        elif gos:
            end = int(gos[-1]) + NTOKENS
        else:
            end = NTOKENS + 1
    else:
        dec_s = 0.0
        dec_tps = 0.0
        end = NTOKENS + 1
    pret = max(end - NTOKENS, 1)
    pre_s = max(total - dec_s, 1e-6)
    pre_tps = pret / pre_s
    return total, pre_tps, dec_tps


def main():
    # 备份 serve.yaml
    sh(f"cp {CFG} {CFG_BAK}")
    try:
        print(f"{'线程T':<6} | {'gen-pre':<9} | {'gen-dec':<9} | "
              + " | ".join(f"r{R}-pre{' '*(7-len(str(R)))}{'':<2} | r{R}-dec" for R in RANKS))
        for t in THREADS:
            gv = gen_vals(t)
            gpre = f"{gv[0]:.1f}" if gv else "?"
            gdec = f"{gv[1]:.1f}" if gv else "?"
            cells = [f"{t}", gpre, gdec]
            for r in RANKS:
                pre_avg, dec_avg = 0.0, 0.0
                for _ in range(RUNS):
                    tot, pt, dt = serve_vals(t, r)
                    pre_avg += pt
                    dec_avg += dt
                cells.append(f"{pre_avg / RUNS:.1f}")
                cells.append(f"{dec_avg / RUNS:.1f}")
            print(f"{cells[0]:<6} | {cells[1]:<9} | {cells[2]:<9} | "
                  + " | ".join(f"{c:<9}" for c in cells[3:]))
    finally:
        kill_all()
        sh(f"mv {CFG_BAK} {CFG}")


if __name__ == "__main__":
    main()