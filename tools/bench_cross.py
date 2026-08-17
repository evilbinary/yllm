#!/usr/bin/env python3
# 跨机 PP 基准: rank0 跑 Windows(hub), rank1..N-1 跑 Linux(ssh 拉起)
#
# 场景: 50-token prompt + 16 decode, 每 rank OMP_NUM_THREADS=T
#   - rank0: yllm hub(Windows, 本机 supervisor+router+server+rank0)
#   - rank1..N-1: yllm rank(Linux, ssh 后台拉起)
# 指标: prefill / decode(均取 rank0 日志, 含整条流水线)
#
# 用法:
#   python3 tools/bench_cross.py
#   环境变量 THREADS / RANKS / RUNS 可覆盖; 需先配好 ssh(192.168.0.23)

import os
import re
import shutil
import subprocess
import time
import urllib.request
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

THREADS = [int(x) for x in os.environ.get("THREADS", "8 16").split()]
RANKS = [int(x) for x in os.environ.get("RANKS", "2 3").split()]
WIN_RANKS = int(os.environ.get("WIN_RANKS", "1"))
RUNS = int(os.environ.get("RUNS", "1"))
PROMPT = os.environ.get(
    "PROMPT",
    "A merchant named Chen walked through the market square at dawn, "
    "carrying two baskets of ripe peaches and humming a song he learned "
    "from his grandfather.",
)
NTOKENS = int(os.environ.get("NTOKENS", "16"))

WIN_IP = os.environ.get("WIN_IP", "192.168.1.161")
LIN_IP = os.environ.get("LIN_IP", "192.168.0.23")
LIN_DIR = os.environ.get("LIN_DIR", "~/develop/yllm")
MODEL = "tinyllama"
CFG = "serve-cross.yaml"
LOG0 = "logs/tinyllama-rank-0.log"
LLF = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf"
VOCAB = "models/tinyllama.vocab.txt"
URL = "http://127.0.0.1:8000/v1/chat/completions"
WIN_BIN = "./build/avx2/yllm.exe"
LIN_BIN = "./build/avx2/yllm"

PAT_PPDONE = re.compile(r"pp done rc=0 \(resume=0 end=(\d+)")
PAT_GENOK = re.compile(r"generate ok \((\d+) delta")
PAT_DEC = re.compile(r"decode:\s+(\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)")

SSH = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", LIN_IP]


def sh(cmd, timeout=120):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def ssh_lin(cmd, timeout=120):
    return subprocess.run(SSH + ["cd %s && %s" % (LIN_DIR, cmd)],
                          capture_output=True, text=True, timeout=timeout)


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "yllm.exe"],
                   capture_output=True, text=True)
    ssh_lin("pkill -9 -f 'build/avx2/yllm' ; sleep 0.3", timeout=30)
    time.sleep(1)


def write_cfg(ranks):
    lines = [
        "log: logs/serve.log",
        "log-level: info",
        "bin: %s" % WIN_BIN,
        "sv-port: 9500",
        "router-port: 9400",
        "server-port: 9420",
        "rank-port-base: 9410",
        "http-port: 8000",
        "cache-dir: ./sessions",
        "sv-host: 127.0.0.1",
        "router-addrs: 127.0.0.1:9400",
        "servers: 1",
        "auto-heal: 1",
        "strategy: least",
        "lease-strategy: request",
        "models:",
        "  - name: %s" % MODEL,
        "    model: %s" % LLF,
        "    vocab: %s" % VOCAB,
        "    ranks: %d             # 本模型总段数" % ranks,
        "    local: 1             # 本机(Windows)只自动拉 rank0, 其余段(含 Windows rank1)手动拉起",
        "",
    ]
    with open(CFG, "w") as f:
        f.write("\n".join(lines))


def peers_csv(ranks, win_ranks):
    parts = [WIN_IP] * win_ranks + [LIN_IP] * (ranks - win_ranks)
    return ",".join(parts)


def start_extra_ranks(t, ranks, win_ranks):
    """拉起 rank1..N-1: rank 1..win_ranks-1 在 Windows(本地 subprocess, 带正确 peers),
    其余 win_ranks..ranks-1 在 Linux(ssh 后台)。每 rank OMP_NUM_THREADS=t。
    注: supervisor 只自动拉 rank0(local=1, peers 是 127.0.0.1×N 连本地 rank1 可通),
    但 rank1 必须手动起并带正确 peers 才能连到远端 rank2。"""
    dist_stats = "YLLM_DIST_STATS=1 " if os.environ.get("YLLM_DIST_STATS") else ""
    if os.environ.get("YLLM_DISTTIMING"):
        dist_stats += "YLLM_DISTTIMING=1 "
    peers = peers_csv(ranks, win_ranks)
    procs = []
    for r in range(1, ranks):
        args = [
            "rank", "--model", LLF, "--vocab", VOCAB, "--model-name", MODEL,
            "--port", str(9410 + r), "--rank", str(r), "--ranks", str(ranks),
            "--supervisor", "%s:9500" % WIN_IP, "--id", "rank-%d" % r,
            "--peers", peers, "--cache-dir", "sessions",
            "--log", "logs/%s-rank-%d.log" % (MODEL, r),
        ]
        if r < win_ranks:
            # Windows 本地 rank(rank1..win_ranks-1)
            p = subprocess.Popen(
                [WIN_BIN] + args,
                env=dict(os.environ, OMP_NUM_THREADS=str(t),
                         **{k: v for k, v in
                            {"YLLM_DIST_STATS": dist_stats.strip(),
                             "YLLM_DISTTIMING": dist_stats.strip()}.items() if v}),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(p)
        else:
            inner = ("rm -rf sessions && mkdir -p sessions logs && "
                     "OMP_NUM_THREADS=%d %s%s %s > logs/rank-%d.out 2>&1"
                     % (t, dist_stats, LIN_BIN, " ".join(args), r))
            cmd = "setsid sh -c '%s &' </dev/null" % inner
            rc = ssh_lin(cmd, timeout=30)
            if rc.returncode != 0:
                print("remote rank%d start rc=%d: %s" % (r, rc.returncode, rc.stderr.strip()))
    return procs


def start_win_hub(t):
    t0 = int(os.environ.get("RANK0_THREADS", str(t)))
    return subprocess.Popen(
        [WIN_BIN, "hub", "--config", CFG],
        env=dict(os.environ, OMP_NUM_THREADS=str(t0)),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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


def serve_vals(t, ranks, win_ranks):
    write_cfg(ranks)
    kill_all()
    if os.path.isdir("sessions"):
        shutil.rmtree("sessions", ignore_errors=True)
    os.makedirs("sessions", exist_ok=True)
    extra = start_extra_ranks(t, ranks, win_ranks)
    # 后台拉起 hub(rank0 由 hub 的 supervisor 拉起)
    p = start_win_hub(t)
    time.sleep(10)
    try:
        curl_payload("Hi", 1)
    except Exception:
        pass
    time.sleep(2)
    with open(LOG0) as f:
        mark = len(f.readlines())
    t0 = time.time()
    curl_payload(PROMPT, NTOKENS)
    total = time.time() - t0
    time.sleep(2)
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
    p.terminate()
    kill_all()
    return total, pre_tps, dec_tps


def main():
    print("%-6s | %-9s | %-9s" % ("线程T", "pre tok/s", "dec tok/s"))
    for t in THREADS:
        for ranks in RANKS:
            pre_avg, dec_avg = 0.0, 0.0
            ok = 0
            for _ in range(RUNS):
                tot, pt, dt = serve_vals(t, ranks, WIN_RANKS)
                if pt > 0 and dt > 0:
                    pre_avg += pt
                    dec_avg += dt
                    ok += 1
            if ok == 0:
                print("%d ranks=%d  FAILED" % (t, ranks))
                continue
            print("%d ranks=%d | %-9.1f | %-9.1f"
                  % (t, ranks, pre_avg / ok, dec_avg / ok))
    kill_all()


if __name__ == "__main__":
    main()