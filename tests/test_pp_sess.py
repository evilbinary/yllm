#!/usr/bin/env python3
"""PP 会话缓存集成测试(2 段): 增量续接 / 双段落盘 / 重启恢复。

前置: 服务已启动且 tinyllama 为 PP 2 段(serve.yaml: ranks: 2),
      --cache-dir ./sessions 已配置。

流程:
  1. 多轮对话 → 断言 rank0 日志 resume 递增(增量 prefill, 非全量)
  2. ctl stop → 断言 sessions/ 出现 .r0.kv + .r1.kv + .sess
  3. ctl start + 继续对话 → 断言 rank0/rank1 均 kv restored 且 resume 续接

用法:
  python3 tests/test_pp_sess.py --bin ./build/avx2/yllm
退出码: 0 = 通过; 1 = 失败
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "tinyllama"
LOGS = "logs"
SESSIONS = "sessions"
KEY_RE = re.compile(r"127\.0\.0\.1_[0-9a-f]{16}")


def chat(messages, max_tokens=8):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def rank0_log():
    return os.path.join(LOGS, f"{MODEL}-rank-0.log")


def wait_service(seconds=40):
    """等 8000 口可服务(启动竞态自愈窗口)。"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            body = json.dumps({"model": MODEL, "messages": [],
                               "max_tokens": 1}).encode()
            req = urllib.request.Request(BASE + "/v1/models", data=None,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run(bin_path, keep_going=False):
    fails = []
    ok = lambda m: print(f"  [ok] {m}")
    bad = lambda m: fails.append(m) or print(f"  [FAIL] {m}")

    if not wait_service():
        bad("服务未就绪(8000 口不可达)")
        return 1

    # ---- 1) 多轮对话, 断言 resume 递增 ----
    print("== ① 多轮对话(增量续接)")
    messages = []
    for t in range(2):
        messages.append({"role": "user", "content": "Once upon a time" if t == 0 else "What happened next?"})
        r = chat(messages, max_tokens=8)
        messages.append({"role": "assistant", "content": r})
        time.sleep(1)

    text = open(rank0_log(), encoding="utf-8", errors="replace").read()
    resumes = [int(m) for m in re.findall(r"pp done rc=0 \(resume=(\d+) end=", text)]
    if len(resumes) < 2:
        bad(f"rank0 日志缺少 PP 会话行: {resumes}")
    else:
        if resumes[-2] < 0 or resumes[-1] <= resumes[-2]:
            bad(f"resume 未递增: {resumes}")
        else:
            ok(f"增量续接 resume 递增: {resumes[-2]} -> {resumes[-1]}")

    # ---- 2) 正常停止, 双段落盘 ----
    print("== ② ctl stop(双段落盘)")
    os.system(f"{bin_path} ctl stop >/dev/null 2>&1")
    time.sleep(3)
    keys = []
    for f in os.listdir(SESSIONS):
        m = KEY_RE.search(f)
        if m:
            keys.append(m.group(0))
    keys = sorted(set(keys))
    if not keys:
        bad("sessions/ 无会话文件")
        return 1
    key = keys[-1]
    files = os.listdir(SESSIONS)
    for expect in (f"{key}.r0.kv", f"{key}.r1.kv", f"{key}.sess"):
        if expect not in files:
            bad(f"缺少落盘文件: {expect}")
        else:
            ok(f"落盘: {expect}")

    # ---- 3) 重启 + 续接 ----
    print("== ③ 重启 + 续接")
    os.system(f"{bin_path} ctl start >/dev/null 2>&1")
    if not wait_service():
        bad("重启后服务未就绪")
        return 1
    # 等 rank0/rank1 都完成 kv 恢复(rank 加载需数秒, 过早请求会触发全量重发)
    start_mark = time.time()
    log0 = rank0_log()
    log1 = os.path.join(LOGS, f"{MODEL}-rank-1.log")
    restored = {"r0": False, "r1": False}
    while time.time() - start_mark < 90:
        t = open(log0, encoding="utf-8", errors="replace").read()
        if "kv restored" in t:
            restored["r0"] = True
        t1 = open(log1, encoding="utf-8", errors="replace").read()
        if "kv restored" in t1:
            restored["r1"] = True
        if restored["r0"] and restored["r1"]:
            break
        time.sleep(2)
    if not (restored["r0"] and restored["r1"]):
        bad(f"rank 恢复日志缺失: r0={restored['r0']} r1={restored['r1']}")
        return 1

    # 会话 key = IP + 首条用户消息哈希: 续接必须带与 ① 相同的完整历史(首条一致)
    messages = [
        {"role": "user", "content": "Once upon a time"},
        {"role": "assistant", "content": "placeholder"},
        {"role": "user", "content": "What happened next?"},
        {"role": "assistant", "content": "placeholder"},
        {"role": "user", "content": "And then?"},
    ]
    # 只统计本次请求后的 PP 会话行(旧日志里的 resume 序列已无关)
    with open(log0, encoding="utf-8", errors="replace") as f:
        mark = f.read().count("\n")
    r = chat(messages, max_tokens=8)
    time.sleep(1)

    with open(log0, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[mark:]
    after = "".join(lines)
    text0_full = open(log0, encoding="utf-8", errors="replace").read()
    text1_full = open(log1, encoding="utf-8", errors="replace").read()
    if "kv restored" not in text0_full:
        bad("rank0 未恢复 kv(kv restored 缺失)")
    else:
        ok("rank0 kv restored")
    if "kv restored" not in text1_full:
        bad("rank1 未恢复 kv(dist worker: kv restored 缺失)")
    else:
        ok("rank1 kv restored")

    # 重启后的请求 resume 应 > 0(免全量重 prefill)
    restarted = [int(m) for m in re.findall(r"pp done rc=0 \(resume=(\d+) end=", after)]
    if not restarted or restarted[-1] == 0:
        bad(f"重启后 resume=0(未续接): {restarted}")
    else:
        ok(f"重启后续接 resume={restarted[-1]}")

    if fails:
        print(f"\nPP 会话缓存测试: {len(fails)} FAILED")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPP 会话缓存测试: all passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description="PP 会话缓存集成测试")
    ap.add_argument("--bin", default="./build/avx2/yllm")
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        print(f"二进制不存在: {args.bin}")
        return 1
    sys.exit(run(args.bin))


if __name__ == "__main__":
    main()