#!/usr/bin/env python3
"""PP 会话缓存集成测试(2 段): 增量续接 / 双段落盘 / 重启恢复。

前置: 服务已启动且 tinyllama 为 PP 2 段(serve.yaml: ranks: 2),
      --cache-dir ./sessions 已配置。

用例:
  ① 基础增量: 两轮对话 → rank0 日志 resume 递增(增量 prefill, 非全量)
  ② 长上下文: 多轮对话累积 200+ tokens → resume 持续递增, 末轮续接大值
  ③ 多会话:   3 个不同首条消息的会话交替轮询 → 各会话 resume 独立递增(互不串扰)
  ④ 并发:     4 会话 × 2 轮并发请求 → 全部成功, 各会话状态独立且续接正常
  ⑤ 落盘+重启恢复: ctl stop → 各会话 .r0.kv/.r1.kv/.sess 齐全 → 重启 → 双段 kv
                 restored → resume 续接(免全量重 prefill)

会话 key = 客户端 IP + 首条用户消息 FNV-1a 64 哈希(与 router_http.c 一致);
续接同一会话必须携带相同首条消息的完整历史。

用法:
  python3 tests/test_pp_sess.py --bin ./build/avx2/yllm       # 完整(约 6 分钟)
  python3 tests/test_pp_sess.py --bin ./build/avx2/yllm --quick  # 核心用例(约 2 分钟)
注: PP 2 段下每轮请求约 13 秒(0.2 tok/s 分布式开销), 完整模式共 ~25 次请求。
退出码: 0 = 通过; 1 = 失败
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "tinyllama"
LOGS = "logs"
SESSIONS = "sessions"
KEY_RE = re.compile(r"127\.0\.0\.1_[0-9a-f]{16}")
PP_RE = re.compile(r"rank: sess (\S+) pp done rc=0 \(resume=(\d+) end=")


def sess_key(first):
    """与会话 key 生成保持一致(IP 固定 127.0.0.1)。"""
    h = 1469598103934665603
    for c in first.encode():
        h ^= c
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"127.0.0.1:{h:016x}"


def chat(messages, max_tokens=2, timeout=180, retries=6):
    """请求并返回回复内容; 服务刚启动 rank 未就绪时会断连, 自动重试。"""
    last = None
    for i in range(retries):
        try:
            body = json.dumps({"model": MODEL, "messages": messages,
                               "max_tokens": max_tokens}).encode()
            req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def rank_log(r):
    return os.path.join(LOGS, f"{MODEL}-rank-{r}.log")


def pp_resumes(key=None, since_line=0):
    """rank0 日志中的 PP 会话行 (key, resume) 序列; 按 key 过滤 + 行号过滤。"""
    with open(rank_log(0), encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[since_line:]
    out = []
    for ln in lines:
        m = PP_RE.search(ln)
        if m and (key is None or m.group(1) == key):
            out.append((m.group(1), int(m.group(2))))
    return out


def log_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().count("\n")


def wait_service(seconds=40):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            req = urllib.request.Request(BASE + "/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def wait_kv_restored(seconds=90):
    """等 rank0/rank1 都出现 kv restored(重启后 rank 加载需数秒)。"""
    t0 = time.time()
    restored = {"r0": False, "r1": False}
    while time.time() - t0 < seconds:
        for r in (0, 1):
            if not restored[f"r{r}"]:
                t = open(rank_log(r), encoding="utf-8", errors="replace").read()
                if "kv restored" in t:
                    restored[f"r{r}"] = True
        if restored["r0"] and restored["r1"]:
            return True
        time.sleep(2)
    return False


class TC:
    def __init__(self, bin_path):
        self.bin = bin_path
        self.fails = []

    def ok(self, m):
        print(f"  [ok] {m}")

    def bad(self, m):
        self.fails.append(m)
        print(f"  [FAIL] {m}")


# ---------------- ① 基础增量 ----------------
def test_incr(tc):
    print("== ① 多轮对话(增量续接)")
    mark = log_lines(rank_log(0))
    messages = []
    for t in range(2):
        messages.append({"role": "user",
                         "content": "Once upon a time" if t == 0 else "What happened next?"})
        r = chat(messages, max_tokens=2)
        messages.append({"role": "assistant", "content": r})
        time.sleep(0.5)
    resumes = [v for _, v in pp_resumes(since_line=mark)]
    if len(resumes) < 2:
        tc.bad(f"缺少 PP 会话行: {resumes}")
        return
    if not (resumes[0] == 0 and all(b > a for a, b in zip(resumes, resumes[1:]))):
        tc.bad(f"resume 未严格递增: {resumes}")
    else:
        tc.ok(f"增量续接 resume: {resumes[0]} -> {resumes[-1]}")


# ---------------- ② 长上下文 ----------------
def test_long_ctx(tc):
    print("== ② 长上下文(多轮累积, 末轮续接大值)")
    mark = log_lines(rank_log(0))
    messages = []
    turns = [
        "In a land far far away",
        "What happened next?",
        "And then?",
        "Tell me more about the journey",
        "What did they discover?",
        "Describe the final battle",
    ]
    for t in range(len(turns)):
        messages.append({"role": "user", "content": turns[t]})
        r = chat(messages, max_tokens=2)
        messages.append({"role": "assistant", "content": r})
        time.sleep(0.5)
    resumes = [v for _, v in pp_resumes(since_line=mark)]
    if len(resumes) < len(turns):
        tc.bad(f"长上下文轮数不足: {len(resumes)}/{len(turns)}")
        return
    if not all(b > a for a, b in zip(resumes, resumes[1:])):
        tc.bad(f"长上下文 resume 未持续递增: {resumes}")
        return
    tc.ok(f"{len(turns)} 轮全部增量续接, resume: {resumes[0]} -> {resumes[-1]}")
    if resumes[-1] < 80:
        tc.bad(f"末轮 resume 过小(上下文未累积?): {resumes[-1]}")
    else:
        tc.ok(f"长上下文已累积 {resumes[-1]} tokens, 仍为增量 prefill")


# ---------------- ③ 多会话 ----------------
def test_multi_sess(tc):
    print("== ③ 多会话(3 个会话交替, 各自独立续接)")
    mark = log_lines(rank_log(0))
    prompts = ["At the edge of the world", "The last lighthouse keeper", "A tiny dragon hatchling"]
    hist = {p: [] for p in prompts}
    for rnd in range(2):
        for p in prompts:
            hist[p].append({"role": "user", "content": p if len(hist[p]) == 0 else "What next?"})
            rep = chat(hist[p], max_tokens=2)
            hist[p].append({"role": "assistant", "content": rep})
            time.sleep(0.3)
    keys = {p: sess_key(p) for p in prompts}
    for p in prompts:
        seq = [v for k, v in pp_resumes(key=keys[p], since_line=mark)]
        if len(seq) < 2 or not all(b > a for a, b in zip(seq, seq[1:])):
            tc.bad(f"会话 [{p[:18]}...] resume 未独立递增: {seq}")
        else:
            tc.ok(f"会话 [{p[:18]}...] 独立续接: {seq[0]} -> {seq[-1]}")
    all_keys = {k for k, _ in pp_resumes(since_line=mark)}
    for p in prompts:
        if keys[p] not in all_keys:
            tc.bad(f"会话 [{p[:18]}...] 未出现在 rank0 日志")
    tc.ok(f"{len(prompts)} 个会话 key 各自独立记账")


# ---------------- ④ 并发 ----------------
def test_concurrent(tc):
    print("== ④ 并发(4 会话 × 2 轮同时请求)")
    mark = log_lines(rank_log(0))
    prompts = ["Under a crimson sky", "The clockmaker's apprentice", "Beyond the ninth wave",
               "In the amber city"]
    hist = {p: [] for p in prompts}

    def one(p, rnd):
        if rnd == 0:
            hist[p].append({"role": "user", "content": p})
        else:
            hist[p].append({"role": "user", "content": "What next?"})
        rep = chat(hist[p], max_tokens=2, timeout=240)
        hist[p].append({"role": "assistant", "content": rep})
        return p, rnd, len(rep)

    ok_cnt = 0
    for rnd in range(2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(one, p, rnd) for p in prompts]
            for f in concurrent.futures.as_completed(futs):
                try:
                    p, r, n = f.result()
                    if n > 0:
                        ok_cnt += 1
                    else:
                        tc.bad(f"并发会话 [{p[:14]}...] 第{r + 1}轮空回复")
                except Exception as e:
                    tc.bad(f"并发请求异常: {e}")
        time.sleep(1)
    keys = {p: sess_key(p) for p in prompts}
    for p in prompts:
        seq = [v for k, v in pp_resumes(key=keys[p], since_line=mark)]
        if len(seq) < 2 or not all(b > a for a, b in zip(seq, seq[1:])):
            tc.bad(f"并发会话 [{p[:14]}...] 续接异常: {seq}")
        else:
            tc.ok(f"并发会话 [{p[:14]}...] 两轮续接: {seq[0]} -> {seq[-1]}")
    if ok_cnt != len(prompts) * 2:
        tc.bad(f"并发请求成功数不足: {ok_cnt}/{len(prompts) * 2}")
    else:
        tc.ok(f"并发 {len(prompts)} 会话 × 2 轮全部成功")


# ---------------- ⑤ 落盘 + 重启恢复 ----------------
def test_persist(tc):
    print("== ⑤ 落盘 + 重启恢复")
    os.system(f"{tc.bin} ctl stop >/dev/null 2>&1")
    time.sleep(3)
    files = sorted(os.listdir(SESSIONS)) if os.path.isdir(SESSIONS) else []
    keys = sorted({m.group(0) for f in files if (m := KEY_RE.search(f))})
    if not keys:
        tc.bad("sessions/ 无会话文件")
        return
    for key in keys:
        for ext in (".r0.kv", ".r1.kv", ".sess"):
            name = f"{key}{ext}"
            if name in files:
                tc.ok(f"落盘: {name}")
            else:
                tc.bad(f"缺少落盘文件: {name}")

    os.system(f"{tc.bin} ctl start >/dev/null 2>&1")
    if not wait_service():
        tc.bad("重启后服务未就绪")
        return
    if not wait_kv_restored():
        tc.bad("rank0/rank1 kv restored 缺失")
        return
    tc.ok("重启后 rank0/rank1 均 kv restored")

    # 续接其中一个会话(完整历史, 首条消息一致)
    target = keys[0]
    mark = log_lines(rank_log(0))
    first = None
    for p, k in [(p_, sess_key(p_)) for p_ in
                 ["Once upon a time", "In a land far far away", "At the edge of the world",
                  "Under a crimson sky"]]:
        if k.replace(":", "_") == target:
            first = p
            break
    if first is None:
        tc.bad(f"无法从落盘 key 定位首条消息: {target}")
        return
    messages = [{"role": "user", "content": first},
                {"role": "assistant", "content": "placeholder"},
                {"role": "user", "content": "Continue"}]
    r = chat(messages, max_tokens=2)
    time.sleep(1)
    seq = [v for k, v in pp_resumes(since_line=mark)]
    if not seq or seq[-1] == 0:
        tc.bad(f"重启后 resume=0(未续接): {seq}")
    else:
        tc.ok(f"重启后会话 [{first[:14]}...] 续接 resume={seq[-1]}")


def run(bin_path, quick=False):
    tc = TC(bin_path)
    if not wait_service():
        tc.bad("服务未就绪(8000 口不可达)")
        return 1
    test_incr(tc)
    if not quick:
        test_long_ctx(tc)
        test_multi_sess(tc)
        test_concurrent(tc)
    test_persist(tc)
    if tc.fails:
        print(f"\nPP 会话缓存测试: {len(tc.fails)} FAILED")
        for f in tc.fails:
            print(f"  - {f}")
        return 1
    print("\nPP 会话缓存测试: all passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description="PP 会话缓存集成测试")
    ap.add_argument("--bin", default="./build/avx2/yllm")
    ap.add_argument("--quick", action="store_true",
                    help="只跑核心用例(①增量+⑤落盘重启恢复, 约 2 分钟)")
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        print(f"二进制不存在: {args.bin}")
        return 1
    sys.exit(run(args.bin, quick=args.quick))


if __name__ == "__main__":
    main()