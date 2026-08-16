#!/usr/bin/env python3
"""会话缓存验证客户端: 通过本地 OpenAI 接口做多轮对话。

验证方式:
  1. 多轮对话(每轮把完整历史 messages 发给 /v1/chat/completions)
  2. server(router)侧: 渲染全部消息 → 前缀命中 → 只把增量 token 发给 rank
  3. 看 rank 日志( logs/tinyllama-rank-0.log )的会话行:
       "session=<key> generate ok (N delta + M gen tokens, X ms)"
     第二轮起 delta 应远小于全量历史 → 缓存生效

用法:
  python3 tools/chat_session_test.py [--port 8000] [--model tinyllama] [--turns 3]
"""
import argparse
import json
import sys
import time
import urllib.request


def chat(base, model, messages, stream=False):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 48,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="OpenAI 接口会话缓存验证")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="tinyllama")
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()

    messages = []
    turns = [
        "Once upon a time",
        "What happened next?",
        "And then?",
    ]
    print(f"base={args.base} model={args.model} turns={args.turns}")
    for t in range(min(args.turns, len(turns))):
        messages.append({"role": "user", "content": turns[t]})
        t0 = time.time()
        reply = chat(args.base, args.model, messages)
        ms = (time.time() - t0) * 1000
        print(f"[turn {t + 1}] {ms:.0f}ms  in={len(messages)}msgs  reply={reply[:60]!r}")
        messages.append({"role": "assistant", "content": reply})
        time.sleep(1)


if __name__ == "__main__":
    main()
