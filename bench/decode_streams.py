#!/usr/bin/env python3
"""Aggregate decode speed with N parallel streams against an OpenAI-compatible endpoint.

Each stream asks for the same ~900-word prose answer (a unique suffix defeats the prefix cache), up to
--max-tokens tokens, thinking off. Prints per-stream tokens/time and the aggregate tok/s (all tokens / wall time).

    ./decode_streams.py 1 2 4 8 --url http://127.0.0.1:18300 --model flashnext-exl3-305
"""
import argparse, json, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("streams", nargs="*", type=int, default=[1, 2, 4, 8])
ap.add_argument("--url", default="http://127.0.0.1:18300")
ap.add_argument("--model", default="flashnext-exl3-305")
ap.add_argument("--max-tokens", type=int, default=1200)
ap.add_argument("--temperature", type=float, default=0.6)
A = ap.parse_args()
PROMPT = ("Write a detailed technical explanation, about 900 words, of how TCP congestion control works "
          "(slow start, congestion avoidance, fast retransmit, fast recovery, CUBIC, BBR). Plain prose.")

def one(res, i):
    body = json.dumps({"model": A.model, "max_tokens": A.max_tokens, "temperature": A.temperature,
                       "reasoning_effort": "none",
                       "messages": [{"role": "user", "content": PROMPT + f" (variant {i} {time.time()})"}]}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", body,
                                         {"Content-Type": "application/json"}), timeout=1800))
    res[i] = (r["usage"]["completion_tokens"], time.time() - t)

for n in A.streams:
    res = {}; th = [threading.Thread(target=one, args=(res, i)) for i in range(n)]; t = time.time()
    [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t
    per = ", ".join(f"{v[0]} tok/{v[1]:.1f}s={v[0] / v[1]:.1f}" for v in res.values())
    print(f"{n} stream(s): {per} | aggregate {sum(v[0] for v in res.values()) / wall:.1f} tok/s", flush=True)
