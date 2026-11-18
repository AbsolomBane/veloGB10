#!/usr/bin/env python3
"""Needle in a haystack near the full context window.

Haystack = Python source files from the local Python install (deterministic order), a 6-digit code inserted at
--depth (fraction of the text). Greedy, thinking off. If the server rejects the prompt as too long (HTTP 400), the
text is shortened (97%, 94%, 90%) and retried. On success the same prompt is sent again to check that the answer
comes back from the prefix cache and is identical.

    ./needle.py --chars 4150000          # ~1M tokens with the Qwen3.x tokenizer
"""
import argparse, glob, hashlib, json, random, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--chars", type=int, default=4150000)
ap.add_argument("--depth", type=float, default=0.55)
ap.add_argument("--url", default="http://127.0.0.1:18300")
ap.add_argument("--model", default="flashnext-exl3-305")
A = ap.parse_args()

files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)); text = ""; i = 0
while len(text) < A.chars and i < len(files): text += open(files[i], errors="ignore").read(); i += 1
code = f"{random.Random(17).randint(100000, 999999)}"

def ask(n, tag):
    t = text[:n]; p = int(len(t) * A.depth)
    hay = t[:p] + f"\n# SECRET_NOTE: the vault access code is {code}\n" + t[p:]
    body = {"model": A.model, "max_tokens": 60, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": hay + "\n\nWhat is the vault access code in SECRET_NOTE? Answer with the number only."}]}
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", json.dumps(body).encode(),
                                          {"Content-Type": "application/json"}), timeout=7200).read())
    a = (d["choices"][0]["message"].get("content") or "").strip(); pt = d["usage"]["prompt_tokens"]; dt = time.time() - t0
    print(f"{tag}: {pt} prompt tokens, {dt:.1f} s (~{pt / dt:.0f} tok/s), answer {a[:40]!r}, expected {code} -> "
          f"{'FOUND' if code in a else 'MISSED'}, sha {hashlib.sha256(a.encode()).hexdigest()[:12]}", flush=True)

for n in (A.chars, int(A.chars * 0.97), int(A.chars * 0.94), int(A.chars * 0.90)):
    try: ask(n, "needle"); ask(n, "re-sent"); break
    except urllib.error.HTTPError as e: print(f"{n} chars: HTTP {e.code} {e.read()[:160]!r}", flush=True)
