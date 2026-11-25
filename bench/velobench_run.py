#!/usr/bin/env python3
"""Run VeloBench (https://github.com/sf-stav/VeloBench, v0.4.0) built-in tests headless and save the reports.

Needs a running VeloBench with the server under test added as a provider (Settings, or data/settings.json).
Uses only VeloBench's own built-in tests at temperature 0, so anyone can reproduce the numbers.

    ./velobench_run.py my-label --provider velo [--vb http://127.0.0.1:13843]
Writes vb-<label>.json and prints endToEndTokS / wall / average TTFT for the concurrent runs.
"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--provider", default="velo")
ap.add_argument("--vb", default="http://127.0.0.1:13843")
ap.add_argument("--workers", default="1,2,4,8")
A = ap.parse_args()
res = {}

def run(test, mode="single", w=1):
    b = {"provider": A.provider, "testname": test, "title": f"{A.label}-{test}-{mode}{w}", "mode": mode, "temperature": 0.0}
    if mode == "concurrent": b["workers"] = w
    t = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(A.vb + "/api/run", json.dumps(b).encode(),
                                          {"Content-Type": "application/json"}), timeout=10800).read())
    res[f"{test}|{mode}{w}"] = d; json.dump(res, open(f"vb-{A.label}.json", "w"))
    line = f"{test} {mode} {w}: {d.get('status')} {time.time() - t:.0f}s"
    if mode == "concurrent" and d.get("report"):
        c = d["report"]["concurrency"]
        line += f" | {c['endToEndTokS']:.1f} tok/s, wall {c['wallMs'] / 1000:.0f} s, avg TTFT {c['avgTtftMs'] / 1000:.1f} s"
    print(line, flush=True)

for t in ("prebuilt-shape-quick", "prebuilt-shape-pp-sweep", "prebuilt-shape-tg-sweep", "prebuilt-prefill-scaling"): run(t)
for t in ("prebuilt-pure-code-c", "prebuilt-creative-scifi"):
    for w in (int(x) for x in A.workers.split(",")): run(t, "concurrent", w)
