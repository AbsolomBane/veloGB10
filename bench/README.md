# bench/ — the scripts behind the numbers in KV_POOL.md

All scripts talk to an OpenAI-compatible endpoint. The defaults (`--url http://127.0.0.1:18300`, `--model flashnext-exl3-305`) are the authors' setup: a proxy on `:18300` in front of the engine on `:18301`. Always pass `--url` / `--model` for yours, and never point the long-context scripts (`needle.py`) at a server that serves users. Python 3 standard library only.

| Script | What it measures | Used for |
|---|---|---|
| `decode_streams.py` | aggregate decode tok/s with N parallel streams (1,200 tokens each, thinking off) | the `par8` tables |
| `needle.py` | needle in a haystack near the full window, plus re-send from the prefix cache | needle at 1M |
| `velobench_run.py` | VeloBench v0.4.0 built-in tests, headless (`/api/run`) | the VeloBench side-by-side |
| `live_stats.py` | live terminal monitor: decode, prefill, memory, GPU utilization, **KV pool occupancy**, temperatures | watching a server |

## Examples

```
./decode_streams.py 1 2 4 8
./needle.py --chars 4150000                 # ~1M tokens; shortens itself on HTTP 400
./velobench_run.py fork-pool30 --provider velo
./live_stats.py --native http://127.0.0.1:18301
```

## Notes

* For a fair A/B, run the same script against both servers on the same box, the same day, with nothing else
  loading the GPU. We report every run, including the slow ones.
* `decode_streams.py` uses temperature 0.6 (like a real chat); single runs vary by a few percent, more with MTP on
  prose. Repeat before drawing conclusions from small differences.
* `live_stats.py` reads native `/metrics` (`--native`), the engine log (newest `~/logs/velo-*.log` or `--log`),
  and `nvidia-smi`. When lane metrics are available, each bar shows active context or dormant KV up to that
  lane's limit. The KV POOL bar splits active and cached tokens against the estimated Ruby q8 capacity.
  For a local engine with SSD spill enabled, KV SPILL SSD shows completed snapshot files and the configured limit
  in GiB. Proxy metrics on `--url` provide fallback request counters.
* Quality checks (perplexity, identical greedy outputs across configurations) were run with the engine's own tools
  and are reported in KV_POOL.md.
