#!/usr/bin/env python3
# Live terminal monitor for a veloGB10 EXL3 server (written on an ASUS GX10 / GB10).
#
# Sources:
#  * an OpenAI proxy's /metrics on --url (optional; without it the REQUESTS block shows a warning),
#  * the engine's native /metrics on --nativ (v0.7.3+): decode, prefill, KV positions, shared KV pool,
#  * the engine log (--log, default: newest ~/logs/velo-*.log): last request, MTP acceptance,
#  * nvidia-smi: GPU utilization, power, SM clock, temperature; /proc/meminfo: memory.
#
# The KV POOL bar appears only when the server runs with --kv-pool lazy --kv-pool-gb N (this fork).
# Usage:  ./live_stats.py [--url http://127.0.0.1:18300] [--native http://127.0.0.1:18301] [--interval 1.0] [--log FILE] [--once]
# Exit:   Ctrl+C
import argparse, glob, os, re, shutil, subprocess, sys, time, urllib.request
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:18300")
ap.add_argument("--interval", type=float, default=1.0)
ap.add_argument("--log", default="")
ap.add_argument("--native", default="http://127.0.0.1:18301", help="native veloGB10 /metrics (v0.7.3+)")
ap.add_argument("--once", action="store_true", help="one reading, then exit")
A = ap.parse_args()
BASE = A.url.rstrip("/")
ESC = "\033" if sys.stdout.isatty() else ""
c = lambda code, s: s if not ESC else f"{ESC}[{code}m{s}{ESC}[0m"
dim = lambda s: c("2", s); bold = lambda s: c("1", s); grn = lambda s: c("32", s)
yel = lambda s: c("33", s); red = lambda s: c("31", s); cya = lambda s: c("36", s)


def scrape():
    try:
        t = urllib.request.urlopen(BASE + "/metrics", timeout=3).read().decode()
        return {l.split()[0]: float(l.split()[1]) for l in t.splitlines() if l and l[0] != "#" and len(l.split()) == 2}
    except Exception:
        return None


_steps = deque(maxlen=60)   # (t, scheduler_steps_total, requests_running) from native /metrics v0.7.3+
_arm = deque(maxlen=60)    # (t, arm_tok_total, arm_ms_total) v0.7.5
_ctx = {}
_kv = deque(maxlen=60)     # (monotonic time, used KV rows, decode tokens, running requests, restore hits)
_pf = {}                   # last request: real prefill = (prompt - cached) / TTFT
def prefill_live(window=12.0):
    """Estimate wall-clock prefill speed from completed chunks within a stable window."""
    now = time.monotonic()
    samples = [sample for sample in _kv if now - sample[0] <= window]
    tokens = elapsed = 0.0
    for prev, curr in zip(samples, samples[1:]):
        dt = curr[0] - prev[0]
        decoded = curr[2] - prev[2]
        new_rows = curr[1] - prev[1] - decoded
        if dt <= 0 or curr[3] <= 0 or prev[3:] != curr[3:] or decoded < 0 or not (0 <= new_rows <= 8000 * dt):
            tokens = elapsed = 0.0
            continue
        # KV positions update at chunk boundaries. Count the time between jumps, too.
        tokens += new_rows
        elapsed += dt
    return tokens / elapsed if elapsed >= window / 2 and tokens > 64 else None
def decode_arm(window=15.0):
    """(tok/s over the last seconds, cumulative tok/s) from the time actually spent decoding."""
    now = time.time(); r = [x for x in _arm if now - x[0] <= window]
    live = None
    if len(r) >= 2 and r[-1][2] > r[0][2]: live = (r[-1][1] - r[0][1]) / (r[-1][2] - r[0][2]) * 1000
    elif len(r) >= 2: live = 0.0
    cum = _arm[-1][1] / _arm[-1][2] * 1000 if _arm and _arm[-1][2] else None
    return live, cum


def native():
    """v0.7.3+: token counters update only at the end, but scheduler_steps_total is live."""
    try:
        t = urllib.request.urlopen(A.native + "/metrics", timeout=2).read().decode()
        d = {l.split()[0]: float(l.split()[1]) for l in t.splitlines() if l.startswith("velogb10_") and len(l.split()) == 2}
        _steps.append((time.time(), d["velogb10_scheduler_steps_total"], d.get("velogb10_requests_running", 0)))
    except Exception:
        pass
    # v0.7.5+: no [spec-pass]/[mtp-stats] in the log; the spec_lanes arm_tok/arm_ms counters are live = pure decode
    try:
        tok = sum(float(l.split()[1]) for l in t.splitlines() if l.startswith("velogb10_spec_lanes_arm_tok_total"))
        ms = sum(float(l.split()[1]) for l in t.splitlines() if l.startswith("velogb10_spec_lanes_arm_ms_total"))
        g = lambda k: next((float(l.split()[1]) for l in t.splitlines() if l.startswith(k + " ")), 0.0)
        _arm.append((time.time(), tok, ms))
        _ctx.update(used=g("velogb10_kv_positions_used"), cap=g("velogb10_kv_positions_capacity"),
                    prompt=g("velogb10_prompt_tokens_total"), cached=g("velogb10_prompt_tokens_cached_total"))
        restore_hits = g("velogb10_prefix_pool_hits_total") + g("velogb10_kv_spill_hits_total")
        _kv.append((time.monotonic(), _ctx["used"], tok, g("velogb10_requests_running"), restore_hits))
        tc, ts = g("velogb10_time_to_first_token_seconds_count"), g("velogb10_time_to_first_token_seconds_sum")
        u = (tc, ts, _ctx["prompt"], _ctx["cached"])
        if _pf.get("u") and tc > _pf["u"][0]:   # a new request reached its first token
            dt = ts - _pf["u"][1]; new = (_ctx["prompt"] - _pf["u"][2]) - (_ctx["cached"] - _pf["u"][3])
            _pf["last"] = (new / dt if dt > 0 else 0.0, new, _ctx["cached"] - _pf["u"][3], dt, time.time())
        _pf["u"] = u
    except Exception:
        pass


def plain_live(rps, window=15.0):
    """Plain steps without MTP (many streams): each step yields one token per active stream. Estimate (also counts prefill steps)."""
    now = time.time(); r = [x for x in _steps if now - x[0] <= window]
    if len(r) < 2 or r[-1][0] == r[0][0]: return None, 0.0
    sps = (r[-1][1] - r[0][1]) / (r[-1][0] - r[0][0]); run = sum(x[2] for x in r) / len(r)
    return max(0.0, sps - (rps or 0.0)) * run, sps


def logfile():
    if A.log: return A.log
    fs = [f for f in glob.glob(os.path.expanduser("~/logs/velo-*.log")) if "proxy" not in f and "paza" not in f]
    return max(fs, key=os.path.getmtime) if fs else None


class EngineLog:
    """Reads only the new lines of the engine log."""
    RE_SPEC = re.compile(r"^\[spec-pass\] rounds (\d+)")
    RE_STREAM = re.compile(r"^\[req\] stream\s+prompt_tokens=(\d+)")
    RE_MTP = re.compile(r"^\[mtp-stats\].*?gen=(\d+).*?accepted=\d+ \(([\d.]+)%\) tok/round=([\d.]+) ms/round=([\d.]+)")
    RE_DONE = re.compile(r"^\[req\] done\s+tok=(\d+) \(([\d.]+) tok/s wall\) finish=(\S+)")

    def __init__(self):
        self.path = None; self.pos = 0
        self.rounds_hist = deque(maxlen=120)        # (t, rounds)
        self.prompt = None; self.t_start = None
        self.mtp = None; self.done = None; self.history = deque(maxlen=60)
        self.tok_per_round = 3.0

    def read_new(self):
        p = logfile()
        if p != self.path:
            self.path, self.pos = p, (max(0, os.path.getsize(p) - 60000) if p else 0)  # priming: the last finished request
        if not p: return
        if os.path.getsize(p) < self.pos: self.pos = 0
        with open(p, errors="replace") as f:
            f.seek(self.pos); linii = f.readlines(); self.pos = f.tell()
        now = time.time(); amorsare = not getattr(self, "amorsat", False); self.amorsat = True
        for l in linii:
            if m := self.RE_SPEC.match(l):
                if not amorsare: self.rounds_hist.append((now, int(m[1])))
            elif m := self.RE_STREAM.match(l): self.prompt, self.t_start = int(m[1]), (None if amorsare else now)
            elif m := self.RE_MTP.match(l):
                self.mtp = (int(m[1]), float(m[2]), float(m[3]), float(m[4])); self.tok_per_round = float(m[3])
            elif m := self.RE_DONE.match(l):
                self.done = (int(m[1]), float(m[2]), m[3], None if amorsare else now)
                if not amorsare: self.history.append((now, int(m[1]), float(m[2])))

    def decode_live(self, window=15.0):
        """rounds/s over the last seconds x tokens/round (last known value)."""
        now = time.time()
        r = [x for x in self.rounds_hist if now - x[0] <= window]
        if len(r) < 2 or r[-1][0] == r[0][0]: return None
        rps = (r[-1][1] - r[0][1]) / (r[-1][0] - r[0][0])
        return rps * self.tok_per_round if rps > 0 else 0.0

    def rps(self, window=15.0):
        now = time.time(); r = [x for x in self.rounds_hist if now - x[0] <= window]
        if len(r) < 2 or r[-1][0] == r[0][0]: return 0.0
        return max(0.0, (r[-1][1] - r[0][1]) / (r[-1][0] - r[0][0]))


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, rest = line.partition(":"); d[k] = int(rest.split()[0]) // 1024
    return d


_tc = {"t": 0.0, "v": (None, None, None)}
_gpu = {"util": None, "pw": None, "clk": None, "ist": []}
def temperatures():
    if time.time() - _tc["t"] < 2.0: return _tc["v"]
    g = s = n = None
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip().split(",")
        o = [x.strip() for x in o]
        g = int(o[0]) if o and o[0].isdigit() else None
        f = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None   # GPU utilization / power / clock
        if len(o) >= 4:
            u = f(o[1]); _gpu.update(util=u, pw=f(o[2]), clk=f(o[3]))
            if u is not None: _gpu["ist"].append((time.time(), u)); _gpu["ist"] = [x for x in _gpu["ist"] if time.time() - x[0] <= 60]
    except Exception: pass
    try:
        z = [int(open(f).read()) / 1000 for f in glob.glob("/sys/class/thermal/thermal_zone*/temp")]
        s = max(z) if z else None
        for h in glob.glob("/sys/class/hwmon/hwmon*"):
            if open(h + "/name").read().strip() == "nvme": n = int(open(h + "/temp1_input").read()) / 1000
    except Exception: pass
    _tc.update(t=time.time(), v=(g, s, n)); return _tc["v"]


def kv_pool():
    """Shared KV pool occupancy (this fork): active + cached positions from native /metrics, against the server's
    --kv-pool-gb (measured: 20 GB ~ 1.04M tokens with q8 KV). None when the server runs without the pool."""
    try:
        pid = subprocess.run(["pgrep", "-f", "gb10_inference --server"], capture_output=True, text=True).stdout.split()[0]
        argv = open(f"/proc/{pid}/cmdline").read().split("\0")
        flag = lambda f: argv[argv.index(f) + 1] if f in argv and argv.index(f) + 1 < len(argv) else None
        if flag("--kv-pool") != "lazy" or not flag("--kv-pool-gb"): return None
        cap = float(flag("--kv-pool-gb")) * 1.04e6 / 20
        t = urllib.request.urlopen(A.native + "/metrics", timeout=2).read().decode()
        g = lambda k: int(float(re.search(r"^" + k + r" ([0-9.e+]+)", t, re.M).group(1)))
        act, cac = g("velogb10_kv_positions_used"), g("velogb10_kv_positions_cached")
        return act + cac, cap, act, cac
    except Exception: return None


def bar(frac, w=22):
    frac = max(0.0, min(1.0, frac)); k = int(round(frac * w))
    return (red if frac >= 0.9 else grn)("█" * k) + dim("░" * (w - k))


def render(m, j, peak):
    L = [bold("  veloGB10 · Flash-Next EXL3") + dim(f"   {BASE}   ") + dim(time.strftime("%H:%M:%S")),
         dim("  kv-pool fork by AbsolomBane · https://github.com/AbsolomBane/veloGB10"), ""]
    alive = subprocess.run(["pgrep", "-f", "gb10_inference --server"], capture_output=True).returncode == 0
    L.append("  ENGINE   " + (grn("running") if alive else red("STOPPED")) + dim(f"   log {os.path.basename(j.path or '-')}"))
    if m is None:
        L.append("  " + red("the proxy does not answer /metrics (velo-proxy.py on :18300?)"))
    else:
        run = int(m.get("velo:num_requests_running", 0))
        L.append("  REQUESTS running " + bold(str(run)) + dim(f"   max concurrent {int(m.get('velo:max_simultaneous', 0))}"
                 f"   finished {int(m.get('velo:request_success_total', 0))}"))
        L.append(dim(f"           prompt total {int(m.get('velo:prompt_tokens_total', 0)):,}   generated total {int(m.get('velo:generation_tokens_total', 0)):,}"))
    L.append("")
    dm = j.decode_live(); pl, sps = plain_live(j.rps())
    dl = (dm or 0.0) + (pl or 0.0) if (dm is not None or pl is not None) else None
    al, ac = decode_arm()
    if j.decode_live() is None and al is not None: dl, pl = al, None   # v0.7.5+: pure decode from native counters
    if dl is not None and dl > 0: peak[0] = max(peak[0], dl)
    L.append(bold("  DECODE LIVE") + dim("  (MTP rounds/s x tokens/round, 15 s window)"))
    L.append("    now      " + (grn(f"{dl:7.1f} tok/s") if dl else dim("      - tok/s")) +
             dim("   peak ") + grn(f"{peak[0]:.1f}") + dim(f"   tok/round {j.tok_per_round:.2f}"))
    if al is not None and pl is None:
        L.append(dim(f"    pure decode from native counters (no prefill)   average since boot {ac or 0:.1f} tok/s"))
    if _ctx.get("cap"):
        pr = _ctx["prompt"] or 1
        L.append(dim(f"    context {_ctx['used']:,.0f} / {_ctx['cap']:,.0f}   prompt cache {_ctx['cached'] / pr * 100:.0f}% of {_ctx['prompt']:,.0f} tokens"))
    pv = prefill_live()
    L.append(bold("  PREFILL") + dim("  (live estimate from KV progress over 12 s)"))
    L.append("    live est " + (cya(f"{pv:7.0f} tok/s") if pv else dim("      - tok/s")))
    if _pf.get("last"):
        v, new, hit, dt, t = _pf["last"]
        L.append(f"    last     {v:,.0f} tok/s" + dim(f"   {new:,.0f} new + {hit:,.0f} from cache · TTFT {dt:.1f} s · {time.time() - t:.0f} s ago"))
    if pl:
        L.append(dim(f"    of which MTP {dm or 0:.1f} + plain steps without MTP ~{pl:.1f} (engine steps {sps:.1f}/s; estimate, includes prefill)"))
    if j.prompt and j.t_start:
        L.append(dim(f"    current request: prompt {j.prompt:,} tokens, for {time.time() - j.t_start:.0f} s"))
    L.append("")
    L.append(bold("  LAST REQUEST") + dim("  (from the log, at finish)"))
    if j.mtp:
        gen, acc, tr, ms = j.mtp
        L.append(f"    pure decode {grn(f'{tr / ms * 1000:.1f} tok/s')}" + dim(f"   MTP acceptance ") + f"{acc:.1f}%" +
                 dim(f"   {tr:.2f} tok/round · {ms:.1f} ms/round · {gen} tokens"))
    if j.done:
        tok, wall, fin, t = j.done
        when = f"{time.time() - t:.0f} s ago" if t else "before the monitor started"
        L.append(f"    per request {wall:.1f} tok/s" + dim(f" (with prefill)   {tok} tokens · {fin} · {when}"))
    if j.history:
        rec = [x for x in j.history if time.time() - x[0] <= 600]
        if rec:
            tok = sum(x[1] for x in rec)
            L.append(dim(f"    last 10 min: {len(rec)} requests, {tok:,} tokens, average {sum(x[2] for x in rec) / len(rec):.1f} tok/s per request"))
    if not (j.mtp or j.done): L.append(dim("    no request finished yet"))
    L.append("")
    mem = meminfo(); tot, av, fr = mem["MemTotal"], mem["MemAvailable"], mem["MemFree"]
    sw = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
    L.append(bold("  MEMORY") + "    " + bar((tot - av) / tot) + f" {(tot - av) / 1024:6.1f}/{tot / 1024:.1f} GiB used")
    L.append("    available " + (red if av < 4096 else grn)(f"{av / 1024:6.2f} GiB") + dim(f"   free {fr / 1024:.2f} GiB") +
             (dim(f"   swap {sw / 1024:.2f} GiB") if sw else ""))
    temperatures()
    u = _gpu["util"]
    if u is not None:
        med = sum(x[1] for x in _gpu["ist"]) / len(_gpu["ist"]) if _gpu["ist"] else u
        L.append(bold("  GPU") + "       " + bar(u / 100) + f" {u:3.0f}%" + dim(f"   60 s avg {med:.0f}%") +
                 (dim(f"   {_gpu['pw']:.1f} W") if _gpu["pw"] is not None else "") +
                 (dim(f"   SM {_gpu['clk']:.0f} MHz") if _gpu["clk"] is not None else ""))
    bz = kv_pool()
    if bz:
        if u is not None:
            L.append("")
        tok, cap, act, cac = bz
        L.append(bold("  KV POOL") + "   " + bar(tok / cap) + f" {tok / 1e6:5.2f}/{cap / 1e6:.2f}M tokens" +
                 dim(f"   active {act:,} + cache {cac:,}   (~18 KB/token, estimate)"))
    pz = os.path.expanduser("~/logs/velo-guard.log")
    if os.path.exists(pz) and os.path.getsize(pz):
        L.append("    " + red("memory guard stepped in: ") + open(pz).read().strip().splitlines()[-1])
    g, s, n = temperatures()
    t = lambda v, p: dim("-") if v is None else (red if v >= p else grn)(f"{v:.0f} °C")
    L.append("")
    L.append(bold("  TEMP") + "   GPU " + t(g, 85) + dim("   SoC/CPU ") + t(s, 90) + dim("   NVMe ") + t(n, 70) + dim("   (every 2 s)"))
    L.append(dim("  Ctrl+C to exit"))
    return L


def main():
    j = EngineLog(); peak = [0.0]
    if A.once:
        j.read_new(); native(); time.sleep(A.interval); j.read_new(); native(); print("\n".join(render(scrape(), j, peak))); return
    try:
        while True:
            j.read_new(); native()
            lines = render(scrape(), j, peak)
            w = shutil.get_terminal_size((100, 30)).columns
            sys.stdout.write("\033[H\033[2J" + "\n".join(l[: w + 40] for l in lines) + "\n"); sys.stdout.flush()
            time.sleep(A.interval)
    except KeyboardInterrupt:
        print()

main()
