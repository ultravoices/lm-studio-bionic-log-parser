#!/usr/bin/env python3
"""logstats — summarize Bionic / Splash local-model session logs.

Parses the harness log lines:
  Loading · <model>            (engine load; starts a model attribution window)
  Ready · <model> · context N
  Done · input N · cached N · output N · ... · TTFT Ns · N tok/s
  Stopping / Server stopped
plus engine-internal lines for OOM, prompt-cache disk usage, and cache
restore efficiency. Generic words like "error" inside logged *content*
are deliberately not counted; only engine-level failure patterns are.

Usage:
  python3 logstats.py [PATH ...]        # files and/or directories (recursed for *.log)
  python3 logstats.py --json [PATH ...] # machine-readable
  python3 logstats.py --model Qwen3.8   # only report requests attributed to a model
                                        # (substring match on the short name)
  python3 logstats.py --buckets 16k,32k,64k,128k,256k
                                        # custom context bucket edges (k/m suffix or
                                        # bare token counts; last bucket runs to infinity)

With no PATH, searches ~/.lmstudio/apps/bionic/server-logs recursively,
falling back to ~/.lmstudio/scratchpads if that directory does not exist.
Stdlib only.
"""

import hashlib
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

# --- line patterns ---------------------------------------------------------

RE_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})\]")
RE_LOAD = re.compile(r"Loading \u00b7 ([\w./@-]+)")
RE_DONE = re.compile(
    r"input ([\d,]+) \u00b7 cached ([\d,]+) \u00b7 output ([\d,]+)"
    r".*?TTFT ([\d.]+)s \u00b7 ([\d.]+) tok/s"
)
RE_STOP = re.compile(r"Stopping|Server stopped")
RE_CACHE_DISK = re.compile(
    r"used_mib=([\d.]+)\s+cap_mib=([\d.]+)\s+lifetime_evicted_mib=([\d.]+)"
    r"(?:.*?records=(\d+))?"
)
RE_CACHE_RESTORE = re.compile(r"lifetime_efficiency=([\d.]+)%")

# Engine-level failure patterns (content dumps like {"error": ...} are ignored).
ERR_PATTERNS = [
    ("gpu_oom", re.compile(r"OutOfMemory|Insufficient Memory")),
    ("engine_error", re.compile(r"\[ERROR\]")),
    ("runtime_error", re.compile(r"RuntimeError|Traceback")),
    ("failure", re.compile(r"Failed to \w+")),
]

# Context-size bucket edges (input token counts); the final bucket runs to
# infinity. Override with --buckets (see parse_bucket_spec).
DEFAULT_BUCKET_EDGES = [4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]


def _unit(n):
    if n >= 1_000_000:
        return "M", 1_000_000
    if n >= 1_000:
        return "K", 1_000
    return "", 1


def _fmt_edge(n):
    u, s = _unit(n)
    return f"{n / s:g}{u}" if u else str(n)


def _fmt_range(lo, hi):
    """Label a [lo, hi) range: '4-8K', '8K-1M', '<4K', '100-100K', '>256K'."""
    if hi == float("inf"):
        return f">{_fmt_edge(lo)}"
    if lo == 0:
        return f"<{_fmt_edge(hi)}"
    su, sm = _unit(hi)
    if lo >= sm:
        return f"{lo / sm:g}-{hi / sm:g}{su}"
    return f"{_fmt_edge(lo)}-{hi / sm:g}{su}"


def build_buckets(edges):
    """Edge list -> [(label, lo, hi), ...]; the last bucket is open-ended."""
    out, prev = [], 0
    for e in edges:
        out.append((_fmt_range(prev, e), prev, e))
        prev = e
    out.append((f">{_fmt_edge(prev)}", prev, float("inf")))
    return out


def parse_bucket_spec(spec):
    """Parse '4k,8k,16k' / '256k,1m' / bare ints into buckets; None if malformed."""
    try:
        edges = []
        for part in spec.split(","):
            p = part.strip().lower()
            if not p:
                return None
            mult = 1
            if p.endswith("k"):
                mult, p = 1_000, p[:-1]
            elif p.endswith("m"):
                mult, p = 1_000_000, p[:-1]
            v = int(float(p)) * mult
            if v <= 0:
                return None
            edges.append(v)
    except ValueError:
        return None
    if not edges or any(a >= b for a, b in zip(edges, edges[1:])):
        return None
    return build_buckets(edges)

# Canonical first; scratchpads is a mirror (same files) used as fallback.
DEFAULT_ROOTS = [
    ".lmstudio/apps/bionic/server-logs",
    ".lmstudio/scratchpads",
]


def default_roots(home=None):
    """Canonical server-logs dir if it exists; else the scratchpad mirror.

    Never both: scratchpad copies are point-in-time snapshots (often stale
    relative to the live server log), so mixing them double-counts requests.
    """
    home = Path.home() if home is None else Path(home)
    canonical = home / DEFAULT_ROOTS[0]
    if canonical.is_dir():
        return [str(canonical)]
    return [str(home / r) for r in DEFAULT_ROOTS[1:] if (home / r).is_dir()]


# --- parsing ---------------------------------------------------------------

def parse_file(path) -> dict:
    path = Path(path)
    reqs = []
    models, loads, stops = [], 0, 0
    cur_model = None
    err_counts = {}
    cache_used = cache_cap = cache_records = cache_evicted = None
    cache_eff = None
    load_times = []

    with open(path, errors="ignore") as fh:
        for line in fh:
            m = RE_LOAD.search(line)
            if m:
                loads += 1
                cur_model = m.group(1)
                if cur_model not in models:
                    models.append(cur_model)
                ts = RE_TS.match(line)
                if ts:
                    load_times.append(ts.group(1) + " " + ts.group(2))
            if RE_STOP.search(line):
                stops += 1
            m = RE_DONE.search(line)
            if m:
                i, c, o = (int(x.replace(",", "")) for x in m.groups()[:3])
                reqs.append({
                    "model": cur_model,
                    "input": i, "cached": c, "output": o,
                    "ttft": float(m.group(4)), "tps": float(m.group(5)),
                })
            for name, pat in ERR_PATTERNS:
                if pat.search(line):
                    err_counts[name] = err_counts.get(name, 0) + 1
                    break  # one bucket per line
            m = RE_CACHE_DISK.search(line)
            if m:
                u, c2, ev = (float(m.group(1)), float(m.group(2)),
                             float(m.group(3)))
                # per-lifetime counters: max across lines == max within one
                # lifetime (counters reset to 0 on each model load)
                cache_cap = max(cache_cap or 0, c2) or None
                cache_used = max(cache_used or 0, u) or None
                cache_evicted = max(cache_evicted or 0, ev) or None
                if m.group(4):
                    cache_records = max(cache_records or 0, int(m.group(4)))
            m = RE_CACHE_RESTORE.search(line)
            if m:
                cache_eff = max(cache_eff or 0.0, float(m.group(1)))

    # Files often start mid-session (the load was logged in the previous day's
    # file). If exactly one model was loaded in this file, attribute the
    # pre-load requests to it; otherwise they stay as "?".
    if len(models) == 1:
        for r in reqs:
            if r["model"] is None:
                r["model"] = models[0]

    return {
        "file": str(path),
        "size_kb": round(path.stat().st_size / 1024),
        "models": [m.rsplit("/", 1)[-1] for m in models],
        "loads": loads, "stops": stops,
        "load_times": load_times,
        "requests": reqs,
        "errors": err_counts,
        "cache": {
            "used_gib": round(cache_used / 1024, 1) if cache_used else None,
            "cap_gib": round(cache_cap / 1024, 1) if cache_cap else None,
            "evicted_gib": round(cache_evicted / 1024, 1) if cache_evicted else None,
            "records": cache_records,
            "restore_eff_pct": round(cache_eff, 2) if cache_eff else None,
        },
    }


# --- reporting -------------------------------------------------------------

def fstat(xs):
    if not xs:
        return "n=0"
    return f"mean {st.mean(xs):6.1f}  min {min(xs):6.1f}  max {max(xs):7.1f}  n={len(xs)}"


def short(model):
    return (model or "?").rsplit("/", 1)[-1][:24]


def watch_items(fs: dict):
    items = []
    e = fs["errors"]
    if e.get("gpu_oom"):
        items.append(f"GPU OOM x{e['gpu_oom']} (kIOGPU...OutOfMemory)")
    if e.get("engine_error") or e.get("runtime_error"):
        items.append(f"engine errors: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(e.items())
            if k in ("engine_error", "runtime_error", "failure")))
    c = fs["cache"]
    if c["cap_gib"] and c["used_gib"] is not None:
        frac = c["used_gib"] / c["cap_gib"]
        if frac > 0.5:
            items.append(f"prompt-cache disk at {frac:.0%} of cap "
                         f"({c['used_gib']:.0f}/{c['cap_gib']:.0f} GiB)")
    if (c["evicted_gib"] or 0) > 10:
        items.append(f"prompt-cache evicted {c['evicted_gib']:.0f} GiB "
                     f"(eviction churn — cache was at its disk cap)")
    slow = [r for r in fs["requests"] if r["tps"] < 30]
    if slow:
        items.append(f"{len(slow)} request(s) below 30 tok/s "
                     f"(min {min(r['tps'] for r in slow):.1f})")
    # crash-restart signature: >=3 loads within 120 s
    ts = [t.split(" ")[1] for t in fs["load_times"]]
    if len(ts) >= 3:
        hms = [tuple(int(x) for x in t.split(":")) for t in ts]
        for a, b, c_ in zip(hms, hms[1:], hms[2:]):
            if (b[0] * 3600 + b[1] * 60 + b[2]) - (a[0] * 3600 + a[1] * 60 + a[2]) <= 120 \
               and (c_[0] * 3600 + c_[1] * 60 + c_[2]) - (a[0] * 3600 + a[1] * 60 + a[2]) <= 120:
                items.append("rapid load/stop cycles (possible crash-restart)")
                break
    return items


def report(fs: dict, model_filter=None, indent="  ", buckets=None):
    reqs = [r for r in fs["requests"]
            if not model_filter or model_filter.lower() in (r["model"] or "").lower()]
    if not reqs:
        print(f"== {os.path.basename(fs['file'])}  ({fs['size_kb']} KB)  no Done lines")
        for w in watch_items(fs):
            print(f"{indent}WATCH: {w}")
        return

    out = sum(r["output"] for r in reqs)
    peak = max(r["input"] + r["output"] for r in reqs)
    cfr = sum(r["cached"] / r["input"] for r in reqs if r["input"]) / len(reqs)
    print(f"== {os.path.basename(fs['file'])}  ({fs['size_kb']} KB)  "
          f"models: {', '.join(fs['models']) or '?'}")
    print(f"{indent}{len(reqs)} reqs \u00b7 {out:,} out \u00b7 peak ctx {peak:,} \u00b7 "
          f"cached {cfr:.0%} \u00b7 loads {fs['loads']} stops {fs['stops']}")

    by_model = {}
    for r in reqs:
        by_model.setdefault(short(r["model"]), []).append(r)
    for m, rs in sorted(by_model.items()):
        big = [r["tps"] for r in rs if r["output"] >= 100]
        print(f"{indent}{m:20s} {fstat([r['tps'] for r in rs])}")
        if big:
            for label, lo, hi in buckets or build_buckets(DEFAULT_BUCKET_EDGES):
                b = [r["tps"] for r in rs if r["output"] >= 100 and lo <= r["input"] < hi]
                if b:
                    print(f"{indent}  ctx {label:9s} {fstat(b)}")

    for w in watch_items(fs):
        print(f"{indent}WATCH: {w}")


def to_json(fs: dict):
    """Compact JSON: per-model aggregates + watch items, raw requests dropped."""
    by_model = {}
    for r in fs["requests"]:
        by_model.setdefault(short(r["model"]), []).append(
            [r["input"], r["cached"], r["output"], r["ttft"], r["tps"]])
    return {
        "file": fs["file"], "models": fs["models"],
        "loads": fs["loads"], "stops": fs["stops"],
        "requests": len(fs["requests"]),
        "output_tokens": sum(r["output"] for r in fs["requests"]),
        "peak_context": max((r["input"] + r["output"] for r in fs["requests"]), default=0),
        "per_model": {m: {"n": len(v),
                          "tps_mean": round(st.mean(x[4] for x in v), 1),
                          "tps_min": round(min(x[4] for x in v), 1),
                          "tps_max": round(max(x[4] for x in v), 1)}
                       for m, v in by_model.items()},
        "errors": fs["errors"], "cache": fs["cache"],
        "watch": watch_items(fs),
    }


def _fingerprint(p: Path):
    """(size, md5) of a file, or None if unreadable."""
    try:
        size = p.stat().st_size
        h = hashlib.md5()
        with open(p, "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return (size, h.hexdigest())
    except OSError:
        return None


def collect(paths):
    out = []
    for p in paths:
        p = Path(os.path.expanduser(p))
        if p.is_dir():
            out.extend(sorted(p.rglob("*.log")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"warn: {p} not found", file=sys.stderr)
    # dedupe by path, then by content (mirror copies of the same file),
    # preserving order
    seen_paths, seen_content, files = set(), set(), []
    for f in out:
        if str(f) in seen_paths:
            continue
        seen_paths.add(str(f))
        fp = _fingerprint(f)
        if fp is not None:
            if fp in seen_content:
                continue
            seen_content.add(fp)
        files.append(f)
    return files


def main(argv):
    args = argv[1:]
    as_json = "--json" in args
    model_filter = None
    buckets = None
    if "--model" in args or "--buckets" in args:
        if "--model" in args:
            i = args.index("--model")
            if i + 1 < len(args):
                model_filter = args[i + 1]
            args = args[:i] + args[i + 2:]
        if "--buckets" in args:
            i = args.index("--buckets")
            if i + 1 < len(args):
                buckets = parse_bucket_spec(args[i + 1])
                if buckets is None:
                    print("bad --buckets spec (want e.g. 16k,32k,64k,128k,256k)",
                          file=sys.stderr)
                    return 1
            args = args[:i] + args[i + 2:]
        args = [a for a in args if a != "--json"]
    else:
        args = [a for a in args if a != "--json"]
    paths = args or default_roots()
    if not paths:
        print("no default log directory found (looked for "
              + ", ".join("~/" + r for r in DEFAULT_ROOTS) + ")",
              file=sys.stderr)
        return 1

    files = collect(paths)
    if not files:
        print("no log files found", file=sys.stderr)
        return 1

    for f in files:
        fs = parse_file(f)
        if as_json:
            print(json.dumps(to_json(fs)))
        else:
            report(fs, model_filter, buckets=buckets)
            if f is not files[-1]:
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
