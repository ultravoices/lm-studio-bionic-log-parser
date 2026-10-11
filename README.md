# lm-studio-bionic-log-parser

Parse and summarize session logs from **LM Studio Bionic** and its **Splash** inference backend — without loading log files into an LLM context.

The parser reads the harness log lines (model loads, per-request `input / cached / output / TTFT / tok/s` stats, engine stops) plus engine-internal diagnostics (Metal OOM, prompt-cache disk usage, cache-restore efficiency), and prints per-file, per-model performance summaries with automatic watch-item flagging.

Single file, standard library only, Python 3.9+. No dependencies, no network, nothing leaves your machine.

## Quickstart

```bash
python3 logstats.py                          # parse everything under ~/.lmstudio/apps/bionic/server-logs (recurses)
python3 logstats.py <file> [file|dir ...]    # specific files or directories (dirs searched for *.log)
python3 logstats.py --model A3B <file>       # only requests attributed to a model (substring match)
python3 logstats.py --json <file>            # machine-readable summary
```

Example output:

```
== 2026-10-11.1.log  (45 KB)  models: Qwen3.8-27B-Splash, Qwen3.6-35B-A3B-Splash
  313 reqs · 137,494 out · peak ctx 115,415 · cached 93% · loads 5 stops 5
  Qwen3.6-35B-A3B-Splash  mean  199.1  min   88.8  max  315.5  n=230
    ctx 8-30K   mean  163.1  min  131.7  max  254.3  n=17
    ctx 30-80K  mean  204.6  min  101.9  max  315.5  n=135
  Qwen3.8-27B-Splash      mean   67.0  min   32.8  max  134.1  n=83
    ctx >80K    mean   64.4  min   38.2  max   91.7  n=37
```

## What it reports

Per file:

- **Volume** — request count, total output tokens, peak context, prefix-cache hit fraction, load/stop counts
- **Speed** — per-model `tok/s` (mean/min/max) overall and in context buckets (`<8K`, `8-30K`, `30-80K`, `>80K`)
- **Cache** — prompt-cache disk usage vs cap, record count, restore efficiency
- **WATCH flags**, auto-detected:
  - GPU out-of-memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`)
  - engine-level errors / runtime errors / failed operations
  - requests below 30 tok/s
  - rapid load/stop cycles (crash-restart signature)
  - prompt-cache disk usage above 50% of cap

Failure detection matches engine-level patterns only; the word "error" appearing inside logged *content* (JSON responses, model output) is deliberately not counted.

## Where the logs live

- Canonical location: `~/.lmstudio/apps/bionic/server-logs/<YYYY-MM>/YYYY-MM-DD.N.log` — the default search root
- Bionic also mirrors copies under `~/.lmstudio/scratchpads/<session>/…` (used as a fallback when `server-logs` does not exist). Do **not** pass both roots in one run — the files are duplicates and requests would be double-counted
- Per-project session data: `~/.lmstudio/apps/bionic/projects/<id>/.internal/ng-sessions.sqlite`
- Splash engine internals (OOM, cache stats) appear interleaved in the same server log files

## Privacy

Session logs can contain conversation content. This tool **only aggregates metrics** — it never prints message text — but treat log files as private data. The `.gitignore` excludes `*.log` (except the synthetic `examples/sample.log`) for exactly this reason.

## Development

```bash
python3 -m unittest discover -s tests -v   # stdlib-only test suite
```

```
.
├── logstats.py              # the CLI (stdlib only)
├── examples/sample.log      # synthetic log exercising every line type
├── tests/test_logstats.py   # parser, JSON, and CLI smoke tests
└── .github/workflows/ci.yml # runs the suite on push/PR
```

## Log line formats parsed

```
[..] Loading · <org/model>                 engine load (starts model attribution)
[..] Ready · <org/model> · context 256K …  engine ready
[..] Done · input N · cached N · output N · tools … · TTFT Ns · N tok/s
[..] Stopping · releasing engine resources / Server stopped
[..][cache_store][INFO] VLM prompt cache disk usage: used_mib=… cap_mib=… records=…
[..][coordinator][INFO] Prompt cache restore: … lifetime_efficiency=…%
[..][ERROR] / RuntimeError / OutOfMemory   engine failures
```

Requests occurring before a file's first `Loading` line are attributed to the file's sole model when exactly one model appears in it; otherwise they are reported as `?`.

## License

MIT — see [LICENSE](LICENSE).
