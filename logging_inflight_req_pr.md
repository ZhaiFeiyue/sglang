## Summary

Add debug-level inflight request counter logging to the PD router, allowing operators to see how many requests are currently in-flight through the gateway.

- Two atomic counters (`REQ_SENT` / `REQ_DONE`) track the request lifecycle
- Each log line prints `sent`, `done`, and `inflight = sent - done`
- Guarded by `enabled!(Level::DEBUG)`: at default `--log-level info`, no atomic ops, no formatting — zero overhead

## How to use

```bash
# Start router with debug level to see inflight counters
python3 -m sglang_router.launch_router \
  --pd-disaggregation --port 30000 \
  --prefill http://prefill:8000 --decode http://decode:8000 \
  --log-level debug
```

No flags needed to disable — at the default `--log-level info`, the counters are completely skipped.

## Log output

**`--log-level info` (default)** — no extra output:

```
[2026-04-10 08:09:03] INFO  Starting router on 0.0.0.0:30000
[2026-04-10 08:09:04] INFO  Activated 1 worker(s) (marked as healthy)
```

**`--log-level debug`** — 3 sequential streaming requests:

```
[2026-04-10 08:19:02] DEBUG [REQ_SENT] sent=1 done=0 inflight=1
[2026-04-10 08:19:03] DEBUG [REQ_DONE] sent=1 done=1 inflight=0
[2026-04-10 08:19:03] DEBUG [REQ_SENT] sent=2 done=1 inflight=1
[2026-04-10 08:19:03] DEBUG [REQ_DONE] sent=2 done=2 inflight=0
[2026-04-10 08:19:03] DEBUG [REQ_SENT] sent=3 done=2 inflight=1
[2026-04-10 08:19:03] DEBUG [REQ_DONE] sent=3 done=3 inflight=0
```

**`--log-level debug` + `SGLANG_LOG_MS=true`** — with millisecond timestamps:

```
[2026-04-10 07:32:33.173] DEBUG [REQ_SENT] sent=1 done=0 inflight=1
[2026-04-10 07:32:33.751] DEBUG [REQ_DONE] sent=1 done=1 inflight=0
[2026-04-10 07:32:33.758] DEBUG [REQ_SENT] sent=2 done=1 inflight=1
[2026-04-10 07:32:34.330] DEBUG [REQ_DONE] sent=2 done=2 inflight=0
[2026-04-10 07:32:34.336] DEBUG [REQ_SENT] sent=3 done=2 inflight=1
[2026-04-10 07:32:34.680] DEBUG [REQ_DONE] sent=3 done=3 inflight=0
```

Per-request latency can be read directly from timestamps:

| Request | REQ_SENT | REQ_DONE | Latency |
|---------|----------|----------|---------|
| 1 | 33.173 | 33.751 | 578ms |
| 2 | 33.758 | 34.330 | 572ms |
| 3 | 34.336 | 34.680 | 344ms |

## Test plan

- [x] `--log-level info` + 3 requests → no REQ_SENT/REQ_DONE in logs
- [x] `--log-level debug` + 3 sequential streaming requests → counters increment correctly, inflight=1→0
- [x] `--log-level debug` + `SGLANG_LOG_MS=true` → ms timestamps + counters work together
- [x] Tested on 1P1D PD disaggregation (MI355X, DeepSeek-R1 MXFP4, EP4 prefill + EP8 decode)
- [x] `rustfmt --check` passes
- [x] Pre-commit hooks pass
