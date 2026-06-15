#!/usr/bin/env python3
"""
SGLang Metrics Trace Recorder

Scrapes Prometheus metrics from SGLang server(s) during a benchmark,
then exports a Chrome Trace JSON file for timeline visualization.

Usage:
    # PD disaggregation: prefill + decode workers + router
    python sglang_metrics_trace.py \
        -t prefill=http://localhost:30000 decode=http://localhost:30001 \
        -t router=http://localhost:29000 \
        -i 0.5 -o pd_trace.json

    # Single server
    python sglang_metrics_trace.py -t http://localhost:30000 -o trace.json

    # Auto-stop after 120 seconds
    python sglang_metrics_trace.py -t http://localhost:30000 -d 120

    # View result: open chrome://tracing  or  https://ui.perfetto.dev
"""

import argparse
import json
import math
import re
import signal
import sys
import time
from collections import defaultdict
from urllib.request import Request, urlopen

# ── Prometheus text format parser ───────────────────────────────────────────

_LINE_RE = re.compile(r"^([\w:.]+)(?:\{([^}]*)\})?\s+(.+)$")
_TYPE_RE = re.compile(r"^#\s+TYPE\s+([\w:.]+)\s+(\w+)")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus(text):
    """Parse Prometheus exposition text.

    Returns:
        gauges: dict of metric_name -> value
        histograms: dict of base_name -> {"sum", "count", "buckets": [(le, cumcount)]}
    """
    types = {}
    gauges = {}
    histograms = defaultdict(lambda: {"sum": 0.0, "count": 0.0, "buckets": []})

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        type_match = _TYPE_RE.match(line)
        if type_match:
            types[type_match.group(1)] = type_match.group(2)
            continue
        if line.startswith("#"):
            continue

        metric_match = _LINE_RE.match(line)
        if not metric_match:
            continue

        name = metric_match.group(1)
        labels_str = metric_match.group(2) or ""
        try:
            value = float(metric_match.group(3))
        except ValueError:
            continue
        if math.isnan(value):
            continue

        labels = dict(_LABEL_RE.findall(labels_str))

        if name.endswith("_bucket"):
            base = name[:-7]
            le_str = labels.get("le", "+Inf")
            try:
                le_val = float("inf") if le_str in ("+Inf", "Inf") else float(le_str)
            except ValueError:
                continue
            histograms[base]["buckets"].append((le_val, value))

        elif name.endswith("_sum") and types.get(name[:-4]) == "histogram":
            histograms[name[:-4]]["sum"] = value

        elif name.endswith("_count") and types.get(name[:-6]) == "histogram":
            histograms[name[:-6]]["count"] = value

        else:
            if "worker" in labels:
                worker = labels["worker"]
                if ":" in worker:
                    worker = worker.rsplit(":", 1)[-1]
                gauges[f"{name}[{worker}]"] = value
            elif name not in gauges:
                gauges[name] = value

    return gauges, dict(histograms)


def histogram_quantile(q, buckets):
    """Approximate quantile from histogram bucket data."""
    if not buckets:
        return 0.0
    sorted_buckets = sorted(buckets)
    total = 0.0
    for le, count in sorted_buckets:
        if le == float("inf"):
            total = count
            break
    if total <= 0:
        return 0.0

    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in sorted_buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return prev_le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    return prev_le


def delta_histogram(current, previous):
    """Compute per-interval histogram by subtracting previous snapshot."""
    if previous is None:
        return current
    prev_map = {le: cnt for le, cnt in previous["buckets"]}
    return {
        "sum": current["sum"] - previous["sum"],
        "count": current["count"] - previous["count"],
        "buckets": [
            (le, cnt - prev_map.get(le, 0.0)) for le, cnt in current["buckets"]
        ],
    }


# ── Metric track definitions ───────────────────────────────────────────────
# (track_name, [(prometheus_name, series_label), ...])

GAUGE_TRACKS = [
    (
        "Requests",
        [
            ("sglang:num_running_reqs", "running"),
            ("sglang:num_queue_reqs", "queued"),
        ],
    ),
    ("Token Usage", [("sglang:token_usage", "ratio")]),
    ("Throughput", [("sglang:gen_throughput", "tok/s")]),
    ("Cache Hit Rate", [("sglang:cache_hit_rate", "rate")]),
    (
        "PD Prefill Queues",
        [
            ("sglang:num_prefill_prealloc_queue_reqs", "prealloc"),
            ("sglang:num_prefill_inflight_queue_reqs", "inflight"),
        ],
    ),
    (
        "PD Decode Queues",
        [
            ("sglang:num_decode_prealloc_queue_reqs", "prealloc"),
            ("sglang:num_decode_transfer_queue_reqs", "transfer"),
        ],
    ),
]

# (track_name, histogram_base_name)
HISTOGRAM_TRACKS = [
    ("TTFT (s)", "sglang:time_to_first_token_seconds"),
    ("TPOT (s)", "sglang:inter_token_latency_seconds"),
    ("E2E Latency (s)", "sglang:e2e_request_latency_seconds"),
    ("KV Transfer Speed (GB/s)", "sglang:kv_transfer_speed_gb_s"),
    ("KV Transfer Latency (ms)", "sglang:kv_transfer_latency_ms"),
]

# Counter-based tracks: compute rate = delta / dt
RATE_TRACKS = [
    (
        "Token Rate",
        [
            ("sglang:prompt_tokens_total", "prompt/s"),
            ("sglang:generation_tokens_total", "gen/s"),
        ],
    ),
    ("Request Rate", [("sglang:num_requests_total", "req/s")]),
    (
        "PD Failure Rate",
        [
            ("sglang:num_bootstrap_failed_reqs_total", "bootstrap/s"),
            ("sglang:num_transfer_failed_reqs_total", "transfer/s"),
        ],
    ),
]


# ── Chrome Trace Recorder ──────────────────────────────────────────────────


class MetricsRecorder:
    def __init__(self, targets, interval, output, duration):
        self.targets = targets  # [(label, url), ...]
        self.interval = interval
        self.output = output
        self.duration = duration
        self.events = []
        self.running = True
        self.start_time = None
        self.prev_snapshots = {}  # label -> (gauges, histograms, timestamp)
        self.summary = defaultdict(list)  # (label, track/series) -> [values]

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, *_):
        self.running = False

    def _relative_ts_us(self, wall_time):
        return (wall_time - self.start_time) * 1e6

    def _add_counter_event(self, name, ts_us, pid, args):
        self.events.append(
            {"name": name, "ph": "C", "ts": ts_us, "pid": pid, "tid": 0, "args": args}
        )

    def _scrape(self, url):
        try:
            req = Request(f"{url}/metrics", headers={"Accept": "text/plain"})
            with urlopen(req, timeout=5) as resp:
                return resp.read().decode()
        except Exception:
            return None

    def _process_snapshot(self, pid, label, gauges, histograms, now):
        ts = self._relative_ts_us(now)
        prev_gauges, prev_hists, prev_time = self.prev_snapshots.get(
            label, ({}, {}, now)
        )
        dt = max(now - prev_time, 1e-9)

        # Gauge tracks
        for track_name, specs in GAUGE_TRACKS:
            args = {}
            for prom_name, series in specs:
                if prom_name in gauges:
                    args[series] = round(gauges[prom_name], 4)
            if args:
                self._add_counter_event(track_name, ts, pid, args)
                for k, v in args.items():
                    self.summary[(label, f"{track_name}/{k}")].append(v)

        # Histogram tracks: per-interval percentiles
        for track_name, base_name in HISTOGRAM_TRACKS:
            if base_name not in histograms:
                continue
            delta = delta_histogram(histograms[base_name], prev_hists.get(base_name))
            if delta["count"] <= 0:
                continue
            avg = delta["sum"] / delta["count"]
            p50 = histogram_quantile(0.5, delta["buckets"])
            p99 = histogram_quantile(0.99, delta["buckets"])
            args = {
                "avg": round(avg, 6),
                "p50": round(p50, 6),
                "p99": round(p99, 6),
            }
            self._add_counter_event(track_name, ts, pid, args)
            self.summary[(label, f"{track_name}/p50")].append(p50)
            self.summary[(label, f"{track_name}/p99")].append(p99)

        # Rate tracks: counter deltas → per-second rate
        for track_name, specs in RATE_TRACKS:
            args = {}
            for prom_name, series in specs:
                curr_val = gauges.get(prom_name)
                prev_val = prev_gauges.get(prom_name)
                if curr_val is not None and prev_val is not None:
                    rate = max(0.0, (curr_val - prev_val) / dt)
                    args[series] = round(rate, 2)
            if args:
                self._add_counter_event(track_name, ts, pid, args)
                for k, v in args.items():
                    self.summary[(label, f"{track_name}/{k}")].append(v)

        # Dynamic per-worker metrics from router (smg_worker_requests_active)
        worker_active = {}
        for key, val in gauges.items():
            if key.startswith("smg_worker_requests_active["):
                worker_port = key.split("[")[1].rstrip("]")
                worker_active[f"w:{worker_port}"] = round(val, 1)
        if worker_active:
            self._add_counter_event("Router Worker Reqs", ts, pid, worker_active)

        self.prev_snapshots[label] = (gauges, histograms, now)

    def run(self):
        self.start_time = time.time()

        print("SGLang Metrics Trace Recorder")
        print(f"  Output:   {self.output}")
        print(f"  Interval: {self.interval}s")
        if self.duration:
            print(f"  Duration: {self.duration}s (auto-stop)")
        print("  Targets:")
        for i, (label, url) in enumerate(self.targets):
            print(f"    [{i}] {label} -> {url}/metrics")
            self.events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": i,
                    "args": {"name": f"{label} ({url})"},
                }
            )
        print("\nPress Ctrl+C to stop and export.\n")

        while self.running:
            loop_start = time.time()

            if self.duration and (loop_start - self.start_time) >= self.duration:
                break

            for i, (label, url) in enumerate(self.targets):
                text = self._scrape(url)
                if text is None:
                    continue
                gauges, histograms = parse_prometheus(text)
                self._process_snapshot(i, label, gauges, histograms, time.time())

            elapsed = time.time() - self.start_time
            sys.stdout.write(
                f"\r  >> {elapsed:7.1f}s | {len(self.events):>7} events collected"
            )
            sys.stdout.flush()

            sleep_time = self.interval - (time.time() - loop_start)
            if sleep_time > 0 and self.running:
                time.sleep(sleep_time)

        print("\n")
        self._export()

    def _export(self):
        trace = {
            "traceEvents": self.events,
            "displayTimeUnit": "ms",
        }
        with open(self.output, "w") as f:
            json.dump(trace, f)

        duration = time.time() - self.start_time
        print(f"Exported {len(self.events)} events ({duration:.1f}s recording)")
        print(f"  File: {self.output}")
        print(f"  View: chrome://tracing  or  https://ui.perfetto.dev")
        print()

        if not self.summary:
            return

        print("-- Summary " + "-" * 60)
        col1 = max(len(lb) for (lb, _) in self.summary)
        col2 = max(len(tk) for (_, tk) in self.summary)
        for (label, track), values in sorted(self.summary.items()):
            if not values:
                continue
            avg = sum(values) / len(values)
            lo, hi = min(values), max(values)
            print(
                f"  {label:>{col1}} | {track:<{col2}} | "
                f"avg={avg:10.4f}  min={lo:10.4f}  max={hi:10.4f}"
            )
        print()


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_target(spec):
    """Parse 'label=url' or plain 'url' into (label, url)."""
    if "=" in spec and not spec.startswith("http"):
        label, url = spec.split("=", 1)
        return label.strip(), url.strip().rstrip("/")
    url = spec.strip().rstrip("/")
    port = url.rsplit(":", 1)[-1].split("/")[0] if ":" in url else "server"
    return f"server:{port}", url


def main():
    parser = argparse.ArgumentParser(
        description="Record SGLang Prometheus metrics and export as Chrome Trace JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single server
  python sglang_metrics_trace.py -t http://localhost:30000

  # PD disaggregation
  python sglang_metrics_trace.py \\
      -t prefill=http://localhost:30000 \\
      -t decode=http://localhost:30001 \\
      -t router=http://localhost:29000 \\
      -i 0.5 -o pd_trace.json

  # Auto-stop after 2 minutes
  python sglang_metrics_trace.py -t http://localhost:30000 -d 120
        """,
    )
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        action="append",
        required=True,
        help='Scrape targets: "http://host:port" or "label=http://host:port"',
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Scrape interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="sglang_trace.json",
        help="Output file path (default: sglang_trace.json)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Auto-stop after N seconds (default: manual Ctrl+C)",
    )

    args = parser.parse_args()

    # Flatten nested lists from append action
    all_specs = []
    for group in args.targets:
        all_specs.extend(group)
    targets = [parse_target(s) for s in all_specs]

    recorder = MetricsRecorder(targets, args.interval, args.output, args.duration)
    recorder.run()


if __name__ == "__main__":
    main()
