"""Endpoint rendezvous for PP stage-disaggregation.

Each stage process stands up its own activation IOEngine on a dynamic ZMQ port
(``MoriActivationTransport.endpoint``). Before transfers can begin, every stage
must learn its neighbors' (and, for tied weights / broadcast, all stages')
endpoints for the *same* attn_tp_rank.

This module provides a tiny, dependency-free HTTP registry:

* The coordinator (stage 0, attn_tp_rank 0) hosts the registry on
  ``--pp-activation-bootstrap-host:--pp-activation-bootstrap-port``.
* Every stage/tp-rank POSTs its entry, then polls GET until all
  ``num_stages * attn_tp_size`` entries are present (a barrier), and receives
  the full table.

The full mori-scheduler router (C7) can later take over this role, but this
keeps stage-disaggregation independently launchable and testable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# (stage_id, attn_tp_rank) -> endpoint
EndpointTable = Dict[Tuple[int, int], str]


class _RegistryState:
    def __init__(self, expected: int):
        self.lock = threading.Lock()
        self.expected = expected
        self.table: Dict[str, str] = {}  # "stage:tp" -> endpoint

    def add(self, key: str, endpoint: str) -> Tuple[Dict[str, str], bool]:
        """Insert/overwrite an entry. Returns (table snapshot, was_present).

        ``was_present`` true with a *changed* endpoint signals a stage restart
        (the prior table was already complete), which the wiring layer uses to
        trigger reconnect handshakes instead of the initial barrier."""
        with self.lock:
            prior = self.table.get(key)
            was_present = prior is not None and prior != endpoint
            self.table[key] = endpoint
            return dict(self.table), was_present


class _Handler(BaseHTTPRequestHandler):
    state: _RegistryState = None  # set on the server instance's handler class

    def log_message(self, *args):  # silence default stderr logging
        return

    def _send(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        table, was_present = self.state.add(payload["key"], payload["endpoint"])
        self._send(
            200,
            {
                "complete": len(table) >= self.state.expected,
                "reregister": was_present,
                "table": table,
            },
        )

    def do_GET(self):
        with self.state.lock:
            table = dict(self.state.table)
        self._send(200, {"complete": len(table) >= self.state.expected, "table": table})


def start_registry(host: str, port: int, expected: int) -> ThreadingHTTPServer:
    """Start the coordinator registry server (non-blocking)."""
    state = _RegistryState(expected)

    handler_cls = type("_BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(
        "Stage rendezvous registry up at %s:%d (expecting %d entries)",
        host,
        port,
        expected,
    )
    return server


def register_and_collect(
    coordinator_host: str,
    coordinator_port: int,
    stage_id: int,
    attn_tp_rank: int,
    endpoint: str,
    expected: int,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
) -> Tuple[EndpointTable, bool]:
    """Register this process's endpoint and block until the full table (all
    ``expected`` entries) is available. Returns (table, reregister) where
    ``reregister`` is True when this endpoint replaced a prior one (i.e. this
    process is a *restart* of an already-registered stage)."""
    base = f"http://{coordinator_host}:{coordinator_port}"
    key = f"{stage_id}:{attn_tp_rank}"
    body = json.dumps({"key": key, "endpoint": endpoint}).encode()

    deadline = time.monotonic() + timeout_s
    table: Dict[str, str] = {}
    reregister = False
    # POST (with retries: the coordinator may not be up yet).
    while True:
        try:
            req = urllib.request.Request(f"{base}/register", data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                table = data["table"]
                reregister = data.get("reregister", False)
            break
        except (urllib.error.URLError, ConnectionError, OSError):
            if time.monotonic() > deadline:
                raise TimeoutError("stage rendezvous: coordinator unreachable")
            time.sleep(poll_interval_s)

    # Poll GET until complete.
    while len(table) < expected:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"stage rendezvous: only {len(table)}/{expected} stages registered"
            )
        time.sleep(poll_interval_s)
        try:
            with urllib.request.urlopen(f"{base}/get", timeout=10) as resp:
                table = json.loads(resp.read())["table"]
        except (urllib.error.URLError, ConnectionError, OSError):
            continue

    out: EndpointTable = {}
    for k, ep in table.items():
        sid, tp = k.split(":")
        out[(int(sid), int(tp))] = ep
    return out, reregister


def _main() -> None:
    """Run the rendezvous registry as a standalone, long-lived process.

    Hosting the registry outside the stage processes (e.g. from a supervisor)
    lets *any* stage -- including stage 0 -- restart and re-rendezvous without
    losing the endpoint table, which is what enables arbitrary PP-rank hot
    restart. Stages auto-detect an external registry (their in-process
    ``start_registry`` no-ops on ``OSError`` when the port is already bound)."""
    import argparse

    ap = argparse.ArgumentParser(description="PP stage rendezvous registry")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9123)
    ap.add_argument(
        "--expected",
        type=int,
        required=True,
        help="num_stages * attn_tp_size (barrier size for initial bring-up)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [rendezvous] %(message)s"
    )
    start_registry(args.host, args.port, args.expected)
    logger.info("standalone rendezvous registry serving; Ctrl-C to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    _main()
