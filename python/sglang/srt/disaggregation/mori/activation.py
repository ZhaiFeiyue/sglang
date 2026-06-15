"""Mori IO activation transport for PP stage disaggregation.

This moves inter-stage activations (PPProxyTensors: ``hidden_states`` +
``residual``, plus optional output tensors) between *independent* pipeline
stages, each running in its own NCCL world. It replaces the cross-stage NCCL
P2P path (``GroupCoordinator.send_tensor_dict`` / ``recv_tensor_dict``) used by
the lockstep PP scheduler.

Design (see mori-scheduler/docs/pp-stage-disaggregation-impl.md, sections 1.2
and 3):

* Each stage owns a *dedicated* mori ``IOEngine`` (engine_key ``act-*``),
  separate from the KV-disaggregation engine, so activation traffic never
  contends with the KV control plane or the intra-stage TP NCCL world.
* Transport is one-sided RDMA ``batch_write`` (optionally xGMI intra-node),
  identical to the KV path in ``mori/conn.py``.
* Because ``batch_write`` requires the destination buffer to be pre-registered
  but proxy tensors have dynamic shapes, we use fixed-size *ring buffer* pools:
  a registered send-staging pool on the producer and a registered recv pool on
  the consumer. Per-transfer layout (keys/shapes/dtypes/slot) is exchanged
  out-of-band over ZMQ using the same ``MORI_GUARD`` framing as the KV path.
* Flow control is credit-based: the consumer grants the producer credits for
  free recv slots; when slots are exhausted, ``push`` blocks. This is the
  concrete realization of the IO-buffer bound on max running requests.

This module lazily imports ``mori`` inside methods so it can be imported in
environments where the native library is not installed.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

import msgspec
import torch
import zmq

logger = logging.getLogger(__name__)

# Reuse the KV path's wire guard so tooling/sniffers treat both alike.
ACT_GUARD = b"MoriActGuard"

# ZMQ control message kinds (payload[0] after ACT_GUARD).
_MSG_REGISTER = b"register"  # downstream -> upstream: engine_desc + recv slot descs
_MSG_CREDIT = b"credit"  # downstream -> upstream: free slot ids
_MSG_DATA = b"data"  # upstream -> downstream: layout + slot id (after write)
_MSG_PYOBJ = b"pyobj"  # control-plane python objects (weight tying / broadcast)
_MSG_REREGISTER = b"rereg"  # restarted producer -> downstream: "re-register to me"


class PeerReconnected(Exception):
    """Raised by a blocked transport op (pull / push / credit wait) when the
    affected link re-handshakes at a new epoch because a neighbor stage
    restarted. Callers (the PP scheduler loop) catch this, drop in-flight
    micro-batch state for the link, and resume at the new epoch."""

    def __init__(self, link: str, epoch: int):
        super().__init__(f"activation peer reconnected on {link} link (epoch={epoch})")
        self.link = link
        self.epoch = epoch


class TensorLayout(msgspec.Struct):
    """Per-tensor placement inside a flat slot buffer."""

    key: str
    # dotted shape; kept as list for msgpack
    shape: List[int]
    dtype: str
    offset: int  # byte offset within the slot
    nbytes: int


class SlotMessage(msgspec.Struct):
    """Describes one logical activation message written into a recv slot."""

    msg_key: str  # rendezvous key, e.g. f"{rid}:{mb_seq}:{msg_type}"
    slot_id: int
    total_bytes: int
    tensors: List[TensorLayout]
    extras: bytes  # msgpack of non-tensor scalars from the tensor_dict


@dataclasses.dataclass
class _Slot:
    """A per-microbatch region inside one contiguous, singly-registered IO
    block (KV-cache style). ``buffer`` is a uint8 view aliasing the block at
    ``offset``; the RDMA ``MemoryDesc`` is held once at the transport level and
    addressed by ``offset``, so there is no per-slot registration."""

    slot_id: int
    offset: int  # byte offset of this region within the contiguous IO block
    buffer: torch.Tensor  # uint8 [max_slot_bytes] view into the block at offset
    in_use: bool = False


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


_STR_TO_DTYPE = {
    _dtype_to_str(d): d
    for d in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
}


def _str_to_dtype(name: str) -> torch.dtype:
    return _STR_TO_DTYPE[name]


class MoriActivationTransport:
    """Dedicated mori IO engine moving activations between adjacent PP stages.

    A single transport instance serves one stage. It can act as a *producer*
    (writes to the downstream stage) and/or a *consumer* (receives from the
    upstream stage). The first stage is consumer-less; the last stage is
    producer-less for proxy tensors (it may still produce "output" messages
    routed back to stage 0 if enabled).
    """

    def __init__(
        self,
        *,
        stage_id: int,
        num_stages: int,
        attn_tp_rank: int,
        attn_tp_size: int,
        gpu_id: int,
        device: str,
        ib_device: Optional[str],
        max_slot_bytes: int,
        num_slots: int,
        bind_host: Optional[str] = None,
        use_xgmi: bool = False,
    ):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.gpu_id = gpu_id
        self.device = device
        self.ib_device = ib_device
        self.max_slot_bytes = int(max_slot_bytes)
        self.num_slots = int(num_slots)
        self.use_xgmi = use_xgmi

        self.is_first = stage_id == 0
        self.is_last = stage_id == num_stages - 1

        self._zmq_ctx = zmq.Context.instance()
        self._socket_local = threading.local()

        # Lazy mori objects (set in _init_engine).
        self.engine = None
        self.engine_desc = None
        self._mori = None

        # Consumer state: one contiguous recv block (registered once) carved into
        # per-microbatch regions, plus ready messages keyed by msg_key.
        self._recv_block: Optional[torch.Tensor] = None
        self._recv_block_desc = None  # single mori MemoryDesc for the whole block
        self._recv_slots: List[_Slot] = []
        self._free_recv: "deque[int]" = deque()
        self._recv_lock = threading.Condition()
        self._ready: Dict[str, SlotMessage] = {}

        # Control-plane python-object inbox keyed by msg_key (small messages
        # for weight tying / broadcast_object_list; not RDMA).
        self._pyobj_lock = threading.Condition()
        self._pyobj_inbox: Dict[str, bytes] = {}

        # Producer state: one contiguous send staging block (registered once)
        # carved into per-microbatch regions, plus per-peer remote block desc +
        # credits.
        self._send_block: Optional[torch.Tensor] = None
        self._send_block_desc = None  # single mori MemoryDesc for the whole block
        self._send_slots: List[_Slot] = []
        self._free_send: "deque[int]" = deque()
        self._send_lock = threading.Condition()
        # downstream peer registration: single remote recv-block MemoryDesc; a
        # remote slot id addresses into it at (slot_id * peer_slot_bytes).
        self._peer_engine_desc = None
        self._peer_endpoint: Optional[str] = None
        self._peer_engine_key: Optional[str] = None
        self._peer_registered = threading.Event()
        self._peer_recv_block_desc = None  # single MemoryDesc for peer's block
        self._peer_slot_bytes: int = self.max_slot_bytes  # remote region stride
        self._credits: "deque[int]" = deque()  # free remote recv slot ids
        self._credit_lock = threading.Condition()
        self._upstream_endpoint: Optional[str] = None

        # Per-link generation counters for restart recovery. ``_recv_epoch``
        # tags the upstream->me (consumer) link; ``_send_epoch`` tags the
        # me->downstream (producer) link. A neighbor restart re-handshakes only
        # the affected link and bumps just that epoch; blocked ops on the link
        # observe the change and raise ``PeerReconnected`` so the scheduler can
        # reset and resume. Keys embed the epoch (see MoriPPGroup), so stale
        # old-epoch messages are never matched.
        self._recv_epoch = 0
        self._send_epoch = 0
        # Set while a producer push has detected its downstream is gone and is
        # waiting for the restarted consumer to re-register.
        self._producer_broken = False

        self.local_ip = bind_host
        self.server_socket = None
        self.endpoint = None

        # Observability counters (lock-free single-writer increments are fine;
        # reads in stats() are best-effort snapshots).
        self._stat_pushes = 0
        self._stat_pulls = 0
        self._stat_bytes_written = 0
        self._stat_credit_stalls = 0
        self._stat_credit_wait_s = 0.0
        self._stat_pyobj_sent = 0
        self._stat_pyobj_recv = 0

        self._init_engine()
        self._alloc_buffers()
        self._start_bootstrap_thread()

    # ------------------------------------------------------------------ setup

    def _init_engine(self):
        import mori.io as mio  # lazy

        self._mori = mio
        from sglang.srt.environ import envs
        from sglang.srt.utils.network import get_local_ip_auto

        if self.ib_device:
            os.environ.setdefault("MORI_RDMA_DEVICES", self.ib_device)

        if self.local_ip is None:
            self.local_ip = get_local_ip_auto()

        engine_key = (
            f"act-s{self.stage_id}-tp{self.attn_tp_rank}-"
            f"pid{os.getpid()}-{self.local_ip}-{uuid.uuid4().hex[:8]}"
        )
        config = mio.IOEngineConfig(host=self.local_ip, port=0)
        engine = mio.IOEngine(engine_key, config)

        backend_type = (
            mio.BackendType.XGMI
            if self.use_xgmi and hasattr(mio.BackendType, "XGMI")
            else mio.BackendType.RDMA
        )
        if backend_type == mio.BackendType.RDMA:
            rdma_cfg = mio.RdmaBackendConfig(
                envs.SGLANG_MORI_QP_PER_TRANSFER.get(),
                envs.SGLANG_MORI_POST_BATCH_SIZE.get(),
                envs.SGLANG_MORI_NUM_WORKERS.get(),
                mio.PollCqMode.POLLING,
                False,
            )
            engine.create_backend(backend_type, rdma_cfg)
        else:
            engine.create_backend(backend_type)

        self.engine_key = engine_key
        self.engine = engine
        self.engine_desc = engine.get_engine_desc()
        port = self.engine_desc.port
        # RDMA binds a TCP control port (must be > 0). XGMI is an intra-node GPU
        # fabric backend and may not expose a TCP port; the engine_desc is still
        # exchanged out-of-band over our ZMQ rendezvous, so port 0 is fine there.
        if backend_type == mio.BackendType.RDMA:
            assert port > 0, f"Failed to bind activation engine {engine_key}"
        logger.info(
            "Mori activation engine %s up at %s:%s backend=%s slots=%s slot_bytes=%s",
            engine_key,
            self.local_ip,
            port,
            backend_type.name,
            self.num_slots,
            self.max_slot_bytes,
        )

    def _alloc_buffers(self):
        """Allocate + register both pools. Stages form a uniform ring: every
        stage receives from its ring-predecessor (proxy for stages>0, output
        for stage 0) and sends to its ring-successor, mirroring the NCCL
        ``send->next_rank`` / ``recv<-prev_rank`` PP semantics. Hence every
        stage needs both a recv ring pool and a send staging pool."""
        mio = self._mori
        # One contiguous block per direction, registered with mori exactly once
        # (KV-cache style). Per-microbatch regions are byte offsets into it, so
        # RDMA addressing is (block_desc, slot_id * max_slot_bytes) and there is
        # no per-slot registration to track.
        total_bytes = self.num_slots * self.max_slot_bytes

        self._recv_block = torch.empty(
            total_bytes, dtype=torch.uint8, device=self.device
        )
        self._recv_block_desc = self.engine.register_memory(
            self._recv_block.data_ptr(),
            total_bytes,
            self.gpu_id,
            mio.MemoryLocationType.GPU,
        )
        self._send_block = torch.empty(
            total_bytes, dtype=torch.uint8, device=self.device
        )
        self._send_block_desc = self.engine.register_memory(
            self._send_block.data_ptr(),
            total_bytes,
            self.gpu_id,
            mio.MemoryLocationType.GPU,
        )

        for i in range(self.num_slots):
            off = i * self.max_slot_bytes
            self._recv_slots.append(
                _Slot(i, off, self._recv_block[off : off + self.max_slot_bytes])
            )
            self._free_recv.append(i)
            self._send_slots.append(
                _Slot(i, off, self._send_block[off : off + self.max_slot_bytes])
            )
            self._free_send.append(i)

    # -------------------------------------------------------------- bootstrap

    def _bind_server(self):
        sock = self._zmq_ctx.socket(zmq.PULL)
        port = sock.bind_to_random_port(f"tcp://{self.local_ip}")
        self.server_socket = sock
        self.endpoint = f"tcp://{self.local_ip}:{port}"
        return self.endpoint

    def _connect(self, endpoint: str):
        cache = getattr(self._socket_local, "cache", None)
        if cache is None:
            cache = {}
            self._socket_local.cache = cache
        if endpoint not in cache:
            s = self._zmq_ctx.socket(zmq.PUSH)
            s.setsockopt(zmq.SNDHWM, 0)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(endpoint)
            cache[endpoint] = s
        return cache[endpoint]

    def _start_bootstrap_thread(self):
        if self.server_socket is None:
            self._bind_server()

        def worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if not msg or msg[0] != ACT_GUARD:
                        logger.warning("activation: malformed control message")
                        continue
                    kind = msg[1]
                    if kind == _MSG_REGISTER:
                        self._on_register(msg[2:])
                    elif kind == _MSG_CREDIT:
                        self._on_credit(msg[2:])
                    elif kind == _MSG_DATA:
                        self._on_data(msg[2:])
                    elif kind == _MSG_PYOBJ:
                        self._on_pyobj(msg[2:])
                    elif kind == _MSG_REREGISTER:
                        self._on_reregister(msg[2:])
                    else:
                        logger.warning("activation: unknown kind %r", kind)
                except Exception:
                    logger.exception("activation bootstrap worker failed")

        threading.Thread(
            target=worker, daemon=True, name=f"act-boot-s{self.stage_id}"
        ).start()

    # ------------------------------------------------- epoch / reconnect state

    @property
    def recv_epoch(self) -> int:
        return self._recv_epoch

    @property
    def send_epoch(self) -> int:
        return self._send_epoch

    @property
    def downstream_endpoint(self) -> Optional[str]:
        """Live ring-successor (consumer) endpoint; updated on every (re)register
        so control-plane routing follows a restarted neighbor automatically."""
        return self._peer_endpoint

    @property
    def upstream_endpoint(self) -> Optional[str]:
        """Live ring-predecessor (producer) endpoint we are registered with."""
        return self._upstream_endpoint

    def _bump_recv_epoch(self, epoch: int) -> None:
        """Advance the upstream->me link epoch and wake any consumer waiters
        (``pull`` / ``pull_pyobj``) so they unwind with ``PeerReconnected``."""
        if epoch <= self._recv_epoch:
            return
        self._recv_epoch = epoch
        with self._recv_lock:
            self._recv_lock.notify_all()
        with self._pyobj_lock:
            self._pyobj_lock.notify_all()

    def _bump_send_epoch(self, epoch: int) -> None:
        """Advance the me->downstream link epoch and wake any producer waiters
        (``_acquire_credit`` / ``_acquire_send_slot``)."""
        if epoch <= self._send_epoch:
            return
        self._send_epoch = epoch
        with self._credit_lock:
            self._credit_lock.notify_all()
        with self._send_lock:
            self._send_lock.notify_all()

    def _reset_consumer_state(self) -> None:
        """Discard recv ring state from a previous epoch and rearm all slots."""
        with self._recv_lock:
            self._ready.clear()
            self._free_recv.clear()
            self._free_recv.extend(range(self.num_slots))
            for s in self._recv_slots:
                s.in_use = False
            self._recv_lock.notify_all()
        with self._pyobj_lock:
            if self._pyobj_inbox:
                logger.info(
                    "[RC] s%d reset_consumer clearing %d inbox keys: %s",
                    self.stage_id,
                    len(self._pyobj_inbox),
                    list(self._pyobj_inbox.keys())[:10],
                )
            self._pyobj_inbox.clear()
            self._pyobj_lock.notify_all()

    def _reset_producer_state(self) -> None:
        """Discard granted credits / staging occupancy from a previous epoch."""
        with self._credit_lock:
            self._credits.clear()
            self._credit_lock.notify_all()
        with self._send_lock:
            self._free_send.clear()
            self._free_send.extend(range(self.num_slots))
            for s in self._send_slots:
                s.in_use = False
            self._send_lock.notify_all()

    # ---------------------------------------------------- consumer-side recv

    def register_with_upstream(
        self, upstream_endpoint: str, epoch: Optional[int] = None
    ):
        """As a consumer, advertise our engine + recv slot buffers to the
        upstream producer so it can ``batch_write`` into them. Called once after
        both stages' transports exist (driven by the launch/bootstrap layer),
        and again on reconnect when our upstream restarts (``epoch`` bumped)."""
        # Uniform ring: every stage (including stage 0) has a ring-predecessor
        # it consumes from -- stage 0 receives the last stage's output tokens.
        if epoch is not None and epoch != self._recv_epoch:
            self._bump_recv_epoch(epoch)
        # Reset consumer-side ring state so we start the (new) epoch clean: any
        # half-delivered slots from a crashed producer are discarded.
        self._reset_consumer_state()
        self._upstream_endpoint = upstream_endpoint
        packed_engine = self.engine_desc.pack()
        # Advertise the single contiguous recv block (one MemoryDesc) plus the
        # region stride and count; the producer addresses region i at
        # (block_base + i * max_slot_bytes).
        packed_block = self._recv_block_desc.pack()
        sock = self._connect(upstream_endpoint)
        sock.send_multipart(
            [
                ACT_GUARD,
                _MSG_REGISTER,
                self.endpoint.encode(),
                str(self.attn_tp_rank).encode(),
                str(self.max_slot_bytes).encode(),
                packed_engine,
                packed_block,
                str(self._recv_epoch).encode(),
                str(self.num_slots).encode(),
            ]
        )
        # grant initial credits for every free slot
        self._grant_credits(upstream_endpoint, list(self._free_recv))

    def announce_restart_to_downstream(
        self, downstream_endpoint: str, epoch: int
    ) -> None:
        """After we (a restarted producer) come back up, ask our downstream
        consumer to re-register its recv buffers with our *new* engine. The
        consumer cannot detect our restart by itself (it only ever waits in
        ``pull``), so we must actively notify it."""
        self._bump_send_epoch(epoch)
        # Our downstream peer registration is stale until the consumer re-registers.
        self._peer_registered.clear()
        sock = self._connect(downstream_endpoint)
        sock.send_multipart(
            [
                ACT_GUARD,
                _MSG_REREGISTER,
                self.endpoint.encode(),
                str(epoch).encode(),
            ]
        )

    def _on_reregister(self, payload: List[bytes]):
        """Our upstream producer restarted and tells us its new endpoint; we
        (the consumer) bump the recv-link epoch, reset, and re-advertise our
        recv buffers to it. Wakes any ``pull`` blocked on the old epoch."""
        new_upstream = payload[0].decode()
        epoch = int(payload[1].decode())
        logger.info(
            "activation stage %d: upstream restarted -> re-registering "
            "(recv epoch %d->%d) at %s",
            self.stage_id,
            self._recv_epoch,
            epoch,
            new_upstream,
        )
        self.register_with_upstream(new_upstream, epoch=epoch)

    def _grant_credits(self, upstream_endpoint: str, slot_ids: List[int]):
        if not slot_ids:
            return
        sock = self._connect(upstream_endpoint)
        # Tag with our recv-link epoch so the producer can discard credits left
        # over from a previous (dead) peer that arrive after its reconnect reset
        # -- those would phantom-inflate the credit pool and let the producer
        # overwrite an occupied remote slot.
        sock.send_multipart(
            [
                ACT_GUARD,
                _MSG_CREDIT,
                msgspec.msgpack.encode(slot_ids),
                str(self._recv_epoch).encode(),
            ]
        )

    def _on_data(self, payload: List[bytes]):
        """Producer told us a slot is filled. Decode header, mark ready."""
        sm = msgspec.msgpack.decode(payload[0], type=SlotMessage)
        with self._recv_lock:
            self._ready[sm.msg_key] = sm
            self._recv_lock.notify_all()

    def pull(
        self, msg_key: str, timeout_s: Optional[float] = None
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        """Block until the message for ``msg_key`` arrives; return
        (tensor_dict, extras). Caller must call ``release_slot`` after copying
        the tensors out (they alias the ring buffer)."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        start_epoch = self._recv_epoch
        with self._recv_lock:
            while msg_key not in self._ready:
                if self._recv_epoch != start_epoch:
                    raise PeerReconnected("recv", self._recv_epoch)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"activation pull timed out for {msg_key}")
                self._recv_lock.wait(timeout=remaining)
            sm = self._ready.pop(msg_key)

        self._stat_pulls += 1
        slot = self._recv_slots[sm.slot_id]
        tensor_dict: Dict[str, torch.Tensor] = {}
        for t in sm.tensors:
            dtype = _str_to_dtype(t.dtype)
            n = 1
            for d in t.shape:
                n *= d
            flat = slot.buffer[t.offset : t.offset + t.nbytes]
            # view raw bytes as the target dtype/shape, then clone so the slot
            # can be recycled independently of the consumer's lifetime.
            view = flat.view(dtype)[:n].reshape(t.shape)
            tensor_dict[t.key] = view.clone()
        extras = msgspec.msgpack.decode(sm.extras) if sm.extras else {}
        # Stamp the handle with the epoch the slot was pulled under. ``pull``
        # already aborts (PeerReconnected) if the epoch bumps mid-wait, so this
        # is the live epoch for this delivery; ``release_slot`` uses it to drop
        # stale releases that race a reconnect reset (see below).
        return tensor_dict, (sm.slot_id, self._recv_epoch, extras)

    def release_slot(self, slot_handle, upstream_endpoint: str):
        """Return a recv region to the producer by granting a fresh credit for
        it upstream.

        Guards against a release that races a reconnect: if the link epoch has
        advanced since the slot was pulled, ``_reset_consumer_state`` has already
        re-granted credits for every region under the new epoch. Re-granting this
        one would over-grant a credit for the same physical region, letting the
        producer write two different microbatches into it (corruption / token
        mismatch). Drop such stale releases."""
        if isinstance(slot_handle, tuple):
            slot_id = slot_handle[0]
            slot_epoch = slot_handle[1] if len(slot_handle) >= 3 else None
        else:
            slot_id = slot_handle
            slot_epoch = None
        with self._recv_lock:
            if slot_epoch is not None and slot_epoch != self._recv_epoch:
                logger.info(
                    "activation stage %d: dropping stale region %d release "
                    "(pulled epoch %s != current %d); reset already re-granted it",
                    self.stage_id,
                    slot_id,
                    slot_epoch,
                    self._recv_epoch,
                )
                return
        self._grant_credits(upstream_endpoint, [slot_id])

    # ---------------------------------------------------- producer-side send

    def _on_register(self, payload: List[bytes]):
        mio = self._mori
        endpoint = payload[0].decode()
        peer_tp_rank = int(payload[1].decode())
        peer_slot_bytes = int(payload[2].decode())
        engine_desc = mio.EngineDesc.unpack(payload[3])
        # Single contiguous recv-block descriptor; remote region i lives at
        # (block_base + i * peer_slot_bytes).
        block_desc = mio.MemoryDesc.unpack(payload[4])
        peer_epoch = int(payload[5].decode()) if len(payload) > 5 else 0
        # We only pair TP rank -> same TP rank (mirrors send_tensor_dict's
        # per-rank slice; cross-TP-size reshaping handled by MoriPPGroup).
        if peer_tp_rank != self.attn_tp_rank:
            return
        if peer_slot_bytes != self.max_slot_bytes:
            logger.warning(
                "activation: peer slot bytes %s != local %s",
                peer_slot_bytes,
                self.max_slot_bytes,
            )
        engine_key = getattr(engine_desc, "key", None)
        is_reconnect = self._peer_engine_desc is not None and (
            endpoint != self._peer_endpoint or peer_epoch > self._send_epoch
        )
        if is_reconnect:
            # Downstream consumer restarted (or re-registered at a new epoch):
            # drop credits granted against its dead recv slots before adopting
            # the new descriptors, then advance the producer link epoch so any
            # push blocked on the old epoch raises PeerReconnected.
            logger.info(
                "activation stage %d: downstream re-registered (send epoch "
                "%d->%d) at %s",
                self.stage_id,
                self._send_epoch,
                peer_epoch,
                endpoint,
            )
            self._reset_producer_state()
        self.engine.register_remote_engine(engine_desc)
        self._peer_engine_desc = engine_desc
        self._peer_recv_block_desc = block_desc
        self._peer_slot_bytes = peer_slot_bytes
        self._peer_endpoint = endpoint
        self._peer_engine_key = engine_key
        self._producer_broken = False
        if peer_epoch > self._send_epoch:
            self._bump_send_epoch(peer_epoch)
        self._peer_registered.set()
        logger.info(
            "activation: registered downstream peer at %s (block stride=%d, "
            "epoch %d)",
            endpoint,
            peer_slot_bytes,
            self._send_epoch,
        )

    def wait_downstream_registered(self, timeout_s: float = 300.0) -> None:
        """Block until our ring-successor has registered its recv buffers, so
        the first ``push`` does not race the registration handshake."""
        if not self._peer_registered.wait(timeout=timeout_s):
            raise TimeoutError("activation: downstream never registered")

    def wait_links_healthy(self, timeout_s: float = 120.0) -> None:
        """Block until the producer link is (re)established. Used by the
        scheduler's reconnect handler before resuming the loop, so the first
        push after recovery does not hit a stale/half-broken downstream."""
        if not self._peer_registered.wait(timeout=timeout_s):
            raise TimeoutError("activation: downstream did not re-register")
        self._producer_broken = False

    def _on_credit(self, payload: List[bytes]):
        slot_ids = msgspec.msgpack.decode(payload[0])
        credit_epoch = int(payload[1].decode()) if len(payload) > 1 else None
        with self._credit_lock:
            if credit_epoch is not None and credit_epoch < self._send_epoch:
                logger.info(
                    "activation stage %d: dropping %d stale credits "
                    "(epoch %d < send epoch %d)",
                    self.stage_id,
                    len(slot_ids),
                    credit_epoch,
                    self._send_epoch,
                )
                return
            for s in slot_ids:
                self._credits.append(s)
            self._credit_lock.notify_all()

    def _acquire_credit(self, timeout_s: Optional[float]) -> int:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        start_epoch = self._send_epoch
        with self._credit_lock:
            if not self._credits:
                # Downstream ring is full: producer is back-pressured. This is
                # the IO-buffer bound on max in-flight microbatches becoming
                # active; surface it for capacity tuning.
                self._stat_credit_stalls += 1
                stall_start = time.monotonic()
                logger.debug(
                    "activation stage %d: stalled on downstream credit "
                    "(total stalls=%d)",
                    self.stage_id,
                    self._stat_credit_stalls,
                )
                while not self._credits:
                    if self._send_epoch != start_epoch:
                        self._stat_credit_wait_s += time.monotonic() - stall_start
                        raise PeerReconnected("send", self._send_epoch)
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("activation push: no downstream credit")
                    self._credit_lock.wait(timeout=remaining)
                self._stat_credit_wait_s += time.monotonic() - stall_start
            return self._credits.popleft()

    def _return_credit(self, slot_id: int):
        """Return a previously-acquired remote recv-slot credit to the pool
        (used when a write fails before the consumer was notified)."""
        with self._credit_lock:
            self._credits.appendleft(slot_id)
            self._credit_lock.notify_all()

    def _acquire_send_slot(self) -> int:
        start_epoch = self._send_epoch
        with self._send_lock:
            while not self._free_send:
                if self._send_epoch != start_epoch:
                    raise PeerReconnected("send", self._send_epoch)
                self._send_lock.wait(timeout=1.0)
            return self._free_send.popleft()

    def _release_send_slot(self, slot_id: int, epoch: Optional[int] = None):
        with self._send_lock:
            if epoch is not None and epoch != self._send_epoch:
                # A reconnect bumped the send epoch while this push was in
                # flight; ``_reset_producer_state`` already re-armed every send
                # slot. Re-appending would duplicate this staging buffer in the
                # free list, so drop the stale release.
                return
            self._free_send.append(slot_id)
            self._send_lock.notify_all()

    def push(
        self,
        msg_key: str,
        tensor_dict: Dict[str, torch.Tensor],
        extras: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ):
        """Copy ``tensor_dict`` into a staging slot, ``batch_write`` it into a
        granted downstream recv slot, then notify the consumer. Blocks on credit
        when the downstream ring is full (back-pressure)."""
        if self._peer_engine_desc is None:
            raise RuntimeError("activation push before downstream registered")
        if self._producer_broken:
            # Downstream is restarting; do not attempt a write against the dead
            # engine. Surface as a reconnect so the scheduler waits for re-reg.
            raise PeerReconnected("send", self._send_epoch)

        # Epoch this push runs under; used to drop stale send-slot releases if a
        # reconnect reset the producer state mid-push (see _release_send_slot).
        push_epoch = self._send_epoch

        # Pack into a local staging slot *before* acquiring a downstream credit:
        # a slot-overflow (misconfiguration) then costs no credit, so it cannot
        # silently deplete the credit pool and deadlock the producer.
        send_slot_id = self._acquire_send_slot()
        send_slot = self._send_slots[send_slot_id]

        # Pack tensors contiguously into the staging buffer; build layout.
        layouts: List[TensorLayout] = []
        offset = 0
        try:
            for key, tensor in tensor_dict.items():
                t = tensor.contiguous()
                nbytes = t.numel() * t.element_size()
                if offset + nbytes > self.max_slot_bytes:
                    raise RuntimeError(
                        f"activation slot overflow: need {offset + nbytes} > "
                        f"{self.max_slot_bytes}; increase "
                        "pp_activation_io_buffer_bytes"
                    )
                dst = send_slot.buffer[offset : offset + nbytes].view(t.dtype)[
                    : t.numel()
                ]
                dst.copy_(t.reshape(-1))
                layouts.append(
                    TensorLayout(
                        key=key,
                        shape=list(t.shape),
                        dtype=_dtype_to_str(t.dtype),
                        offset=offset,
                        nbytes=nbytes,
                    )
                )
                offset += nbytes
        except Exception:
            self._release_send_slot(send_slot_id, push_epoch)
            raise

        total = offset
        torch.cuda.current_stream().synchronize()

        # Now reserve a downstream recv slot (blocks under back-pressure).
        remote_slot_id = self._acquire_credit(timeout_s)

        # One-sided RDMA write from our send block (at this region's offset) into
        # the peer's single recv block (at the granted region's offset). Both
        # sides are one registered allocation; the slot id selects the byte
        # offset within it.
        uid = self.engine.allocate_transfer_uid()
        statuses = self.engine.batch_write(
            [self._send_block_desc],
            [[send_slot.offset]],
            [self._peer_recv_block_desc],
            [[remote_slot_id * self._peer_slot_bytes]],
            [[total]],
            [uid],
        )
        self.engine.wait_all(statuses)
        for st in statuses:
            if st.Failed():
                # The remote slot was never notified to the consumer, so it is
                # still free from its perspective; return the credit instead of
                # leaking it. A write failure means the downstream consumer is
                # gone (crashed/restarting): mark the producer link broken so
                # the scheduler treats this as a reconnect, not a hard error.
                self._return_credit(remote_slot_id)
                self._release_send_slot(send_slot_id, push_epoch)
                self._producer_broken = True
                self._peer_registered.clear()
                logger.warning(
                    "activation stage %d: downstream write failed (%s); link "
                    "broken, awaiting re-register",
                    self.stage_id,
                    st.Message(),
                )
                raise PeerReconnected("send", self._send_epoch)

        self._release_send_slot(send_slot_id, push_epoch)

        sm = SlotMessage(
            msg_key=msg_key,
            slot_id=remote_slot_id,
            total_bytes=total,
            tensors=layouts,
            extras=msgspec.msgpack.encode(extras) if extras else b"",
        )
        self._connect(self._peer_endpoint).send_multipart(
            [ACT_GUARD, _MSG_DATA, msgspec.msgpack.encode(sm)]
        )
        self._stat_pushes += 1
        self._stat_bytes_written += total

    # ----------------------------------------------------------- observability

    def stats(self) -> dict:
        """Best-effort snapshot of transport counters for metrics/logging."""
        with self._credit_lock:
            free_credits = len(self._credits)
        with self._recv_lock:
            ready_backlog = len(self._ready)
            # Recv regions not currently holding undelivered data. Bounded by
            # num_slots (unlike the old raw free-list which double-counted on
            # release and could exceed the physical region count).
            free_recv = max(0, self.num_slots - ready_backlog)
        with self._send_lock:
            free_send = len(self._free_send)
        return {
            "stage_id": self.stage_id,
            "pushes": self._stat_pushes,
            "pulls": self._stat_pulls,
            "bytes_written": self._stat_bytes_written,
            "credit_stalls": self._stat_credit_stalls,
            "credit_wait_s": round(self._stat_credit_wait_s, 4),
            "pyobj_sent": self._stat_pyobj_sent,
            "pyobj_recv": self._stat_pyobj_recv,
            "num_slots": self.num_slots,
            "free_credits": free_credits,
            "free_recv_slots": free_recv,
            "free_send_slots": free_send,
            "ready_backlog": ready_backlog,
            "recv_epoch": self._recv_epoch,
            "send_epoch": self._send_epoch,
        }

    # ---------------------------------------------------- control-plane pyobj

    def _on_pyobj(self, payload: List[bytes]):
        msg_key = payload[0].decode()
        with self._pyobj_lock:
            self._pyobj_inbox[msg_key] = payload[1]
            self._pyobj_lock.notify_all()

    def push_pyobj(self, endpoint: str, msg_key: str, obj) -> None:
        """Send a small python object to ``endpoint`` (any neighbor stage)."""
        import pickle

        self._connect(endpoint).send_multipart(
            [ACT_GUARD, _MSG_PYOBJ, msg_key.encode(), pickle.dumps(obj)]
        )
        self._stat_pyobj_sent += 1

    def pull_pyobj(self, msg_key: str, timeout_s: Optional[float] = None):
        """Block until a python object tagged ``msg_key`` arrives."""
        import pickle

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        start_epoch = self._recv_epoch
        with self._pyobj_lock:
            while msg_key not in self._pyobj_inbox:
                if self._recv_epoch != start_epoch:
                    raise PeerReconnected("recv", self._recv_epoch)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"activation pull_pyobj timed out: {msg_key}")
                self._pyobj_lock.wait(timeout=remaining)
            self._stat_pyobj_recv += 1
            return pickle.loads(self._pyobj_inbox.pop(msg_key))
