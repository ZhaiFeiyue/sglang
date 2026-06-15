"""Virtual, mori-backed pipeline-parallel group for PP stage disaggregation.

In stage-disaggregation mode each PP stage runs in its own NCCL world that only
contains the intra-stage TP/attn ranks. There is no cross-stage NCCL PP group.
``MoriPPGroup`` impersonates the subset of the ``GroupCoordinator`` surface that
the PP code paths (models, KV sizing, scheduler) actually use, so ~70 model
files keep working unchanged:

* **Identity** (``rank_in_group``/``world_size``/``is_first_rank``/...): pure
  integer math from ``(stage_id, num_stages)``. This is what ``make_layers`` /
  ``get_pp_indices`` and the embed/norm/lm_head gating read.
* **Tensor transport** (``send_tensor_dict``/``recv_tensor_dict_typed``):
  delegated to :class:`MoriActivationTransport`, keyed by ``f"{msg_type}:{seq}"``
  so it is a drop-in for the order-based NCCL demux used by the lockstep loop.
  The attn-TP slice/all-gather semantics of the NCCL path are reproduced here.
* **Control plane** (``send``/``recv`` for tied embeddings,
  ``broadcast_object_list``): small python objects over the transport's ZMQ
  side-channel.

See mori-scheduler/docs/pp-stage-disaggregation-impl.md §1.1 / §2.4.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MoriPPGroup:
    def __init__(
        self,
        *,
        stage_id: int,
        num_stages: int,
        transport,  # MoriActivationTransport
        device: str,
    ):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.transport = transport
        self.device = device

        # Virtual rank mapping (no real torch ranks across stages).
        self.ranks: List[int] = list(range(num_stages))
        self.rank: int = stage_id
        self.rank_in_group: int = stage_id
        self.world_size: int = num_stages
        self.local_rank: int = stage_id

        # Real ProcessGroups do not exist across stages.
        self.cpu_group = None
        self.device_group = None

        # Per (direction, msg_type) monotonic sequence counters for rendezvous.
        self._send_seq: Dict[str, int] = defaultdict(int)
        self._recv_seq: Dict[str, int] = defaultdict(int)
        self._p2p_seq = 0
        self._bcast_seq = 0
        self._ring_send_seq: Dict[str, int] = defaultdict(int)
        self._ring_recv_seq: Dict[str, int] = defaultdict(int)
        # Last link epoch observed on each direction. A neighbor restart bumps
        # the transport's per-link epoch; when we notice the change we reset the
        # corresponding sequence counters so post-restart keys start at seq 0 at
        # the new epoch (old-epoch keys can never collide).
        self._last_send_epoch = 0
        self._last_recv_epoch = 0

        # Endpoint tables (populated by the launch / bootstrap layer).
        # adjacency for activations, full table for broadcast / tied weights.
        self.upstream_endpoint: Optional[str] = None  # stage_id - 1
        self.downstream_endpoint: Optional[str] = None  # stage_id + 1
        self.stage_endpoints: Dict[int, str] = {}

    # ----------------------------------------------------------- wiring hooks

    def set_neighbor_endpoints(
        self, upstream: Optional[str], downstream: Optional[str]
    ):
        self.upstream_endpoint = upstream
        self.downstream_endpoint = downstream

    def set_stage_endpoints(self, endpoints: Dict[int, str]):
        self.stage_endpoints = dict(endpoints)

    # -------------------------------------------------------------- identity

    @property
    def first_rank(self) -> int:
        return self.ranks[0]

    @property
    def last_rank(self) -> int:
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last_rank(self) -> bool:
        return self.stage_id == self.num_stages - 1

    @property
    def next_rank(self) -> int:
        return self.ranks[(self.stage_id + 1) % self.num_stages]

    @property
    def prev_rank(self) -> int:
        return self.ranks[(self.stage_id - 1) % self.num_stages]

    # ------------------------------------------------------ no-op collectives

    @contextmanager
    def graph_capture(self, graph_capture_context=None, stream=None):
        # PP group does not participate in CUDA-graph capture collectives.
        yield graph_capture_context

    def barrier(self):
        # No cross-stage barrier; stages are independent.
        return

    # --------------------------------------------------------- tensor dict IO

    @staticmethod
    def _msg_type_of(tensor_dict: Dict[str, Any]) -> str:
        return tensor_dict.get("__msg_type__", "default")

    def _send_epoch(self) -> int:
        """Current producer-link epoch; reset send-side seq counters on bump."""
        ep = getattr(self.transport, "send_epoch", 0)
        if ep != self._last_send_epoch:
            self._last_send_epoch = ep
            self._send_seq.clear()
            self._ring_send_seq.clear()
        return ep

    def _recv_epoch(self) -> int:
        """Current consumer-link epoch; reset recv-side seq counters on bump."""
        ep = getattr(self.transport, "recv_epoch", 0)
        if ep != self._last_recv_epoch:
            self._last_recv_epoch = ep
            self._recv_seq.clear()
            self._ring_recv_seq.clear()
        return ep

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
        dst: Optional[int] = None,
        all_gather_group=None,
        async_send: bool = False,
        batch_key: Optional[str] = None,
    ):
        """Drop-in for GroupCoordinator.send_tensor_dict over mori.

        Reproduces the NCCL path's attn-TP slicing: with an all_gather_group,
        only this rank's contiguous slice of each tensor is shipped; the
        receiver all-gathers it back. Returns [] (push is synchronous), which
        the scheduler's ``_pp_commit_comm_work`` treats as nothing to wait on.

        ``batch_key`` (content-addressing): when supplied, the rendezvous key
        embeds the *identity* of the micro-batch (e.g. its ordered rids +
        per-req token counts) instead of a positional sequence number. This
        decouples the two stages -- the consumer pulls the proxy belonging to
        the exact batch it built, regardless of arrival order or any post-
        restart phase drift -- which is what makes hot-restart "first-ready-
        first-serve" without the rotary token-mismatch crash.
        """
        if self.downstream_endpoint is None:
            raise RuntimeError("MoriPPGroup.send_tensor_dict before downstream wired")

        msg_type = self._msg_type_of(tensor_dict)
        ag_size = 1 if all_gather_group is None else all_gather_group.world_size
        ag_rank = 0 if all_gather_group is None else all_gather_group.rank_in_group

        payload: Dict[str, torch.Tensor] = {}
        orig_shapes: Dict[str, List[int]] = {}
        scalars: Dict[str, Any] = {}
        for key, value in tensor_dict.items():
            if key == "__msg_type__":
                continue
            if not isinstance(value, torch.Tensor):
                scalars[key] = value
                continue
            if value.numel() == 0:
                payload[key] = value
                orig_shapes[key] = list(value.shape)
                continue
            t = value
            if all_gather_group is not None and t.numel() % ag_size == 0:
                orig_shapes[key] = list(t.shape)
                t = t.reshape(ag_size, -1)[ag_rank]
            else:
                orig_shapes[key] = list(t.shape)
            payload[key] = t.contiguous()

        epoch = self._send_epoch()
        if batch_key is not None:
            # Identity-addressed: do NOT advance the positional counter so the
            # proxy stream is fully order-independent.
            key = f"{epoch}:{msg_type}:bk={batch_key}"
        else:
            seq = self._send_seq[msg_type]
            self._send_seq[msg_type] += 1
            key = f"{epoch}:{msg_type}:{seq}"
        extras = {"msg_type": msg_type, "orig_shapes": orig_shapes, "scalars": scalars}
        self.transport.push(key, payload, extras=extras)
        return []

    def recv_tensor_dict_typed(
        self,
        msg_type: str,
        all_gather_group=None,
        batch_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Receive a tensor dict of ``msg_type``.

        With ``batch_key`` the pull is content-addressed by micro-batch
        identity (see ``send_tensor_dict``); otherwise it is sequence-matched.
        """
        epoch = self._recv_epoch()
        if batch_key is not None:
            key = f"{epoch}:{msg_type}:bk={batch_key}"
        else:
            seq = self._recv_seq[msg_type]
            self._recv_seq[msg_type] += 1
            key = f"{epoch}:{msg_type}:{seq}"
        tensors, slot_handle = self.transport.pull(key)
        extras = slot_handle[-1]
        try:
            ag_size = 1 if all_gather_group is None else all_gather_group.world_size
            orig_shapes = extras.get("orig_shapes", {})
            out: Dict[str, Any] = {}
            for k, t in tensors.items():
                orig = orig_shapes.get(k)
                if (
                    all_gather_group is not None
                    and orig is not None
                    and t.numel() > 0
                    and _numel(orig) % ag_size == 0
                    and t.numel() == _numel(orig) // ag_size
                ):
                    gathered = all_gather_group.all_gather(t, dim=0)
                    out[k] = gathered.reshape(orig)
                else:
                    out[k] = t if orig is None else t.reshape(orig)
            for k, v in extras.get("scalars", {}).items():
                out[k] = v
            out["__msg_type__"] = extras.get("msg_type", msg_type)
            return out
        finally:
            up = (
                getattr(self.transport, "upstream_endpoint", None)
                or self.upstream_endpoint
            )
            self.transport.release_slot(slot_handle, up)

    # ------------------------------------------------- ring pyobj (reqs/rids)

    def send_pyobj_next(self, obj, tag: str = "ring") -> None:
        """Send a python object to the next stage in the ring (wraps last->0).

        Used by the PP scheduler to relay requests / consensus rid lists that
        previously travelled over world_group P2P. ``tag`` separates logical
        streams so their per-stream sequence numbers stay aligned."""
        target = (self.stage_id + 1) % self.num_stages
        # Route to the transport's *live* downstream (ring-successor) endpoint so
        # a restarted neighbor is followed automatically; fall back to the static
        # bootstrap table before the first handshake.
        ep = getattr(self.transport, "downstream_endpoint", None) or (
            self._resolve_endpoint(target)
        )
        epoch = self._send_epoch()
        seq = self._ring_send_seq[tag]
        self._ring_send_seq[tag] += 1
        key = f"ring:{epoch}:{tag}:{self.stage_id}->{target}:{seq}"
        self.transport.push_pyobj(ep, key, obj)

    def recv_pyobj_prev(self, tag: str = "ring", timeout_s=None):
        """Receive the next python object from the previous stage in the ring."""
        src = (self.stage_id - 1) % self.num_stages
        epoch = self._recv_epoch()
        seq = self._ring_recv_seq[tag]
        self._ring_recv_seq[tag] += 1
        key = f"ring:{epoch}:{tag}:{src}->{self.stage_id}:{seq}"
        return self.transport.pull_pyobj(key, timeout_s=timeout_s)

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        raise NotImplementedError(
            "MoriPPGroup requires typed receives; use recv_tensor_dict_typed("
            "msg_type). The stage scheduler demuxes proxy/output explicitly."
        )

    # ----------------------------------------------- control-plane (pyobj/p2p)

    def send(self, tensor: torch.Tensor, dst: int):
        """Tied-weight P2P send to stage ``dst`` (may be non-adjacent)."""
        endpoint = self._resolve_endpoint(dst)
        seq = self._p2p_seq
        self._p2p_seq += 1
        key = f"p2p:{self.stage_id}->{dst}:{seq}"
        self.transport.push_pyobj(
            endpoint,
            key,
            {"cpu": tensor.detach().to("cpu"), "dtype": str(tensor.dtype)},
        )

    def recv(self, size, dtype, src: int) -> torch.Tensor:
        """Tied-weight P2P recv from stage ``src``."""
        seq = self._p2p_seq
        self._p2p_seq += 1
        key = f"p2p:{src}->{self.stage_id}:{seq}"
        obj = self.transport.pull_pyobj(key)
        return obj["cpu"].to(self.device, dtype=dtype).reshape(tuple(size))

    def broadcast_object_list(self, obj_list: List[Any], src: int = 0):
        """Broadcast ``obj_list`` from stage ``src`` to all stages."""
        seq = self._bcast_seq
        self._bcast_seq += 1
        key = f"bcast:{src}:{seq}"
        if self.stage_id == src:
            for sid, endpoint in self.stage_endpoints.items():
                if sid == self.stage_id:
                    continue
                self.transport.push_pyobj(endpoint, key, obj_list)
            return obj_list
        received = self.transport.pull_pyobj(key)
        obj_list[:] = received
        return obj_list

    # ----------------------------------------------------------- observability

    def transport_stats(self) -> Dict[str, Any]:
        """Snapshot of the underlying mori activation transport counters
        (pushes/pulls/bytes/credit-stalls/slot occupancy) for metrics."""
        stats = self.transport.stats()
        stats["stage"] = f"{self.stage_id}/{self.num_stages}"
        return stats

    # ----------------------------------------------------------------- helper

    def _resolve_endpoint(self, stage_id: int) -> str:
        if stage_id == self.stage_id + 1 and self.downstream_endpoint:
            return self.downstream_endpoint
        if stage_id == self.stage_id - 1 and self.upstream_endpoint:
            return self.upstream_endpoint
        ep = self.stage_endpoints.get(stage_id)
        if ep is None:
            raise RuntimeError(
                f"MoriPPGroup: no endpoint for stage {stage_id}; "
                "call set_stage_endpoints() during bootstrap"
            )
        return ep


def _numel(shape: List[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n
