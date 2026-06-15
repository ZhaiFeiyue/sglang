"""Unit tests for PP stage-disaggregation plumbing that do not require the mori
native library or GPUs:

* ``MoriPPGroup`` tensor-dict transport semantics (msg_type+seq rendezvous,
  attn-TP slice on send / all-gather on recv, ring pyobj relay) against an
  in-memory fake transport.
* The dependency-free HTTP endpoint rendezvous registry.

The RDMA data path itself is exercised by the GPU e2e test
``test/registered/amd/disaggregation/test_pp_stage_disagg_e2e.py``.
"""

import threading
import unittest

import torch

from sglang.srt.distributed.mori_pp_group import MoriPPGroup


class _FakeWire:
    """Shared in-memory medium between fake transports (one per stage)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.data = {}  # msg_key -> (tensors, extras)
        self.pyobj = {}  # msg_key -> obj

    def put_data(self, key, tensors, extras):
        with self.cond:
            self.data[key] = ({k: v.clone() for k, v in tensors.items()}, extras)
            self.cond.notify_all()

    def get_data(self, key):
        with self.cond:
            while key not in self.data:
                self.cond.wait()
            return self.data.pop(key)

    def put_pyobj(self, key, obj):
        with self.cond:
            self.pyobj[key] = obj
            self.cond.notify_all()

    def get_pyobj(self, key):
        with self.cond:
            while key not in self.pyobj:
                self.cond.wait()
            return self.pyobj.pop(key)


class _FakeTransport:
    """Implements the slice of MoriActivationTransport that MoriPPGroup uses."""

    def __init__(self, send_wire: _FakeWire, recv_wire: _FakeWire):
        self.send_wire = send_wire  # where push() writes
        self.recv_wire = recv_wire  # where pull() reads
        self.last_pushed = {}  # key -> payload tensors (for slice inspection)
        self.released = []

    def push(self, key, tensor_dict, extras=None, timeout_s=None):
        self.last_pushed[key] = {k: v.clone() for k, v in tensor_dict.items()}
        self.send_wire.put_data(key, tensor_dict, extras or {})

    def pull(self, key, timeout_s=None):
        tensors, extras = self.recv_wire.get_data(key)
        return tensors, (0, extras)

    def release_slot(self, slot_handle, upstream_endpoint):
        self.released.append(slot_handle)

    def push_pyobj(self, endpoint, key, obj):
        self.send_wire.put_pyobj(key, obj)

    def pull_pyobj(self, key, timeout_s=None):
        return self.recv_wire.get_pyobj(key)

    def stats(self):
        return {"pushes": len(self.last_pushed), "released": len(self.released)}


class _FakeAG:
    """Fake all_gather group for attn-TP slice/gather simulation."""

    def __init__(self, world_size, rank_in_group, full_tensors=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        # On recv, all_gather must reconstruct the full tensor. We cheat by
        # handing it the known full tensor keyed by the local slice's id.
        self._full = full_tensors or {}

    def all_gather(self, tensor, dim=0):
        # Reconstruct: the test registers the expected full flattened tensor.
        key = tensor.numel()
        if key in self._full:
            return self._full[key]
        # Fallback: tile the slice (only valid when world_size==1).
        return tensor


def _make_pair():
    """Two MoriPPGroups (stage 0 -> stage 1) sharing one directional wire."""
    wire = _FakeWire()
    t0 = _FakeTransport(send_wire=wire, recv_wire=wire)
    t1 = _FakeTransport(send_wire=wire, recv_wire=wire)
    g0 = MoriPPGroup(stage_id=0, num_stages=2, transport=t0, device="cpu")
    g1 = MoriPPGroup(stage_id=1, num_stages=2, transport=t1, device="cpu")
    g0.set_neighbor_endpoints(upstream="ep1", downstream="ep1")
    g1.set_neighbor_endpoints(upstream="ep0", downstream="ep0")
    g0.set_stage_endpoints({0: "ep0", 1: "ep1"})
    g1.set_stage_endpoints({0: "ep0", 1: "ep1"})
    return g0, g1, t0, t1


class TestMoriPPGroupTensorDict(unittest.TestCase):
    def test_round_trip_no_slice(self):
        g0, g1, _, _ = _make_pair()
        hidden = torch.randn(5, 8)
        residual = torch.randn(5, 8)
        td = {"hidden_states": hidden, "residual": residual, "__msg_type__": "proxy"}
        g0.send_tensor_dict(dict(td), all_gather_group=None)
        out = g1.recv_tensor_dict_typed("proxy", all_gather_group=None)
        self.assertTrue(torch.equal(out["hidden_states"], hidden))
        self.assertTrue(torch.equal(out["residual"], residual))
        self.assertEqual(out["__msg_type__"], "proxy")

    def test_scalar_passthrough(self):
        g0, g1, _, _ = _make_pair()
        td = {"hidden_states": torch.zeros(2, 3), "flag": 7, "__msg_type__": "proxy"}
        g0.send_tensor_dict(dict(td), all_gather_group=None)
        out = g1.recv_tensor_dict_typed("proxy", all_gather_group=None)
        self.assertEqual(out["flag"], 7)

    def test_batch_key_content_addressing(self):
        # With an explicit batch_key the rendezvous is identity-addressed: the
        # consumer pulls the proxy belonging to the exact batch it built,
        # independent of arrival order / positional sequence.
        g0, g1, _, _ = _make_pair()
        a = torch.full((1, 4), 1.0)
        b = torch.full((1, 4), 2.0)
        # Producer ships two batches, keyed by identity, in order a then b.
        g0.send_tensor_dict(
            {"hidden_states": a, "__msg_type__": "proxy"},
            all_gather_group=None,
            batch_key="rA#0#1",
        )
        g0.send_tensor_dict(
            {"hidden_states": b, "__msg_type__": "proxy"},
            all_gather_group=None,
            batch_key="rB#0#1",
        )
        # Consumer pulls them out of order, by key -> must still get the right one.
        out_b = g1.recv_tensor_dict_typed("proxy", batch_key="rB#0#1")
        out_a = g1.recv_tensor_dict_typed("proxy", batch_key="rA#0#1")
        self.assertTrue(torch.equal(out_a["hidden_states"], a))
        self.assertTrue(torch.equal(out_b["hidden_states"], b))

    def test_batch_key_does_not_consume_positional_seq(self):
        # A keyed proxy send must not advance the positional counter, so a
        # later non-keyed stream (e.g. "output") stays aligned at seq 0.
        g0, g1, _, _ = _make_pair()
        k = torch.full((1, 2), 9.0)
        g0.send_tensor_dict(
            {"hidden_states": k, "__msg_type__": "proxy"},
            all_gather_group=None,
            batch_key="rid#0#1",
        )
        o = torch.full((1, 2), 3.0)
        g0.send_tensor_dict({"next_token_ids": o, "__msg_type__": "output"})
        # output is seq-based; recv without a key must find it at seq 0.
        out = g1.recv_tensor_dict_typed("output")
        self.assertTrue(torch.equal(out["next_token_ids"], o))
        got_k = g1.recv_tensor_dict_typed("proxy", batch_key="rid#0#1")
        self.assertTrue(torch.equal(got_k["hidden_states"], k))

    def test_msg_type_sequencing(self):
        g0, g1, _, _ = _make_pair()
        a = torch.full((1, 4), 1.0)
        b = torch.full((1, 4), 2.0)
        g0.send_tensor_dict(
            {"hidden_states": a, "__msg_type__": "proxy"}, all_gather_group=None
        )
        g0.send_tensor_dict(
            {"hidden_states": b, "__msg_type__": "proxy"}, all_gather_group=None
        )
        out_a = g1.recv_tensor_dict_typed("proxy")
        out_b = g1.recv_tensor_dict_typed("proxy")
        self.assertTrue(torch.equal(out_a["hidden_states"], a))
        self.assertTrue(torch.equal(out_b["hidden_states"], b))

    def test_send_slices_by_attn_tp(self):
        # With all_gather_group of size 2, rank 1 should ship only its slice.
        g0, _, t0, _ = _make_pair()
        full = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        ag = _FakeAG(world_size=2, rank_in_group=1)
        g0.send_tensor_dict(
            {"hidden_states": full, "__msg_type__": "proxy"}, all_gather_group=ag
        )
        pushed = t0.last_pushed["proxy:0"]["hidden_states"]
        expected = full.reshape(2, -1)[1]
        self.assertTrue(torch.equal(pushed, expected))
        self.assertEqual(pushed.numel(), full.numel() // 2)

    def test_recv_all_gather_restores_shape(self):
        g0, g1, _, _ = _make_pair()
        full = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        send_ag = _FakeAG(world_size=2, rank_in_group=0)
        g0.send_tensor_dict(
            {"hidden_states": full, "__msg_type__": "proxy"}, all_gather_group=send_ag
        )
        # recv side: all_gather reconstructs the flat full tensor.
        recv_ag = _FakeAG(
            world_size=2, rank_in_group=0, full_tensors={6: full.reshape(-1)}
        )
        out = g1.recv_tensor_dict_typed("proxy", all_gather_group=recv_ag)
        self.assertEqual(tuple(out["hidden_states"].shape), (2, 6))
        self.assertTrue(torch.equal(out["hidden_states"], full))

    def test_ring_pyobj_relay(self):
        g0, g1, _, _ = _make_pair()
        reqs = [{"rid": "a"}, {"rid": "b"}]
        g0.send_pyobj_next(reqs, tag="reqs")
        got = g1.recv_pyobj_prev(tag="reqs")
        self.assertEqual(got, reqs)

    def test_ring_pyobj_tags_are_independent_fifos(self):
        # The PD overlay multiplexes many logical streams (reqs / bootstrap /
        # xfer / consensus_* / retract / prealloc / release) on one edge. Each
        # tag must be an independent FIFO so a different number/order of sends
        # on one tag never desynchronizes another.
        g0, g1, _, _ = _make_pair()
        # Interleave sends across tags in a scrambled order.
        g0.send_pyobj_next(["reqs0"], tag="reqs")
        g0.send_pyobj_next(["boot0"], tag="bootstrap")
        g0.send_pyobj_next(["reqs1"], tag="reqs")
        g0.send_pyobj_next(["xfer0"], tag="xfer")
        g0.send_pyobj_next(["boot1"], tag="bootstrap")
        # Receive each tag in its own order; values must match per-tag FIFO.
        self.assertEqual(g1.recv_pyobj_prev(tag="bootstrap"), ["boot0"])
        self.assertEqual(g1.recv_pyobj_prev(tag="reqs"), ["reqs0"])
        self.assertEqual(g1.recv_pyobj_prev(tag="xfer"), ["xfer0"])
        self.assertEqual(g1.recv_pyobj_prev(tag="reqs"), ["reqs1"])
        self.assertEqual(g1.recv_pyobj_prev(tag="bootstrap"), ["boot1"])

    def test_transport_stats_passthrough(self):
        g0, g1, _, _ = _make_pair()
        g0.send_pyobj_next(["x"], tag="reqs")
        stats = g0.transport_stats()
        self.assertEqual(stats["stage"], "0/2")
        self.assertIn("pushes", stats)

    def test_identity_surface(self):
        g0, g1, _, _ = _make_pair()
        self.assertTrue(g0.is_first_rank)
        self.assertFalse(g0.is_last_rank)
        self.assertFalse(g1.is_first_rank)
        self.assertTrue(g1.is_last_rank)
        self.assertEqual(g0.rank_in_group, 0)
        self.assertEqual(g1.world_size, 2)


class TestStageRendezvous(unittest.TestCase):
    def test_register_and_collect(self):
        from sglang.srt.disaggregation.mori.stage_rendezvous import (
            register_and_collect,
            start_registry,
        )

        num_stages, tp_size = 2, 1
        expected = num_stages * tp_size
        server = start_registry("127.0.0.1", 0, expected)
        port = server.server_address[1]

        results = {}

        def worker(stage_id):
            results[stage_id] = register_and_collect(
                "127.0.0.1",
                port,
                stage_id=stage_id,
                attn_tp_rank=0,
                endpoint=f"tcp://127.0.0.1:{5000 + stage_id}",
                expected=expected,
                timeout_s=30,
            )

        threads = [
            threading.Thread(target=worker, args=(s,)) for s in range(num_stages)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        server.shutdown()

        for stage_id in range(num_stages):
            table = results[stage_id]
            self.assertEqual(len(table), expected)
            self.assertEqual(table[(0, 0)], "tcp://127.0.0.1:5000")
            self.assertEqual(table[(1, 0)], "tcp://127.0.0.1:5001")


if __name__ == "__main__":
    unittest.main()
