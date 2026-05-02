# RFC: Wave-Based DP Coordinator for Unified Multi-Rank Scheduling

- **Author**: ZhaiFeiyue
- **Date**: 2026-05-02
- **Status**: Draft
- **Related branch**: `tbo_awared_chunked_prefill`

---

## 1. Motivation

### 1.1 Current Architecture

SGLang's Data Parallel (DP) attention mode runs one independent `Scheduler`
process per DP rank. Requests are dispatched to ranks by
`DataParallelController` using coarse-grained load-balancing (round-robin,
total-tokens, etc.).

Each step follows this sequence:

```
[Each rank independently]
  get_new_batch_prefill()          # rank schedules its own waiting_queue
      ↓
  maybe_prepare_mlp_sync_batch()   # all_gather: first global view
      ↓ learns: global_num_tokens, can_run_tbo, forward_mode
  TboForwardBatchPreparer          # decides balanced-split vs two-chunk
      ↓
  execute_overlapped_operations    # TBO pipeline
```

### 1.2 Problems

#### P1 — TBO on/off is decided *after* scheduling

Two-Batch Overlap (TBO) requires all DP ranks to agree on whether to run TBO
for a given step. The decision is made via `all_gather` **after** each rank has
already built its batch:

```
rank0: scheduled bs=4  →  expects TBO
rank1: scheduled bs=1  →  local_can_run_tbo=False
                          ↓ all_gather min() = 0
                          All ranks fall back to non-TBO
                          rank0's 4-req batch wasted its TBO-aware chunking
```

#### P2 — TBO-aware chunked prefill is a local approximation

When `--enable_two_batch_overlap` and `--chunked_prefill_size` are both set,
`PrefillAdder` splits the chunk budget into two halves (`_tbo_half_target`) to
guide TBO's balanced-split. But this split is computed locally per rank,
without knowing whether other ranks will also satisfy TBO's minimum bs
requirement. If any rank cannot TBO, the whole half-budget accounting is
wasted.

#### P3 — Information arrives late, causing cascading patches

The scheduler accumulates global context across multiple stages:

| Stage | New information | Resulting patch |
|-------|----------------|-----------------|
| `get_new_batch_prefill` | local waiting_queue | chunk split heuristic |
| `all_gather` | global num_tokens, TBO feasibility | `tbo_split_seq_index` |
| `TboForwardBatchPreparer` | `extend_lens` distribution | balanced vs two-chunk |
| `ops_strategy` | layer config, forward_mode | delta_stages layout |

Each patch tries to correct a decision made with incomplete information.
This makes the codebase increasingly difficult to reason about and extend.

#### P4 — No prefix-cache-aware routing

`DataParallelController` routes requests by load alone; it does not consider
which rank already holds the prefix cache for a given request. This causes
redundant cache misses and uneven radix-tree utilization across ranks.

---

## 2. Proposal: Wave-Based DP Coordinator

### 2.1 Core Idea

Replace the current "each rank schedules independently, then reconcile via
all_gather" model with a **centralized Wave Coordinator** that:

1. Holds a global view of all pending requests and per-rank state.
2. Produces one `WavePlan` per step containing fully-specified per-rank batches.
3. Distributes `WavePlan` to each rank before forward.
4. Ranks execute without any scheduling-level decision or collective.

### 2.2 Wave Definition

A **Wave** is the atomic unit of work for one forward step across all DP ranks.

```python
@dataclass
class RankWave:
    rank_id: int
    reqs: List[Req]               # requests assigned to this rank
    extend_lens: List[int]        # per-req extend token counts (already chunked)
    tbo_enabled: bool             # same value for ALL ranks in a wave
    tbo_split_seq_index: int      # pre-computed, 0 if tbo_enabled=False
    global_num_tokens: List[int]  # token counts for all dp ranks (for MoE EP buffer sizing)
    forward_mode: ForwardMode     # same value for ALL ranks in a wave

@dataclass
class Wave:
    wave_id: int
    rank_waves: List[RankWave]    # one per DP rank, including idle ranks
```

**Wave invariants**:
- All `RankWave` entries share the same `tbo_enabled` and `forward_mode`.
- If any rank would have `bs < 2`, all ranks get `tbo_enabled=False`.
- Idle ranks receive a `RankWave` with `reqs=[]`, `forward_mode=IDLE`.
- All ranks advance wave index in lockstep.

### 2.3 Coordinator Responsibilities

```
DPCoordinator (replaces DataParallelController)
  │
  ├── Global waiting queue (all requests, all ranks)
  ├── Per-rank state cache:
  │     last_global_num_tokens  (from previous wave result)
  │     prefix_cache_digest     (hash-to-rank mapping for routing)
  │     available_kv_budget     (token pool availability per rank)
  │
  ├── Wave construction algorithm (one call per step):
  │     1. Route requests to ranks (prefix-cache-aware)
  │     2. Apply chunked_prefill_size budget per rank
  │     3. Check TBO feasibility: all ranks bs >= 2?
  │          yes → tbo_enabled=True, compute tbo_split_seq_index per rank
  │               with TBO-aware half-budget (no _tbo_half_target patch needed)
  │          no  → tbo_enabled=False for entire wave
  │     4. Fill idle ranks with RankWave(reqs=[], forward_mode=IDLE)
  │     5. Broadcast Wave to all ranks via shared memory or zmq
  │
  └── Receives wave completion signals from ranks
        (token counts, cache updates, finished requests)
```

### 2.4 Rank Scheduler (simplified)

With a `WavePlan` in hand, each rank's Scheduler reduces to:

```python
def run_one_step(self, rank_wave: RankWave):
    batch = ForwardBatch.from_wave(rank_wave)   # no scheduling logic
    # No TboForwardBatchPreparer conditions
    # No all_gather for TBO decision
    # tbo_split_seq_index already set by Coordinator
    result = self.model_worker.forward(batch)
    self.report_completion(result)              # feed back to Coordinator
```

The only remaining collective is MoE EP AlltoAll (dispatch/combine) —
this is pure compute, not a scheduling decision.

---

## 3. Detailed Design

### 3.1 TBO-Aware Wave Construction

```python
def _construct_wave(self, pending_reqs, per_rank_budgets):
    # Step 1: prefix-cache-aware routing
    rank_assignments = self._route_by_prefix_cache(pending_reqs)

    # Step 2: apply chunk budget per rank
    rank_batches = {}
    for rank, reqs in rank_assignments.items():
        budget = per_rank_budgets[rank]
        rank_batches[rank] = self._apply_chunk_budget(reqs, budget)

    # Step 3: TBO feasibility — single decision point
    all_bs = [len(b.reqs) for b in rank_batches.values()]
    tbo_enabled = (
        is_tbo_enabled()
        and all(bs >= 2 for bs in all_bs)
        and self._all_same_forward_mode(rank_batches)
    )

    # Step 4: if TBO enabled, split extend_lens with half-budget
    if tbo_enabled:
        for rank, batch in rank_batches.items():
            batch.tbo_split_seq_index = compute_split_seq_index(
                forward_mode=batch.forward_mode,
                extend_lens=batch.extend_lens,
            )
            # Apply TBO-aware half-budget (no per-rank guesswork)
            batch.extend_lens = self._apply_tbo_half_budget(
                batch.extend_lens, chunk_size=per_rank_budgets[rank]
            )

    # Step 5: fill idle ranks
    for rank in range(self.dp_size):
        if rank not in rank_batches:
            rank_batches[rank] = IdleRankWave(rank)

    global_num_tokens = [
        sum(b.extend_lens) for b in rank_batches.values()
    ]
    return Wave(rank_waves=rank_batches, tbo_enabled=tbo_enabled,
                global_num_tokens=global_num_tokens)
```

### 3.2 Prefix-Cache-Aware Routing

The Coordinator maintains a `prefix_routing_table`:

```
prefix_hash → preferred_dp_rank
```

When a new request arrives:
1. Compute prefix hash (same as radix cache key).
2. If hash exists in table → route to that rank (cache hit).
3. Else → route by least-loaded rank, record in table.
4. On request completion → update table with new prefix coverage.

This is the same information radix cache already computes; the Coordinator
just needs a lightweight shadow copy (hashes only, no token data).

### 3.3 Eliminating all_gather for Scheduling

| Information | Current source | Wave source |
|---|---|---|
| `global_num_tokens` | all_gather | Coordinator-computed, in `Wave` |
| `can_run_tbo` | all_gather min() | Coordinator decision, in `Wave` |
| `tbo_split_seq_index` | post-allgather TboForwardBatchPreparer | Coordinator-computed, in `Wave` |
| `forward_mode_agree` | all_gather check | Coordinator enforces in wave construction |

The existing `all_gather` in `prepare_mlp_sync_batch` can be removed entirely.
MoE EP AlltoAll (dispatch/combine) remains unchanged — it is a compute
collective, not a scheduling collective.

### 3.4 Idle Wave Handling

When a rank has no pending requests, Coordinator issues an `IdleRankWave`.
The rank executes an idle forward (existing `ForwardMode.IDLE` path),
contributing 0 tokens to MoE AlltoAll while staying in sync with other ranks.

This is cleaner than the current approach where idle ranks are discovered
implicitly via `max(global_num_tokens) > 0` after all_gather.

---

## 4. Migration Plan

Given the scope of the change, a phased approach is recommended:

### Phase 0 (done) — Local TBO-aware chunking
- Branch: `tbo_awared_chunked_prefill`
- `PrefillAdder._tbo_half_target` approximates TBO balance locally.
- all_gather still exists.
- Improvement: reduces two-chunk fallback frequency for multi-req batches.

### Phase 1 — Coordinator-side TBO feasibility pre-check
- `DataParallelController` tracks per-rank queue depth (1 int per rank).
- Before dispatching a new request, check: "will all ranks have bs >= 2?"
- If not, suppress TBO-aware chunking for this wave via a flag in the request.
- Still uses all_gather; no structural change to Scheduler.

### Phase 2 — Wave plan with pre-computed TBO fields
- Coordinator constructs `WavePlan` with `tbo_enabled`, `tbo_split_seq_index`,
  `global_num_tokens` embedded.
- Rank Scheduler reads these from the plan instead of re-computing.
- all_gather still runs but its TBO-related outputs are ignored (used for
  sanity checking, then removed in Phase 3).

### Phase 3 — Remove all_gather
- MoE EP buffer sizing uses `global_num_tokens` from WavePlan.
- `prepare_mlp_sync_batch` is removed.
- Ranks run fully plan-driven.

### Phase 4 — Prefix-cache-aware routing
- Coordinator maintains `prefix_routing_table`.
- Requests routed to cache-warm ranks.

---

## 5. Impact Analysis

### Benefits

| Area | Current | Wave |
|------|---------|------|
| TBO feasibility | decided post-scheduling, may waste chunking | decided pre-scheduling, always consistent |
| Two-chunk fallback | triggered by imbalanced local batches | eliminated for cross-rank imbalance |
| Prefix cache locality | random/load-based routing | routing follows prefix hash |
| Code complexity | cascading conditions in Scheduler | Scheduler is pure executor |
| all_gather overhead | 1× per step for scheduling | removed (only compute collectives remain) |

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Coordinator becomes scheduling bottleneck | Wave construction is O(dp_size × batch_size), CPU-only, sub-millisecond |
| Coordinator single point of failure | Existing watchdog/process-restart applies |
| Cross-rank prefix cache consistency | Shadow table is best-effort; miss falls back to recompute |
| Phased migration breaks existing tests | Phase 0-1 are backward-compatible; Phase 2+ behind feature flag |

---

## 6. Open Questions

1. **Coordinator placement**: Same process as `DataParallelController`, or a
   separate thread? Co-locating with the controller is simplest and avoids IPC.

2. **Mixed prefill/decode waves**: When some ranks are in decode and others
   have new prefill requests, should the Coordinator issue a prefill wave
   (new reqs only) or a mixed wave? Current SGLang behavior prioritizes
   prefill; Wave model should preserve this.

3. **Chunked request continuity across waves**: A chunked request spans
   multiple waves. The Coordinator must track `chunked_req` state per rank and
   resume it in the next wave. Equivalent to current `scheduler.chunked_req`,
   just owned by Coordinator.

4. **Speculative decoding interaction**: Spec decoding has its own DP sync
   logic (`speculative_skip_dp_mlp_sync`). Phase 1-2 should leave spec
   decoding paths unchanged.

5. **CUDA Graph compatibility**: With Wave, the Coordinator can guarantee
   consistent `bs` across waves that trigger graph capture, potentially
   improving capture rate and eliminating the current
   `capture_one_batch_size` complexity in `TboCudaGraphRunnerPlugin`.

---

## 7. References

- SGLang TBO implementation: `python/sglang/srt/batch_overlap/two_batch_overlap.py`
- DP attention sync: `python/sglang/srt/managers/scheduler_dp_attn_mixin.py`
- Current DP controller: `python/sglang/srt/managers/data_parallel_controller.py`
- Chunked prefill policy: `python/sglang/srt/managers/schedule_policy.py`
- Phase 0 implementation: branch `tbo_awared_chunked_prefill`, commit `be3841208`
