"""Scheduler Wave: batched dispatch with pre-computed TBO decisions.

The DP coordinator accumulates incoming requests, assigns them to DP ranks,
computes TBO split metadata, and dispatches them as a "wave". Each scheduler
receives its portion of requests followed by a WaveInfo marker containing
the pre-computed TBO split decision, allowing it to skip the all_gather.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class WaveInfo:
    """Marker sent after a wave's requests to carry TBO metadata.

    The scheduler checks for this in recv_requests. When present, the
    pre-computed TBO split is used instead of running all_gather.
    """

    wave_id: int
    # Number of reqs dispatched to this rank in this wave
    num_reqs: int
    # Pre-computed TBO split (None = TBO disabled or can't run balanced)
    tbo_split_seq_index: Optional[int]
    # Whether TBO can run for this wave
    can_run_tbo: bool
    # Timestamp for debugging
    dispatch_time: float = 0.0


def compute_tbo_split_seq_index(
    extend_lens: Sequence[int],
    threshold: float = 0.48,
) -> Optional[int]:
    """Compute the TBO balanced split index for a list of extend lengths.

    Returns the seq index where we split, or None if two-chunk is needed.
    Mirrors the logic in two_batch_overlap._split_extend_seqs.
    """
    if not extend_lens or len(extend_lens) < 2:
        return None

    # Balanced split: find index where left/right sums are closest
    overall_sum = sum(extend_lens)
    left_sum = 0
    min_diff = float("inf")
    best_index = 0

    for i in range(1, len(extend_lens)):
        left_sum += extend_lens[i - 1]
        right_sum = overall_sum - left_sum
        diff = abs(left_sum - right_sum)
        if diff <= min_diff:
            min_diff = diff
            best_index = i
        else:
            break

    # Check if balanced split is good enough (not triggering two-chunk)
    left_sum = sum(extend_lens[:best_index])
    if left_sum < overall_sum * threshold or left_sum > overall_sum * (1 - threshold):
        # Would trigger two-chunk fallback — balanced split is too uneven
        return None

    return best_index


class WavePlanner:
    """Plans waves of requests for DP ranks.

    Used by the DP coordinator to accumulate requests, assign them to
    ranks, and compute TBO split decisions.
    """

    def __init__(
        self,
        dp_size: int,
        chunk_budget_per_rank: int,
        page_size: int = 1,
        tbo_enabled: bool = False,
        tbo_threshold: float = 0.48,
        wave_timeout_s: float = 0.001,
        min_reqs_per_wave: int = 1,
    ):
        self.dp_size = dp_size
        self.chunk_budget = chunk_budget_per_rank
        self.page_size = page_size
        self.tbo_enabled = tbo_enabled
        self.tbo_threshold = tbo_threshold
        self.wave_timeout_s = wave_timeout_s
        # For TBO pairing, we want at least 2 reqs per rank
        self.min_reqs_per_wave = 2 if tbo_enabled else min_reqs_per_wave

        self.wave_counter = 0
        self.pending_reqs: list = []
        self.last_flush_time = time.monotonic()

    def add_req(self, req) -> None:
        """Add a request to the pending buffer."""
        self.pending_reqs.append(req)

    def should_flush(self) -> bool:
        """Check if we should flush pending reqs as a wave."""
        if not self.pending_reqs:
            # No reqs but timeout elapsed — flush empty wave for sync
            return time.monotonic() - self.last_flush_time > self.wave_timeout_s * 10
        if len(self.pending_reqs) >= self.dp_size * self.min_reqs_per_wave:
            return True
        if time.monotonic() - self.last_flush_time > self.wave_timeout_s:
            return True
        return False

    def _get_req_tokens(self, req) -> int:
        """Get input token count for a request (TokenizedGenerateReqInput)."""
        return len(req.input_ids)

    def _pack_pairs_for_tbo(self, reqs: list) -> List[List]:
        """Pack reqs into balanced pairs for TBO.

        Goal: each pair has 2 reqs with similar ISL that together fit in
        chunk_budget, so TBO balanced split at the seq boundary yields
        ~equal micro-batches. Never add a 3rd req — that risks breaking
        the balance.

        Strategy:
        1. Sort by ISL
        2. Pair adjacent reqs (most similar ISL) if they fit in budget
        3. Unpaired reqs go solo (two-chunk, unavoidable)
        """
        if not self.tbo_enabled or self.chunk_budget <= 0:
            return [[r] for r in reqs]

        sorted_reqs = sorted(reqs, key=self._get_req_tokens)
        used = [False] * len(sorted_reqs)
        groups: List[List] = []

        # Pair adjacent (similar ISL) reqs that fit together
        i = 0
        while i < len(sorted_reqs) - 1:
            if used[i]:
                i += 1
                continue
            j = i + 1
            while j < len(sorted_reqs) and used[j]:
                j += 1
            if j >= len(sorted_reqs):
                break

            isl_i = self._get_req_tokens(sorted_reqs[i])
            isl_j = self._get_req_tokens(sorted_reqs[j])

            if isl_i + isl_j <= self.chunk_budget:
                groups.append([sorted_reqs[i], sorted_reqs[j]])
                used[i] = True
                used[j] = True
                i = j + 1
            else:
                i += 1

        # Collect unpaired reqs as solo groups
        for k in range(len(sorted_reqs)):
            if not used[k]:
                groups.append([sorted_reqs[k]])

        return groups

    def flush_wave(self, round_robin_counter: int) -> tuple:
        """Build a wave from pending reqs with TBO-aware packing.

        Returns:
            (assignments, wave_infos, new_round_robin_counter)
            assignments: list of (rank, req) pairs
            wave_infos: list of WaveInfo, one per rank
        """
        wave_id = self.wave_counter
        self.wave_counter += 1
        self.last_flush_time = time.monotonic()

        if self.tbo_enabled and len(self.pending_reqs) >= 2:
            rank_reqs = self._flush_tbo_aware(round_robin_counter)
        else:
            rank_reqs = self._flush_round_robin(round_robin_counter)

        self.pending_reqs.clear()

        # Compute TBO split for each rank
        wave_infos = []
        now = time.monotonic()

        for rank in range(self.dp_size):
            reqs = rank_reqs[rank]

            tbo_split = None
            can_run_tbo = False

            if self.tbo_enabled and len(reqs) >= 2:
                extend_lens = [len(r.input_ids) for r in reqs]
                tbo_split = compute_tbo_split_seq_index(extend_lens, self.tbo_threshold)
                can_run_tbo = tbo_split is not None

            wave_infos.append(
                WaveInfo(
                    wave_id=wave_id,
                    num_reqs=len(reqs),
                    tbo_split_seq_index=tbo_split,
                    can_run_tbo=can_run_tbo,
                    dispatch_time=now,
                )
            )

        # Build flat assignment list
        assignments = []
        for rank, reqs in enumerate(rank_reqs):
            for req in reqs:
                assignments.append((rank, req))

        # Advance round-robin counter by total reqs dispatched
        new_rr = (round_robin_counter + sum(len(r) for r in rank_reqs)) % self.dp_size

        return assignments, wave_infos, new_rr

    def _flush_round_robin(self, rr: int) -> List[list]:
        """Simple round-robin assignment."""
        rank_reqs: List[list] = [[] for _ in range(self.dp_size)]
        for req in self.pending_reqs:
            rank_reqs[rr % self.dp_size].append(req)
            rr = (rr + 1) % self.dp_size
        return rank_reqs

    def _flush_tbo_aware(self, rr: int) -> List[list]:
        """TBO-aware assignment: pair reqs that fit in chunk_budget together.

        1. Pack reqs into pairs (largest + smallest that fits)
        2. Distribute pairs to ranks via round-robin
        """
        groups = self._pack_pairs_for_tbo(self.pending_reqs)

        rank_reqs: List[list] = [[] for _ in range(self.dp_size)]
        for group in groups:
            rank = rr % self.dp_size
            rank_reqs[rank].extend(group)
            rr = (rr + 1) % self.dp_size

        return rank_reqs
