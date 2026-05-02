"""Wave dispatch: DP coordinator batches requests into waves.

A wave is a group of requests dispatched together to a DP rank.
When wave dispatch is enabled, the scheduler strictly processes only
the requests in the current wave before accepting new ones.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class WaveInfo:
    """A wave of requests dispatched to a single DP rank.

    Sent as a single zmq message from the DP coordinator to the scheduler.
    Contains the actual request objects (TokenizedGenerateReqInput, etc.)
    bundled together, plus metadata.
    """

    wave_id: int
    reqs: list
    dispatch_time: float = 0.0


# ---------------------------------------------------------------------------
# Assignment strategies
#
# Signature:  (buffer: list, dp_size: int, **kwargs) -> (List[list], int)
#   Input:
#     buffer  – the full pending request buffer; strategy may inspect
#               buffer length, request attributes (ISL, etc.) to decide
#               how many and which reqs to schedule in this wave.
#     dp_size – number of DP ranks
#     **kwargs – strategy-specific parameters forwarded from WavePlanner
#   Output:
#     rank_reqs   – List of length dp_size, where element[rank] is the
#                   list of reqs assigned to that rank.
#     num_consumed – how many reqs from the front of buffer were consumed.
#                   WavePlanner will remove buffer[:num_consumed] after
#                   the call.  This allows the strategy to leave some
#                   reqs in the buffer for the next wave.
# ---------------------------------------------------------------------------

AssignmentStrategy = Callable[..., Tuple[List[list], int]]


def assign_one_per_rank(buffer: list, dp_size: int, **kwargs) -> Tuple[List[list], int]:
    """Assign at most one req per rank, round-robin. Default strategy."""
    rr = kwargs.get("round_robin_counter", 0)
    rank_reqs: List[list] = [[] for _ in range(dp_size)]
    n = min(len(buffer), dp_size)
    for i in range(n):
        rank_reqs[(rr + i) % dp_size].append(buffer[i])
    return rank_reqs, n


def assign_round_robin(buffer: list, dp_size: int, **kwargs) -> Tuple[List[list], int]:
    """Consume all buffered reqs and distribute via round-robin."""
    rr = kwargs.get("round_robin_counter", 0)
    rank_reqs: List[list] = [[] for _ in range(dp_size)]
    for req in buffer:
        rank_reqs[rr % dp_size].append(req)
        rr = (rr + 1) % dp_size
    return rank_reqs, len(buffer)


class WavePlanner:
    """Accumulates requests and flushes them as waves to DP ranks.

    Used by the DP coordinator. Incoming requests are appended to an
    internal buffer.  On each flush, the assignment strategy inspects
    the buffer (size, req attributes, etc.) and decides how many reqs
    to consume and how to distribute them across ranks.  Unconsumed
    reqs remain in the buffer for the next wave.

    The assignment strategy determines how reqs are distributed across
    ranks. Pass a different strategy to the constructor to customise.
    """

    def __init__(
        self,
        dp_size: int,
        strategy: AssignmentStrategy = assign_one_per_rank,
    ):
        self.dp_size = dp_size
        self.strategy = strategy
        self.wave_counter = 0
        self.buffer: list = []

    def add_req(self, req) -> None:
        self.buffer.append(req)

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)

    def flush_wave(self, round_robin_counter: int) -> tuple:
        """Let the strategy consume reqs from the buffer and build per-rank WaveInfo.

        Returns:
            (wave_infos, new_round_robin_counter)
            wave_infos: list of WaveInfo, one per rank (may have empty reqs list)
        """
        wave_id = self.wave_counter
        self.wave_counter += 1
        now = time.monotonic()

        rank_reqs, num_consumed = self.strategy(
            self.buffer,
            self.dp_size,
            round_robin_counter=round_robin_counter,
        )
        del self.buffer[:num_consumed]

        wave_infos = [
            WaveInfo(wave_id=wave_id, reqs=rank_reqs[rank], dispatch_time=now)
            for rank in range(self.dp_size)
        ]

        new_rr = (round_robin_counter + num_consumed) % self.dp_size
        return wave_infos, new_rr
