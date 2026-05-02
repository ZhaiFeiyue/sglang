"""Monkey-patch Scheduler to handle WaveInfo markers in process_input_requests.

Called explicitly by data_parallel_controller when wave dispatch is enabled.
Since the controller and schedulers run in separate processes, we apply
the patch in each scheduler process via a sitecustomize-style approach:
the controller sets an env var, and each scheduler checks for it on init.
"""

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_scheduler():
    """Patch Scheduler to respect wave boundaries.

    1. process_input_requests: intercept WaveInfo markers
    2. _get_new_batch_prefill_raw: limit batch to wave's num_reqs
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_wave import WaveInfo

    # Patch 1: intercept WaveInfo in process_input_requests
    _original_process = Scheduler.process_input_requests

    def _patched_process_input_requests(self, recv_reqs):
        filtered = []
        for req in recv_reqs:
            if isinstance(req, WaveInfo):
                if not hasattr(self, "current_wave_info"):
                    self.current_wave_info = None
                self.current_wave_info = req
            else:
                filtered.append(req)
        return _original_process(self, filtered)

    Scheduler.process_input_requests = _patched_process_input_requests

    # Patch 2: limit batch to wave's num_reqs
    _original_get_batch = Scheduler._get_new_batch_prefill_raw

    def _patched_get_new_batch_prefill_raw(self, *args, **kwargs):
        wave = getattr(self, "current_wave_info", None)
        saved_max = self.server_args.prefill_max_requests
        if wave is not None and wave.num_reqs > 0:
            # Cap this round to only the wave's reqs
            self.server_args.prefill_max_requests = wave.num_reqs
        try:
            result = _original_get_batch(self, *args, **kwargs)
        finally:
            self.server_args.prefill_max_requests = saved_max
            # Clear wave after consuming
            if hasattr(self, "current_wave_info"):
                self.current_wave_info = None
        return result

    Scheduler._get_new_batch_prefill_raw = _patched_get_new_batch_prefill_raw

    logger.info("Wave patch: Scheduler patched for WaveInfo + batch size limit")


def maybe_patch():
    """Auto-patch if SGLANG_WAVE_DISPATCH=1 is set."""
    if os.environ.get("SGLANG_WAVE_DISPATCH") == "1":
        patch_scheduler()
