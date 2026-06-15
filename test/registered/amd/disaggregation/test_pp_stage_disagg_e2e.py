"""GPU end-to-end smoke for PP stage disaggregation.

Launches ``NUM_STAGES`` independent SGLang stage microservices (each its own
NCCL world, TP only) connected over the mori activation transport, then drives
a generation against stage 0 (the entry/exit stage in the ring). Skips unless
enough GPUs are visible. RDMA device is taken from SGLANG_TEST_RDMA_DEVICE.
"""

import os
import subprocess
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    try_cached_model,
)

register_amd_ci(est_time=900, suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd")


class TestPPStageDisaggE2E(unittest.TestCase):
    num_stages = 2
    tp = 1
    base_port = 31100
    coord_port = 9123
    host = "127.0.0.1"

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("torch.cuda is not available.")
            if torch.cuda.device_count() < cls.num_stages * cls.tp:
                raise unittest.SkipTest(
                    f"PP stage disagg test needs >= {cls.num_stages * cls.tp} GPUs."
                )
        except Exception as e:
            raise unittest.SkipTest(f"torch unavailable: {e}")

        cls.model = try_cached_model(
            os.environ.get(
                "SGLANG_PP_STAGE_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            )
        )
        rdma = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
        cls.ib_args = (
            ["--pp-activation-ib-device", rdma, "--disaggregation-ib-device", rdma]
            if rdma
            else []
        )

        cls.procs = []
        for k in range(cls.num_stages):
            cls.procs.append(cls._launch_stage(k))

        cls.stage0_url = f"http://{cls.host}:{cls.base_port}"
        cls._wait_ready(cls.stage0_url, cls.procs[0])

    @classmethod
    def _launch_stage(cls, k: int):
        args = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--host",
            "0.0.0.0",
            "--port",
            str(cls.base_port + k),
            "--tp-size",
            str(cls.tp),
            "--base-gpu-id",
            str(k * cls.tp),
            "--trust-remote-code",
            "--disable-custom-all-reduce",
            "--pp-stage-disaggregation",
            "--pp-stage-id",
            str(k),
            "--pp-num-stages",
            str(cls.num_stages),
            "--pp-activation-bootstrap-host",
            cls.host,
            "--pp-activation-bootstrap-port",
            str(cls.coord_port),
        ] + cls.ib_args
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    @classmethod
    def _wait_ready(cls, url, proc):
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"stage process exited early ({proc.returncode})")
            try:
                if requests.get(url + "/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(2)
        raise TimeoutError("stage 0 did not become ready")

    @classmethod
    def tearDownClass(cls):
        for p in getattr(cls, "procs", []):
            p.terminate()
        for p in getattr(cls, "procs", []):
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()

    def test_generate_smoke(self):
        resp = requests.post(
            self.stage0_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        out = resp.json()
        self.assertIn("text", out)
        self.assertGreater(len(out["text"]), 0)


if __name__ == "__main__":
    unittest.main()
